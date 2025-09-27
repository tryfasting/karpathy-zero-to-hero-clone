'''
fineweb-bbc-news dataset (for srs pretraining)
https://huggingface.co/datasets/permutans/fineweb-bbc-news
Downloads and tokenizes the data and saves data shard to disk. (shard : 조각)
Run simply as:
$ uv run fineweb-bbc-news.py
Will save shards to the local directory 'fineweb-bbc-news'.

그 밖에 참고하면 좋을 데이터셋
https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
'''

import os
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset, get_dataset_config_names
from tqdm import tqdm
import logging

# ---------------------------------------------------
# Logging 설정 (print 중복 방지 및 디버깅 용이)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

output_dir = 'fineweb_bbc_news_tokens'      # 저장될 폴더 이름
dataset_name = "permutans/fineweb-bbc-news" # 사용할 데이터셋 이름
file_prefix = "bbc_news"               # 저장될 파일 접두사
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards
# shard_size는 1억(1e8) 토큰으로
# 100,000,000 토큰 * 2 바이트/토큰 = 200,000,000 바이트 ≈ 200 MB
# 한 파일당 약 200MB입니다. 일반적인 환경에 적합한 크기입니다.


# 전역 변수: 워커 프로세스에서 사용할 토크나이저 캐시
_TOKENIZER = None
_EOT = None

def _init_worker_tokenizer():
    # 멀티프로세싱 워커 초기화 시 1회만 토크나이저 생성 (오버헤드 감소)
    global _TOKENIZER, _EOT
    if _TOKENIZER is None:
        _TOKENIZER = tiktoken.get_encoding('gpt2')
        _EOT = _TOKENIZER.eot_token

def tokenize(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    # tokens = [eot] # the special <|endoftext|> token delimits all documents
    # tokens.extend(enc.encode_ordinary(doc["text"])) # list.extend : a = [1,2]; a.extend([3,4,5]); a >>> [1,2,3,4,5]
    global _TOKENIZER, _EOT
    if _TOKENIZER is None:
        # 안전장치: initializer가 실패했을 때 대비
        _TOKENIZER = tiktoken.get_encoding('gpt2')
        _EOT = _TOKENIZER.eot_token
    tokens = [_EOT]
    tokens.extend(_TOKENIZER.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), 'token dictionary too large for uint16'
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def _iter_streaming_with_pool(pool, iterable, chunksize=16):
    """
    스트리밍 모드에서 datasets iterable을 받아
    Pool.imap을 사용해 순차적으로 토크나이즈 결과를 yield.
    - datasets streaming은 길이를 알 수 없는 generator이므로,
      Pool.imap(tokenize, iterable, chunksize) 형태로 직접 연결.
    """
    for tok in pool.imap(tokenize, iterable, chunksize=chunksize):
        yield tok

if __name__ == '__main__':
    # 사용자 설정: 스트리밍 모드 on/off
    STREAMING = True  # True: RAM 사용량 최소화, 빠른 시작 / False: 전체 로드(메모리 사용 증가)

    # create the cache the local directory if it doesn't exist yet
    DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), output_dir)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # 다운로드 설정 목록 가져오기
    # 데이터셋의 모든 설정 이름 가져오기
    all_configs = get_dataset_config_names(dataset_name)
    # donwload 받을 리스트 생성
    configs_to_download = [c for c in all_configs if 'sample-350' in c]

    logger.info(f"Total {len(configs_to_download)} configs will be downloaded (samples excluded).")
    logger.info(f"Configs: {configs_to_download}")  # 한 번만 출력 (중복 방지)

    # tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
    nprocs = max(1, os.cpu_count()//2)
    with mp.Pool(nprocs, initializer=_init_worker_tokenizer) as pool:
        shard_index = 0
        # preallocate buffer to hold current shard
        # 현재 샤드에 채워질 토큰들을 임시로 담아둘 버퍼를 메인 프로세스의 메모리에 할당
        all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
        token_count = 0
        progress_bar = None

        # multiprocessing의 핵심 코드
        # 다운로드할 각 설정에 대해 반복
        for config_name in configs_to_download:
            print(f"\nProcessing config: {config_name}...")

            if STREAMING:
                # 현재 설정을 스트리밍 모드로 로드 (메모리 사용 최소화)
                # num_proc는 streaming과 함께 사용할 수 없으므로 제거
                fw_iter = load_dataset(
                    dataset_name,
                    name=config_name,
                    split='train',
                    streaming=True
                )
                # multiprocessing을 사용하여 현재 설정의 문서를 토큰화
                token_iter = _iter_streaming_with_pool(pool, fw_iter, chunksize=16)
            else:
                # 현재 설정을 로드. num_proc를 추가하여 로딩 속도를 높일 수 있음 (메모리 사용 증가)
                fw = load_dataset(dataset_name, name=config_name, split='train', num_proc=nprocs)
                # multiprocessing을 사용하여 현재 설정의 문서를 토큰화
                token_iter = pool.imap(tokenize, fw, chunksize=16)

            for tokens in token_iter:
                # is there enough space in the current shard for the new tokens?
                if token_count + len(tokens) < shard_size:
                    # simply append tokens to current shard
                    all_tokens_np[token_count: token_count + len(tokens)] = tokens
                    token_count += len(tokens)
                    # update progress bar
                    if progress_bar is None:
                        progress_bar = tqdm(total=shard_size, unit='tokens', desc=f'Shard {shard_index}')
                    progress_bar.update(len(tokens))
                else:
                    # write the current shard and start a new one
                    split = "val" if shard_index == 0 else 'train'
                    filename = os.path.join(DATA_CACHE_DIR, f'{file_prefix}_{split}_{shard_index:06d}')

                    # split the document into whatever fits in this shard; the remainder goes to next one
                    remainder = shard_size - token_count
                    if remainder > 0:
                        progress_bar.update(remainder)
                        all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]

                    write_datafile(filename, all_tokens_np)
                    shard_index += 1

                    # 현재 샤드의 progress bar를 닫고 다음 샤드를 위해 초기화
                    if progress_bar:
                        progress_bar.close()
                    progress_bar = None

                    # populate the next shard with the leftovers of the current doc
                    leftover_tokens = tokens[remainder:]
                    all_tokens_np[0:len(leftover_tokens)] = leftover_tokens
                    token_count = len(leftover_tokens)

        # write any remaining tokens as the last shard
        # 모든 설정 처리가 끝난 후 마지막 남은 토큰들을 저장
        if token_count != 0:
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{file_prefix}_{split}_{shard_index:06d}")
            write_datafile(filename, all_tokens_np[:token_count])
            if progress_bar:
                progress_bar.close()

    print("\nAll tasks completed.")

