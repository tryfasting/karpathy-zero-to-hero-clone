### 모델 최적화를 위해 더 해보고 싶은 것.

1. Gradient Accumulation에서 발생할 수 있는 문제상황의 수식을 이해하고 코드로 구현 해결해보자.
https://unsloth.ai/blog/gradient

2. torch.nn.parallel.DistributedDataParallel, DDP 설정이 다른 LLM 모델은 어떻게 되어 있는지 확인하고, 비교 구현해보자.
https://docs.pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel
https://docs.pytorch.org/docs/2.5/notes/cuda.html#cuda-nn-ddp-instead

3. 데이터셋을 어떻게 building하는가?
https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1