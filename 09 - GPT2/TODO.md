### 모델 최적화를 위해 더 해보고 싶은 것.

1. Gradient Accumulation에서 발생할 수 있는 문제상황의 수식을 이해하고 코드로 구현 해결해보자.
https://unsloth.ai/blog/gradient

2. torch 2.5.0 버전에서는, torch.nn.parallel.distributed 클래스를 사용할 것을 권장한다.
아래의 공식문서를 보고 코드를 update해보자.
https://docs.pytorch.org/docs/2.5/generated/torch.nn.parallel.DistributedDataParallel.html#torch.nn.parallel.DistributedDataParallel