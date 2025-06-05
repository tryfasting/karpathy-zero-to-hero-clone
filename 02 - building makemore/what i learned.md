### zip
: If the passed iterables have different lengths, the iterable with the least items decides the length of the new iterator.
=> if any one of these lists is shorter than the other then it will just halt and return.

example : in word 'emma' -> zip(emma, mma) -> e m / m m / m a


### dict.get
: .get argument '0' is Optional, 
A value to return if the specified key does not exist. 
values can be other thing's which can be added in dict.

example : b[bigram] = b.get(bigram, 0) + 1


### multinomial
: the probability of counts for each side of a k-sided die(:singular word of 'dice') rolled n times.


### keepdim
If keepdim is True, the output tensor is of the same size as input except in the dimension(s) dim where it is of size 1


### model smoothing
Model smoothing is a technique that uses algorithms to remove noise or outliers from data, making important patterns and trends easier to see. 
It helps create a smoother curve or result so you can better understand the overall direction of the data without being distracted by random fluctuations.


### torch.tensor Vs. torch.Tensor
torch.tensor : 'infers the dtype automatically,'
                입력된 데이터의 타입을 자동으로 추론해서 텐서를 만든다.

torch.Tensor : 'returns a torch.FloatTensor'
                항상 float타입으로 텐서를 만든다.