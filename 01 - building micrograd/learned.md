### 연산자 오버로딩

연산자 오버로딩이란, +, * 같은 연산자를 우리가 만든(사용자 정의) 클래스에 맞게 동작하도록 직접 정의하는 것을 말한다.

#### 왜 __radd__, __rmul__이 필요할까?

문제는 순서가 바뀌면(즉, 내 클래스가 오른쪽에 오면) Python이 먼저 왼쪽(기본 타입)의 메서드를 호출한다는 점이다.

a + b는 a.__add__(b) 호출
b + a는 b.__add__(a) 호출

예를 들어, 3 + MyNumber(5)를 하면, int의 __add__는 MyNumber를 모른다. 그래서 실패한다.
이때 Python은 자동으로 MyNumber(5).__radd__(3)을 찾아서 호출한다.

#### 결론
따라서 연산자 오버로딩을 할 거면 모두 구현해야만 한다.


### 자연상수 e
https://m.blog.naver.com/galaxyenergy/222511035255

e는 '1에 한없이 가까운 수 자기 자신을 무한히 거듭제곱 한 수'이다.

정의 상 e의 미분값은 e 자기 자신이다.


### We can defining new Autograd functions by PYTORCH
https://docs.pytorch.org/tutorials/beginner/pytorch_with_examples.html#autograd
