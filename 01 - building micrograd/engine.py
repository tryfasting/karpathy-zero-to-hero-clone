import math
import numpy as np

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        '''
        x1 = Value(2.0, label='x1')
        x2 = Value(0.0, label='x2')
        '''
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label


    def __repr__(self):
        return f'Value(data={self.data})'


    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other),'+') 
        def _backward():
            ''' 
            (d out / d self.data) = 1.0 
            d ??? / d out = out.grad
            '''
            self.grad += 1.0 * out.grad 
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    

    def __radd__(self, other): # other + self
        return self + other 
    

    def __neg__(self): # -self
        return self * -1


    def __sub__(self, other): # self - other
        # '-other' is implemented by negation
        return self + (-other)
    

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other),'*') 

        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        
        out._backward = _backward
        return out


    def __rmul__(self,other): # other * self
        return self * other
    

    def __truediv__(self,other): # self / other
        '''나눗셈을 재정의'''
        return self * other** -1


    def __pow__(self,other):
        assert isinstance(other, (int,float)), 'only supporting int/float powers for now'
        out = Value(self.data**other, (self,), f'**{other}')

        def _backward():
            self.grad += out.grad * other * (self.data **(other- 1))
        out._backward = _backward

        return out 


    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Value(t, (self,), 'tanh')

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
    
 
    def relu(self):
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out


    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp') # = e ** x

        def _backward():
            self.grad += out.grad * out.data 
            # d (backward start point) / d(exp(x)) = d(backward start point) / d out * d out / d exp(x)
            # d(backward start point) / d out = out.grad
            # d out / d exp(x) = e ** x, 
            # 즉, e의 미분은 자기 자신이므로 out.data 유지
        out._backward = _backward
    
        return out
    

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            '''
            v : root node
            starting at root node, 
            go through all of its children(= leaf node), 
            lay them out from left to right.

            Recursively traverses all child nodes before adding the current node to the topo list.
            '''
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
