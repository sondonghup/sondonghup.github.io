---
# multilingual page pair id, this must pair with translations of this page. (This name must be unique)
lng_pair: id_Examples
title: deep-learning-from-scratch2-chapter1

# post specific
# if not specified, .name will be used from _data/owner/[language].yml
author: Mr. Green's Workshop
# multiple category is not supported
category: jekyll
# multiple tag entries are possible
tags: [jekyll, sample, example post]
# thumbnail image for post
img: ":post_pic1.jpg"
# disable comments on this page
#comments_disable: true

# publish date
date: 2023-07-03 18:38:06 +0900

# seo
# if not specified, date will be used.
#meta_modify_date: 2022-08-16 18:38:06 +0900
# check the meta_common_description in _data/owner/[language].yml
#meta_description: ""

# optional
# if you enabled image_viewer_posts you don't need to enable this. This is only if image_viewer_posts = false
#image_viewer_on: true
# if you enabled image_lazy_loader_posts you don't need to enable this. This is only if image_lazy_loader_posts = false
#image_lazy_loader_on: true
# exclude from on site search
#on_site_search_exclude: true
# exclude from search engines
#search_engine_exclude: true
# to disable this page, simply set published: false or delete this file
#published: false
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### 1.1 수학과 파이썬 복습

***

#### 1.1.1 벡터화 행렬

벡터 : 크기와 방향을 가진 양(질량) -> 1차원 배열

행렬 : 2차원 배열

***

#### 1.1.2 행렬의 원소별 연산

형상이 같은 행렬의 연산

```
import numpy as np

W = np.array([[1, 3, 5], [2, 4, 6]])
X = np.array([[0, 1, 2], [1, 1, 1]])

print(f"W + X :\n{W + X}")

print(f"W * X :\n{W * X}")

>>> W + X :
[[1 4 7]
 [3 5 7]]
>>> W * X :
[[ 0  3 10]
 [ 2  4  6]]
```

***

#### 1.1.3 브로드캐스트

형상이 다른 행렬의 연산

```
import numpy as np

A = np.array([[1, 2], [3, 4]])

print(f"A * 10 :\n{A * 10}")

>>> A * 10 :
[[10 20]
 [30 40]]
```

```
import numpy as np

A = np.array([[1, 2], [3, 4]])
b = np.array([5, 1000])

print(f"A * b :\n{A * b}")

>>> A * b :
[[   5 2000]
 [  15 4000]]
```

***

#### 1.1.4 벡터의 내적과 행렬의 곱

벡터의 내적

```
import numpy as np

a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

print(f"a dot b :\n{np.dot(a, b)}")

>>> a dot b :
20
```

행렬의 곱

```
import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [5, 4]])

print(f"a matmul b :\n{np.matmul(a, b)}")

>>> a matmul b :
[[15 14]
 [35 34]]
```

### 1.2 신경망의 추론

***

#### 1.2.1 신경망 추론 전체 그림

입력층 : 입력을 받아들이는 층 -> 입력데이터의 크기를 설정

은닉층 : 특성을 추출하는 층

출력층 : 결과를 출력하는 층

각 층에는 적당한 개수의 뉴런이 배치 되어 있으며 가중치와 뉴런의 값을 각각 곱한 값과 bias를 합한 결과를 활성화 함수를 적용한 값이 다음 뉴런으로 전달

활성화 함수 : 선형 변환을 비선형 변환으로 바꿔줌

비선형 이여야하는 이유 : 딥러닝은 층이 깊어질 수록 좋은 효과를 볼 수 있는데 선형 변환 이면 한층으로 변환 할 수 있기 때문

-> y = 3x, y = 2x, y = 7x 이와 같이 3개의 층이 있다면 첫 층의 y가 y = 2x의 x로 전달되어 최종적으로 y = 7x의 y값으로 나온다

그러나 위 층은 y = 42x 와 같은 결과를 내므로 단 하나의 층으로 볼 수 있어 층이 깊어지지 않는다.

***

#### 1.2.2 계층으로 클래스화 및 순전파 구현

순전파 : 신경망 각 계층이 입력으로 부터 출력 방향으로 차례로 전파

역전파 : 순전파와 반대 방향으로 전파

### 1.3 신경망의 학습

***

#### 1.3.1 손실함수

손실 : 학습 단계의 특정 시점에서 신경망의 성능을 나타내는 척도 (실제 정답과의 차이)

손실 함수 : 신경망의 손실을 구하기 위한 함수

cross entropy loss : 다중 클래스 분류시에  사용

binary cross entropy loss : 이진 클래스 분류시에 사용

***

#### 1.3.2 오차역전파

가중치의 기울기를 효과적으로 계산하는 법

1. 결과값을 손실함수로 변환

2. 손실함수의 기울기를 수치 미분

3. 기울기가 0이 되는 지점까지 weight를 변화

오차역전파는 국소적 미분을 연쇄법칙에 따라 순전파의 반대 방향으로 전달한다.

```
# 곱셈 계층

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    # 순전파
    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    # 역전파
    def backward(self, dout):
        dx = dout * self.y   # x와 y를 교체
        dy = dout * self.x

        return dx, dy
```

```
# 덧셈 계층

class AddLayer:
    def __init__(self):
        pass

    # 순전파
    def forward(self, x, y):
        out = x + y

        return out

    # 역전파
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy
```

***

#### 1.3.6 가중치 갱신

예측값과 정답값 간의 차이인 손실 함수의 크기를 최소화시키는 파라미터를 찾는 것

경사하강법(SGD) : 1차 근삿값 발견용 최적화 알고리즘 경사의 절대값이 가장 낮은 극값에 이를때까지 반복하는 것

경사하강법의 단점 

local minima 

<img width="739" alt="image" src="https://github.com/sondonghup/programmers/assets/42092560/33d1975a-6167-45be-9395-6bc78570a18f">

global minima를 찾아야 하지만 local minima에 빠지는 경우

### 1.4 실습

```
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size

        # 가중치와 편향 초기화
        W1 = 0.01 * np.random.randn(I, H)
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H, O)
        b2 = np.zeros(O)

        # 계층 생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]
        self.loss_layer = SoftmaxWithLoss()

        # 모든 가중치와 기울기를 리스트에 모은다.
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
```

```
import sys
sys.path.append('..')  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

# 하이퍼파라미터 설정
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

# 데이터 읽기, 모델과 옵티마이저 생성
x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 학습에 사용하는 변수
data_size = len(x)
max_iters = data_size // batch_size
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 데이터 뒤섞기
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        batch_x = x[iters*batch_size:(iters+1)*batch_size]
        batch_t = t[iters*batch_size:(iters+1)*batch_size]

        # 기울기를 구해 매개변수 갱신
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 정기적으로 학습 경과 출력
        if (iters+1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| 에폭 %d |  반복 %d / %d | 손실 %.2f'
                  % (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0
```

<img width="572" alt="image" src="https://github.com/sondonghup/programmers/assets/42092560/c4bc433c-4662-4546-af94-d672e3a6da50">

<img width="535" alt="image" src="https://github.com/sondonghup/programmers/assets/42092560/fcf2e1fc-b9ec-4626-90e8-0a807f9f87a0">