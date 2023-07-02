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