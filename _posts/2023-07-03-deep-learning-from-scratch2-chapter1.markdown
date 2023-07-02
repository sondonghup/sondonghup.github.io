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

#### 1.1.1 벡터화 행렬

벡터 : 크기와 방향을 가진 양(질량) -> 1차원 배열

행렬 : 2차원 배열


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
#### 