---
layout: post
title: deep-learning-from-scratch2-chapter2
subtitle: deep-learning-from-scratch2-chapter2
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### 2.1 수학과 파이썬 복습

***

#### 2.1.1 단어의 의미

말의 의미는 단어로 구성되므로 컴퓨터에게 단어의 의미를 이해시켜야 합니다. 

***

### 2.2 시소러스

시소러스 : 유의어  사전

***

#### 2.2.1 WordNet

WordNet : 가장 유명한 시소러스 

-> 유의어를 얻거나 단어 네트워크를 이용하여 단어 간의 유사도를 측정할 수 있습니다.

#### 2.2.2 시소러스의 문제점

1. 신조어 대응의 문제점
2. 엄청난 인적 비용
3. 단어의 미묘한 차이를 표현 불가능 

***

### 2.3 통계 기반 기법

말뭉치 : 자연어처리 연구를 위해 수집된 대량의 텍스트 데이터입니다.

***

#### 2.3.1 파이썬으로 말뭉치 전처리하기

***

#### 2.3.2 단어의 분산 표현

단어의 분산 표현 : 단어의 의미를 정확하게 파악할 수 있도록 하는 벡터 표현

#### 2.3.3 분포 가설

분포 가설 : 단어의 의미는 주변 단어에 의해 형성된다

-> 단어가 사용된 맥락이 그 의미를 형성 한다는 것

***

#### 2.3.4 동시발생 행렬

통계 기반 기법 : 각 단어의 맥락에 해당 되는 단어의 빈도를 세어 집계 하는 방법

<img src="./2023-07-05/fig 2-7.png">

***

#### 2.3.5 벡터 간 유사도

코사인 유사도 : 두벡터가 가리키는 방향

-> 코사인이 0도일때 1 : 두 벡터가 향하는 방향이 같다 (유사하다)

-> 코사인이 180도일때 -1 : 두벡터가 향하는 방향이 반대다 (유사하지 않다)

```
from laserembeddings import Laser
from scipy.spatial import distance

laser = Laser()

A = "고양이는 밥을 먹고 버스를 타고 집에 갑니다."
B = "개는 운동을 하다가 지쳐 잠에 듭니다."
C = "고양이가 버스를 운전하고 있습니다."

embeddings = laser.embed_sentences([A, B, C], ['ko', 'ko', 'ko'])

def cosine_similarity(source, target):
    return 1 - distance.cosine(source, target)

print(f'{A}\n{B}\n유사도 : {cosine_similarity(embeddings[0], embeddings[1])} \n')
print(f'{B}\n{C}\n유사도 : {cosine_similarity(embeddings[1], embeddings[2])} \n')
print(f'{A}\n{C}\n유사도 : {cosine_similarity(embeddings[0], embeddings[2])} \n')

>>>
고양이는 밥을 먹고 버스를 타고 집에 갑니다.
개는 운동을 하다가 지쳐 잠에 듭니다.
유사도 : 0.6368371844291687 

개는 운동을 하다가 지쳐 잠에 듭니다.
고양이가 버스를 운전하고 있습니다.
유사도 : 0.5950261354446411 

고양이는 밥을 먹고 버스를 타고 집에 갑니다.
고양이가 버스를 운전하고 있습니다.
유사도 : 0.7729980945587158 

```

#### 2.3.6 유사 단어의 랭킹 표시

bm25를 이용한 유사 단어의 랭킹

```
from rank_bm25 import BM25Okapi

corpus = [
    "고양이는 밥을 먹고 버스를 타고 집에 갑니다.",
    "개는 운동을 하다가 지쳐 잠에 듭니다.",
    "고양이가 버스를 운전하고 있습니다.",
]

tokenized_corpus = [doc.split(" ") for doc in corpus]

input = "버스 타고 집에가서 자야지"
tokenized_input = input.split(" ")

bm25 = BM25Okapi(tokenized_corpus)

bm25.get_top_n(tokenized_input, corpus, n = 1)

>>>
['고양이는 밥을 먹고 버스를 타고 집에 갑니다.']
```

***

### 2.4 통계 기반 기법 개선하기

***

#### 2.4.1 상호정보량

발생 횟수가 좋은 것이 아님을 보여줍니다.

-> 그, 버스를 생각해 보면 많은 문장에서 그 라는 단어가 많이 나올테지만 정보량을 따지면 버스가 더 중요합니다.

***

#### 2.4.2 차원감소

벡터의 차원을 줄이나 중요한 정보는 최대한 보존 하면서 줄이는 방법

원소 대부분이 0 인 행렬 또는 벡터의 중요한 축을 찾아 축소 하는 것