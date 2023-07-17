---
layout: post
title: deep-learning-from-scratch2-chapter3
subtitle: deep-learning-from-scratch2-chapter3
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### 3.1 추론 기반 기법과 신경망

단어를 벡터로 표현하는 방법 : 통계 기반 기법, 추론 기반 기법

***

#### 3.1.1 통계 기반 기법의 문제점

통계 기반 기법은 주변 단어의 빈도를 기초롤 단어를 표현 -> 이 방법은 대규모 말뭉치에서는 효과적이지 않음

통계 기반 기법 -> 모든 학습데이터를 한꺼번에 처리 합니다.

추론 기반 기법 -> 학습데이터의 일부분만을 사용하여 순차적으로 학습합니다.

***

#### 3.1.2 추론 기반 기법 개요

나는 [밥을] 먹고 있어 

[밥을]양 옆의 맥락을 이용해 추론을 하는 방식으로 학습을 합니다.


***

#### 3.1.3 신경망에서의 단어 처리

단어를 고정 길이의 벡터로 변환 -> 원핫 벡터로 변환

***

### 3.2 단순한 word2vec

CBOW : continuous bag of words

***

#### 3.2.1 CBOW 모델의 추론 처리 

<img width="616" alt="post-images_dscwinterstudy_5c3fa260-4517-11ea-a244-8f351b0c9082_fig-3-12" src="https://github.com/sondonghup/music_vae/assets/42092560/3f2ebbbd-0486-4f55-96f4-3640949b4f5b">

***

#### 3.5.2 skip-gram 모델

skip gram : 맥락과 타깃을 역전 시키는 모델

나는 [밥을] 먹고 있다. -> [나는] 밥을 [먹고] 있다.