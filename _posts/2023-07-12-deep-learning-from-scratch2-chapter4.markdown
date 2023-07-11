---
layout: post
title: deep-learning-from-scratch2-chapter4
subtitle: deep-learning-from-scratch2-chapter4
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### 4.1 word2vec 개선

<img width="616" alt="post-images_dscwinterstudy_5c3fa260-4517-11ea-a244-8f351b0c9082_fig-3-12" src="https://github.com/sondonghup/music_vae/assets/42092560/3f2ebbbd-0486-4f55-96f4-3640949b4f5b">

CBOW 모델은 단어 2개를 맥락으로 사용해 이를 바탕으로 하나의 단어를 추측

-> 거대한 말뭉치에서는 수많은 뉴런 때문에 많은 시간이 소요
-> 행렬 곱과 softmax 계층의 계산이 병목

***

#### 4.1.1 Embedding 계층

embedding : 걔층에 단어 임베딩 (분산 표현)을 저장하는 것

원핫 벡터의 모든 값을 matmul 할필요가 없고 단어 ID에 해당되는 행을 추출

***

### 4.2 word2vec 개선 2

은닉층 이후의 병목은 네거티브 샘플링으로 해결

softmax 대신 네거티브 샘플링을 사용 -> 어휘가 아무리 많아져도 계산량을 낮은 수준에서 일정하게 억제 가능

***

#### 4.2.2 다중 분류에서 이진 분류로

네거티브 샘플링의 핵심 아이디어는 이진 분류에 있습니다. 

-> 다중 분류를 이진 분류로 근사하는 것

<img width="931" alt="스크린샷 2023-07-12 오전 1 40 24" src="https://github.com/sondonghup/music_vae/assets/42092560/e4af73e7-13a1-48f5-ae34-c415758e8659">