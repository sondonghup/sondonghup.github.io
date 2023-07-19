---
layout: post
title: deep-learning-from-scratch2-chapter6
subtitle: deep-learning-from-scratch2-chapter6
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### Rnn의 문제점
1. 벡터가 순차적으로 입력되어 병렬화가 불가능 합니다.
2. gradient exploding 또는 gradient vanishing
-> 데이터의 시퀀스가 길어지면 과거 정보가 현재 학습에 영향을 미칠수 없습니다.

### LSTM
데이터의 시퀀스가 길어지면 일어나는 기울기 손실을 해결하기 위한 3가지 게이트를 가진 LSTM

<img width="589" alt="스크린샷 2023-07-20 오전 3 05 55" src="https://github.com/sondonghup/programmers/assets/42092560/e87cf182-2abb-46c2-b660-7773e1d94676">

- forget gate layer
정보를 버릴지 말지 결정 
sigmoid를 통해 값을 출력

- input gate layer
새로운 정보가 cell state에 저장이 될지 결정하는 게이트

- output gate layer
어느 부분을 출력으로 보낼지 결정하는 게이트

```
ft = sigmoid(np.dot(xt, Wf) + np.dot(ht_1, Uf) + bf)  # forget gate
it = sigmoid(np.dot(xt, Wi) + np.dot(ht_1, Ui) + bi)  # input gate
ot = sigmoid(np.dot(xt, Wo) + np.dot(ht_1, Uo) + bo)  # output gate
Ct = ft * Ct_1 + it * np.tanh(np.dot(xt, Wc) + np.dot(ht_1, Uc) + bc)
ht = ot * np.tanh(Ct)
```