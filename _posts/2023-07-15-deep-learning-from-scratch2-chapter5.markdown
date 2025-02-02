---
layout: post
title: deep-learning-from-scratch2-chapter5
subtitle: deep-learning-from-scratch2-chapter5
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/thumb.png
share-img: /assets/img/path.jpg
tags: [books, test]
---

<!-- outline-start -->

This is an example page to display markdown related styles for Mr. Green Jekyll Theme.

<!-- outline-end -->

### Rnn
{:data-align="center"}
word2vec의 CBOW모델이 있다. 이 모델을 언어 모델로 사용하게 되면 어떻게 될까?  
  
문맥고려 문제와 순서고려 문제가 있다. 이럴때 사용하는 것이 Rnn 이다.  

Rnn은 Recurrent neural network의 줄임말로 순환 신경망이라 한다.  
  
  
### Rnn 식
![image](https://user-images.githubusercontent.com/42092560/184859664-f0c4a3da-9b39-48a7-92dd-4ae93ba804ca.png)
  
![image](https://user-images.githubusercontent.com/42092560/184860250-5f0d3341-6e0e-419f-8cf9-65902afbd158.png) : 은닉 상태 벡터  

![image](https://user-images.githubusercontent.com/42092560/184861191-f059d65e-08c8-401d-9a28-9916440cdd9f.png) : 이전 은닉 상태 벡터  

![image](https://user-images.githubusercontent.com/42092560/184861343-8f54af0c-90fa-43e4-95aa-2e3ddb5164fd.png) : 단어 벡터  
ex) 나는 동협이다. -> 나 / 는 / 동협 / 이다 -> '나'를 벡터화 한 것 (그다음에는 '는'을 벡터화 한 것이 들어갈것 )  
  
![image](https://user-images.githubusercontent.com/42092560/184865256-dcb171c7-5393-4bdb-8441-b1be6fab4105.png) : 가중치 

![image](https://user-images.githubusercontent.com/42092560/184865381-7eb3f216-3190-4bcb-bf18-44e664da26ce.png)는 sigmoid도 사용가능하다 tanh가 기울기가 커서 수렴이 더 빠르다  
  
### Rnn  
![image](https://user-images.githubusercontent.com/42092560/184868778-ff33051b-9133-48d0-b252-baca65ee8bc8.png)

![image](https://user-images.githubusercontent.com/42092560/184869250-10261f37-fc24-49f3-8d4a-bdfe2d36d518.png)
  
  
순전파와 역전파의 그림이다.

나는 동협이다를 예로 순서를 알아보자.  
처음 x에는 '나'를 벡터화한 값이 들어간다. hprev는 처음이라 영벡터가 들어간다. 위의 그림에 따라 내적도 하고 tanh도 취해서 hnext값이 나온다 (벡터값)  
  
그러면 이제 이 hnext값이 다시 hprev값으로 들어가며 x에는 '나'다음인 '는'의 벡터화 된값이 들어가며 반복이된다.  
  
앞에서 말했듯 rnn은 한개의 구조에 순환하는 방식이다. 나는 펼쳐진 그림을 봐서 여러개라 생각했었다...  
  
이제 역전파를 한번 생각해보자 위의 예시처럼 나는 동협이다를 기준으로 하면 4번 역전파를 하면 된다.
  
그런데 엄청 긴 문장이라면 어떨까 역전파를 계속 하면서 0에서 1사이의 값이 계속 곱해져 점점 작아진다.  
  
이렇게 되면 손실이 잃어나게 되어 가중치가 업데이트가 원활하게 업데이트 되지 않을 것이다.  
  
  
![image](https://user-images.githubusercontent.com/42092560/184878031-8606e1f7-8c6c-4dd1-af3c-181619f91f8a.png)

![image](https://user-images.githubusercontent.com/42092560/184878783-1f32e1fe-bc5f-41b6-90b7-7687d5f3992e.png)

그러면 어떻게 해야 되나 바로 위의 그림처럼 중간에 끊어내는 것이다.  
  
이렇게 잘라내면 역전파가 중간에 끊기는게 아닌가 생각이 들수 있지만 Rnn은 앞에서 말했듯이 하나라 실제로는 끊기는게 아니다.

하지만 순전파일때는 자르면 안된다.  
  
affine (= linear layer ) : 차원을 맞춰주는 레이어다.  
ex) 만약 10개 카테고리를 분류 한다고 생각하면 affine에서 10개의 차원으로 맞춰줘야 할것이다.
  
  
### rnn 구현  
  
```python
class Rnn:
    def __init__(self, wx, wh, b):
        self.params = [wx, wh, b]
        self.grads = [np.zeros_like(wx),np.zeros_like(wh),np.zeros_like(b)]
    
    def forward(self, x, h_pre):
        wx, wh, b = self.params
        t = np.matmul(x, wx) + np.matmul(wh, h_pre) + b
        h_next = np.tanh(t)
        
        return h_next
```  
  
pytorch에서 forward만 작성을 해주면 loss.backward()를 통해 역전파도 제공을 한다.
