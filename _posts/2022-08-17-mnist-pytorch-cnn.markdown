---
# multilingual page pair id, this must pair with translations of this page. (This name must be unique)
lng_pair: id_Examples
title: mnist pytorch cnn실습

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
date: 2022-08-16 18:38:06 +0900

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

### mnist pytorch 실습  
  
mnist 손글씨 데이터 셋으로 학습해보자  
저번 시간엔 선형분류모델로 했었지만 이번엔 cnn으로 해보자  


```python
import sys

import sklearn.datasets
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

mnist = sklearn.datasets.fetch_openml('mnist_784', data_home="mnist_784")
```  
  
mnist 손글씨 데이터를 가져온다. 
  
  
```python
class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame(), target: pd.Series()):
        super(ImageDataset, self).__init__()
        self.data = torch.tensor(data.values/255, dtype = torch.float).reshape((-1, 1, 28, 28))
        self.labels = torch.tensor(target.astype(int).values, dtype = torch.long)

    def __getitem__(self, idx):
        return {"image" : self.data[idx], "label" : self.labels[idx]}

    def __len__(self):
        return len(self.labels)
```  
  
우리는 data와 label만 필요하므로 두개를 가져온다.  
  
getitem에서 data를 하나씩 가져오는 방식이다.  
  
cnn layer에 맞게 reshape를 해준다.  

```python
dataset = ImageDataset(mnist.data, mnist.target)
```  
  
dataset 객체를 만들어준다.  
  
```python
fig, axes = plt.subplots(2, 4, constrained_layout=True)

for i, ax in enumerate(axes.flat):
    ax.imshow(1 - dataset[i]["image"].reshape((28, 28)), cmap="gray", vmin=0, vmax=1)
    ax.set(title=f"{dataset[i]['label']}")
    ax.set_axis_off()
```  
  
![image](https://user-images.githubusercontent.com/42092560/185364287-0a109065-ef9a-42ba-b43d-dd42f2208370.png)  
  
  
데이터가 잘 들어가 있는지 확인 해본다.  
  
```python
train_dataset, test_dataset = train_test_split(dataset, test_size=0.05)
```  
  
dataset을 train과 test set으로 나눈다. 비율은 0.95 : 0.05  
  
  
```python
batch_size = 32
n_epochs = 10
lr = 3e-4
```  
  
미니배치와 학습 횟수 학습률을 입력해준다.  
  
```python
train_data = DataLoader(dataset=train_dataset,
                        batch_size=batch_size,
                        drop_last=False)
test_data = DataLoader(dataset=test_dataset,
                        batch_size=batch_size,
                        drop_last=False)
```  
  
dataloader에 train과 test를 넣어준다.  
  
```python
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # input shape(28, 28, 1)
        # conv output shape(8, 8, 64)
        # maxpooling output shape (2, 2, 64)
        self.conv_layer = nn.Sequential( 
            nn.Conv2d(1, 64, kernel_size=7, stride=3), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4)
        )
        self.fc_layer = nn.Linear(2 * 2 * 64, 10)

    def forward(self, x):
        output = self.conv_layer(x)
        output = nn.Flatten()(output)
        output = self.fc_layer(output)
        return output
```  
  
cnn 모델을 구성한다. 
  
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```  
  
device를 선언한다. gpu가 있을땐 gpu를 아니면 cpu를 쓴다.  
  
```python
model = LinearModel()
model.eval().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr)
```  
  
모델을 선언하고 loss_fn optimizer를 선언한다.  
  
```python
for epoch in range(n_epochs):
    for data in train_data:
        images, labels = data["image"], data["label"]
        optimizer.zero_grad()

        output = model(images.to(device))
        loss = loss_fn(output, labels)
        
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_prediction_list = list()
        test_labels_list = list()
        test_loss = 0.0
        for data in test_data:
            images, labels = data["image"], data["label"]

            test_output = model(images.to(device))
            test_loss += loss_fn(test_output, labels) / len(labels)
            test_prediction = test_output.argmax(-1)

            test_prediction_list.extend(test_prediction)
            test_labels_list.extend(labels)

    print("-" * 30)
    accuracy = accuracy_score(test_prediction_list,test_labels_list)
    loss = test_loss / len(test_data)
    print(f"epoch {epoch} is done ...")
    print(f"accuracy is {100 * accuracy:.2f}")
    print(f"loss is {100 * loss:.4f}")
```  
  
epoch 마다 acc와 loss를 계산한다.  
  
![image](https://user-images.githubusercontent.com/42092560/185372980-1d77cd23-55d9-445f-8937-716125e2c687.png)


