---
# multilingual page pair id, this must pair with translations of this page. (This name must be unique)
lng_pair: id_Examples
title: linear-regression

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
date: 2023-05-14 18:38:06 +0900

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

### linear regression

선형 회귀란 

데이터(파란 점)를 가장 잘 대변 해주는 선(빨간 선, line of best fit)을 찾아 내는 것

h(x) = a + bx

<img width="605" alt="image" src="https://github.com/sondonghup/Game_project/assets/420925609b8e3ec2-9fc4-4887-92ac-3078cc3e797c">

목표 변수 : 맞추려고 하는 값
(target variable / output variable)

입력 변수 : 맞추는데 사용하는 값
(input variable / feature)

### 가설 함수 평가법

평균 제곱 오차 (MSE)

![image](https://github.com/sondonghup/Game_project/assets/42092560/e4959281-c15c-415d-9bad-df1950af0e1e)

제곱을 하는 이유 

오차가 양수 또는 음수로 나올 수도 있으므로 상쇄되는것을 방지

오차를 좀더 부각시켜주기 위해서

### 손실 함수 (비용 함수)

![image](https://github.com/sondonghup/Game_project/assets/42092560/82c96b87-9781-4122-9499-5c76036344d3)

가설 함수의 성능을 평가하는 함수

손실 함수가 작으면 가설 함수가 데이터에 잘 맞으며 반대일 경우 잘 맞지 않는 것