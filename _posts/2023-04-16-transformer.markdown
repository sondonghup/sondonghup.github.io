---
# multilingual page pair id, this must pair with translations of this page. (This name must be unique)
lng_pair: id_Examples
title: transformer

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
date: 2023-04-16 18:38:06 +0900

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

### encoder-decoder framework

앞서 배웠던 rnn은 nlp작업, 음성처리, 시계열 작업에 널리 사용이 되었다.
rnn은 한 언어에서 다른 언어로 매핑하는 작업을 할때 중요한 역할을 했고, 이런 종류의 작업은 대개 encoder-decoder 또는 sequence-to-sequence 구조로 처리하며
입력과 출력의 길이가 같을때 잘 맞는다. 
디코더는 인코더의 마지막 은닉상태만을 참조해 출력을 만드므로 입력이 길어질 수록 마지막 은닉상태에서 시작 부분의 정보가 손실될 확률이 크다.

### attention 메커니즘

마지막 만 참고하면 정보 손실이 일어날 확률이 크므로 각 스텝마다 은닉상태를 출력 하도록 하자.
하지만 모든 상태를 동시에 사용하려면 입력이 너무 많으니까 어떤 은닉상태를 먼저 사용할지 가중치를 할당한다.
<img width="813" alt="image" src="https://user-images.githubusercontent.com/42092560/232316094-97f9125c-72fe-4c61-b225-7a2bbb29f7f1.png">

### self attention

위 처럼 어텐션을 사용해도 연산이 순차적으로 이뤄지기 때문에 병렬화를 할 수가 없다.
