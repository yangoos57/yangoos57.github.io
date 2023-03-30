---
title: "Transformer Positional Encoding 이해하기"
category: "DeepLearning"
date: "2023-03-30"
thumbnail: "./img/transformer.png"
desc: pytorch를 활용해 Transformer 논문을 코드로 구현하며 모델의 상세 작동원리를 설명하였다. 구현한 Transformer 모델을 활용해 학습과 평가하는 과정을 경험할 수 있도록 튜토리얼을 제작했으며, 튜토리얼을 통해 모델 내부에서 어떻게 데이터가 흐르는지, 어떠한 과정을 거쳐 입력 데이터에 대한 결과물을 산출하는지를 이해할 수 있다. 논문에 포함된 Transformer의 도식화 그림을 활용해 Transformer 구조 전반에 대한 이해에 도움을 준다.
---

### 들어가며

이 글은 Transformer의 구조 중 Positional Encoding에 대한 설명과 이에 대한 참고자료를 정리하였습니다.

### 단어의 위치 정보 생성하기(what)

Poisional Encoding은 문장 내 단어의 "위치 정보"를 벡터로 표현한 한 것입니다. 이러한 벡터 값이 왜 필요로 하는지, 어째서 Transformer 모델에 중요한지는 Transformer가 탄생한 배경을 알게 된다면 이해하실 수 있습니다.

기본적으로 Transfoermer의 큰 구조인 Encoder, Decoder는 Transformer 모델에서 새롭게 소개된 구조가 아닙니다. Transformer는 RNN 기반의 seq2seq모델을 Attention을 활용해 구현한 모델일 뿐입니다.

물론 Attention만을 활용해 seq2seq 모델을 구현하는 것 자체가 기술적인 도전이었고, 병렬연산, 문장 내 모든 단어 정보 참조가 가능한 Attention의 장점을 많은 사람들에게 각인 시킨 모델이었기에 Transformer 기반 모델이 사실상 NLP 모델의 표준으로 자리잡게 된 것이라 할 수 있습니다.

따라서 Positional Encoding은 Seq2Seq 구조를 Attention으로 구현하기 위한 기술적 과제 중 하나인 병렬 연산이 가능해짐에 따라 문장 내 단어의 위치 정보가 소실되는 문제를 해결한 방법으로 이해할 수 있습니다.

정의에 대한 이해를 돕기 위해 예시를 하나 들어보겠습니다. "나는 학교에 갔다"라는 문장이 있습니다. 이 문장을 토크나이징 하게되면 [나, -는, 학교, -에, 갔다.]로 구분 됩니다. 토큰화 된 개별 단어는 단어의 원 뜻 뿐만 아니라 문장의 위치에 따른 정보도 포함하고 있습니다.

### 중복되지 않은 벡터를 생성하려면(How)
