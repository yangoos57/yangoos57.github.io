---
title: "[NLP] Further Pre-training 및 Fine-tuning 정리"
category: "DeepLearning"
date: "2022-12-08"
thumbnail: "./img/Fine-tuning.png"
---

<br/>

Bert 모델 Fine-tuning 방법을 찾던 중 Fine-tuning을 다양한 용어로 부르는 것 같아 Fine-tuning 과정을 이해하는데 어려움이 있었다. Fine-tuning에 대해 찾다보면 MLM Fine-tuning, Feature Extraction ,Domain Adpatation, Further Pre-training 등 다양한 용어가 등장하는데 이러 용어들이 Fine-tuning을 다르게 부르는 용어인지 아니면 Fine-tuning의 또 다른 방법인지 혼란이 있었다. 이 글은 흔히 Fine-tuning으로 부르는 용어를 정리하여 이러한 혼란을 줄이기 위해 작성했으며, Pretrained Language Model(PLM)을 어떻게 학습시켜야할지 감이 잡히지 않거나,하고싶은 방법은 아는데 이를 지칭하는 용어가 헷갈리는 사람들을 대상으로 작성했다.

### Fine-tuning의 범위는 어디까지일까

일반적으로 Fine-tuning을 하는 행위는 Pretrained Language Model(PLM)을 Text classification, Sentiment Analysis와 같은 Downstream task를 수행할 수 있도록 훈련시키는 과정을 의미한다. 하지만 Fine-tuning에 대해 찾다보면 알고있는 방법과 다른 Fine-tuning 방법이 있는 것 같다는 느낌을 지우기 어렵다. 일반적으로 PLM을 학습시키는 방법 전부를 Fine-tuning한다라고 간주하는 경향이 있는 것 같다. 일례로 MLM Fine-tuning이라는 용어가 있는데, MLM을 학습하는 Fine-tuning과 일반 Fine-tuning이 같은 방법을 의미하는 건지, 아니면 새로운 Fine-tuning 방법을 의미하는건지 NLP에 익숙하지 않은 사용자들에게 혼란을 가중시킨다. 다양한 용어를 접하다보면 오히려 기존에 알고 있던 Fine-tuning에 대해서도 맞게 알고 있는건지 의문이 생기기도 한다.

아래의 그림은 PLM을 가지고 어떤 훈련을 수행할 수 있는지에 대해 도식화한 그림이다. 많은 경우 검은색 네모 박스 쳐져있는 영역을 Fine-tuning의 영역으로 판단하는 것 같다. 원래의 Fine-tuning은 Downstream task를 위한 훈련 방법 중 하나이지만, 모델을 훈련 하는 방법이 발전함에 따라 모든 훈련 과정을 지칭하는 의미로도 확장된 것 같다.

<br/>

<img src='img/img1.png' alt='img1' width ='600px'>

<br/><br/>

네모 박스 내 훈련 방식은 Further Pre-training과 Training Downstream Task으로 구분된다. 이미 학습이 완료된 모델(PLM)을 활용해 전문 domain을 학습한다는 의미로 Further Pre-training이라는 용어를 사용하고 있으며, 앞서 예시로 들었던 MLM Fine-tuning은 Further Pre-traing의 다른 표현이다. Training Downstream Task는 Pretrained Language Model(PLM)을 특정 task를 수행할 수 있도록하는 학습 과정이다 Fine-tuning이 이를 위한 방법이며 Fine-tuning외에도 Feature Extraction이라는 방법도 존재한다.

먼저 Further Pre-traning을 설명한 다음 Training Downstream Task에 대해 설명하겠다.

## Further Pre-training

도식에서 볼 수 있듯 Further Pre-training은 필수가 아니다. 해결하려는 문제의 domain이 전문 영역인 경우 관련 용어 학습을 위해 Further Pre-training을 권하지만, 학습에 필요한 domain 관련 Corpus를 확보하지 못하거나, 일반분야의 문제 해결이라면 해당 과정을 생략하고 Downstream Task 학습으로 넘어가도 무방하다.

### 용어정리

지금까지 자료 통해 이해한바로는 2020년에 발행된 [Don't Stop Pre-training. Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)이라는 논문이 나오기 전까지는 Further Pre-training이 곧 Domain Adaptation을 의미했다. 해당 논문에서 Task Adaptive Pre-training(TAPT)이라는 개념을 새로 소개하면서 Domain Adapatation을 Domain Adaptive Pre training(DAPT)로 통일했고, 현재는 Domain Adaptation과 DAPT를 혼용해서 사용하고 있다.

일반적인 경우 Domain Adaptation, Domain Adaptive Pre-training(DAPT), Further Pre-training, MLM Fine-tuning 이렇게 4개의 용어는 모두 같은 방법을 의미한다. Further Pre-training에는 DAPT 말고도 TAPT나 Extend Vocab 방법도 있지만 흔히 활용하지 않는 방법이다. 그렇지만 TAPT는 효율성면에서 장점이 있고, Extend Vocab은 Vocab에 단어를 넣고 학습한다는 점에서 장점이 있는 방법이다.

### Further Pre-training이 가능한 이유 : Subword Embedding

Further Pre-training이 가능한 이유는 Bert 학습이 Subword 방식으로 이루어졌기 때문이다. Subword embedding은 Word Embedding에서 발생하는 OOV(Out of Vocabulary) 문제를 해결하기 위해 제안된 방식이다. NLP 모델을 학습하기 위해서는 학습에 쓰이는 데이터를 통해 단어집(Vocab)을 추출해야한다. 이렇게 추출된 단어집(Vocab)은 문장들을 단어로 분리하고, 이를 숫자로 encoding하는 과정에 활용된다. 따라서 단어집(Vocab)에 없는 단어의 경우 토크나이징과 encoding 절차를 수행할 수 없다.

이러한 이유로 Word embedding으로 학습한 모델은 단어집(Vocab)에 없는 단어가 포함된 문장을 활용할 수 없다. 이를 Out of Vocabulary(OOV)라 하는데, 이를 해결하기 위해서는 단어집(Vocab)이 무수히 커져야 하는데, 단어 수가 증감함에 따라 연산 수도 증가하고, 이 방법으로도 끊임없이 변형되는 언어의 변화에 대처하는데에는 한계가 있다.

반면 Subword embedding의 경우 단어를 한번 더 나누어 학습하므로 단어집(Vocab)에 없는 단어라 하위 단어의 조합을 통해 단어를 생성할 수 있으므로 OOV를 해결 할 수 있다. 예로들어 ‘왼손’, ‘왼편’은 ‘왼’과, 손’, ‘편’의 합성이므로 ‘왼’,’-손’,’-편’ 3 단어를 단어집(Vocab)에 넣으면 '왼손’, ‘왼편’을 단어집(Vocab)에 넣지 않아도 토큰화가 가능하다. 이때 ‘왼쪽’이라는 Vocab에 없는 단어가 존재하면 ‘왼’과 ‘-쪽’으로 토크나이징하여 encoding이 가능하다. ‘쪽’이 단어집(Vocab)에 없더라도 ‘ㅉ’,’ㅗ’,’ㄱ’으로 토크나이징 가능하다.

이러한 Subword의 장점으로 인해 단어집(Vocab)에 없는 단어라도 학습 가능해졌고 학습이 마무리된 모델에 대해 재학습이 가능해졌다.

### Further Pre-training 방법

Further Pre-training은 학습을 이어나가는 방법이므로 PLM을 초기 학습시키는 과정과 동일하다. 대신 Bert 초기학습 시 NSP와 MLM 학습을 필요로 하는데, Further Pre-training에서는 MLM만 학습할 수 있다.

모델을 새롭게 학습할 때의 방식과 동일하다. 대신 PLM 학습에 활용된 Tokenizer를 불러와 활용해야한다. 임의로 Tokenizer를 만들거나 다른 모델 학습에 활용된 Tokenizer를 사용하면 단어별 맵핑된 정수가 다르기 때문에 전혀 다른 학습을 하게 된다.

이렇게 Further Pre-training이 완료되면 Downstream task에 맞도록 Fine-tuning을 수행하면 된다.

### DAPT, TAPT, Extend Vocab 차이

<br/>

<img src='img/img2.png' alt='img2' width ='600px'>

<br/><br/>

**DAPT와 TAPT**

용어의 유사성에서 보듯 DAPT와 TAPT의 차이는 domain과 task의 차이에서 비롯한다. domain은 Pre-train 수준의 방대한 학습, task는 Fine-tuning 수준의 효율적인 학습을 수행한다. 데이터 크기를 보면 두 방법의 차이를 느낄 수 있는데, DAPT로 학습하면 40GB 수준의 방대한 Corpus를 활용하고 TAPT는 Fine-tuning을 위해 사용되는 데이터로 약 80kb 수준의 몇만건 정도 되는 데이터를 학습한다.

학습 데이터 크기만 보더라도 효율면에 있어서 Task Adaptive가 압도적임을 알 수 있다. 성능면에서도 효율 대비 성능이 좋다고 한다. 다만 task 특화된 사전학습 방법이므로 A task 용 데이터로 학습한 모델에 B task를 부여하면 성능 향상이 거의 없다고 한다. 이는 Text Classification에 필요한 학습 데이터와 Sentiment Analysis에 필요한 학습 데이터가 기본적으로 다르다는 점을 고려하면 이해 가능한 결과이다.

<br/>

<img src='img/img4.png' alt='img4' width ='600px'>

<br/><br/>

위의 표는 일반 모델(RoBERTa)과 DAPT, TAPT, DAPT+TAPT(DAPT 수행 후 TAPT를 수행한 모델)의 Downstream task에서의 성능 차이를 보여준다. 표를 통해 알 수 있는 사실은,

1. 어떤 방법이든 일반 모델보다 더 나은 성능을 보장한다.
2. 거의 모든 task에서 DAPT+TAPT 성능이 우수하다.
3. TAPT는 매우 효율적인 방법이다.(일부 task에서는 DAPT 보다 나은 성능을 보임)

특히 1번의 경우 모델의 Parameter가 커지고 학습량 또한 급증하는 추세임에도 Pre-training이 여전히 효과적인 성능향상 방법임을 나타낸다.

**DAPT와 Extend Vocab**

Extend Vocab은 전문용어를 Tokenizing하지 않고 그대로 학습하기 위한 방법이다. 새롭게 추가하는 단어 크기에 따라 방법이 나뉘는데, 추가하는 단어가 적은 경우 기존 Vocab에 추가해 학습하는 방법을 사용하고, 추가해야할 단어가 많은 경우 새로운 Vocab을 생성한 뒤 기존 Vocab과 Module로 연계하여 학습하는 방법을 사용한다.

추가할 단어가 적은 경우(Vocab 대비 최대 1%) [How to add a domain-specific Vocabulary (new tokens) to a subword tokenizer already trained](https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41) 에서 자세한 방법을 소개하고 있다.

단어 개수가 많은 경우 다음의 논문을 참고하여 학습 가능하다. [exBERT: Extending Pre-trained Models with Domain-specific Vocabulary Under Constrained Training Resources](https://aclanthology.org/2020.findings-emnlp.129) 논문에서 자세히 소개하고 있으며 이에 대한 리뷰는 \_\_\_를 참고하자.

### Pre-training From Sratch과 Further Pre-training

세부 domain에 특화된 모델을 만드는 방법 중 하나로 domain에 특화된 모델을 처음부터 만드는 방법도 있다. 전문용어를 subword로 나눠서 학습하고 싶지 않은 경우 domain에 맞게 새로운 모델을 제작하는 것이 일반적이다. 실제로 많은 기업에서 자신의 domain에 특화된 PLM을 만들어 사용하는데, 해당 domain을 처음부터 학습시킬 수 있는 충분한 데이터가 축적되었고 이를 만들 수 있는 충분한 역량이 있거나, 특정 용어가 subword로 토크나이징 된 후 학습되는 것을 방지하기 위해 선택한다.

domain 특화된 PLM 제작에 관심이 있다면 화해 기술블로그에 방문해 글을 읽어보는 것을 추천한다.

> ”화해”가 domain specific PLM을 제작한 이유
>
> **1.Beauty domain에 좋은 성능을 내기 위해서는 domain에 적합한 Vocab이 필요하기 때문**
>
> **좋은 성능의 PLM을 만들기 위해서는 좋은 Vocab이 필요**하고, 좋은 Vocab을 만들기 위해서는 전처리 과정에 공을 들여야 하는데요. 여러가지 전처리 기준을 정함에 있어서 무엇보다도 해당 domain 데이터의 특성을 잘 고려하는 것이 가장 중요하다고 할 수 있습니다. 화해팀의 경우 문장 분류, 검색, 추천 등 다양한 영역에서 PLM을 활용하고자 하는 목적이 있었기 때문에 세부적인 전처리 기준을 정하기에 앞서 아래와 같은 정성적인 기준을 정했습니다.
>
> **2.Domai에 특화된 용어가 subword로 토크나이징 되는 것을 방지하기 위해**
>
> <br/>
>
> <img src='img/img5.png' alt='img5' width ='600px'>
>
> <br/>
>
> 출처 : [Beauty domain-Specific Pre-trained Language Model 개발하기(화해 기술 블로그)](http://blog.hwahae.co.kr/all/tech/tech-tech/5876/)

하지만 Vocab에 전문용어를 넣지 않아도 되는 경우 Domain Adaptation이 보다 효과적인 선택지가 될 수 있다. DAPT가 오히려 성능면에서 더 우수한 결과를 보인다는 논문 결과도 있으며, 비용면에서도 저렴한 방법이기 때문이다. Domain Adaptation 장점에 대한 내용은 Data Rabbit님의 블로그에 잘 정리되어있다. Data Rabbit님께서 Further Pre-training 관련 논문을 읽고 핵심만 종합했다.

> **Domain을 위한 Language Model further Pre-training(Data Rabbit 블로그)**
>
> ### 케이스
>
> - 금융 Domain, FinBERT (2019)[https://arxiv.org/abs/1908.10063](https://arxiv.org/abs/1908.10063)
> - 텍스트북 별 Domain적용(채점 문제), 별칭없음 BERT (2019)[https://www.aclweb.org/anthology/D19-1628/](https://www.aclweb.org/anthology/D19-1628/)
> - 법률, LEGAL-BERT (2020)[https://arxiv.org/abs/2010.02559](https://arxiv.org/abs/2010.02559)
> - 그외 외 논문들에서 relative works에 참고된 것들
>   - BioBERT (2019)
>   - SciBERT (2019)
>   - Clinical BioBERT (2019)
>
> 위 BERT들 모두 다 further Pre-training 방식임. 아무도 pre training from scratch 하지 않음
>
> ### 결론
>
> - Full Pre-training보다는 further Pre-training으로 Domain을 보다 효율적으로 주입. 그래서 대부분 further Pre-training을 함
> - Further Pre-training은 domain knowledge를 LM(Language Model)에 부여하는 효과를 봄
> - 복잡한 task일 수록 더 큰 효과를 봄 Binary classification < Multi label classification < Question & Answer
> - Domain을 한정 할 수록 한정된 Domain의 문제에서 더 큰 효과를 봄
> - 작은 사이즈의 BERT 모델을 기반으로 Domain을 추가 학습한 경우가 풀 사이즈의 기본 BERT와 Domain 특정지어지는 문제에서의 성능은 비슷하지만 더 가볍기 때문에 효율적임
>
> 출처: [Domain을 위한 Language Model further Pre-training](https://flonelin.wordpress.com/2021/03/21/domain%EC%9D%84-%EC%9C%84%ED%95%9C-language-model-further-Pre-training/)

## Training Downstream Task

<br/>

<img src='img/img3.png' alt='img3' width ='600px'>

<br/><br/>

Pre-training이 끝난 모델은 task에 맞는 학습을 거쳐 관련 업무를 수행한다. PLM을 사용하는 방법에는 크게 Fine-tuning과 Feature extraction이 있다. 세부 task를 학습하면서 Language Model의 Weight를 조정하는 방법을 Fine-tuning이라 하고 Language Model의 Weight 조정없이 새로 학습하는 ML 모델의 Weight를 조정하는 방법을 Feature Extraction이라 한다.

설명만 이해하면 Feature extraction이 pretrained model의 weight 조정을 하지 않았기 때문에 Fine-tuning보다 성능이 저조할 것으로 생각할 수 있지만, 두 방법의 성능 차이는 거의 없다고 한다. Feature extraction과 Fine-tuning의 차이를 연구한 논문인 [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://www.aclweb.org/anthology/W19-4302.pdf)에서는 두 방법의 성능 차이가 거의 없다고 한다.

### Fine-tuning

일반적으로 Fine-tuning이라 하면 text classficiation,sentiment analysis 등 NLP로 해결 할 수 있는 Downstream task에 적합하게 모델을 훈련하는 과정을 말한다. pretrained Language Model(PLM)을 불러온 뒤 그 윗단에 task 수행을 위한 새로운 layer를 쌓은 구조를 만든 다음 label이 있는 관련 데이터셋을 활용해 모델 전체를 학습시킨다. 이때 task layer 뿐만 아니라 하단 구조인 Language Model에서도 weight이 조정된다.

Language Model에 task layer를 쌓아 학습하는 이유는 적은 양의 데이터 셋을 활용 하더라도 상당히 좋은 성능을 보장하기 때문이다. Labeled 된 데이터를 수집하기는 많은 비용과 시간을 필요로 하며, 축적된 데이터가 많아도 ML 모델로는 성능 향상의 한계가 있기 때문이다. 이미 대량의 Corpus를 학습해 단어 간 관계, 연관성을 학습한 모델은 Language Model은 적은량의 Labeled 데이터를 가지고도 ML 모델을 압도한다. 이러한 이유에서 기업에서도 pretrained model을 Fine-tuning하는 방법을 통해 라벨링에 드는 비용,시간 등을 절약한다.

또한 기업에서는 pretrained Langumodel을 만들 수 있을만큼의 충분한 기술력이 있고 해당 산업에 적합한 데이터가 많기 때문에, 기업에 맞는 Pretrained Language Model을 제작한 다음 Fine-tuning을 진행한다.

다음은 mathpress 기술 블로그에 있는 내용 일부이다.

> **Language Model 을 활용한 문제 분류(Mathpress 블로그)**
> 문제 유형을 분류하는 Language Model을 학습시키기 위해서는 실제 문제 유형이 태깅되어있는 데이터가 필요합니다. 이런 데이터는 구체적으로 정의하기 어렵고, 구축하는 과정이 노동 집약적이기 때문에 대용량으로 확보하기가 어렵습니다. 반면에, 태깅이 되어있지 않은 수학 문제는 비교적 수집이 용이하기 때문에, 이를 Pre-training에 활용하고, 태깅된 소수의 데이터를 이용해 Fine-tuning 하는 방식을 취하였습니다.
>
> 출처 : [Language Model 을 활용한 문제 분류](https://blog.mathpresso.com/language-model-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%AC%B8%EC%A0%9C-%EB%B6%84%EB%A5%98-cb339ed1bc26)

### Feature extraction(=Language Model Frozen)

이 방법은 Language Model을 활용해 task를 수행한다는 점에서 Fine-tuning과 동일하지만 구현하는 방법에 차이가 있다. Feature extraction 방식은 Language Model을 Encoder로 사용하는 것이라 이해할 수 있다. Feature extraction은 Layer 끝단의 Embedding만을 사용한다. 간단하 설명하면 문장을 embedding으로 변환하는 용도로 Language Model을 사용한다.

```python
sen = '파이썬은 재밌어'

token = ['-파이','썬','-은','재미','-있어']

embedding = [
	     [0.05905652, 0.2062039 , ... ,0.28419164, 0.97168385], < -파이
             [0.09039539, 0.35047797, ... ,0.73923901, 0.80421832], < 썬
             [0.68320781, 0.54474656, ... ,0.06239229, 0.15109788], < 은
             [0.76254785, 0.2298931 , ... ,0.79764905, 0.99467024], < 재미
             [0.49704732, 0.60449829, ... ,0.60479266, 0.40783498]  < -있어
	    ]

```

이렇게 얻은 embedding vector를 가지고 새로운 ML 모델을 학습시켜 weight를 구해서 해당 모델을 활용한다.

Language Model의 weight 조정 없이 output layer만을 사용하기 때문에 Feature Extraction 방법을 Language Model Frozen 방법이라고도 한다.

### 참고자료

- [[Data Rabbit] Domain을 위한 Language Model further Pre-training](https://flonelin.wordpress.com/2021/03/21/domain%EC%9D%84-%EC%9C%84%ED%95%9C-language-model-further-Pre-training/)

- [[MathPresso] Language Model 을 활용한 문제 분류](https://blog.mathpresso.com/language-model-%EC%9D%84-%ED%99%9C%EC%9A%A9%ED%95%9C-%EB%AC%B8%EC%A0%9C-%EB%B6%84%EB%A5%98-cb339ed1bc26)

- [[화해] Beauty domain-Specific Pre-trained Language Model 개발하기](http://blog.hwahae.co.kr/all/tech/tech-tech/5876/)

- [How to add a domain-specific Vocabulary (new tokens) to a subword tokenizer already trained](https://medium.com/@pierre_guillou/nlp-how-to-add-a-domain-specific-vocabulary-new-tokens-to-a-subword-tokenizer-already-trained-33ab15613a41)

- [To Tune or Not to Tune? Adapting Pretrained Representations to Diverse Tasks](https://www.aclweb.org/anthology/W19-4302.pdf)

- [Don't Stop Pre-training. Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)

- [exBERT: Extending Pre-trained Models with Domain-specific Vocabulary Under Constrained Training Resources](https://aclanthology.org/2020.findings-emnlp.129)
