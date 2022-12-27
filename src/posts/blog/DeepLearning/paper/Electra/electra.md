---
title: "Huggingface로 ELECTRA 학습하기 : Domain Adaptation"
category: "DeepLearning"
date: "2022-12-22"
thumbnail: "./img/electra.png"
desc: "pytorch를 활용해 ELECTRA 논문을 코드로 구현하며 Generator와 Descriminator 간 연결 방법 및 Replace Token Detection(RTD)에 대해 설명한다.
Huggingface의 trainer를 활용하여 모델을 학습하는 방법을 소개하고, 이에 대한 튜토리얼을 제작해 ELECTRA 뿐만 아니라 Huggingface 사용법을 손쉽게 익힐 수 있도록 하였다. 직접 Domain Adapatation을 경험하며 ELECTRA 학습 방법 및 데이터 흐름에 대해 이해할 수 있다."
---

### 왜 ELECTRA인가?

- ELECTRA는 Masked Language Model(MLM)의 비효율적인 학습 방법에 새로운 대안을 제시하는 모델임. 지금껏 언어 모델은 통계 기반 모델, Vector Space 내 단어, 문장을 배치하는 모델, Mask를 예측하는 모델이 등장하며 변화를 불러왔음. 개인적인 생각으로 TPD의 등장은 동일한 연산 대비 더 나은 성능을 보장하는 최적화로서 방향이 확대된다는 생각임.

- ELECTRA가 제시하는 replaced token detection(RTD) 학습 방법은 동일 환경, 동일 시간 대비 MLM 보다 더 좋은 성능을 보장함. 동일한 성능을 내기 위해서 RTD가 MLM에 비해 더 적은 컴퓨팅 시간을 소모한다는 의미이므로 효율적인 학습 방법이라 할 수 있음.

- ELECTRA는 BERT를 학습시키는 새로운 방법론을 제시하는 모델이므로 BERT 구조를 이해하고 있다면 어렵지 않게 논문을 이해할 수 있음.

<br/>
<br/>

### ELECTRA 특징

- MLM 모델이 비효율적인 이유는 문장의 15%만을 학습에 활용하기 때문임. 모델은 [Mask]된 토큰을 예측하는 과정에서 학습을 진행하는데, 문장의 약 15% 토큰이 임의로 선택 된 뒤 전환되므로 [Mask] 되지 않은 나머지 문장은 학습을 하지 않게 됨.

- 이러한 비효율을 개선하고자 모델이 문장 내 토큰을 전부 학습할 수 있는 방법이 RTD임. RTD는 아래와 절차로 진행됨
  - Generator 모델에서 문장 토큰 중 약 15%를 바꿔 가짜 문장을 만듬. Generator는 기존 MLM 학습 방법대로 학습을 수행함.
  - Discriminator는 모든 문장에 대해 진짜 토큰인지, 가짜 토큰인지 구별하는 과정에서 학습을 수행함.
- RTD는 MLM의 도움을 받아 학습하지만, 학습이 완료되면 Generator는 사용하지 않고 RTD로 학습된 Discriminator만을 활용함.

<br/>
<br/>

### Domain Adaptation을 위한 ELECTRA 학습구조 설계

> Domain Adaptation에 대한 설명이 필요한 경우 [[NLP] Further Pre-training 및 Fine-tuning 정리
> ](https://yangoos57.github.io/blog/DeepLearning/paper/Finetuning/Finetuning)를 참고

- 학습 구조는 [lucidrains의 electra-pytorch](https://github.com/lucidrains/electra-pytorch) 코드를 활용했으며, Huggingface와 함께 사용할 수 있도록 일부 코드를 수정하였음.
- Base 모델로 monologg님의 `koelectra-v-base`모델을 활용했음.
- 모델 학습 및 평가에 대한 `튜토리얼`은 [해당 github](https://github.com/yangoos57/Electra_for_fine_tuning)를 참고
- 모든 코드는 Huggingface의 `Transformers`와 `pytorch` 를 기반으로 작성하였음.

<br/>

### 1. Huggingface Transformers로 Pre-trained Model 불러오기

- Discriminator를 불러오는 모듈은 `ElectraForPreTraining` , Generator를 불러오는 모듈은 `ElectraForMaskedLM` 이어야 함.

- koElectra hugging face에 있는 사용예시에서는 `ElectraModel`사용하지만 학습을 위해서는 `ElectraModel` 를 활용해 모델을 불러오지 않음. `ElectraModel`는 output으로 마지막 Encoder에서 나오는 output을 제공하기 때문임. 모델의 output shape은 (batch_size,src_token,embed_size)임.

- `ElectraForPreTraining` 와 `ElectraForMaskedLM` 는 `ElectraModel` 위에 훈련에 활용될 layer를 쌓아 제공하는 모델임.

- `ElectraForPretraining`은 token의 진위여부를 판별하는 classification layer가 더해져있는 모델이고 `ElectraForMaskedLM`은 [MASK] 부분에 들어가기 적합한 토큰을 판단하는 모델임.

  ```python

  from transformers import ElectraForPreTraining, ElectraTokenizer, ElectraForMaskedLM

  tokenizer = ElectraTokenizer.from_pretrained("monologg/koelectra-base-v3-generator")

  generator = ElectraForMaskedLM.from_pretrained('monologg/koelectra-base-v3-generator')

  discriminator = ElectraForPreTraining.from_pretrained("monologg/koelectra-base-v3-discriminator")

  ```

- `ElectraForMaskedLM` 은 Electra model의 output과 [Mask]에 들어갈 단어를 예측하는 layer를 덮은 model임. 모델의 output은 [Mask]에 적합한 단어를 확률로 제공하며 output의 shape은 (batch_size, src_token_len, vocab_size)임.

```python

# Huggingface Transformers 내부 ElectraForMaskedLM 코드

class ElectraForMaskedLM(ElectraPreTrainedModel):
    _keys_to_ignore_on_load_missing = ["generator_lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)

        # ElectraForMaskedLM 내부에서 ElectraModel 모듈을 불러와 사용함.
        self.electra = ElectraModel(config)

        # [Mask] 토큰에 적합한 토큰을 예측하는 generator predictor도 포함되어 있음을 확인할 수 있음.
        self.generator_predictions = ElectraGeneratorPredictions(config)

        self.generator_lm_head = nn.Linear(config.embedding_size, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(...) :

            # electra를 통해 마지막 encoder의 output(=last_hidden_state)를 받음.
            generator_hidden_states = self.electra(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            # last_hidden_states를 prediction scores에 input 데이터로 활용함.
            generator_sequence_output = generator_hidden_states[0]

            # last_hidden_states를 MLM prediction layer의 input 데이터로 활용함.
            prediction_scores = self.generator_predictions(generator_sequence_output)
            prediction_scores = self.generator_lm_head(prediction_scores)

    return MaskedLMOutput(
                loss=loss,
                # Prediction_score을 리턴
                # logits의 shape은 (batch_size, src_token_len, vocab_size)
                logits=prediction_scores,
                hidden_states=generator_hidden_states.hidden_states,
                attentions=generator_hidden_states.attentions,
            )

```

- `ElectraForPreTraining` 도 마찬가지로 `ElectraModel` 모듈을 베이스로 하고 문장 내 개별 token이 진짜 token인지를 구분하는 classification layer가 연결되어 있음. 개별 토큰의 진위여부를 0과 1로 판단함. output이 1인 경우 모델이 가짜 토큰으로 판별함을 의미함. 개별 token에 대한 진위여부를 판단하므로 shape는 (batch_size, src_token_len)임.

```python

# Huggingface Transformers 내부 ElectraForPreTraining 코드

class ElectraForPreTraining(ElectraPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # ElectraForPreTraining 내부에서 ElectraModel 모듈을 불러와 사용함.
        self.electra = ElectraModel(config)

        # Token의 진위여부를 판별하는 Precdiction 모델을 불러옴.
        self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

    ....


    def forward(...)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # electra를 통해 마지막 encoder의 output(=last_hidden_state)를 받음.
        discriminator_hidden_states = self.electra(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # last_hidden_states를 classification layer의 input 데이터로 활용함.
        discriminator_sequence_output = discriminator_hidden_states[0]

        logits = self.discriminator_predictions(discriminator_sequence_output)

        return ElectraForPreTrainingOutput(
            loss=loss,
            # logits을 리턴
            # logits의 shape은 (batch_size, src_token_len)
            logits=logits,
            hidden_states=discriminator_hidden_states.hidden_states,
            attentions=discriminator_hidden_states.attentions,
        )
```

<br>

> **학습을 완료한 경우에는 `ElectraForPreTraining` 내부에 있는 ElectraModel를 추출해 finetuning에 활용함. 학습 완료한 모델에서 ElectraModel을 추출하는 방법은 아래와 같음.**
>
> ```python
> discriminator = ElectraForPreTraining.from_pretrained('...')
>
> # Electra Model 추출
> trained_electra = discriminator.electra
>
> # 추출된 electra 모델은 Encoder 출력 끝단(=last_hidden_states)를 output으로 제공함.
> # ElectraModel을 활용해 Finetuning 수행
> ```

<br/>

### 2. 학습 모델 설계하기

- Transformers에서 불러온 Generator와 Discriminator를 학습하기 위해서는 학습용 모델 설계가 필수임.
  > 해당 모델은 Domain Adaptation 또는 pre-traing from scratch를 위한 모델이며, Fine tuning을 수행할 경우 아래의 학습 모델 설계 없이 discriminator만 활용하면 됨.
- 아래의 학습 모델은 아래 논문의 구조를 구현한 것임. electra의 학습 방식은 세 단계로 구분할 수 있음.

<img src='img/electra_sm.png'/>

- 1단계 : input data masking
- 2단계 : Generator 학습 및 fake sentence 생성
- 3단계 : Discriminator 학습

```python
import math
from functools import reduce
from collections import namedtuple

import torch
from torch import nn
import torch.nn.functional as F

# constants

Results = namedtuple(
    "Results",
    [
        "loss",
        "mlm_loss",
        "disc_loss",
        "gen_acc",
        "disc_acc",
        "disc_labels",
        "disc_predictions",
    ],
)

# 모델 내부에서 활용되는 함수 정의

def log(t, eps=1e-9):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature=1.0):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim=-1)

def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob

def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask

def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = mask.cumsum(dim=-1) > (num_tokens * prob).ceil()
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()

# main electra class

class Electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        tokenizer,
        *,
        num_tokens=35000,
        mask_prob=0.15,
        replace_prob=0.85,
        mask_token_id=4,
        pad_token_id=0,
        mask_ignore_token_ids=[2, 3],
        disc_weight=50.0,
        gen_weight=1.0,
        temperature=1.0,
    ):
        super().__init__()

        """
        num_tokens: 모델 vocab_size
        mask_prob: 토큰 중 [MASK] 토큰으로 대체되는 비율
        replace_prop:  토큰 중 [MASK] 토큰으로 대체되는 비율(?????)
        mask_token_i: [MASK] Token id
        pad_token_i: [PAD] Token id
        mask_ignore_token_id: [CLS],[SEP] Token id
        disc_weigh: discriminator loss의 Weight 조정을 위한 값
        gen_weigh: generator loss의 Weight 조정을 위한 값
        temperature: gumbel_distribution에 활용되는 arg, 값이 높을수록 모집단 분포와 유사한 sampling 수행
        """

        self.generator = generator
        self.discriminator = discriminator
        self.tokenizer = tokenizer

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

        # sampling temperature
        self.temperature = temperature

        # loss weights
        self.disc_weight = disc_weight
        self.gen_weight = gen_weight

    def forward(self, input_ids, **kwargs):

        input = input_ids["input_ids"]

        # ------ 1단계 Input Data Masking --------#

        """
        - Generator는 Bert와 구조도 동일하고 학습하는 방법도 동일함.

        - Generator 학습을 위해선 [Masked] 토큰이 필요하므로 input data를 Masking하는 과정이 필요함.

        """

        replace_prob = prob_mask_like(input, self.replace_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        # also do not include these special tokens in the tokens chosen at random
        no_mask = mask_with_tokens(input, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # get mask indices
        mask_indices = torch.nonzero(mask, as_tuple=True)

        # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        masked_input = input.clone().detach()

        # set inverse of mask to padding tokens for labels
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        # clone the mask, for potential modification if random tokens are involved
        # not to be mistakened for the mask above, which is for all tokens, whether not replaced nor replaced with random tokens
        masking_mask = mask.clone()

        # [mask] input
        masked_input = masked_input.masked_fill(
            masking_mask * replace_prob, self.mask_token_id
        )

        # ------ 2단계 Masking 된 문장을 Generator가 학습하고 가짜 Token을 생성 --------#

        """
        - Generator를 학습하여 MLM_loss 계산(combined_loss 계산에 활용)
        - Generator에서 예측한 문장을 Discriminator 학습에 활용
        - ex) 원본 문장 : ~~~
              마스킹 문장 :
              가짜 문장 :
        """

        # get generator output and get mlm loss(수정)
        logits = self.generator(masked_input, **kwargs).logits

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2), gen_labels, ignore_index=self.pad_token_id
        )

        # use mask from before to select logits that need sampling
        sample_logits = logits[mask_indices]

        # sample
        sampled = gumbel_sample(sample_logits, temperature=self.temperature)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.detach()

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float().detach()

        # ------ 3단계 Token의 진위여부를 Discriminator가 예측하고 이를 학습 --------#

        """
        - 가짜 문장을 학습해 개별 토큰에 대해 진위여부를 판단
        - 진짜 token이라 판단하면 0, 가짜 토큰이라 판단하면 1을 부여
        - 정답과 비교해 disc_loss를 계산(combined_loss 계산에 활용)
        - combined_loss : 학습의 최종 loss임. 모델은 combined_loss의 최솟값을 얻기 위한 방식으로 학습 진행
        """

        # get discriminator predictions of replaced / original
        non_padded_indices = torch.nonzero(input != self.pad_token_id, as_tuple=True)

        # get discriminator output and binary cross entropy loss
        disc_logits = self.discriminator(disc_input, **kwargs).logits
        disc_logits_reshape = disc_logits.reshape_as(disc_labels)

        disc_loss = F.binary_cross_entropy_with_logits(
            disc_logits_reshape[non_padded_indices], disc_labels[non_padded_indices]
        )

        # combined loss 계산
        # disc_weight을 50으로 주는 이유는 discriminator의 task가 복잡하지 않기 떄문임.
        # mlm loss의 경우 vocab_size(=35000) 만큼의 loos 계산을 수행하지만
        # disc_loss의 경우 src_token_len 만큼의 loss 계산을 수행한만큼
        # loss 값에 큰 차이가 발생함. disc_weight은 이를 보완하는 weight임.
        combined_loss = (self.gen_weight * mlm_loss + self.disc_weight * disc_loss,)

        # ------ 모델 성능 및 학습 과정을 추적하기 위한 지표(Metrics) 설계 --------#

        with torch.no_grad():
            # gen mask 예측
            gen_predictions = torch.argmax(logits, dim=-1)

            # fake token 진위 예측
            disc_predictions = torch.round(
                (torch.sign(disc_logits_reshape) + 1.0) * 0.5
            )
            # generator_accuracy
            gen_acc = (gen_labels[mask] == gen_predictions[mask]).float().mean()
            # discriminator_accuracy
            disc_acc = (
                0.5 * (disc_labels[mask] == disc_predictions[mask]).float().mean()
                + 0.5 * (disc_labels[~mask] == disc_predictions[~mask]).float().mean()
            )


        return Results(
            combined_loss,
            mlm_loss,
            disc_loss,
            gen_acc,
            disc_acc,
            disc_labels,
            disc_predictions,
        )
```

<br/>

### 3. Huggingface Datasets으로 학습 데이터 불러오기

- 이 글에서는 Hugging face의 Trainer API를 활용해 모델을 학습할 예정임. Trainer API를 사용한다면 데이터를 Huggingface의 Datasets으로 불러오는 것을 강력하게 권함.
- pytorch의 Dataset을 활용할 수 있긴 하지만 Trainer와 함께 사용하기에는 원인을 찾기 힘든 에러가 많아 디버깅에 어려움이 있음.

```python
from datasets import load_dataset

# local file을 불러오고 싶을땐 '확장자명', '경로'를 적으면 됨
train = load_dataset('csv',data_files='data/book_train_128.csv')
validation = load_dataset('csv',data_files='data/book_validation_128.csv')

# tokenizing 방법 정의
def tokenize_function(examples):
    return tokenizer(examples['sen'], max_length=128, padding=True, truncation=True)

# datasets의 map 매서드를 활용해 모든 문장에 대한 토크나이징 수행
train_data_set = train['train'].map(tokenize_function,batch_size=True)
validation_data_set = validation['train'].map(tokenize_function,batch_size=True)
```

- datasets 기본 매서드 소개

```python
train = load_dataset('csv',data_files='data/book_train_128.csv')

train

>>> DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'sen'],
        num_rows: 175900
    })
})

#----------

# column 제거
train = train.remove_columns('Unnamed: 0')

>>> DatasetDict({
    train: Dataset({
        features: ['sen'],
        num_rows: 175900
    })
})

#----------

# train 데이터셋으로 이동
train_data_set = train['train']

>>> Dataset({
    features: ['Unnamed: 0', 'sen'],
    num_rows: 175900
})

#----------

# 데이터 불러오기
train_data_set[0]

>> {'sen': '이 책의 특징ㆍ코딩의 기초 기초수학 논리의 ... 기초수학'}

#----------

# 데이터 추출
train_data_set[0]['sen']

>>> '이 책의 특징ㆍ코딩의 기초 기초수학 논리의 ... 기초수학'

#----------

# type 확인
train_data_set.feature

>>> {'sen': Value(dtype='string', id=None)}

#----------

# 저장
train_data_set.to_csv('')

```

<br/>

### 4. Transformers Trainer API로 모델 학습하기

#### ❖ 훈련 옵션 설정(선택사항)

훈련에 사용되는 모든 Argument를 수정할 수 있음. 이중 `logging_steps` 에 대해서만 설명하겠음. step은 1회 batch 진행을 의미함. logging_steps = 2는 2회의 step이 끝나면 log를 print 하라는 명령어임. log에 대한 내용은 callback 함수를 설명하며 다루겠음.

```python
from transformers import TrainingArguments

training_args = TrainingArguments(
    output_dir="test_trainer",
    per_device_eval_batch_size=8,
    per_device_train_batch_size=8,
    logging_steps=2,
    num_train_epochs=2,
    evaluation_strategy='steps'
)
```

<br/>

#### ❖ Callback 설정(선택사항)

> 아래의 내용과 공식 홈페이지의 [Callback 페이지](https://huggingface.co/docs/transformers/main/en/main_classes/callback#transformers.integrations.CometCallback)와 함께 읽으면 callback에 대해 빠르게 이해할 수 있음

- callback은 훈련 과정 중 Trainer API가 추가로 수행해야하는 내용을 정의하는 함수임.
- 예로들어 step이 시작할때 마다 몇번째 step인지 print하고 싶을때 활용할 수 있음.
- callback class를 정의 한 뒤 callback이 필요한 순서를 함수로 정의하여 사용함.
  - callback이 가능한 순서는 `on_init_end`, `on_train_begin`, `on_train_end`, `on_epoch_begin`, `on_epoch_end`, `on_step_begin`, `on_substep_end`, `on_step_end`, `on_evaluate`, `on_save`, `on_log`, `on_prediction_step` 이 있음
- callback 내부 함수는 `arg`, `state`, `control`, `logs`, `**kwargs`로 모두 동일함.

  - arg는 훈련 옵션으로 설정한 값을 불러옴.
  - state는 현재 step, epoch 등 진행 상태에 대한 값을 불러옴
  - control은 훈련 과정을 통제하는 변수를 불러옴
  - logs는 loss, lr, epoch 등 기본적인 정보를 불러옴
    ```python
    # logs output
    {'loss': 1.5284, 'learning_rate': 4.995452064762598e-05, 'epoch': 0.0}
    ```
  - `**kwargs` 는 model, tokenizer, optimizer, dataloader 등을 불러 올 수 있음.

    ```python
    ### trainer_callback.py 참고

    class CallbackHandler(TrainerCallback):
        """Internal class that just calls the list of callbacks in order."""

        def __init__(self, callbacks, model, tokenizer, optimizer, lr_scheduler):
            self.callbacks = []
            for cb in callbacks:
                self.add_callback(cb)
            # kwargs로 불러올 수 있는 함수들
            self.model = model
            self.tokenizer = tokenizer
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
            self.train_dataloader = None
            self.eval_dataloader = None
    ```

- 미리 정의 된 Callback을 사용할 수도 있음.
  - Transformers 라이브러리에서 해당 callback명을 불러와 사용
  - `ProgressCallback` 은 on_train_begin 단계에서 진행 상태바를 callback하도록 설정
  - `PrinterCallback` 은 on_log 순서에서 logs 내용을 callback하도록 설정
  - `EarlyStoppingCallback` 은 on_evaluate 순서에서 EarlyStop을 callback하도록 설정

```python
from transformers import TrainerCallback

# custom callback 만들기, 이때 TrainerCallback을 상속 받아야함.
class myCallback(TrainerCallback):

  def on_step_begin(self, args, state, control, logs=None, **kwargs):
    # step은 1회 batch 진행을 의미함. step의 시작일 때 아래의 내용을 실행

      if state.global_step % args.logging_steps == 0:
        # state는 현재 step, epoch 등 진행 상태에 대한 값을 불러옴
        # arg는 훈련 옵션으로 설정한 값을 불러옴.
          print("")
          print(
              f"{int(state.epoch)}번째 epoch 진행 중 --- {state.global_step}번째 step 결과"
          )
```

<br/>

#### ❖ Custom Trainer(선택사항)

- Trainer를 필요에 맞게 수정할 수 있음. optimizer 설정, loss 계산 등 훈련 진행 방법에 대한 방법을 수정하는데도 사용하지만, 모델이 정확한 예측을 수행하는지 아래와 같은 방법으로 출력이 필요한 경우에도 사용할 수 있음.

  ```js
  0번째 epoch 진행 중 ------- 0번째 step 결과
  input 문장 : 장 수학 기호 수식에 많이 쓰이는 그리스 알 [MASK] [MASK] [MASK] 읽고 쓰는 법을 배웁니다
  output 문장 : 장 수학 기호 수식에 많이 쓰이는 그리스 알 [##파] [##벳] [##을]을 쓰는 법을 배웁니다

  0번째 epoch 진행 중 ------- 20번째 step 결과
  input 문장 : [MASK]이 출간된지 꽤 됬다고 생각하는데 실습하는데 전혀 [MASK]없습니다
  output 문장 : [책]이 출간된지 꽤 됬다고 생각하는데 실습하는데 전혀 [문제]없습니다

  ```

- Train 단계에서 모델에 input data를 넣고 output data를 추출하는 과정은 compute_loss 매서드에서 이뤄짐. 따라서 compute_loss 매서드를 덮어쓰기하여 필요한 데이터를 활용할 수 있음.

```python
class customtrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # compute_loss 함수 덮어쓰기
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer.

        By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        prds,outputs = model(inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        # ############# 모델 학습 과정 확인을 위한 코드 추가

        if self.state.global_step % self.args.logging_steps == 0:
            # self.state.global_step = 현 step 파악
            # args.logging_steps = argument에서 지정한 logging_step

            # batch 중 0 번째 위치한 문장 선택
            num = 1
            input_id = inputs.input_ids[num].reshape(-1).data.tolist()
            output_id = outputs.logits[num].argmax(dim=-1).reshape(-1).data.tolist()
            attention_mask = inputs.attention_mask[num]

            # mask가 위치한 idx 추출하기
            mask_idx = (inputs.input_ids[num] == 4).nonzero().data.reshape(-1).tolist()

            # padding 제거
            input_id_without_pad = [
                input_id[i] for i in range(len(input_id)) if attention_mask[i]
            ]
            output_id_without_pad = [
                output_id[i] for i in range(len(output_id)) if attention_mask[i]
            ]

            # id to token
            # [1:-1] [CLS,SEP] 제거
            inputs_tokens = self.tokenizer.convert_ids_to_tokens(input_id_without_pad)[
                1:-1
            ]
            outputs_tokens = self.tokenizer.convert_ids_to_tokens(
                output_id_without_pad
            )[1:-1]

            # output mask 부분 표시하기
            for i in mask_idx:
                # [CLS,SEP 위치 조정]
                outputs_tokens[i - 1] = "[" + outputs_tokens[i - 1] + "]"

            inputs_sen = self.tokenizer.convert_tokens_to_string(inputs_tokens)
            outputs_sen = self.tokenizer.convert_tokens_to_string(outputs_tokens)

            print(f"input 문장 : {''.join(inputs_sen)}")
            print(f"output 문장 : {''.join(outputs_sen)}")

            # input 문장 : 수학 기호 수식에 많이 쓰이는 그리스 알 [MASK] [MASK] [MASK] 읽고 쓰는 법을 배웁니다
            # output 문장 : 수학 기호 수식에 많이 쓰이는 그리스 알 [##파] [##벳] [##을]을 쓰는 법을 배웁니다

        return (loss, outputs) if return_outputs else loss
```

<br/>

#### ❖ Trainer 설정

- 앞서 설정했던 옵션, 데이터셋, callback 함수 등을 trainer로 통합하는 과정임.
- customtrainer를 Trainer로 사용했고, 모델 학습 과정 확인 단계에서 tokenizer가 필요하므로 tokenizer를 포함했음.
- callback 함수는 1개를 불러오더라도 list 타입으로 불러와야함.
- trainer를 정의한 뒤 .train() 매서드를 실행하면 학습 시작

```python
trainer = customtrainer(
    model=model.to(device),
    train_dataset=train_data_set,
    eval_dataset=validation_data_set,
    data_collator=data_collator_BERT,
    args=training_args,
    tokenizer=tokenizer,
    callbacks=[myCallback,PrinterCallback],
)

trainer.train()
```

- Trainer는 학습 과정과 Training loss를 한눈에 볼 수 있도록 interface를 지원함.

<img src='img/interface.png'/>
