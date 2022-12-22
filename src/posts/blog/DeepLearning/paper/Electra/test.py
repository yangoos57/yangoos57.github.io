# Huggingface Transformers 내부 ElectraForPreTraining 코드

class ElectraForPreTraining(ElectraPreTrainedModel):
def __init__(self, config):
    super().__init__(config)
            
            # ElectraForPreTraining 내부에서 ElectraModel 모듈을 불러와 사용함.
    self.electra = ElectraModel(config)
            
            # Token의 진위여부를 판별하는 Precdiction 모델을 불러옴.
    self.discriminator_predictions = ElectraDiscriminatorPredictions(config)

....


def forward(......)
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