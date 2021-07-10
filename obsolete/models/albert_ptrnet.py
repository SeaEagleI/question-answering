# import torch
# import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AlbertPreTrainedModel, AlbertModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput

from .pointer_network import PointerNetwork


class AlbertPtrForQuestionAnswering(AlbertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super(AlbertPreTrainedModel, self).__init__(config)
        self.num_labels = config.num_labels
        self.max_question_len = 64

        self.albert = AlbertModel(config, add_pooling_layer=False)
        # self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)
        self.qa_outputs = PointerNetwork(config.hidden_size, config.hidden_size)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict else self.config.use_return_dict

        outputs = self.albert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]                            # [CLS], [Question], [SEP], [Context], [SEP], ##[PAD]##
        # split AlbertModel output => question_emb, passage_att
        question_emb, passage_att = sequence_output[:, :self.max_question_len + 2, :], sequence_output
        # padding for each sample
        for sample_id in range(question_emb.size(0)):
            first_sep_pos = list(input_ids[sample_id]).index(3)  # locate first [SEP]
            question_emb[sample_id, first_sep_pos + 1:, :] = 0   # [CLS], [Question], [SEP], ##[PAD]##
            passage_att[sample_id, 1:first_sep_pos, :] = 0       # [CLS], ##[PAD]##, [SEP], [Context], [SEP], ##[PAD]##
        # get start & end logits
        start_logits, end_logits = self.qa_outputs(passage_att, question_emb, batch_first=True)
        start_logits, end_logits = start_logits.squeeze(-1), end_logits.squeeze(-1)  # shape: [batch_size, max_seq_len]

        # compute loss
        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
