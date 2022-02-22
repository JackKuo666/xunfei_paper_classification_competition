from typing import Dict

import torch
from allennlp.data import Vocabulary
from allennlp.data import TextFieldTensors
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy


import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#
# class Attention(nn.Module):
#     def __init__(self, hidden_size):
#         super(Attention, self).__init__()
#         self.weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
#         self.weight.data.normal_(mean=0.0, std=0.05)
#
#         self.bias = nn.Parameter(torch.Tensor(hidden_size))
#         b = np.zeros(hidden_size, dtype=np.float32)
#         self.bias.data.copy_(torch.from_numpy(b))
#
#         self.query = nn.Parameter(torch.Tensor(hidden_size))
#         self.query.data.normal_(mean=0.0, std=0.05)
#
#     def forward(self, encoded_title, encoded_abstract):
#         # 1、思路一：将两者分别乘以w,再加和，输出
#         # 2、思路二：将两者合并，然后进行attention
#         # batch_hidden: b x len x hidden_size (2 * hidden_size of lstm)
#         # batch_masks:  b x len
#
#         # Shape: (batch_size, encoding_dim)
#
#         # linear
#         key = torch.matmul(batch_hidden, self.weight) + self.bias  # b x len x hidden
#
#         # compute attention
#         outputs = torch.matmul(key, self.query)  # b x len
#
#         masked_outputs = outputs.masked_fill((1 - batch_masks).bool(), float(-1e32))
#
#         attn_scores = F.softmax(masked_outputs, dim=1)  # b x len
#
#         # 对于全零向量，-1e32的结果为 1/len, -inf为nan, 额外补0
#         masked_attn_scores = attn_scores.masked_fill((1 - batch_masks).bool(), 0.0)
#
#         # sum weighted sources
#         batch_outputs = torch.bmm(masked_attn_scores.unsqueeze(1), key).squeeze(1)  # b x hidden
#
#         return batch_outputs, attn_scores


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.weight_title = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_title.data.normal_(mean=0.0, std=0.05)

        self.bias_title = nn.Parameter(torch.Tensor(hidden_size))
        b = np.zeros(hidden_size, dtype=np.float32)
        self.bias_title.data.copy_(torch.from_numpy(b))

        self.weight_abstract = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.weight_abstract.data.normal_(mean=0.0, std=0.05)

        self.bias_abstract = nn.Parameter(torch.Tensor(hidden_size))
        self.bias_abstract.data.copy_(torch.from_numpy(b))
        self.output_dim = hidden_size

    def forward(self, encoded_title, encoded_abstract):
        # 1、思路一：将两者分别乘以w,再加和，输出
        # 2、思路二：将两者合并，然后进行attention
        # Shape: (batch_size, encoding_dim)

        # linear
        encoded_title = torch.matmul(encoded_title, self.weight_title) + self.bias_title  # b x hidden
        encoded_abstract = torch.matmul(encoded_abstract, self.weight_abstract) + self.bias_abstract  # b x hidden
        batch_outputs = encoded_title + encoded_abstract
        # todo 这里不单单是加和，也可以是拼接
        # 1、C=torch.cat((A,B),0)#按维数0（行）拼接  b*2 X hidden
        # 2、C=torch.cat((A,D),1)#按维数1（列）拼接  b X hidden*2
        self.output_dim = batch_outputs.shape[-1]

        return batch_outputs

    def get_output_dim(self):
        return self.output_dim


@Model.register("sent_attention")
class SentAttention(Model):
    def __init__(
        self, vocab: Vocabulary,
            embedder_title: TextFieldEmbedder, encoder_title: Seq2VecEncoder,
            embedder_abstract: TextFieldEmbedder, encoder_abstract: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder_title = embedder_title
        self.encoder_title = encoder_title
        self.embedder_abstract = embedder_abstract
        self.encoder_abstract = encoder_abstract
        self.hidden_size = self.encoder_title.get_output_dim()
        self.sent_attention = Attention(self.hidden_size)

        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(self.sent_attention.get_output_dim(), num_labels)
        self.accuracy = CategoricalAccuracy()

    def forward(
        self, title: TextFieldTensors, abstract: TextFieldTensors, label: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        # todo: 这里两个使用同一个参数，可以:
        #  1、使用相同的embedder,但是使用不同的encoder
        #  2、使用不同的embedder和不同的encoder      【doing】

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_title = self.embedder_title(title)
        # Shape: (batch_size, num_tokens)
        mask_title = util.get_text_field_mask(title)
        # Shape: (batch_size, encoding_dim)
        encoded_title = self.encoder_title(embedded_title, mask_title)

        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_abstract = self.embedder_abstract(abstract)
        # Shape: (batch_size, num_tokens)
        mask_abstract = util.get_text_field_mask(abstract)
        # Shape: (batch_size, encoding_dim)
        encoded_abstract = self.encoder_abstract(embedded_abstract, mask_abstract)

        # 1、Shape: (batch_size, encoding_dim)
        sent_hiddens = self.sent_attention(encoded_title, encoded_abstract)

        # 2、Shape: (batch_size, encoding_dim*2)
        # todo 直接拼接 为 (batch_size, encoding_dim*2)
        # sent_hiddens = torch.cat([encoded_title, encoded_abstract], dim=-1)

        # Shape: (batch_size, num_labels)
        logits = self.classifier(sent_hiddens)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=1)
        # Shape: (1,)
        output = {"probs": probs}
        if label is not None:
            self.accuracy(logits, label)
            output["loss"] = torch.nn.functional.cross_entropy(logits, label)
        return output

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}

    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        label = torch.argmax(output_dict["probs"], dim=1)
        label = [self.vocab.get_token_from_index(int(i), "labels") for i in label]
        output_dict["label"] = label
        return output_dict
