"""
https://github.com/codertimo/BERT-pytorch/
https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py

"""
import math

import torch
import torch.nn as nn


class GELU(nn.Module):
    """
    Paper Section 3.4, last paragraph notice that BERT used the GELU instead of RELU
    """

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, num_heads, dim_model):
        super(MultiHeadSelfAttention, self).__init__()
        assert dim_model % num_heads == 0
        self.num_heads = num_heads
        self.dim_model = dim_model
        self.head_size = self.dim_model // self.num_heads
        self.all_head_size = self.head_size * self.num_heads  # ==dim_model
        self.query_affine = nn.Linear(self.dim_model, self.all_head_size)
        self.key_affine = nn.Linear(self.dim_model, self.all_head_size)
        self.value_affine = nn.Linear(self.dim_model, self.all_head_size)

    def _split_heads(self, x):
        """
        :param x: (batch_size, seq_length, dim)
        :return: (batch_size, num_heads, seq_length, head_size)
        """
        x = x.view((x.shape[0], x.shape[1], self.num_heads, self.head_size))
        return x.premute(0, 2, 1, 3)

    def _merge_heads(self):
        pass

    def forward(self, hidden_states=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None, ):
        """
        :param hidden_states: (batch_size, seq_length, dim)
        :param attention_mask: (batch_size, seq_length)
        :param token_type_ids:
        :param position_ids:
        :return:
        """
        # (batch_size, num_heads, seq_length, head_size)
        multi_query = self._split_heads(self.query_affine(hidden_states))
        multi_key = self._split_heads(self.key_affine(hidden_states))
        multi_value = self._split_heads(self.value_affine(hidden_states))
        #
        attention_score = torch.matmul(multi_query, multi_key.transpose(-1, -2))
        attention_score = attention_score / torch.sqrt(
            self.dim_model)  # (batch_size, num_heads, seq_length, seq_length)
        if attention_mask is not None:
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0  # 使得对应位置在softmax后对应的概率值很小
            attention_score = attention_score + extended_attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_score)  # (batch_size, num_heads, seq_length, seq_length)
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        # (batch_size, num_heads, seq_length, head_size)
        attention_multi_value = torch.matmul(attention_probs, multi_value)
        attention_multi_value = attention_multi_value.permute(0, 2, 1, 3).contiguous()
        new_shape = attention_multi_value.size()[:2] + (self.dim_model,)
        attention_value = attention_multi_value.view(*new_shape)
        return attention_value, attention_probs


class TransformerEncoderBlock(nn.Module):
    def __init__(self, num_heads, dim_model):
        super(TransformerEncoderBlock, self).__init__()
        self.feed_forward = nn.Linear(dim_model, dim_model)
        self.gelu = GELU()
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-10)
        self.dropout = nn.Dropout(p=0.2)
        #
        self.bert_attention = MultiHeadSelfAttention(num_heads, dim_model)

    def forward(self, hidden_states=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None, ):
        x = self.bert_attention(hidden_states, attention_mask,
                                token_type_ids, position_ids)
        x = self.feed_forward(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer_norm(hidden_states + x)
        return x


class BertEncoder(nn.Module):
    def __init__(self, num_attention_heads, dim_model, num_hidden_layers):
        super(BertEncoder, self).__init__()
        self.block_layers = [TransformerEncoderBlock(num_heads=num_attention_heads, dim_model=dim_model)
                             for _ in range(num_hidden_layers)]

    def forward(self, hidden_states=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None, ):
        for layer in self.block_layers:
            hidden_states = layer(self, hidden_states,
                                  attention_mask,
                                  token_type_ids,
                                  position_ids)
        return hidden_states


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, dim_model, max_seq_length):
        super(BertEmbedding, self).__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim_model)
        self.position_embeddings = nn.Embedding(max_seq_length, dim_model)
        self.token_type_embeddings = nn.Embedding(vocab_size, dim_model)
        # self.position_ids = ...
        self.register_buffer("position_ids", torch.arange(max_seq_length).expand((1, -1)))
        #
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-12)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None):
        """
        :param input_ids:  (batch_size, seq_length)
        :param token_type_ids:
        :param position_ids:
        :return:
        """
        token_embeddings = self.token_embedding(input_ids)
        batch_size, seq_length = input_ids.size()
        if position_ids is None:
            position_ids = self.position_ids[:, None: seq_length]
        position_embedding = self.position_embeddings(position_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros((batch_size, seq_length), dtype=torch.long, device=self.position_ids.device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = token_embeddings + position_embedding + token_type_embeddings
        return self.dropout(self.layer_norm(embeddings))


class BertModel(nn.Module):
    def __init__(self, num_attention_heads, dim_model, num_hidden_layers,
                 vocab_size, max_seq_length):
        super(BertModel, self).__init__()
        self.embedding = BertEmbedding(vocab_size, dim_model, max_seq_length)
        self.block_layers = [TransformerEncoderBlock(num_heads=num_attention_heads, dim_model=dim_model)
                             for _ in range(num_hidden_layers)]

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None):
        embeddings = self.embedding(input_ids)
        hidden_states = embeddings
        for layer in self.block_layers:
            hidden_states = layer(self, hidden_states,
                                  attention_mask,
                                  token_type_ids,
                                  position_ids)
        cls_output = hidden_states[:, 0]
        return cls_output, hidden_states


class BertPooler(nn.Module):
    def __init__(self, dim_model):
        super().__init__()
        self.dense = nn.Linear(dim_model, dim_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        """
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        :param hidden_states:  (batch_size, seq_length, dim_model)
        :return:
        """
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForPreTraining(nn.Module):
    def __init__(self, num_attention_heads, dim_model, num_hidden_layers,
                 vocab_size, max_seq_length):
        super().__init__()
        self.bert = BertModel(num_attention_heads, dim_model, num_hidden_layers,
                              vocab_size, max_seq_length)
        #
        self.mlm_score = nn.Linear(dim_model, vocab_size)
        self.nsp_score = nn.Linear(dim_model, 2)
        #
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                mlm_labels=None,
                next_sentence_label=None):
        sequence_output, pooled_output = self.bert(input_ids, attention_mask, token_type_ids,
                                                   position_ids)
        loss = 0.0
        if mlm_labels is not None:
            prediction_scores = self.mlm_score(sequence_output)
            mlm_loss = self.cross_entropy(prediction_scores, mlm_labels)
            loss += mlm_loss
        if next_sentence_label is not None:
            seq_relationship_score = self.nsp_score(pooled_output)
            nsp_loss = self.cross_entropy(seq_relationship_score, next_sentence_label)
            loss += nsp_loss
        return loss


