"""vBert modeling"""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import math
import os
import sys
import time

from typing import Optional, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import transformers
from transformers import BertPreTrainedModel, BertModel
from transformers.models.bert.modeling_bert import BertEmbeddings, BertSelfAttention, BertIntermediate, BertEncoder, BertPooler, BertLayer
from transformers.models.bert.modeling_bert import BertPreTrainingHeads, BertOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from . import plot

import multiprocessing as mp

import logging
logger = logging.getLogger(__name__)

class BertSelfAttention(BertSelfAttention):

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):

        device = hidden_states.device
        mixed_query_layer = self.query(hidden_states)

        # most codes are copied from transformers v4.3.3
        
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
            #attention_scores = attention_scores * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_scores) if output_attentions else (context_layer,) # hacked: replace attention_probs with attention_scores

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs
    

class ShallowSkipping(nn.Module):
    
    def __init__(self, model):
        super().__init__()
#         self.model = model # do not register
        self.config = model.config
        self.shallow_config = model.shallow_config
        # current only support trigram
        self.ngram = 3
        
        if self.shallow_config.hidden_size != self.config.hidden_size:
            self.linear = nn.Linear(self.shallow_config.hidden_size, self.config.hidden_size)
            
        self.plot = plot.Plot(self.config.max_num_entries, self.config.hidden_size)
        
#         self.input_q = mp.Queue()
#         self.output_q = mp.Queue()
#         # self.input_ids_buffer = torch.randn(1)
        
#         def search_process(max_num_entries, hidden_size, input_q, output_q):
#             _plot = plot.Plot(self.config.max_num_entries, self.config.hidden_size)
#             while True:
#                 input_ids = input_q.get()
#                 hidden_states = _plot.retrieve_data(input_ids)
#                 # hidden_states.share_memory_()
#                 output_q.put(1)
        
#         self.p = mp.Process(target=search_process, args=(
#             self.config.max_num_entries, self.config.hidden_size, 
#             self.input_q, self.output_q,
#         ))
#         self.p.start()
        
        
    def _build_tri_gram_ids(self, input_ids:torch.Tensor) -> torch.Tensor:
        return torch.from_numpy(
            self.plot.input_ids_to_tri_grams(input_ids.cpu().numpy())
        ).to(input_ids.device)
        
    def build_input_ngrams(self, input_ids:torch.Tensor, token_type_ids:torch.Tensor):
        
        input_ngram_ids = self._build_tri_gram_ids(input_ids)
        
        token_ngram_type_ids = None #
        
        attention_mask = (input_ngram_ids > 0).float()
        
        if self.training:
            _mask = torch.rand(attention_mask.shape).to(attention_mask.device)
            _mask = (_mask > self.config.ngram_masking)
            attention_mask *= _mask

        attention_mask[:, self.ngram//2] = 1 # avoid masking all tokens in a tri-gram
        return input_ngram_ids, token_ngram_type_ids, attention_mask
    
    @torch.jit.script
    def merge_ngrams(input_ids, ngram_hidden_states, aux_embeddings):
        batch_size, seq_length = input_ids.shape
        lens = (input_ids!=0).sum(1)
        hidden_state = torch.zeros([batch_size, seq_length, ngram_hidden_states.size(-1)], dtype=ngram_hidden_states.dtype, device=ngram_hidden_states.device)
        
        # assert to be trigrams
        flat_hidden_state = ngram_hidden_states[:, 1]
        flat_hidden_state[:-1] = flat_hidden_state[:-1] + ngram_hidden_states[1:, 0]
        flat_hidden_state[1:] = flat_hidden_state[1:] + ngram_hidden_states[:-1, 2]
        k = 0
        for i in range(batch_size):
            hidden_state[i, :lens[i]] = flat_hidden_state[k: k+lens[i]]
            k += 1 + lens[i] # 1 for skipping one padding tri-gram
        hidden_state = hidden_state + aux_embeddings
        return hidden_state
    
    def forward_shallow_layers(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        ngram_mask_position=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        output_hidden_states=True,
        model=None,
    ):
        device = model.device
        
        input_ngram_ids, token_ngram_type_ids, attention_mask = self.build_input_ngrams(input_ids, token_type_ids)
        ngram_attention_mask = attention_mask.clone()
        
        if ngram_mask_position is not None:
            input_ngram_ids[:, ngram_mask_position] = 0
            ngram_attention_mask[:, ngram_mask_position] = 0

        extended_attention_mask = model.get_extended_attention_mask(attention_mask, input_ngram_ids.shape, device)

        ngram_index=(input_ngram_ids[:, self.ngram//2] > 0)

        embedding_output = model.embeddings(input_ids=input_ngram_ids, token_type_ids=token_ngram_type_ids)

        hidden_states = embedding_output
        attention_mask = extended_attention_mask

        for i, layer_module in enumerate(
                model.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]):
            layer_head_mask = head_mask[i] if head_mask is not None else None

            layer_outputs = layer_module(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                head_mask=layer_head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
            )

            hidden_states = layer_outputs[0]
            
        if self.shallow_config.hidden_size != self.config.hidden_size:
            hidden_states = self.linear(hidden_states)
            
        # Set zero the padding ngrams: (..., [PAD], ...)
        hidden_states = hidden_states * ngram_index[:, None, None]
            
        hidden_states = hidden_states * model.attn(hidden_states).sigmoid() * ngram_attention_mask.unsqueeze(-1)
        
        return input_ngram_ids, hidden_states

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=True,
        output_hidden_states=True,
        model=None,
    ):
        
        device = model.device
        
        batch_size, seq_length = input_ids.shape
        aux_embeddings = model.embeddings.position_embeddings2.weight[:seq_length].unsqueeze(0)
        aux_embeddings = aux_embeddings + model.embeddings.token_type_embeddings2(token_type_ids)
        
        if self.config.plot_mode == 'force_compute':
            '''
            compute only, ignore PLOT
            '''
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=None,
                model=model,
            )
            
        elif self.config.plot_mode == 'update_all':
            '''
            build PLOT
            '''
            # uni-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=(0,2),
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
            # bi-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=0,
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
            # tri-grams
            input_ngram_ids, hidden_states = self.forward_shallow_layers(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                ngram_mask_position=None,
                model=model,
            )
            self.plot.update_data(input_ngram_ids, hidden_states)
            
        elif self.config.plot_mode == 'plot_passive':
            '''
            use plot if no oov
            '''
            
            if input_ids.is_cuda:
                input_ids = input_ids.cpu()
            if not self.plot.has_oov(input_ids):
                hidden_states = self.plot.retrieve_data(input_ids)
                hidden_states = hidden_states.to(device)
            else:
                input_ids = input_ids.to(device)
                input_ngram_ids, hidden_states = self.forward_shallow_layers(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    ngram_mask_position=None,
                    model=model,
                )
                self.plot.update_data(input_ngram_ids, hidden_states)
                
        elif self.config.plot_mode == 'plot_only':
            '''
            plot only
            looking up order: trigram -> bigram -> unigram -> 0
            '''
            if input_ids.is_cuda:
                logger.warn("'input_ids' is better to placed in CPU.")
                input_ids = input_ids.cpu()
            hidden_states = self.plot.retrieve_data(input_ids)
            # self.input_q.put(input_ids)
            # _ = self.output_q.get()
            # hidden_states = torch.zeros([25800, 3, 768], dtype=torch.float16)
            hidden_states = hidden_states.to(device)
                
        hidden_states = F.dropout(hidden_states, self.config.hidden_dropout_prob, self.training)
        hidden_states = self.merge_ngrams(input_ids, hidden_states, aux_embeddings)
        hidden_states = model.norm(hidden_states)

        return hidden_states
    
    

class BertLayer(nn.Module):
    __constants__ = ['hidden_size', 'in_size']
    hidden_size: int
    in_size: int
    adapter_fi_w: torch.Tensor
    adapter_se_w: torch.Tensor
    
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertSelfAttention(config) #BertAttention(config, location_key="self")
        self.is_decoder = config.is_decoder
        self.add_cross_attention = config.add_cross_attention
        if self.add_cross_attention:
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BertAttention(config, position_embedding_type="absolute", location_key="cross")
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

        #lz:params for adapter [hidden_size&in_size]
        try:
            self.has_adapter = config.has_adapter
        except:
            self.has_adapter = False
        if self.has_adapter:
            self.in_size     = config.hidden_size
            self.hidden_size = 64
            
            #lz:two layer for adapter & activation function
            self.adapter_fi_w = torch.nn.parameter.Parameter(torch.empty(self.hidden_size,self.in_size))
            self.adapter_fi_b = torch.nn.parameter.Parameter(torch.empty(self.hidden_size))
            self.adapter_se_w = torch.nn.parameter.Parameter(torch.empty(self.in_size,self.hidden_size))
            self.adapter_se_b = torch.nn.parameter.Parameter(torch.empty(self.in_size))
            #lz:参数初始化
            self.reset_parameters()
        
        self.acfun      = nn.GELU()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(self.adapter_fi_w, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.adapter_fi_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.adapter_fi_b, -bound, bound)
        torch.nn.init.kaiming_uniform_(self.adapter_se_w, a=math.sqrt(5))
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.adapter_se_w)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        torch.nn.init.uniform_(self.adapter_se_b, -bound, bound)

    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_adapter_param:Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # lz:增加adapter层
        if self.has_adapter:
            batch_num = attention_output.size(0)
            line_num = attention_output.size(1)

            if layer_adapter_param is not None: # 将adapter作为参数传入
                input_tensor = torch.bmm(attention_output,torch.transpose(input=layer_adapter_param[0], dim0=2, dim1=1)[0:batch_num])
                input_tensor = input_tensor + layer_adapter_param[1].unsqueeze(1)[:batch_num]

                input_tensor = self.acfun(input_tensor)         # activation function
                
                input_tensor = torch.bmm(input_tensor,torch.transpose(input=layer_adapter_param[2], dim0=2, dim1=1)[0:batch_num])
                input_tensor = input_tensor + layer_adapter_param[3].unsqueeze(1)[:batch_num]

                attention_output = attention_output + input_tensor  # residual connection
            else:   # 使用初始化的adapter参数
                attention_output = attention_output.view([batch_num*line_num,self.in_size])
                input_tensor = torch.mm(attention_output,self.adapter_fi_w.t())
                input_tensor = input_tensor + self.adapter_fi_b

                input_tensor = self.acfun(input_tensor)         # activation function
                
                input_tensor = torch.mm(input_tensor,self.adapter_se_w.t())
                input_tensor = input_tensor + self.adapter_se_b
                attention_output = attention_output + input_tensor  # residual connection
                attention_output = attention_output.view([batch_num,line_num,self.in_size])

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # add cross attentions if we output attention weights

            # add cross-attn cache to positions 3,4 of present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
    
import numpy as np
import cupy as cp
import json
import time
import threading
import queue

def pin_memory(array):
    mem = cp.cuda.alloc_pinned_memory(array.nbytes)
    ret = np.frombuffer(mem, array.dtype, array.size).reshape(array.shape)
    ret[...] = array
    return ret


def flatten_tensors(tensor_set, chunk=None):
    tensors = [p for p in tensor_set]
    weights = [p.data for p in tensors]
    sizes = [p.numel() for p in tensors]
    total_size = sum(sizes)
    if chunk:
        total_size = ((total_size+chunk-1)//chunk)*chunk

    flatten_weights_tensor = torch.zeros(total_size, dtype=weights[0].dtype).to(weights[0].device)
    flatten_weights_storage = flatten_weights_tensor.storage()

    def set_storage(param, weight_storage, storage_offset):
        with torch.no_grad():
            z = torch.zeros_like(param.data)
            z.set_(weight_storage, storage_offset, param.shape)
            param.data = z

    offset = 0
    for i in range(len(tensors)):
        flatten_weights_tensor[offset: offset + weights[i].numel()] = weights[i].reshape(-1)
        set_storage(tensors[i], flatten_weights_storage, offset)
        offset += sizes[i]

    return flatten_weights_tensor
    
class vBertEncoder(BertEncoder):
    def __init__(self, shallow_config, config):
        super(BertEncoder, self).__init__()
        self.config = config
        self.shallow_config = shallow_config
        self.layer = nn.ModuleList(
            [
                BertLayer(shallow_config) for _ in range(config.num_hidden_layers - config.num_full_hidden_layers)
            ] + [
                BertLayer(config) for _ in range(config.num_full_hidden_layers)
            ])
    
class vBertModel(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config
        self.shallow_config = copy.deepcopy(config)
        
        self.shallow_config.hidden_size = getattr(config, 'shallow_hidden_size', 768)
        self.shallow_config.intermediate_size = getattr(config, 'shallow_intermediate_size', 3072)

        self.embeddings = BertEmbeddings(self.shallow_config)
        self.encoder = vBertEncoder(self.shallow_config, config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.embeddings.position_embeddings2 = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        self.embeddings.token_type_embeddings2 = nn.Embedding(self.config.type_vocab_size, self.config.hidden_size)
        
        self.norm = nn.LayerNorm(self.config.hidden_size)
        self.attn = nn.Linear(self.config.hidden_size, 1)
        self.shallow_skipping = ShallowSkipping(self)
        
        self.init_weights()
        self.init_pipeline_related()
        
        
    def init_pipeline_related(self):
        
        config = self.config
        
        try:
            batch_size = config.batch_size
        except:
            batch_size = 32
        try:
            bottleneck_size = config.bottleneck_size
        except:
            bottleneck_size = 64
        try:
            device = config.device
        except:
            device = torch.device("cuda:0")

        # GPU RAM Buffers
        self.max_num_batches = 4
        self.cur_batch = 0
        
        self.np_buffer_list = [[] for _ in range(self.max_num_batches)]
        self.cp_buffer_list = []
        self.torch_buffer_list = []
        
        self.numel_adapter = bottleneck_size * 768 + bottleneck_size + 768 * bottleneck_size + 768
        
        for _ in range(config.num_full_hidden_layers):
            
            for inp in range(self.max_num_batches):
                np_buffer = pin_memory(np.empty(
                    (batch_size, self.numel_adapter ), 
                    dtype=np.float32))
                self.np_buffer_list[inp].append(np_buffer)
            
            torch_buffer = []
            for i in range(batch_size):
                torch_buffer.append(
                    [
                        torch.empty(( bottleneck_size, 768), dtype=torch.float32, device=device),
                        torch.empty(( bottleneck_size), dtype=torch.float32, device=device),
                        torch.empty((768, bottleneck_size), dtype=torch.float32, device=device),
                        torch.empty((768), dtype=torch.float32, device=device)
                    ]
                )
            
            flatten_torch_buffer = flatten_tensors(sum(torch_buffer, [])).view(batch_size, -1)
            
            cp_buffer = cp.fromDlpack(torch.to_dlpack(flatten_torch_buffer))
            
            self.cp_buffer_list.append(cp_buffer)
            self.torch_buffer_list.append(torch_buffer)
            
        self.skipping_start_events = [
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(self.max_num_batches)
        ]
        self.skipping_ready_events = [
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(self.max_num_batches)
        ]
            
        self.adapter_start_events = [[
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(config.num_full_hidden_layers)
        ] for _ in range(self.max_num_batches)]
        self.adapter_ready_events = [[
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(config.num_full_hidden_layers)
        ] for _ in range(self.max_num_batches)]
        self.forward_start_events = [[
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(config.num_full_hidden_layers)
        ] for _ in range(self.max_num_batches)]
        self.forward_ready_events = [[
            torch.cuda.Event(enable_timing=True, blocking=False) for _ in range(config.num_full_hidden_layers)
        ] for _ in range(self.max_num_batches)]
        
        self.torch_comp_stream = torch.cuda.default_stream(device=device)
        self.torch_search_stream = torch.cuda.Stream(device=device, priority=-1)
        self.torch_load_stream = torch.cuda.Stream(device=device, priority=-1)
        self.cupy_load_stream = cp.cuda.ExternalStream(self.torch_load_stream.cuda_stream)
        
        self.init_forward()
        
        self.processing_queue = queue.Queue()
        self.do_copy_queue = queue.Queue()
        self.done_copy_queue = queue.Queue()
        
        def process_result(self):
            
            while True:
                batch_id = self.processing_queue.get()
                time = self.add_profiling_result(batch_id)
                print(f"{time:.2f}ms")
                self.locks[batch_id].release()
                
        def process_copy(self):
            
            while True:
                adapter_param, batch_size, batch_id = self.do_copy_queue.get()
                for i in range(self.config.num_full_hidden_layers):
                    self.np_buffer_list[batch_id][i][:batch_size] = adapter_param[i][:batch_size]
                self.done_copy_queue.put(batch_id)
            
        t = threading.Thread(target=process_result, args=(self,))
        t.start()
        self.t = t
        
        t = threading.Thread(target=process_copy, args=(self,))
        t.start()
        self.t_copy = t
        
        self.locks = [threading.Lock() for i in range(self.max_num_batches)]
        
    def get_ts(self, event):
        return self.init_time_stamp + self.init_event.elapsed_time(event) * 1e+3
    
    def add_profiling_result(self, batch_id=0):
        
        profiling_log = self.profiling_log
        
        self.skipping_ready_events[batch_id].synchronize()
        forward_slot = self.skipping_start_events[batch_id].elapsed_time(self.skipping_ready_events[batch_id]) * 1e+3
        forward_log = {"name": f"B{batch_id}", "ph": "X", "pid": batch_id, "tid": "representation retrieval",
                    "ts": self.get_ts(self.skipping_start_events[batch_id]), "dur": forward_slot,
                    "args": {}, "cname": "startup"}
        profiling_log.append(forward_log)
        
        for i in range(self.config.num_full_hidden_layers):
            
            self.adapter_ready_events[batch_id][i].synchronize()
            load_slot = self.adapter_start_events[batch_id][i].elapsed_time(self.adapter_ready_events[batch_id][i]) * 1e+3
            load_log = {"name": f"B{batch_id}", "ph": "X", "pid": batch_id, "tid": "adapter loading",
                        "ts": self.get_ts(self.adapter_start_events[batch_id][i]), "dur": load_slot,
                        "args": {"layer": i}, "cname": "grey"}
            profiling_log.append(load_log)
            
            self.forward_ready_events[batch_id][i].synchronize()
            forward_slot = self.forward_start_events[batch_id][i].elapsed_time(self.forward_ready_events[batch_id][i]) * 1e+3
            forward_log = {"name": f"B{batch_id}", "ph": "X", "pid": batch_id, "tid": "transformer calculation",
                        "ts": self.get_ts(self.forward_start_events[batch_id][i]), "dur": forward_slot,
                        "args": {"layer": i}, "cname": "good"}
            profiling_log.append(forward_log)
            
        return self.skipping_start_events[batch_id].elapsed_time(self.forward_ready_events[batch_id][-1])
    
    def export_profiling_result(self, filename):
        
        profiling_log = self.profiling_log
        with open(filename, 'w') as outfile:
            json.dump(profiling_log, outfile)
            
    def init_forward(self):
        self.init_time_stamp = time.time() * 1e+6
        self.init_event = torch.cuda.Event(enable_timing=True, blocking=False)
        self.init_event.record()
        self.profiling_log = []
        
    def do_copy_adapter_params(self, adapter_param, batch_size, batch_id):
        for i in range(self.config.num_full_hidden_layers):
            self.np_buffer_list[batch_id][i][:batch_size] = adapter_param[i]
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
        output_hidden_states=True,
        adapter_param = None,
        # batch_id = 0
    ):
        
        batch_id = self.cur_batch
        batch_size = input_ids.size(0)
        
        self.cur_batch = (self.cur_batch + 1) % self.max_num_batches
        
        self.locks[batch_id].acquire()
        
        self.do_copy_queue.put((adapter_param, batch_size, batch_id))
        
        with torch.no_grad():

            with torch.cuda.stream(self.torch_search_stream):

                ### ori init
                output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
                output_hidden_states = (
                    output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
                )

                input_shape = input_ids.size()
                device = self.device

                if attention_mask is None:
                    attention_mask = (input_ids != 0).float()
                if token_type_ids is None:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

                extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
                head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)
                ### ori init

                self.torch_search_stream.record_event(self.skipping_start_events[batch_id])

                # self.do_copy_queue.put((adapter_param, batch_size, batch_id))
                # for i in range(self.config.num_full_hidden_layers):
                #     self.np_buffer_list[i][:batch_size] = adapter_param[i][:batch_size]
                # t = threading.Thread(target=self.do_copy_adapter_params, args=(adapter_param, batch_size, batch_id))
                # t.start()
                
                hidden_states = self.shallow_skipping(
                    input_ids=input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    model=self,
                )
                
                # t.join()

                self.torch_search_stream.record_event(self.skipping_ready_events[batch_id])
                
                attention_mask = extended_attention_mask.to(device)
                
            all_hidden_states = ()
            all_self_attentions = ()
            
            with torch.cuda.stream(self.torch_load_stream):
                
                assert self.done_copy_queue.get() == batch_id
                
                self.torch_load_stream.wait_event(self.forward_ready_events[batch_id-1][0])
                self.torch_load_stream.record_event(self.adapter_start_events[batch_id][0])
                self.cp_buffer_list[0][:batch_size].set(self.np_buffer_list[batch_id][0][:batch_size], stream=self.cupy_load_stream)
                self.torch_load_stream.record_event(self.adapter_ready_events[batch_id][0])

            with torch.cuda.stream(self.torch_comp_stream):
                self.torch_comp_stream.wait_event(self.skipping_ready_events[batch_id])

            for i, layer_module in enumerate(self.encoder.layer[-self.config.num_full_hidden_layers:]):

                if i+1 < self.config.num_full_hidden_layers:
                    with torch.cuda.stream(self.torch_load_stream):
                        self.torch_load_stream.wait_event(self.forward_ready_events[batch_id-1][i+1])
                        self.torch_load_stream.record_event(self.adapter_start_events[batch_id][i+1])
                        self.cp_buffer_list[i+1][:batch_size].set(self.np_buffer_list[batch_id][i+1][:batch_size], stream=self.cupy_load_stream)
                        self.torch_load_stream.record_event(self.adapter_ready_events[batch_id][i+1])

                # layer_adapter_param = self.torch_buffer_list[i]

                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_head_mask = head_mask[i + self.config.num_hidden_layers - self.config.num_full_hidden_layers] if head_mask is not None else None

                with torch.cuda.stream(self.torch_comp_stream):

                    self.torch_comp_stream.wait_event(self.adapter_ready_events[batch_id][i])
                    self.torch_comp_stream.record_event(self.forward_start_events[batch_id][i])

                    layer_adapter_param = [
                        torch.stack([param_list[0] for param_list in self.torch_buffer_list[i][:batch_size]], 0),
                        torch.stack([param_list[1] for param_list in self.torch_buffer_list[i][:batch_size]], 0),
                        torch.stack([param_list[2] for param_list in self.torch_buffer_list[i][:batch_size]], 0),
                        torch.stack([param_list[3] for param_list in self.torch_buffer_list[i][:batch_size]], 0)
                    ]

                    layer_outputs = layer_module(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        head_mask=layer_head_mask,
                        encoder_hidden_states=encoder_hidden_states,
                        encoder_attention_mask=encoder_attention_mask,
                        past_key_value=None,
                        output_attentions=output_attentions,
                        layer_adapter_param=layer_adapter_param,
                    )

                    self.torch_comp_stream.record_event(self.forward_ready_events[batch_id][i])

                hidden_states = layer_outputs[0]

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            with torch.cuda.stream(self.torch_comp_stream):
                sequence_output = hidden_states
                pooled_output = self.pooler(sequence_output)

            self.processing_queue.put(batch_id)
        
        return (sequence_output, pooled_output, all_hidden_states, all_self_attentions)
    
    
    def freeze_shallow_layers(self):
        for p in self.embeddings.parameters():
            p.requires_grad = False
        for layer in self.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]:
            for p in layer.parameters():
                p.requires_grad = False
        try:
            for p in self.shallow_skipping.linear.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
        try:
            for p in self.attn.parameters():
                p.requires_grad = False
        except Exception as e:
            pass
                
        self.embeddings.dropout.p = 0.
        for layer in self.encoder.layer[:self.config.num_hidden_layers - self.config.num_full_hidden_layers]:
            for m in layer.modules():
                if isinstance(m, torch.nn.Dropout):
                    m.p = 0.
    

class vBertForPreTraining(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        fit_size = getattr(config, 'fit_size', 768)
        self.bert = vBertModel(config)
        self.cls = BertPreTrainingHeads(config)
        
        if self.fit_size != config.hidden_size:
            self.fit_denses = nn.ModuleList(
                [nn.Linear(config.hidden_size, self.fit_size) for _ in range(config.num_hidden_layers + 1)]
            )

    def forward(self, input_ids, token_type_ids=None,
                attention_mask=None, masked_lm_labels=None,
                next_sentence_label=None, labels=None,
                output_attentions=True, output_hidden_states=True,):
        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states)
        
        if self.fit_size != self.config.hidden_size:
            tmp = []
            for s_id, sequence_layer in enumerate(sequence_output):
                tmp.append(self.fit_denses[s_id](sequence_layer))
            sequence_output = tmp

        return att_output, sequence_output

    
    def freeze_shallow_layers(self):
        self.bert.freeze_shallow_layers()

    
class vBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, do_fit=False, share_param=True):
        super().__init__(config)
        num_labels = config.num_labels
        self.hidden_size = config.hidden_size
        self.num_labels = num_labels
        self.bert = vBertModel(config)
        self.dropout = nn.Dropout(
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        self.do_fit, self.share_param = do_fit, share_param
        if self.do_fit:
            fit_size = getattr(config, 'fit_size', 768)
            self.fit_size = fit_size
            if self.share_param:
                self.share_fit_dense = nn.Linear(config.hidden_size, fit_size)
            else:
                self.fit_denses = nn.ModuleList(
                    [nn.Linear(config.hidden_size, fit_size) for _ in range(config.num_hidden_layers + 1)]
                )

    def do_fit_dense(self, sequence_output):
        
        tmp = []
        if self.do_fit:
            for s_id, sequence_layer in enumerate(sequence_output):
                if self.share_param:
                    tmp.append(self.share_fit_dense(sequence_layer))
                else:
                    tmp.append(self.fit_denses[s_id](sequence_layer))
            sequence_output = tmp
            
        return sequence_output

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):

        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
        
        sequence_output = self.do_fit_dense(sequence_output)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits, att_output, sequence_output
    
    def freeze_shallow_layers(self):
        self.bert.freeze_shallow_layers()
    

class vBertForSequenceClassificationPrediction(vBertForSequenceClassification):
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        
        assert not self.training
        
        _, pooled_output, sequence_output, att_output = self.bert(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
            output_hidden_states=True, output_attentions=True)
        
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            loss = torch.tensor(0.)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )