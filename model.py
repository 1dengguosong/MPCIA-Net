from torch import softmax

from torch_scatter import scatter_add
from transformers import (ElectraModel, ElectraPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel,
                          BertConfig)
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, Conv1d
import torch.nn.functional as F
import math
import torch.nn.utils.rnn as rnn_utils
from torch.autograd import Variable

BertLayerNorm = torch.nn.LayerNorm

ACT2FN = {"gelu": F.gelu, "relu": F.relu}


class DCDM(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1,
                         3)  # (batch_size, seq_len, all_head_size) -> (batch_size, num_attention_heads, seq_len, attention_head_size)

    def forward(self, input_ids_a, input_ids_b, attention_mask=None, head_mask=None, output_attentions=False):
        mixed_query_layer = self.query(input_ids_a)
        mixed_key_layer = self.key(input_ids_b)
        mixed_value_layer = self.value(input_ids_b)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size) #batch_size,num_head,512,512
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1)
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

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        # (batch_size * num_choice, num_attention_heads, seq_len, attention_head_size) -> (batch_size * num_choice, seq_len, num_attention_heads, attention_head_size)

        # Should find a better way to do this
        w = (
            self.dense.weight.t()
            .view(self.num_attention_heads, self.attention_head_size, self.hidden_size)
            .to(context_layer.dtype)
        )
        b = self.dense.bias.to(context_layer.dtype)

        projected_context_layer = torch.einsum("bfnd,ndh->bfh", context_layer, w) + b
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_ids_a + projected_context_layer_dropout)
        return layernormed_context_layer


# 使用双向 GRU 管理可变长度的输入序列，该序列对于通过填充和打包序列导致的序列长度变化具有鲁棒性
class GRUWithPadding(nn.Module):
    def __init__(self, config, num_rnn=1):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_layers = num_rnn
        self.biGRU = nn.GRU(config.hidden_size, config.hidden_size, self.num_layers, batch_first=True,
                            bidirectional=True)

    def forward(self, inputs):
        batch_size = len(inputs)

        # Sort inputs by sequence length in descending order
        sorted_inputs = sorted(enumerate(inputs), key=lambda x: x[1].size(0), reverse=True)
        idx_inputs = [i[0] for i in sorted_inputs]
        inputs = [i[1] for i in sorted_inputs]
        inputs_lengths = [len(i[1]) for i in sorted_inputs]

        # Padding the sequences to make them of equal length
        inputs = rnn_utils.pad_sequence(inputs, batch_first=True)
        inputs = rnn_utils.pack_padded_sequence(inputs, inputs_lengths, batch_first=True)

        # Initialize hidden states for GRU (2 directions * num_layers)
        h0 = torch.rand(2 * self.num_layers, batch_size, self.hidden_size).to(inputs.data.device)

        # Flatten parameters for efficiency
        self.biGRU.flatten_parameters()

        # Forward pass through the bidirectional GRU
        out, _ = self.biGRU(inputs, h0)  # (batch_size, seq_len, 2 * hidden_size)

        # Unpacking the padded sequences
        out_pad, out_len = rnn_utils.pad_packed_sequence(out,
                                                         batch_first=True)  # (batch_size, seq_len, 2 * hidden_size)

        # Reordering output to match original batch order
        _, idx2 = torch.sort(torch.tensor(idx_inputs))
        idx2 = idx2.to(out_pad.device)
        output = torch.index_select(out_pad, 0, idx2)

        # Option 1: Use only the forward hidden state (first half of the output)
        # output = output[:, :, :self.hidden_size]  # Forward hidden state only

        # Option 2: Combine both forward and backward states (by averaging)
        output = (output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]) / 2  # Averaging the forward and backward states

        # Option 3: Concatenate both directions (if you want to keep bidirectional features)
        # output = torch.cat((output[:, :, :self.hidden_size], output[:, :, self.hidden_size:]), dim=-1)  # Concatenate forward and backward

        return output

class FuseLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.linear1 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(4 * config.hidden_size, config.hidden_size)
        self.linear3 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        self.activation = nn.ReLU()
        self.gate = nn.Sigmoid()

    def forward(self, orig, input1, input2):
        out1 = self.activation(self.linear1(torch.cat([orig, input1, orig - input1, orig * input1], dim=-1)))
        out2 = self.activation(self.linear2(torch.cat([orig, input2, orig - input2, orig * input2], dim=-1)))
        fuse_prob = self.gate(self.linear3(torch.cat([out1, out2], dim=-1)))

        return fuse_prob * input1 + (1 - fuse_prob) * input2

def pad(tensor, length, cuda_flag):
    if isinstance(tensor, Variable):
        var = tensor
        if length > var.size(0):
            if cuda_flag:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:]).cuda()])
            else:
                return torch.cat([var, torch.zeros(length - var.size(0), *var.size()[1:])])
        else:
            return var
    else:
        if length > tensor.size(0):
            if cuda_flag:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:]).cuda()])
            else:
                return torch.cat([tensor, torch.zeros(length - tensor.size(0), *tensor.size()[1:])])
        else:
            return tensor
def feature_transfer(bank_s_, bank_p_, seq_lengths, cuda_flag=False):
    input_conversation_length = torch.tensor(seq_lengths)
    start_zero = input_conversation_length.data.new(1).zero_()
    if cuda_flag:
        input_conversation_length = input_conversation_length.cuda()
        start_zero = start_zero.cuda()

    max_len = max(seq_lengths)
    start = torch.cumsum(torch.cat((start_zero, input_conversation_length[:-1])), 0)
    # (l,b,h)
    bank_s = torch.stack(
        [pad(bank_s_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)
    bank_p = torch.stack(
        [pad(bank_p_.narrow(0, s, l), max_len, cuda_flag) for s, l in zip(start.data.tolist(), input_conversation_length.data.tolist())], 0
    ).transpose(0, 1)

    return bank_s, bank_p

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import ipdb
from config import args, device
import torch.nn.functional as F
import random
from config import args


class SpeakerUtteranceAttention(nn.Module):
    def __init__(self, hidden_size, dk):
        super(SpeakerUtteranceAttention, self).__init__()
        self.W_q = nn.Linear(hidden_size, dk)
        self.W_k = nn.Linear(hidden_size, dk)
        self.W_v = nn.Linear(hidden_size, hidden_size)
        self.dk = dk
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, context_final_states, sa_final_states):
        # 计算注意力权重
        h_u_q = self.W_q(context_final_states)  # 话语上下文 h_u* 映射为 q
        h_s_k = self.W_k(sa_final_states)  # 说话者上下文 h_s* 映射为 k

        scores = torch.matmul(h_u_q, h_s_k.transpose(-2, -1)) / torch.sqrt(torch.tensor(float(self.dk)))
        alpha = torch.softmax(scores, dim=-1)  # 注意力权重

        # 计算融合结果
        attn_output = torch.matmul(sa_final_states, alpha.transpose(-2, -1))
        attn_output = self.W_v(attn_output)
        final_state = attn_output + context_final_states  # 残差连接
        final_state = self.layer_norm(final_state)
        return final_state
class REModel(nn.Module):
    def __init__(self, hidden_size=768,num_decoupling=1,num_rnn=1):
        super(REModel, self).__init__()
        config = BertConfig.from_pretrained(args.model, output_hidden_states=True, output_attentions=True)
        tokenizer = BertTokenizer.from_pretrained(args.model,config=config)
        tokenizer.add_tokens(["[s1]", "[s2]"])
        self.bert = BertModel.from_pretrained(args.model,output_hidden_states=True, output_attentions=True)
        self.bert.resize_token_embeddings(len(tokenizer))
        self.fc = nn.Linear(2 * hidden_size, hidden_size)
        self.proj_relation = nn.Linear(2 * hidden_size, 37)
        self.proj_trigger = nn.Linear(hidden_size, 2)
        self.proj_binary = nn.Linear(hidden_size, 2)
        self.num_decoupling = num_decoupling
        self.SASelfMHA = SpeakerUtteranceAttention(hidden_size,hidden_size)

        # 初始化聚合模块
        self.localMHA = nn.ModuleList([DCDM(args) for _ in range(num_decoupling)])
        self.globalMHA = nn.ModuleList([DCDM(args) for _ in range(num_decoupling)])
        self.SASelfMHA = nn.ModuleList([DCDM(args) for _ in range(num_decoupling)])
        self.SACrossMHA = nn.ModuleList([DCDM(args) for _ in range(num_decoupling)])

        self.fuse1 = FuseLayer(args)
        self.fuse2 = FuseLayer(args)

        self.gru1 = GRUWithPadding(args, num_rnn)
        self.gru2 = GRUWithPadding(args, num_rnn)

        self.pooler = nn.Linear(2 * hidden_size, hidden_size)
        self.pooler_activation = nn.Tanh()
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.proj_reduce = nn.Linear(2 * hidden_size, hidden_size)

    def reset(self, class_cnt, hidden_size=768):
        self.proj_relation = nn.Linear(2 * hidden_size, class_cnt)

    def forward(self, inputs):
        # Phase 1: feed inputs to self.bert and extract the hidden states
        # Phase 2-1: keep the context part unmasked, and apply start & end prediction
        # Phase 2-2: concatenate hid_trigger, x, and y,
        # then pass this concatenated tensor to self.proj_relation
        last_hidden_states = self.bert(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        last_hidden_states = last_hidden_states["last_hidden_state"]

        # 计算话语内和话语间注意力 话语感知
        local_word_level = self.localMHA[0](last_hidden_states, last_hidden_states,
                                            attention_mask=inputs['utt_local_mask']) #[seqlen,hidden]

        global_word_level = self.globalMHA[0](last_hidden_states, last_hidden_states,
                                              attention_mask=inputs['utt_global_mask'])

        # 计算说话者内和说话者间的注意力 说话者感知
        sa_self_word_level = self.SASelfMHA[0](last_hidden_states, last_hidden_states,
                                               attention_mask=inputs['sa_self_mask'])
        sa_cross_word_level = self.SACrossMHA[0](last_hidden_states, last_hidden_states,
                                                 attention_mask=inputs['sa_cross_mask'])

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level, attention_mask=inputs['utt_local_mask'])
            global_word_level = self.globalMHA[t](global_word_level, global_word_level, attention_mask=inputs['utt_global_mask'])
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level, attention_mask=inputs['sa_self_mask'])
            sa_cross_word_level = \
                self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask=inputs['sa_cross_mask'])

        context_word_level = self.fuse1(last_hidden_states, local_word_level, global_word_level) #[batch_size,seqlen,hidden]
        sa_word_level = self.fuse2(last_hidden_states,  sa_cross_word_level,sa_self_word_level) ##[batch_size,seqlen,hidden]

        context_final_states = self.gru1(context_word_level)  #[batch_size,seqlen,hidden]

        sa_final_states = self.gru2(sa_word_level)   #[batch_size,seqlen,hidden]

        """ w_c = torch.nn.Parameter(torch.ones(1)).to(device) # 权重w_a
        w_s = torch.nn.Parameter(torch.ones(1)).to(device)  # 权重w_b
        #这里的并不可靠
        final_state = w_c * context_final_states + w_s * sa_final_states #[batch_size,seqlen,hidden]"""


        final_state = self.SASelfMHA(context_final_states,sa_final_states)
        pooled_output = self.pooler_activation(final_state) #[batch_size,seqlen,hidden]
        pooled_output = self.dropout(pooled_output) #[batch_size,seqlen,hidden]

        #预测有无触发次词
        start_end_logit = self.proj_trigger(pooled_output)  # shape: (batch_size, seq_len, 2) context_word_level #[batch_size,seqlen,hidden]

        ids = []
        for x_idx in inputs['x_idx']:
            ids.append([0, x_idx[0] - 1])
        masked_start_end_logit = self.get_masked(
            start_end_logit,
            torch.tensor(ids),
            mask_val=float('-inf')
        )
        #触发词预测
        ids = self.get_triggers_ids(masked_start_end_logit)
        #触发词向量计算
        trigger = self.attention(pooled_output, torch.tensor(ids))
        #x 向量是指在触发词之后的一个 token 的表示
        x = []
        for b_idx in range(len(pooled_output)):
            x.append(pooled_output[b_idx][inputs['x_idx'][b_idx][1] + 1, :])
        x = torch.vstack(x)
        #将触发词表示和 CLS token 的表示拼接在一起
        a = pooled_output[:, 0, :]
        concat_hid = torch.hstack((trigger,pooled_output[:, 0, :]))
        relation_logit = self.proj_relation(concat_hid)
        binary_logit = self.proj_binary(x)

        #[batch_size,37] - [batch_size,512,2] - [batch_size,2]
        return relation_logit, start_end_logit, binary_logit  # , distributions#, tri_len_logit

    def get_triggers_ids(self, masked_start_end_logit, tri_len=None):
        ids = []

        if tri_len is None:
            for batch_idx, sample in enumerate(masked_start_end_logit):
                start = sample[:, 0]  # start: shape (512)
                end = sample[:, 1]
                start_candidates = torch.topk(start, k=30)
                end_candidates = torch.topk(end, k=30)
                ans_candidates = [(0, 1)]
                scores = [-100]
                start_logits = F.softmax(start_candidates[0])
                end_logits = F.softmax(end_candidates[0])
                for i, s in enumerate(start_candidates[1]):
                    for j, e in enumerate(end_candidates[1]):
                        if s == 0:
                            ans_candidates.append((s, s + 1))
                            # scores.append(start_candidates[0][i]+end_candidates[0][j])
                            scores.append(start_logits[i] * end_logits[j])
                        if s < e and e - s <= 10:
                            ans_candidates.append((s, e))
                            # scores.append(start_candidates[0][i]+end_candidates[0][j])
                            scores.append(start_logits[i] * end_logits[j])
                results = list(zip(scores, ans_candidates))
                results.sort()
                results.reverse()

                ids.append([int(results[0][1][0]), int(results[0][1][1])])
            return ids
        else:
            for batch_idx, sample in enumerate(masked_start_end_logit):
                start = sample[:, 0]  # start: shape (512)
                end = sample[:, 1]
                start_logits = F.softmax(start)
                end_logits = F.softmax(end)
                # start_candidates = torch.topk(start, k=30)
                # end_candidates = torch.topk(end, k=30)
                max_score = float('-inf')
                cand = None
                for i in range(len(start_logits) - tri_len[batch_idx]):
                    # for j in range(i+1, len(end_logits)):
                    #     if j - i <= 14:
                    cur_score = start_logits[i] + end_logits[i + tri_len[batch_idx]]
                    if cur_score > max_score:
                        max_score = cur_score
                        cand = [i, i + tri_len[batch_idx]]
                ids.append(cand)
            return ids

    def infer(self, inputs):
        last_hidden_states = self.bert(inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids'])
        last_hidden_states = last_hidden_states["last_hidden_state"]

        # 计算话语内和话语间注意力 话语感知
        local_word_level = self.localMHA[0](last_hidden_states, last_hidden_states,
                                            attention_mask=inputs['utt_local_mask'])  # [seqlen,hidden]

        global_word_level = self.globalMHA[0](last_hidden_states, last_hidden_states,
                                              attention_mask=inputs['utt_global_mask'])

        # 计算说话者内和说话者间的注意力 说话者感知
        sa_self_word_level = self.SASelfMHA[0](last_hidden_states, last_hidden_states,
                                               attention_mask=inputs['sa_self_mask'])
        sa_cross_word_level = self.SACrossMHA[0](last_hidden_states, last_hidden_states,
                                                 attention_mask=inputs['sa_cross_mask'])

        for t in range(1, self.num_decoupling):
            local_word_level = self.localMHA[t](local_word_level, local_word_level,
                                                attention_mask=inputs['utt_local_mask'])
            global_word_level = self.globalMHA[t](global_word_level, global_word_level,
                                                  attention_mask=inputs['utt_global_mask'])
            sa_self_word_level = self.SASelfMHA[t](sa_self_word_level, sa_self_word_level,
                                                   attention_mask=inputs['sa_self_mask'])
            sa_cross_word_level = \
                self.SACrossMHA[t](sa_cross_word_level, sa_cross_word_level, attention_mask=inputs['sa_cross_mask'])

        context_word_level = self.fuse1(last_hidden_states, local_word_level,
                                        global_word_level)  # [batch_size,seqlen,hidden]
        sa_word_level = self.fuse2(last_hidden_states, sa_cross_word_level,
                                   sa_self_word_level)  ##[batch_size,seqlen,hidden]

        context_final_states = self.gru1(context_word_level)  # [batch_size,seqlen,hidden]

        sa_final_states = self.gru2(sa_word_level)  # [batch_size,seqlen,hidden]

        final_state = self.r * context_final_states + (1 - self.r) * sa_final_states  # [batch_size,seqlen,hidden]

        pooled_output = self.pooler_activation(final_state)  # [batch_size,seqlen,hidden]
        pooled_output = self.dropout(pooled_output)  # [batch_size,seqlen,hidden]

        #这里是原始代码

        start_end_logit = self.proj_trigger(pooled_output)  # shape: (batch_size, seq_len, 2)

        ids = []
        for x_idx in inputs['x_idx']:
            ids.append([0, x_idx[0] - 1])
        masked_start_end_logit = self.get_masked(
            start_end_logit,
            torch.tensor(ids),
            mask_val=float('-inf')
        )


        ids = self.get_triggers_ids(masked_start_end_logit)
        tokenizer = BertTokenizer.from_pretrained(args.model)
        tokenizer.add_tokens(["[s1]", "[s2]"])
        p_trigs, gt_trigs = [], []
        for i in range(len(inputs['input_ids'])):
            p_trig = tokenizer.decode(inputs['input_ids'][i][ids[i][0]: ids[i][1]])
            p_trigs.append(p_trig)
            gt_trig = tokenizer.decode(inputs['input_ids'][i] \
                                           [inputs['t_idx'][i][0]: inputs['t_idx'][i][1]])
            gt_trigs.append(gt_trig)

        # trigger = self.attention(last_hidden_states, inputs['t_idx'])
        trigger = self.attention(pooled_output, torch.tensor(ids))

        x = []
        for b_idx in range(len(pooled_output)):
            x.append(pooled_output[b_idx][inputs['x_idx'][b_idx][1] + 1, :])
        x = torch.vstack(x)
        # concat_hid = torch.hstack((trigger, x))
        binary_logit = self.proj_binary(x)
        bin_pred = torch.argmax(binary_logit, dim=1)

        # cls = []
        for batch_idx in range(len(bin_pred)):
            if bin_pred[batch_idx] == 0:
                trigger[batch_idx][:] = torch.zeros(len(pooled_output[batch_idx, 0, :]))


        concat_hid = torch.hstack((trigger, pooled_output[:, 0, :]))

        relation_logit = self.proj_relation(concat_hid)
        # return torch.argmax(relation_logit, dim=1), p_trigs, gt_trigs
        argmax = torch.argmax(relation_logit, dim=1)
        has_trigger = torch.argmax(binary_logit, dim=1)

        return argmax, has_trigger, ids, p_trigs, gt_trigs

    def get_masked(self, mat, ids, mask_val=0):
        batch_size, seq_len, cls = mat.shape
        mask = torch.ones(batch_size, seq_len, cls)
        for i in range(batch_size):
            mask[i, ids[i][0]:ids[i][1], :] = 0
        mask = mask.bool()
        return mat.masked_fill(mask.to(device), mask_val)

    def get_trigger(self, mat, ids, mask_val=0, length=15):
        batch_size, seq_len, cls = mat.shape
        triggers = []
        for b_id in range(batch_size):
            # self.total += 1
            trigger = mat[b_id][ids[b_id][0]: ids[b_id][1]][:]
            if len(trigger) < length:
                padding = torch.zeros(length - len(trigger), cls)
                padding = padding.to(device)
                trigger = torch.vstack((trigger, padding))
            else:
                # self.long_trig += 1
                trigger = trigger[:length]
            try:
                assert len(trigger) == length
            except:
                ipdb.set_trace()
            triggers.append(trigger)

        return torch.vstack(triggers).view(batch_size, -1, cls)

    def attention(self, mat, ids):
        triggers = []
        batch_size, _, _ = mat.shape
        for b_id in range(batch_size):
            trigger = mat[b_id][ids[b_id][0]: ids[b_id][1]][:]
            score = []
            cls = mat[b_id, 0, :]
            # score.append(torch.dot(cls, cls))
            for j in range(len(trigger)):
                score.append(torch.dot(cls, trigger[j]))
            score = torch.tensor(score, device=device)
            score = F.softmax(score)
            # trigger = torch.vstack((cls, trigger))
            triggers.append(torch.matmul(trigger.T, score))
        return torch.vstack(triggers)

    def get_trigger_and_lengths(self, mat, ids, mask_val=0):
        lengths = []
        batch_size, seq_len, cls = mat.shape
        mask = torch.ones(batch_size, seq_len, cls)
        for i in range(batch_size):
            mask[i, ids[i][0]:ids[i][1], :] = 0
        mask = mask.bool()
        for b_id in range(batch_size):
            # self.total += 1
            lengths.append(ids[b_id][1] - ids[b_id][0])
        return mat.masked_fill(mask.to(device), mask_val), \
            1 / torch.vstack(lengths)