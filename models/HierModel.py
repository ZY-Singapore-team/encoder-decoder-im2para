#= This file contains Att2in2, AdaAtt, AdaAttMO, TopDown model

# AdaAtt is from Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning
# https://arxiv.org/abs/1612.01887
# AdaAttMO is a modified version with maxout lstm

# Att2in is from Self-critical Sequence Training for Image Captioning
# https://arxiv.org/abs/1612.00563
# In this file we only have Att2in2, which is a slightly different version of att2in,
# in which the img feature embedding and word embedding is the same as what in adaatt.

# TopDown is from Bottom-Up and Top-Down Attention for Image Captioning and VQA
# https://arxiv.org/abs/1707.07998
# However, it may not be identical to the author's architecture.
import torch
import torch.nn as nn
import torch.nn.functional as F
import misc.utils as utils
import copy
import math
import numpy as np
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence, pad_packed_sequence
from transformers import BertModel
from .CaptionModel import CaptionModel


def sort_pack_padded_sequence(input, lengths):
    sorted_lengths, indices = torch.sort(lengths, descending=True)
    tmp = pack_padded_sequence(input[indices], sorted_lengths, batch_first=True)
    inv_ix = indices.clone()
    inv_ix[indices] = torch.arange(0,len(indices)).type_as(inv_ix)
    return tmp, inv_ix

def pad_unsort_packed_sequence(input, inv_ix):
    tmp, _ = pad_packed_sequence(input, batch_first=True)
    tmp = tmp[inv_ix]
    return tmp

def pack_wrapper(module, att_feats, att_masks):
    if att_masks is not None:
        packed, inv_ix = sort_pack_padded_sequence(att_feats, att_masks.data.long().sum(1))
        return pad_unsort_packed_sequence(PackedSequence(module(packed[0]), packed[1]), inv_ix)
    else:
        return module(att_feats)

class HierModel(CaptionModel):
    def __init__(self, opt):
        super(HierModel, self).__init__()
        self.vocab_size = opt.vocab_size
        self.input_encoding_size = opt.input_encoding_size
        self.rnn_size = opt.rnn_size
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm
        self.seq_length = opt.seq_length
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.fs_index = opt.fs_index

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.relu_mod = getattr(opt, 'relu_mod', 'relu')
        self.rela_index = getattr(opt, 'rela_index', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        if self.relu_mod == 'relu':
            self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
            self.bert_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn else ())+
                                        (nn.Linear(self.input_encoding_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))           
            self.att_embed = nn.Sequential(*(
                                        ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ())+
                                        (nn.Linear(self.att_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))+
                                        ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn==2 else ())))
        elif self.relu_mod == 'leaky_relu':
            self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                          nn.LeakyReLU(0.1, inplace=True),
                                          nn.Dropout(self.drop_prob_lm))
            self.att_embed = nn.Sequential(*(
                    ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) +
                    (nn.Linear(self.att_feat_size, self.rnn_size),
                     nn.LeakyReLU(0.1, inplace=True),
                     nn.Dropout(self.drop_prob_lm)) +
                    ((nn.BatchNorm1d(self.rnn_size),) if self.use_bn == 2 else ())))

        self.logit_layers = getattr(opt, 'logit_layers', 1)
        if self.logit_layers == 1:
            self.logit = nn.Linear(self.rnn_size, self.vocab_size + 1)
        self.ctx2att = nn.Linear(self.rnn_size, self.att_hid_size)
        # BERT model
        self.bert_model = BertModel.from_pretrained("bert-base-cased")

    def init_BERT_embeddings(self, embedding_weights):
        self.embed.weight.data.copy_(torch.from_numpy(embedding_weights))
        # self.weight.requires_grad = False

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        return (weight.new_zeros(self.num_layers, bsz, self.rnn_size),
                weight.new_zeros(self.num_layers, bsz, self.rnn_size))

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature(self, fc_feats, att_feats, att_masks):

        # embed fc and att feats
        fc_feats = self.fc_embed(fc_feats)

        if self.rela_index:
            att_feats = self.rela_mod(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        # Project the attention feats first to reduce memory and computation comsumptions.
        p_att_feats = self.ctx2att(att_feats)

        return fc_feats, att_feats, p_att_feats

    def _forward(self, bert_feats, sen_bert_feats, fc_feats, att_feats, seq, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)
        bert_feats = self.bert_embed(bert_feats)
        sen_bert_feats = self.bert_embed(sen_bert_feats)

        for i in range(seq.size(1) - 1):
            if self.training and i >= 1 and self.ss_prob > 0.0: # otherwiste no need to sample
                sample_prob = fc_feats.new(batch_size).uniform_(0, 1)
                sample_mask = sample_prob < self.ss_prob
                if sample_mask.sum() == 0:
                    it = seq[:, i].clone()
                else:
                    sample_ind = sample_mask.nonzero().view(-1)
                    it = seq[:, i].data.clone()
                    prob_prev = torch.exp(outputs[:, i-1].detach()) # fetch prev distribution: shape Nx(M+1)
                    it.index_copy_(0, sample_ind, torch.multinomial(prob_prev, 1).view(-1).index_select(0, sample_ind))
            else:
                it = seq[:, i].clone()          
            # break if all the sequences end
            if i >= 1 and seq[:, i].sum() == 0:
                break

            output, state = self.get_logprobs_state(it, bert_feats[:,i,:], sen_bert_feats[:,i,:], fc_feats, att_feats, p_att_feats, att_masks, state)
            outputs[:, i] = output
            # outputs.append(output)

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, bert_feats, sen_bert_feats, fc_feats, att_feats, p_att_feats, att_masks, state):
        # 'it' contains a word index
        # xt = self.embed(it)

        # BERT version of this; xt:(batch_size * 1 * 768)
        st = sen_bert_feats
        xt = bert_feats
        st.requires_grad = False
        xt.requires_grad = False            
        cs_index = it==int(self.fs_index)

        output, state = self.core(st, xt, cs_index, fc_feats, att_feats, p_att_feats, state, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state

    # This is for inference, should apply none teacher-forcing. 
    def inference_logprobs_state(self, it, previous_seq, fc_feats, att_feats, p_att_feats, att_masks, state):
        # previous_seq: [batch_size, pre_len]
        pass

    def _sample_beam(self, fc_feats, att_feats, att_masks=None, opt={}):
        beam_size = opt.get('beam_size', 10)
        batch_size = fc_feats.size(0)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        assert beam_size <= self.vocab_size + 1, 'lets assume this for now, otherwise this corner case causes a few headaches down the road. can be dealt with in future if needed'
        seq = torch.LongTensor(self.seq_length, batch_size).zero_()
        seqLogprobs = torch.FloatTensor(self.seq_length, batch_size)
        # lets process every image independently for now, for simplicity

        self.done_beams = [[] for _ in range(batch_size)]
        for k in range(batch_size):
            state = self.init_hidden(beam_size)
            tmp_fc_feats = fc_feats[k:k+1].expand(beam_size, fc_feats.size(1))
            tmp_att_feats = att_feats[k:k+1].expand(*((beam_size,)+att_feats.size()[1:])).contiguous()
            tmp_p_att_feats = p_att_feats[k:k+1].expand(*((beam_size,)+p_att_feats.size()[1:])).contiguous()
            tmp_att_masks = att_masks[k:k+1].expand(*((beam_size,)+att_masks.size()[1:])).contiguous() if att_masks is not None else None

            it = fc_feats.new_zeros([beam_size], dtype=torch.long) # input <bos>
            logprobs, state = self.get_logprobs_state(it, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, state)

            self.done_beams[k] = self.beam_search(state, logprobs, tmp_fc_feats, tmp_att_feats, tmp_p_att_feats, tmp_att_masks, opt=opt)
            seq[:, k] = self.done_beams[k][0]['seq'] # the first beam has highest cumulative score
            seqLogprobs[:, k] = self.done_beams[k][0]['logps']
        # return the samples and their log likelihoods
        return seq.transpose(0, 1), seqLogprobs.transpose(0, 1)

    def _sample(self, bert_info, fc_feats, att_feats, att_masks=None, opt={}):

        # bert_tokens: (batch_size, max_seq_len)
        bert_tokens = bert_info['tokens']
        ix_to_BERT = bert_info['vocab']
        att_feats, att_masks = self.clip_att(att_feats, att_masks)

        sample_max = opt.get('sample_max', 1)
        beam_size = opt.get('beam_size', 1)
        temperature = opt.get('temperature', 1.0)
        decoding_constraint = opt.get('decoding_constraint', 0)
        block_trigrams = opt.get('block_trigrams', 0)
        if beam_size > 1:
            return self._sample_beam(fc_feats, att_feats, att_masks, opt)

        batch_size = fc_feats.size(0)
        state = self.init_hidden(batch_size)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        # BEG MODIFIED
        trigrams = [] # will be a list of batch_size dictionaries
        # END MODIFIED

        # seq = []
        # seqLogprobs = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)

        # the initial bert hidden state
        current_bert_tokens = bert_tokens[:, 0:1]
        # current_bert_tokens = torch.from_numpy(current_bert_tokens).long()     
        current_bert_feats = self.bert_model(current_bert_tokens)[0]
        # current_bert_feats = embeddings.data.cpu().numpy().astype(np.float32)
        # full_bert_tokens = current_bert_tokens.copy()
        full_bert_tokens = current_bert_tokens.clone().detach()
        current_bert_feats = current_bert_feats.squeeze(1)

        for t in range(self.seq_length + 1):
            if t == 0: # input <bos>
                it = fc_feats.new_zeros(batch_size, dtype=torch.long)
            elif sample_max:
                sampleLogprobs, it = torch.max(logprobs.data, 1)
                it = it.view(-1).long()
            else:
                if temperature == 1.0:
                    prob_prev = torch.exp(logprobs.data) # fetch prev distribution: shape Nx(M+1)
                else:
                    # scale logprobs by temperature
                    prob_prev = torch.exp(torch.div(logprobs.data, temperature))
                it = torch.multinomial(prob_prev, 1)
                sampleLogprobs = logprobs.gather(1, it) # gather the logprobs at sampled positions
                it = it.view(-1).long() # and flatten indices for downstream processing

            if t >= 1:
                # stop when all finished
                if t == 1:
                    unfinished = it > 0
                else:
                    unfinished = unfinished * (it > 0)
                if unfinished.sum() == 0:
                    break
                it = it * unfinished.type_as(it)
                seq[:,t-1] = it
                # seq.append(it) #seq[t] the input of t+2 time step

                # seqLogprobs.append(sampleLogprobs.view(-1))
                seqLogprobs[:,t-1] = sampleLogprobs.view(-1)

            logprobs, state = self.get_logprobs_state(it, current_bert_feats, current_bert_feats, fc_feats, att_feats, p_att_feats, att_masks, state)
            
            # decode the current prediction word
            # predicts: [batch_size, 1]
            _, predicts = torch.max(torch.softmax(logprobs, 1), 1)
            predicts = predicts.unsqueeze(1)
            # should map predicts to BERT token ids
            predicts_bert = np.array([ix_to_BERT[e] for e in predicts.flatten().cpu().numpy()]).reshape(predicts.shape)
            predicts_bert = torch.from_numpy(predicts_bert).long().cuda()
            # get current contexts
            full_bert_tokens = torch.cat((full_bert_tokens, predicts_bert), 1)
            # extract bert feats
            current_bert_feats = self.bert_model(full_bert_tokens)[0][:, -1, :]

            if decoding_constraint and t > 0:
                tmp = output.new_zeros(output.size(0), self.vocab_size + 1)
                tmp.scatter_(1, seq[:,t-1].data.unsqueeze(1), float('-inf'))
                logprobs = logprobs + tmp


            # BEG MODIFIED ----------------------------------------------------

            # Mess with trigrams
            if block_trigrams and t >= 3 and sample_max:
                # Store trigram generated at last step
                prev_two_batch = seq[:,t-3:t-1]
                for i in range(batch_size): # = seq.size(0)
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    current  = seq[i][t-1]
                    if t == 3: # initialize
                        trigrams.append({prev_two: [current]}) # {LongTensor: list containing 1 int}
                    elif t > 3:
                        if prev_two in trigrams[i]: # add to list
                            trigrams[i][prev_two].append(current)
                        else: # create list
                            trigrams[i][prev_two] = [current]
                # Block used trigrams at next step
                prev_two_batch = seq[:,t-2:t]
                mask = torch.zeros(logprobs.size(), requires_grad=False).cuda() # batch_size x vocab_size
                for i in range(batch_size):
                    prev_two = (prev_two_batch[i][0].item(), prev_two_batch[i][1].item())
                    if prev_two in trigrams[i]:
                        for j in trigrams[i][prev_two]:
                            mask[i,j] += 1
                # Apply mask to log probs
                #logprobs = logprobs - (mask * 1e9)
                alpha = 2.0 # = 4
                logprobs = logprobs + (mask * -0.693 * alpha) # ln(1/2) * alpha (alpha -> infty works best)
                
            # END MODIFIED ----------------------------------------------------

        return seq, seqLogprobs


class HTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(HTopDownCore, self).__init__()
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.satt_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size) 
        self.sent_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.watt_lstm = nn.LSTMCell(opt.rnn_size * 3, opt.rnn_size)
        self.word_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, st, xt, cs_index, fc_feats, att_feats, p_att_feats, state, att_masks=None):

        ####################sentence level######################
        prev_h = state[0][1]
        att_lstm_input = torch.cat([prev_h, st, fc_feats], 1)

        h_att, c_att = self.satt_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        sen_lstm_input = torch.cat([att, h_att], 1)
        h_sen, c_sen = self.sent_lstm(sen_lstm_input, (state[0][1], state[1][1]))

        # get topic vectors when a sentence is ended at [SEP]
        # batch_size = fc_feats.size(0)
        # for batch_id in range(batch_size):
        #     if cs_index[batch_id]==1:
        #         old_state[batch_id] = h_sen[batch_id]
        topic_vector = h_sen

        ####################word level##########################
        if self.num_layers == 3:
            prev_h_word = state[0][2]
            word_lstm_input = torch.cat([xt, prev_h_word, topic_vector], 1)
            h_word, c_word = self.watt_lstm(word_lstm_input, (state[0][2], state[1][2]))
            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_word]), torch.stack([c_att, c_sen, c_word]))

        elif self.num_layers == 4:
            prev_h_word = state[0][3]
            watt_lstm_input = torch.cat([xt, prev_h_word, topic_vector], 1)
            h_watt, c_watt = self.watt_lstm(watt_lstm_input, (state[0][2], state[1][2]))
            watt = self.attention(h_watt, att_feats, p_att_feats, att_masks)
            word_lstm_input = torch.cat([watt, h_watt], 1)
            h_word, c_word = self.word_lstm(word_lstm_input, (state[0][3], state[1][3]))

            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_watt, h_word]), torch.stack([c_att, c_sen, c_watt, c_word]))

        return output, state


class Attention(nn.Module):
    def __init__(self, opt):
        super(Attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.att_hid_size = opt.att_hid_size

        self.h2att = nn.Linear(self.rnn_size, self.att_hid_size)
        self.alpha_net = nn.Linear(self.att_hid_size, 1)

    def forward(self, h, att_feats, p_att_feats, att_masks=None):
        # The p_att_feats here is already projected
        att_size = att_feats.numel() // att_feats.size(0) // self.rnn_size
        att = p_att_feats.view(-1, att_size, self.att_hid_size)
        
        att_h = self.h2att(h)                        # batch * att_hid_size
        att_h = att_h.unsqueeze(1).expand_as(att)            # batch * att_size * att_hid_size
        dot = att + att_h                                   # batch * att_size * att_hid_size
        dot = F.tanh(dot)                                # batch * att_size * att_hid_size
        dot = dot.view(-1, self.att_hid_size)               # (batch * att_size) * att_hid_size
        dot = self.alpha_net(dot)                           # (batch * att_size) * 1
        dot = dot.view(-1, att_size)                        # batch * att_size
        
        weight = F.softmax(dot, dim=1)                             # batch * att_size
        if att_masks is not None:
            weight = weight * att_masks.view(-1, att_size).float()
            weight = weight / weight.sum(1, keepdim=True) # normalize to 1
        att_feats_ = att_feats.view(-1, att_size, self.rnn_size) # batch * att_size * att_feat_size
        att_res = torch.bmm(weight.unsqueeze(1), att_feats_).squeeze(1) # batch * att_feat_size

        return att_res

class HTopDownModel(HierModel):
    def __init__(self, opt):
        super(HTopDownModel, self).__init__(opt)
        self.num_layers = opt.num_layers
        self.core = HTopDownCore(opt)
