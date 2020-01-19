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
        self.st_index = opt.st_index

        self.use_bn = getattr(opt, 'use_bn', 0)
        self.relu_mod = getattr(opt, 'relu_mod', 'relu')
        self.rela_index = getattr(opt, 'rela_index', 0)

        self.ss_prob = 0.0 # Schedule sampling probability

        if self.relu_mod == 'relu':
            self.embed = nn.Embedding(self.vocab_size + 1, self.input_encoding_size)
            self.fc_embed = nn.Sequential(nn.Linear(self.fc_feat_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))
            self.bert_embed = nn.Sequential(nn.Linear(self.input_encoding_size, self.rnn_size),
                                        nn.ReLU(inplace=True),
                                        nn.Dropout(self.drop_prob_lm))                       
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
        old_state = fc_feats.new_zeros(batch_size, self.rnn_size)
        old_att = fc_feats.new_zeros(batch_size, self.rnn_size)

        # outputs = []
        outputs = fc_feats.new_zeros(batch_size, seq.size(1) - 1, self.vocab_size+1)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)


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

            output, state, old_state, old_att = self.get_logprobs_state(it, bert_feats[:,i,:], sen_bert_feats[:,i,:], fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att)
            outputs[:, i] = output
            # outputs.append(output)

        return outputs
        # return torch.cat([_.unsqueeze(1) for _ in outputs], 1)

    def get_logprobs_state(self, it, bert_feats, sen_bert_feats, fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att):
        # 'it' contains a word index
        # xt = self.embed(it)

        # BERT version of this; xt:(batch_size * 1 * 768)
        st = sen_bert_feats.detach()
        xt = bert_feats.detach()
        cs_index = it == int(self.fs_index)

        output, state, old_state, old_att = self.core(cs_index, st, xt, fc_feats, att_feats, p_att_feats, state, old_state, old_att, att_masks)
        logprobs = F.log_softmax(self.logit(output), dim=1)

        return logprobs, state, old_state, old_att

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

    def _sample(self, bert_info, bert_feats, sen_bert_feats, fc_feats, att_feats, att_masks=None, opt={}):

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
        old_state = fc_feats.new_zeros(batch_size, self.rnn_size)
        old_att = fc_feats.new_zeros(batch_size, self.rnn_size)

        fc_feats, att_feats, p_att_feats = self._prepare_feature(fc_feats, att_feats, att_masks)

        # BEG MODIFIED
        trigrams = [] # will be a list of batch_size dictionaries
        # END MODIFIED

        # seq = []
        # seqLogprobs = []
        seq = fc_feats.new_zeros((batch_size, self.seq_length), dtype=torch.long)
        seqLogprobs = fc_feats.new_zeros(batch_size, self.seq_length)

        # the initial bert hidden state
        total_token_list = [[] for _ in range(batch_size)]
        out_records = [[] for _ in range(batch_size)]
        current_pred = bert_feats[:, 0, :]
        current_sen_pred = bert_feats[:, 0, :]
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

            logprobs, state, old_state, old_att = self.get_logprobs_state(it, current_pred, current_sen_pred, fc_feats, att_feats, p_att_feats, att_masks, state, old_state, old_att)

            _, predicts = torch.max(torch.softmax(logprobs, 1), 1)
            predicts = predicts.unsqueeze(1)
            new_predicts = predicts.data.cpu().numpy()
            bert_feats = np.zeros((new_predicts.shape[0], new_predicts.shape[1], 768), dtype='float32')
            for b in range(new_predicts.shape[0]):
                for w in range(new_predicts.shape[1]):
                    bert_feats[b, w, :] = self.loader.word_bert_feats[new_predicts[b,w]]
            bert_feats = bert_feats.astype(np.float32)
            current_pred = torch.from_numpy(bert_feats).float().squeeze(1).cuda()                        

            # # udpate dynamic sentence BERT prediction
            predicts_bert = [ix_to_BERT[e] for e in predicts.flatten().cpu().numpy()]
            for i, index in enumerate(predicts_bert):
                total_token_list[i].append(index) 
                if index==ix_to_BERT[self.fs_index]:
                    out_records[i].append(t)
                    if len(out_records[i])==1:
                        out_list = ix_to_BERT[self.st_index] + [total_token_list[i][n] for n in range(t+1)]
                    else:
                        previous_ind = out_records[i][-2]
                        out_list = ix_to_BERT[self.st_index] + [total_token_list[i][n] for n in range(previous_ind+1, t+1)]
                else:
                    if len(out_records[i])==0:
                        out_list = ix_to_BERT[self.st_index] + [total_token_list[i][n] for n in range(t+1)] + ix_to_BERT[self.fs_index]
                    elif len(out_records[i])==1:
                        previous_ind = out_records[i][0]
                        out_list = ix_to_BERT[self.st_index] + [total_token_list[i][n] for n in range(previous_ind+1,t+1)] + ix_to_BERT[self.fs_index]                      
                    else:
                        previous_ind = out_records[i][-2]
                        out_list = ix_to_BERT[self.st_index] + [total_token_list[i][n] for n in range(previous_ind+1,t+1)] + ix_to_BERT[self.fs_index] 
                out_token = torch.from_numpy(np.array(out_list)).long().unsqueeze(0).cuda()
                current_sen_pred[i] = self.bert_model(out_token)[0][:,0,:].squeeze(0)      

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

class AdaAtt_lstm(nn.Module):
    def __init__(self, opt, use_maxout=True):
        super(AdaAtt_lstm, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.fc_feat_size = opt.fc_feat_size
        self.att_feat_size = opt.att_feat_size
        self.att_hid_size = opt.att_hid_size
        self.input_encoding_size = opt.input_encoding_size

        self.use_maxout = use_maxout

        # Build a LSTM
        self.w2h = nn.Linear(self.input_encoding_size, (4+(use_maxout==True)) * self.rnn_size)
        self.v2h = nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size)

        self.i2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(0)])
        self.h2h = nn.ModuleList([nn.Linear(self.rnn_size, (4+(use_maxout==True)) * self.rnn_size) for _ in range(1)])

        # Layers for getting the fake region
        self.r_w2h = nn.Linear(self.input_encoding_size, self.rnn_size)
        self.r_v2h = nn.Linear(self.rnn_size, self.rnn_size)
        self.r_h2h = nn.Linear(self.rnn_size, self.rnn_size)


    def forward(self, xt, img_fc, state):

        # c,h from previous timesteps
        prev_h = state[0][0]
        prev_c = state[1][0]

        x = xt
        i2h = self.w2h(x) + self.v2h(img_fc)
        all_input_sums = i2h+self.h2h[0](prev_h)
        sigmoid_chunk = all_input_sums.narrow(1, 0, 3 * self.rnn_size)
        sigmoid_chunk = F.sigmoid(sigmoid_chunk)
        # decode the gates
        in_gate = sigmoid_chunk.narrow(1, 0, self.rnn_size)
        forget_gate = sigmoid_chunk.narrow(1, self.rnn_size, self.rnn_size)
        out_gate = sigmoid_chunk.narrow(1, self.rnn_size * 2, self.rnn_size)
        # decode the write inputs
        if not self.use_maxout:
            in_transform = F.tanh(all_input_sums.narrow(1, 3 * self.rnn_size, self.rnn_size))
        else:
            in_transform = all_input_sums.narrow(1, 3 * self.rnn_size, 2 * self.rnn_size)
            in_transform = torch.max(\
                in_transform.narrow(1, 0, self.rnn_size),
                in_transform.narrow(1, self.rnn_size, self.rnn_size))
        # perform the LSTM update
        next_c = forget_gate * prev_c + in_gate * in_transform
        # gated cells form the output
        tanh_nex_c = F.tanh(next_c)
        next_h = out_gate * tanh_nex_c

        i2h = self.r_w2h(x) + self.r_v2h(img_fc)
        gt = i2h+self.r_h2h(prev_h)
        fake_region = F.sigmoid(gt) * tanh_nex_c
        cs = [next_c]
        hs = [next_h]

        # set up the decoder
        top_h = hs[-1]
        top_h = F.dropout(top_h, self.drop_prob_lm, self.training)
        fake_region = F.dropout(fake_region, self.drop_prob_lm, self.training)

        state = (torch.cat([_.unsqueeze(0) for _ in hs], 0), 
                torch.cat([_.unsqueeze(0) for _ in cs], 0))
        return top_h, fake_region, state


class AdaAtt_attention(nn.Module):
    def __init__(self, opt):
        super(AdaAtt_attention, self).__init__()
        self.rnn_size = opt.rnn_size
        self.drop_prob_lm = opt.drop_prob_lm
        self.att_hid_size = opt.att_hid_size

        # fake region embed
        self.fr_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.ReLU(), 
            nn.Dropout(self.drop_prob_lm))
        self.fr_embed = nn.Linear(self.rnn_size, self.att_hid_size)

        # h out embed
        self.ho_linear = nn.Sequential(
            nn.Linear(self.rnn_size, self.rnn_size),
            nn.Tanh(), 
            nn.Dropout(self.drop_prob_lm))
        self.ho_embed = nn.Linear(self.rnn_size, self.att_hid_size)

        self.alpha_net = nn.Linear(self.att_hid_size, 1)
        self.att2h = nn.Linear(self.rnn_size, self.rnn_size)

    def forward(self, h_out, fake_region, conv_feat, conv_feat_embed):

        # View into three dimensions
        att_size = conv_feat.numel() // conv_feat.size(0) // self.rnn_size
        conv_feat = conv_feat.view(-1, att_size, self.rnn_size)
        conv_feat_embed = conv_feat_embed.view(-1, att_size, self.att_hid_size)

        # view neighbor from bach_size * neighbor_num x rnn_size to bach_size x rnn_size * neighbor_num
        fake_region = self.fr_linear(fake_region)
        fake_region_embed = self.fr_embed(fake_region)

        h_out_linear = self.ho_linear(h_out)
        h_out_embed = self.ho_embed(h_out_linear)

        txt_replicate = h_out_embed.unsqueeze(1).expand(h_out_embed.size(0), att_size + 1, h_out_embed.size(1))

        img_all = torch.cat([fake_region.view(-1,1,self.rnn_size), conv_feat], 1)
        img_all_embed = torch.cat([fake_region_embed.view(-1,1,self.rnn_size), conv_feat_embed], 1)

        hA = F.tanh(img_all_embed + txt_replicate)
        hA = F.dropout(hA,self.drop_prob_lm, self.training)
        
        hAflat = self.alpha_net(hA.view(-1, self.att_hid_size))
        PI = F.softmax(hAflat.view(-1, att_size + 1))

        visAtt = torch.bmm(PI.unsqueeze(1), img_all)
        visAttdim = visAtt.squeeze(1)

        atten_out = visAttdim + h_out_linear

        h = F.tanh(self.att2h(atten_out))
        h = F.dropout(h, self.drop_prob_lm, self.training)
        return h


class HTopDownCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(HTopDownCore, self).__init__()
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

        self.satt_lstm = nn.LSTMCell(opt.input_encoding_size+opt.rnn_size * 2, opt.rnn_size) 
        self.sent_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.watt_lstm = nn.LSTMCell(opt.input_encoding_size+opt.rnn_size * 3, opt.rnn_size)
        self.word_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = Attention(opt)

    def forward(self, cs_index, st, xt, fc_feats, att_feats, p_att_feats, state, old_state, old_att, att_masks=None):

        ####################sentence level######################
        prev_h = state[0][0]
        att_lstm_input = torch.cat([prev_h, st, fc_feats], 1)

        h_att, c_att = self.satt_lstm(att_lstm_input, (state[0][0], state[1][0]))
        att = self.attention(h_att, att_feats, p_att_feats, att_masks)

        sen_lstm_input = torch.cat([att, h_att], 1)
        h_sen, c_sen = self.sent_lstm(sen_lstm_input, (state[0][1], state[1][1]))

        # get topic vectors when a sentence is ended at [SEP]
        batch_size = fc_feats.size(0)
        for batch_id in range(batch_size):
            if cs_index[batch_id]==1:
                old_state[batch_id] = state[0][1][batch_id]
                old_att[batch_id]   = att[batch_id]
        topic_vector = old_state

        ####################word level##########################
        if self.num_layers == 3:
            prev_h_word = state[0][2]
            word_lstm_input = torch.cat([xt, prev_h_word, fc_feats, topic_vector], 1)
            h_word, c_word = self.watt_lstm(word_lstm_input, (state[0][2], state[1][2]))
            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_word]), torch.stack([c_att, c_sen, c_word]))

        elif self.num_layers == 4:
            prev_h_word = state[0][2]
            watt_lstm_input = torch.cat([xt, prev_h_word, fc_feats, topic_vector], 1)
            h_watt, c_watt = self.watt_lstm(watt_lstm_input, (state[0][2], state[1][2]))
            word_lstm_input = torch.cat([old_att, h_watt], 1)
            h_word, c_word = self.word_lstm(word_lstm_input, (state[0][3], state[1][3]))

            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_watt, h_word]), torch.stack([c_att, c_sen, c_watt, c_word]))

        return output, state, old_state, old_att

class HAdaAttCore(nn.Module):
    def __init__(self, opt, use_maxout=False):
        super(HAdaAttCore, self).__init__()
        self.satt_lstm = AdaAtt_lstm(opt, use_maxout)
        self.sent_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)        
        self.watt_lstm = nn.LSTMCell(opt.input_encoding_size+opt.rnn_size * 3, opt.rnn_size)
        self.word_lstm = nn.LSTMCell(opt.rnn_size * 2, opt.rnn_size)
        self.attention = AdaAtt_attention(opt)
        self.num_layers = opt.num_layers
        self.drop_prob_lm = opt.drop_prob_lm

    def forward(self, cs_index, st, xt, fc_feats, att_feats, p_att_feats, state, old_state, old_att, att_masks=None):
        ####################sentence level######################
        h_out, p_out, new_state = self.satt_lstm(st, fc_feats, state)
        att_out = self.attention(h_out, p_out, att_feats, p_att_feats)
        h_att, c_att = new_state[0][0], new_state[1][0]
        sen_lstm_input = torch.cat([att_out, h_att], 1)
        h_sen, c_sen = self.sent_lstm(sen_lstm_input, (state[0][1], state[1][1]))

        # get topic vectors when a sentence is ended at [SEP]
        batch_size = fc_feats.size(0)
        for batch_id in range(batch_size):
            if cs_index[batch_id]==1:
                old_state[batch_id] = state[0][1][batch_id]
                old_att[batch_id]   = att_out[batch_id]
        topic_vector = old_state
        ####################word level##########################
        if self.num_layers == 3:
            prev_h_word = state[0][2]
            word_lstm_input = torch.cat([xt, fc_feats, prev_h_word, topic_vector], 1)
            h_word, c_word = self.watt_lstm(word_lstm_input, (state[0][2], state[1][2]))
            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_word]), torch.stack([c_att, c_sen, c_word]))

        elif self.num_layers == 4:
            prev_h_word = state[0][2]
            watt_lstm_input = torch.cat([xt, prev_h_word, fc_feats, topic_vector], 1)
            h_watt, c_watt = self.watt_lstm(watt_lstm_input, (state[0][2], state[1][2]))
            word_lstm_input = torch.cat([old_att, h_watt], 1)
            h_word, c_word = self.word_lstm(word_lstm_input, (state[0][3], state[1][3]))

            output = F.dropout(h_word, self.drop_prob_lm, self.training)
            state = (torch.stack([h_att, h_sen, h_watt, h_word]), torch.stack([c_att, c_sen, c_watt, c_word]))            

        return att_out, state, old_state, old_att

class HTopDownModel(HierModel):
    def __init__(self, opt):
        super(HTopDownModel, self).__init__(opt)
        self.num_layers = opt.num_layers
        self.core = HTopDownCore(opt)

class HAdaAttMOModel(HierModel):
    def __init__(self, opt):
        super(HAdaAttMOModel, self).__init__(opt)
        self.core = HAdaAttCore(opt, True)
