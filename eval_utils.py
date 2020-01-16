import torch
import torch.nn as nn

import numpy as np
import json
from json import encoder
import random
import string
import time
import os
import sys
import misc.utils as utils
from pprint import pprint
# from pycocoevalcap.eval import COCOEvalCap

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

import ipdb

def eval_multi_metrics(ground_turth, predictions):
    # scorers = [
    #     (Bleu(4), "Bleu_4"),
    #     (Meteor(),"METEOR"),
    #     (Rouge(), "ROUGE_L"),
    #     (Cider(), "CIDEr")
    # ]

    scorers = [
        (Bleu(4), "Bleu_4"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr")
    ]

    gts = {}
    res = {}
    if len(predictions) == len(ground_turth):
        for ind, value in enumerate(predictions):
            res[ind] = [value]

        for ind, value in enumerate(ground_turth):
            gts[ind] = [value]
    else:
        Min_Len = min(len(predictions), len(ground_turth))
        for ind in range(Min_Len):
            res[ind] = [predictions[ind]]
            gts[ind] = [ground_turth[ind]]

    # param gts: Dictionary of reference sentences (id, sentence)
    # param res: Dictionary of hypothesis sentences (id, sentence)

    print('samples: {} / {}'.format(len(res.keys()), len(gts.keys())))

    scores = {}
    for scorer, name in scorers:
        score, all_scores = scorer.compute_score(gts, res)
        if isinstance(score, list):
            for i, sc in enumerate(score, 1):
                scores[name + str(i)] = sc
        else:
            scores[name] = score
    pprint(scores)
    return scores

def language_eval(bert_tokens, preds, model_id, split):
    # get all IDs in the preds
    val_ids = []
    predictions = []
    ground_turth = []
    for sample in preds:
        val_ids.append(sample['image_id'])
        predictions.append(sample['caption'])
    
    print('using %d/%d predictions' % (len(predictions), len(predictions)))
    # get ground-truth data
    for ID in val_ids:
        ground_turth.append(bert_tokens[ID]['raw'])

    lang_stat = eval_multi_metrics(ground_turth, predictions)
    return lang_stat

def eval_split(model, crit, loader, eval_kwargs={}):
    verbose = eval_kwargs.get('verbose', True)
    verbose_beam = eval_kwargs.get('verbose_beam', 1)
    verbose_loss = eval_kwargs.get('verbose_loss', 1)
    num_images = eval_kwargs.get('num_images', eval_kwargs.get('val_images_use', -1))
    # split = eval_kwargs.get('split', 'val')
    split = eval_kwargs.get('split', 'test')
    lang_eval = eval_kwargs.get('language_eval', 0)
    dataset = eval_kwargs.get('dataset', 'coco')
    beam_size = eval_kwargs.get('beam_size', 1)

    # Make sure in the evaluation mode
    model.eval()

    loader.reset_iterator(split)

    n = 0
    loss = 0
    loss_sum = 0
    loss_evals = 1e-8
    predictions = []
    while True:
        data = loader.get_batch(split)
        n = n + loader.batch_size

        if data.get('labels', None) is not None and verbose_loss:
            # forward the model to get loss
            tmp = [data['bert_labels'], data['bert_feats'], data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
            tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
            bert_labels, bert_feats, fc_feats, att_feats, labels, masks, att_masks = tmp

            with torch.no_grad():
                loss = crit(model(bert_feats, fc_feats, att_feats, labels, att_masks), labels[:,1:], masks[:,1:]).item()
            loss_sum = loss_sum + loss
            loss_evals = loss_evals + 1

        # forward the model to also get generated samples for each image
        # Only leave one feature for each image, in case duplicate sample
        tmp = [data['bert_labels'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['bert_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['fc_feats'][np.arange(loader.batch_size) * loader.seq_per_img], 
            data['att_feats'][np.arange(loader.batch_size) * loader.seq_per_img],
            data['att_masks'][np.arange(loader.batch_size) * loader.seq_per_img] if data['att_masks'] is not None else None]
        tmp = [torch.from_numpy(_).cuda() if _ is not None else _ for _ in tmp]
        bert_labels, bert_feats, fc_feats, att_feats, att_masks = tmp
        # forward the model to also get generated samples for each image
        # seq: [batch_size, max_seq_len]
        bert_info = {'vocab': loader.ix_to_BERT, 'tokens': bert_labels}
        with torch.no_grad():
            seq = model(bert_info, fc_feats, att_feats, att_masks, opt=eval_kwargs, mode='sample')[0].data
        
        # import ipdb; ipdb.set_trace()

        # Print beam search
        if beam_size > 1 and verbose_beam:
            for i in range(loader.batch_size):
                print('\n'.join([utils.decode_sequence(loader.get_vocab(), _['seq'].unsqueeze(0))[0] for _ in model.done_beams[i]]))
                print('--' * 10)
        sents = utils.decode_sequence(loader.get_vocab(), seq)

        for k, sent in enumerate(sents):
            entry = {'image_id': data['infos'][k]['id'], 'caption': sent}
            if eval_kwargs.get('dump_path', 0) == 1:
                entry['file_name'] = data['infos'][k]['file_path']
            predictions.append(entry)
            if eval_kwargs.get('dump_images', 0) == 1:
                # dump the raw image to vis/ folder
                cmd = 'cp "' + os.path.join(eval_kwargs['image_root'], data['infos'][k]['file_path']) + '" vis/imgs/img' + str(len(predictions)) + '.jpg' # bit gross
                print(cmd)
                os.system(cmd)

        # if we wrapped around the split or used up val imgs budget then bail
        ix0 = data['bounds']['it_pos_now']
        ix1 = data['bounds']['it_max']
        if num_images != -1:
            ix1 = min(ix1, num_images)
        for i in range(n - ix1):
            predictions.pop()

        if verbose:
            print('evaluating validation preformance... %d/%d (%f)' %(ix0 - 1, ix1, loss))

        if data['bounds']['wrapped']:
            break
        if num_images >= 0 and n >= num_images:
            break

    # Evaluate BLEU, MENTOR, etc.
    lang_stats = None
    if lang_eval == 1:
        # dataset: 'data/paratalk/paratalk.json'
        # each element of predictions: {'image_id': XXX, 'caption': XXX}
        # split: 'val'; eval_kwargs['id']: 'bert2'
        lang_stats = language_eval(loader.bert_tokens, predictions, eval_kwargs['id'], split)

    # Print a few sample outputs
    Top_K = 5
    print('Sample Predictions:')
    for i in range(Top_K):
        entry = predictions[i]
        print('\timage %s: %s' %(entry['image_id'], entry['caption']))

    # Switch back to training mode
    model.train()
    return loss_sum/loss_evals, predictions, lang_stats
