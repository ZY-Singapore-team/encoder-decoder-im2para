import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

import time
import os
# from six.moves import cPickle

import opts
import models
from dataloader import *
import eval_utils
import misc.utils as utils
import pickle
from misc import process_bert

import ipdb

def train(opt):
    # Load data
    print('Loading dataset...')
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    loader.input_encoding_size = opt.input_encoding_size

    BERT_features = None
    if opt.cached_bert_features=="":
        # Extract BERT features
        print('Extracting pretrained BERT features...')
        BERT_features = process_bert.extract_BERT_features(loader, opt)
        with open(opt.data_path + 'BERT_features.pkl', 'wb') as f:
            pickle.dump(BERT_features, f)
    else:
        # Load BERT tokenization results
        print('Loading pretrained BERT features...')
        with open(opt.data_path + 'BERT_features.pkl', 'rb') as f:
            BERT_features = pickle.load(f)

    bert_vocab_path = opt.data_path + 'bert-base-cased-vocab.txt'
    opt.vocab_size,opt.fs_index = loader.update_bert_tokens(bert_vocab_path, BERT_features)
    print('Vocabulary size: ' + str(opt.vocab_size))

    # Load pretrained model, info file, histories file
    infos = {}
    histories = {}
    if opt.start_from is not None:
        with open(os.path.join(opt.start_from, 'infos_'+opt.id+'.pkl'), 'rb') as f:
            infos = pickle.load(f)
            saved_model_opt = infos['opt']
            need_be_same=["rnn_type", "rnn_size", "num_layers"]
            for checkme in need_be_same:
                assert vars(saved_model_opt)[checkme] == vars(opt)[checkme], "Command line argument and saved model disagree on '%s' " % checkme
        if os.path.isfile(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl')):
            with open(os.path.join(opt.start_from, 'histories_'+opt.id+'.pkl'), 'rb') as f:
                histories = pickle.load(f)
    iteration = infos.get('iter', 0)
    epoch = infos.get('epoch', 0)
    val_result_history = histories.get('val_result_history', {})
    loss_history = histories.get('loss_history', {})
    lr_history = histories.get('lr_history', {})
    ss_prob_history = histories.get('ss_prob_history', {})
    loader.iterators = infos.get('iterators', loader.iterators)
    loader.split_ix = infos.get('split_ix', loader.split_ix)
    if opt.load_best_score == 1:
        best_val_score = infos.get('best_val_score', None)

    # Create model
    model = models.setup(opt).cuda()
    dp_model = torch.nn.DataParallel(model,device_ids=[0,1])
    dp_model.train()

    # Loss function
    crit = utils.LanguageModelCriterion()
    rl_crit = utils.RewardCriterion()

    # Optimizer and learning rate adjustment flag
    optimizer = utils.build_optimizer(model.parameters(), opt)
    update_lr_flag = True

    # Load the optimizer
    if vars(opt).get('start_from', None) is not None and os.path.isfile(os.path.join(opt.start_from,"optimizer.pth")):
        optimizer.load_state_dict(torch.load(os.path.join(opt.start_from, 'optimizer.pth')))

    # Training loop
    while True:
        # Update learning rate once per epoch
        if update_lr_flag:

            # Assign the learning rate
            if epoch > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0:
                frac = (epoch - opt.learning_rate_decay_start) // opt.learning_rate_decay_every
                decay_factor = opt.learning_rate_decay_rate  ** frac
                opt.current_lr = opt.learning_rate * decay_factor
            else:
                opt.current_lr = opt.learning_rate
            utils.set_lr(optimizer, opt.current_lr)

            # Assign the scheduled sampling prob
            if epoch > opt.scheduled_sampling_start and opt.scheduled_sampling_start >= 0:
                frac = (epoch - opt.scheduled_sampling_start) // opt.scheduled_sampling_increase_every
                opt.ss_prob = min(opt.scheduled_sampling_increase_prob  * frac, opt.scheduled_sampling_max_prob)
                model.ss_prob = opt.ss_prob

            update_lr_flag = False
                
        # Load data from train split (0)
        start = time.time()
        data = loader.get_batch('train')
        data_time = time.time() - start
        start = time.time()

        # Unpack data
        torch.cuda.synchronize()
        tmp = [data['bert_feats'], data['sent_bert_feats'], data['fc_feats'], data['att_feats'], data['labels'], data['masks'], data['att_masks']]
        tmp = [_ if _ is None else torch.from_numpy(_).cuda() for _ in tmp]
        bert_feats, sent_bert_feats, fc_feats, att_feats, labels, masks, att_masks = tmp
        bert_feats.requires_grad = False

        # Forward pass and loss
        optimizer.zero_grad()
        outputs = dp_model(bert_feats, sent_bert_feats, fc_feats, att_feats, labels, att_masks)
        loss = crit(outputs, labels[:,1:], masks[:,1:])

        # Backward pass
        loss.backward()
        utils.clip_gradient(optimizer, opt.grad_clip)
        optimizer.step()
        train_loss = loss.item()
        torch.cuda.synchronize()

        # Print 
        total_time = time.time() - start
        if iteration % opt.print_freq == 1:
            print("iter {} (epoch {}), train_loss = {:.3f}, data_time = {:.3f}, time/batch = {:.3f}" \
                .format(iteration, epoch, train_loss, data_time, total_time))

        # Update the iteration and epoch
        iteration += 1
        if data['bounds']['wrapped']:
            epoch += 1
            update_lr_flag = True

        # Validate and save model 
        if (iteration % opt.save_checkpoint_every == 0):

            # Evaluate model
            eval_kwargs = {'split': 'val',
                            'dataset': opt.input_json}
            eval_kwargs.update(vars(opt))
            val_loss, predictions, lang_stats = eval_utils.eval_split(dp_model, crit, loader, eval_kwargs)

            # Our metric is CIDEr if available, otherwise validation loss
            if opt.language_eval == 1:
                current_score = lang_stats['CIDEr']
            else:
                current_score = - val_loss

            # Save model in checkpoint path 
            best_flag = False
            if best_val_score is None or current_score > best_val_score:
                best_val_score = current_score
                best_flag = True
            if not os.path.exists(os.path.join(opt.checkpoint_path, opt.caption_model)):
                os.mkdir(os.path.join(opt.checkpoint_path, opt.caption_model))
            checkpoint_path = os.path.join(opt.checkpoint_path, opt.caption_model, 'model.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print("model saved to {}".format(checkpoint_path))
            optimizer_path = os.path.join(opt.checkpoint_path, opt.caption_model, 'optimizer.pth')
            torch.save(optimizer.state_dict(), optimizer_path)

            # Dump miscalleous informations
            infos['iter'] = iteration
            infos['epoch'] = epoch
            infos['iterators'] = loader.iterators
            infos['split_ix'] = loader.split_ix
            infos['best_val_score'] = best_val_score
            infos['opt'] = opt
            infos['vocab'] = loader.get_vocab()
            histories['val_result_history'] = val_result_history
            histories['loss_history'] = loss_history
            histories['lr_history'] = lr_history
            histories['ss_prob_history'] = ss_prob_history
            with open(os.path.join(opt.checkpoint_path, opt.caption_model, 'infos_'+opt.id+'.pkl'), 'wb') as f:
                pickle.dump(infos, f)
            with open(os.path.join(opt.checkpoint_path, opt.caption_model, 'histories_'+opt.id+'.pkl'), 'wb') as f:
                pickle.dump(histories, f)

            # Save model to unique file if new best model
            if best_flag:
                model_fname = 'model-best-i{:05d}-score{:.4f}.pth'.format(iteration, best_val_score)
                infos_fname = 'model-best-i{:05d}-infos.pkl'.format(iteration)
                checkpoint_path = os.path.join(opt.checkpoint_path, opt.caption_model, model_fname)
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved to {}".format(checkpoint_path)) 
                with open(os.path.join(opt.checkpoint_path, opt.caption_model, infos_fname), 'wb') as f:
                    pickle.dump(infos, f)

        # Stop if reaching max epochs
        if epoch >= opt.max_epochs and opt.max_epochs != -1:
            break


if __name__ == "__main__":
    opt = opts.parse_opt()
    os.environ["CUDA_VISIBLE_DEVICES"]=opt.gpu
    train(opt)