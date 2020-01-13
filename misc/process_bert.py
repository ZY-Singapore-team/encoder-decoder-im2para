import opts
import numpy as np
from dataloader import *
from transformers import BertTokenizer
import pickle
from tqdm import tqdm

import ipdb

def get_sent_captions_from_loader(loader, opt):
    id_to_word = loader.get_vocab()
    # map from sample_id to word_list
    sent_captions = {}
    for idx in range(len(loader)):
        sample_id = loader.info['images'][idx]['id']
        # caption_ids: np.array[1 * 175]
        caption_ids = loader.get_captions(idx, seq_per_img = opt.seq_per_img)
        word_list = []
        for ID in caption_ids[0, :]:
            if not ID == 0:
                ori_word = str(id_to_word[str(ID)])
                if ori_word == '\n':
                    continue
                elif ori_word == '.':
                    word_list.append('[SEP]')
                else:
                    word_list.append(str(id_to_word[str(ID)]))
        sent_captions[sample_id] = ' '.join(word_list)
    return sent_captions

def extract_BERT_features(loader, opt):
    sent_captions = get_sent_captions_from_loader(loader, opt)

    print('Extracting BERT features...')
    sent_embeddings = {}
    for sent in sent_captions:
        # sent_embeddings[sent] = {'raw': "", 'tokens': [], 'features': np.zeros((opt.seq_length, 768))}
        sent_embeddings[sent] = {'raw': "", 'tokens': []}

    sent_id_list = []
    corpus = []
    for sent_id, texts in sent_captions.items():
        corpus.append(texts)
        sent_id_list.append(sent_id)

    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    # model = BertModel.from_pretrained("bert-base-cased")

    count = 0
    batch_size = 64
    num_batch = int(len(sent_id_list)/batch_size)

    for ind in tqdm(range(num_batch)):
        sents = corpus[ind*batch_size: (ind+1)*batch_size] if (ind+1)*batch_size<len(sent_id_list) else corpus[ind*batch_size:]
        padded_sents = np.zeros((len(sents), opt.seq_length))
        for i, text in enumerate(sents):
            tokens = tokenizer.encode(text, add_special_tokens=True)
            # Padding
            padded_sents[i, :len(tokens)] = np.array(tokens)[:opt.seq_length]
        tokens = torch.from_numpy(padded_sents).long()

        # embeddings = model(tokens)[0]
        for index in range(padded_sents.shape[0]):
            # sent_emb = embeddings[index, :, :]
            sent_ID = sent_id_list[ind*batch_size + index]
            # sent_embeddings[sent_ID]['features'] = sent_emb.data.cpu().numpy()
            sent_embeddings[sent_ID]['tokens'] = padded_sents[index, :]
            sent_embeddings[sent_ID]['raw'] = sents[index]

    return sent_embeddings

def main(opt):
    # Load data
    loader = DataLoader(opt)
    opt.vocab_size = loader.vocab_size
    opt.seq_length = loader.seq_length
    sent_embeddings = extract_BERT_features(loader, opt)

    with open('coco_data/BERT_features.pkl', 'wb') as f:
        pickle.dump(sent_embeddings, f)

if __name__ == "__main__":
    opt = opts.parse_opt()
    main(opt)