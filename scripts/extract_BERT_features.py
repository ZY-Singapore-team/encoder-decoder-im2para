from bert_embedding import BertEmbedding
# import mxnet as mx
import numpy as np
from numpy import dot
from numpy.linalg import norm
import json
import ipdb

# def read_vocab(vocab_file):
#     with open(vocab_file, 'r') as f:
#        return set([w.strip() for w in f])

def build_vocab(corpus):
    return set([word for line in corpus for word in line.strip().split()])

def get_corpus_BERT_embeddings(corpus, save_path):
    with open(corpus, 'r') as f:
        docs = [line.strip() for line in f]

    vocab = build_vocab(docs)

    vocab_embeddings = {}
    for word in vocab:
        vocab_embeddings[word] = np.zeros(768)

    # ctx = mx.gpu(0)
    bert_embedding = BertEmbedding()
    count = 0
    batch_size = 128
    num_batch = int(len(docs)/batch_size) - 1
    for i in range(num_batch):
        if count%10==0:
            print(count * batch_size)
        count += 1
        sents = docs[i: i+batch_size]
        result = bert_embedding(sents, 'sum')
        for ind, _ in enumerate(result):
            sent, embeds = result[ind]
            for w, emb in zip(sent, embeds):
                if w in vocab:
                    vocab_embeddings[w] = vocab_embeddings[w] + emb

    vocab_size = len(vocab_embeddings)
    embeddings = np.random.randn(vocab_size, 768)
    index = 0
    with open(save_path + 'BERT_emb.index', 'w') as f:
        for key, value in vocab_embeddings.items():
            f.write(key + '\n')
            embeddings[index] = value
            index += 1

    np.save(save_path + 'BERT_emb.npy', embeddings)
    return vocab_embeddings

def test_BERT_embeddings(query, vocab_embeddings):
    vec_q = vocab_embeddings[query]
    memory = []
    for key, vec in vocab_embeddings.items():
        cos_sim = 0.0
        if norm(vec_q)!=0 and norm(vec)!=0:
            cos_sim = (dot(vec_q, vec)/(norm(vec_q)*norm(vec)) + 1.0)/2
        memory.append((key, cos_sim))
    ranks = sorted(memory, key=lambda x: x[1], reverse=True)
    print(ranks[:10])

def get_coco_corpus(coco_file, output_file):
    with open(coco_file, 'r') as f:
        decode = json.load(f)

    with open(output_file, 'w') as f:
        for sample in decode['images']:
            for sent in sample['sentences']:
                line = ' '.join(sent['tokens'])
                if not line.strip() == '':
                    f.write(line + '\n')

if __name__ == "__main__":
    #coco_file = '../data/captions/para_karpathy_format.json'
    corpus_file = '../data/coco_corpus.txt'
    save_path = '../data/'
    # get_coco_corpus(coco_file, corpus_file)
    vocab_embeddings = get_corpus_BERT_embeddings(corpus_file, save_path)
    test_BERT_embeddings('equipment', vocab_embeddings)

