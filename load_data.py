from collections import Counter
import datetime
import io
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F

PAD_IDX = 0
UNK_IDX = 1

MAX_SENTENCE_LENGTH = 30

def prepare_data(data):
    '''Split sentences by space and map labels to int
    data: pd.DataFrame
    
    Return:
    df: pd.DataFrame
    '''
    df = data.copy()
    df['label'] = df['label'].map({'contradiction': 0, 'neutral': 1, 'entailment': 2})
    df['sentence1'] = df['sentence1'].str.split()
    df['sentence2'] = df['sentence2'].str.split()
    return df

def load_vectors(f_name, vocabulary):
    '''Load vectors from .vec file
    code taken from https://fasttext.cc/docs/en/english-vectors.html
    '''
    f_in = io.open(f_name, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, f_in.readline().split())
    vectors = {}
    for line in tqdm(f_in):
        tokens = line.strip().split(' ')
        if tokens[0] in vocabulary:
            vectors[tokens[0]] = list(map(float, tokens[1:]))
    return vectors

def build_vocabulary(train_data, vector_path, vocab_size):
    '''Build vocabulary from training data and pre-trained vectors, constrained by vocab size.
    Returns:
    id2token: list of tokens, where id2token[i] returns token that corresponds to token i
    token2id: dictionary where keys represent tokens and corresponding values represent indices
    vectors: dict with tokens as keys, vectors as values
    '''
    print('Building vocabulary... ', end='', flush=True)
    all_tokens = []
    for row in (train_data['sentence1']+train_data['sentence2']).iteritems():
        all_tokens += row[1]
    vocab, count = zip(*Counter(all_tokens).most_common(vocab_size))
    print('Done.')
    print('Loading vocabulary vectors... ', end='', flush=True)
    vectors = load_vectors(vector_path, vocab)
    vocab = [word for word in vocab if word in vectors]
    print('Done.')
    id2token = list(vocab)
    token2id = dict(zip(vocab, range(2, 2+len(vocab))))
    id2token = ['<pad>', '<unk>'] + id2token
    token2id['<pad>'] = PAD_IDX
    token2id['<unk>'] = UNK_IDX
    return token2id, id2token, vectors

def token2index_dataset(tokens_data, token2id):
    '''Turn tokens into indices
    tokens_data: pd.DataFrame, with sentence1 and sentence2 columns consist of tokenized data
                               and label column containing target value
    Returns:
    indices_data: dict, with sentence1 and sentence2 columns consist of indiced data
    '''
    all_indices_data = {'sentence1':[],'sentence2':[]}
    for col in all_indices_data.keys():
        indices_data = []
        for tokens in tokens_data[col].to_list():
            index_list = [token2id[token] if token in token2id else UNK_IDX 
                          for i, token in enumerate(tokens) if i < MAX_SENTENCE_LENGTH]
            indices_data.append(index_list)
        all_indices_data[col] = indices_data
    return all_indices_data, tokens_data.label.to_list()

class SNLIDataset(Dataset):
    """Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """
    def __init__(self, data, target):
        """
        @param data: dict, with `sentence1` and `sentence2` columns consist of indiced data
        @param target: list 
        """
        self.x1 = data['sentence1']
        self.x2 = data['sentence2']
        self.y = target
        assert (len(self.x1) == len(self.x2) == len(self.y))

    def __len__(self):
        return len(self.x1)
        
    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        Turn sentence1[key] and sentence2[key] into indices
        """
        token_idx_1 = self.x1[key]
        token_idx_2 = self.x2[key]
                
        label = self.y[key]
        return [token_idx_1, len(token_idx_1), token_idx_2, len(token_idx_2), label]
    
def SNLI_collate_func(batch):
    """Customized function for DataLoader that dynamically pads the batch so that all 
    data have the same length
    """
    x1_list = []
    x1_length_list = []
    x2_list = []
    x2_length_list = []
    label_list = []
    for datum in batch:
        x1_padded_vec = np.pad(np.array(datum[0]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[1])), 
                                mode="constant", constant_values=0)
        x1_list.append(x1_padded_vec)
        x1_length_list.append(datum[1])
        
        x2_padded_vec = np.pad(np.array(datum[2]), 
                                pad_width=((0,MAX_SENTENCE_LENGTH-datum[3])), 
                                mode="constant", constant_values=0)
        x2_list.append(x2_padded_vec)
        x2_length_list.append(datum[3])
        
        label_list.append(datum[4])

    return [torch.from_numpy(np.array(x1_list)), torch.LongTensor(x1_length_list),
            torch.from_numpy(np.array(x2_list)), torch.LongTensor(x2_length_list),
            torch.LongTensor(label_list)]

def create_weights(vectors, id2token):
    '''Create weights metrics from vectors and id2token
    Returns:
    weights_matrix: torch.Tensor, dimension of (vocab size x embedding dim)
    '''
    weights_matrix = torch.from_numpy(np.array([vectors[id2token[i]] for i in range(2, len(id2token))]))
    zero = torch.zeros(2, weights_matrix.size()[1], dtype=torch.float64)
    weights_matrix = torch.cat([zero, weights_matrix])
    return weights_matrix

def create_emb_layer(weights_matrix, non_trainable=False):
    '''Create embedding layer that's used in a PyTorch model
    Returns:
    emb_layer: nn.Embedding()
    num_embeddings: int
    embedding_dim: int
    '''
    num_embeddings, embedding_dim = weights_matrix.size()
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim