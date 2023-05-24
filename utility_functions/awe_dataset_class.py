import itertools as it
import os

import numpy as np
import pandas as pd
import torch
from scipy.special import comb
from torch.utils.data import Dataset

from utility_functions.feature_extractor import (SSL_features, clip_features,
                                                 load_model)
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()
tokenizer = Tokenizer.from_file("../data/tokenizer-bpe200.json")


def split_string(strs):
    return strs.split(sep="/")[-1].split("_")[0]

class awe_dataset_SSL(Dataset):
    def __init__(self, root, feature_df, model_name, layer, device):
        self.root = root
        self.metadata = pd.read_csv(feature_df)
        self.model_name = model_name
        self.layer = layer
        self.device = device
    

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        model, sr = load_model(self.model_name,self.device)
        SSL_feature_path = os.path.join(self.root, self.metadata.iloc[idx, 8].strip("./").strip("\n"))
        features = SSL_features(SSL_feature_path, model, sr, layer=self.layer, device=self.device)
        word_features = clip_features(features,self.metadata.iloc[idx,2],self.metadata.iloc[idx,3],self.layer)
        
        return torch.squeeze(word_features),torch.tensor(word_features.size()[1])
        
# class awe_dataset_mfcc(Dataset):
#     def __init__(self, root, feature_df, device):
#         self.root = root
#         self.metadata = pd.read_csv(feature_df)
#         self.device = device
#         self.check = torch.cuda.is_available()
    
        

#     def __len__(self):
#         return len(self.metadata)


#     def __getitem__(self, idx):
#         word_name = self.metadata.iloc[idx,4]
#         sp_ch_ut_id = self.metadata.iloc[idx,7]
#         SSL_feature_path = os.path.join(self.root, self.metadata.iloc[idx, 8].strip("./").strip("\n"))
#         features = get_mfcc(SSL_feature_path, device=self.device)
#         word_features = clip_features_spec_mfcc(features,self.metadata.iloc[idx,2],self.metadata.iloc[idx,3])
#         return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_ch_ut_id

class awe_dataset_pre_computed(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]
        # self.check = torch.cuda.is_available()
    
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        SSL_feature_path = self.metadata.iloc[idx, 0]
        word_name = SSL_feature_path.split("/")[-1].split("_")[0]
        sp_ch_ut_id = SSL_feature_path.split("/")[-1].split("_")[-1].split(".")[0]
        word_features = torch.load(SSL_feature_path)
        # if self.check:
        #     word_features = torch.load(SSL_feature_path)
        # else:
        #     word_features = torch.load(SSL_feature_path,map_location=torch.device('cpu'))
            

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_ch_ut_id

class awe_dataset_pre_computed_dummy(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]
        self.check = torch.cuda.is_available()
    
        

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        SSL_feature_path = self.metadata.iloc[idx, 0]
        word_name = SSL_feature_path.split("/")[-1].split("_")[0]
        sp_ch_ut_id = SSL_feature_path.split("/")[-1].split("_")[-1].split(".")[0]
        if self.check:
            word_features = torch.load(SSL_feature_path)
        else:
            word_features = torch.load(SSL_feature_path,map_location=torch.device('cpu'))
            

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_ch_ut_id

# For correspondece autoencoder 
# use only for train loader that's it.
class cae_awe_dataset_pre_computed(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]
        # self.check = torch.cuda.is_available()
        if self.partition=="train":
            self.metadata_copy = self.metadata.copy()
            self.metadata_copy["word_name"] = self.metadata_copy["path"].apply(split_string)
            self.x_idx, self.y_idx  = np.arange(len(self.metadata_copy)), np.arange(len(self.metadata_copy))
            labels = self.metadata_copy["word_name"].values
            num_examples = len(labels)
            num_pairs = int(comb(num_examples, 2))
            # build up binary array of matching examples
            matches = np.zeros(num_pairs, dtype= bool)
            i = 0
            for n in range(num_examples):
                j = i + num_examples - n - 1
                matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
                i = j
            num_same = np.sum(matches)
            pairs_numerical = it.combinations(np.arange(len(labels)), 2)
            matched_pairs = []
            for i, pair in enumerate(pairs_numerical):
                if matches[i]:
                    matched_pairs.append(pair)
            matched_pairs = np.array(matched_pairs)  
            #if condition returns True, then nothing happens:
            assert len(matched_pairs)==num_same
            self.x_idx = np.concatenate((self.x_idx, matched_pairs[:,0]))
            #self.x_idx = np.concatenate((self.x_idx, matched_pairs[:,1]))
            self.y_idx = np.concatenate((self.y_idx, matched_pairs[:,1]))
            #self.y_idx = np.concatenate((self.y_idx, matched_pairs[:,0]))
    
        

    def __len__(self):
        return len(self.x_idx)


    def __getitem__(self, idx):
        SSL_feature_path_x = self.metadata.iloc[self.x_idx[idx], 0]
        SSL_feature_path_y = self.metadata.iloc[self.y_idx[idx], 0]

        word_name_x = SSL_feature_path_x.split("/")[-1].split("_")[0]
        word_name_y = SSL_feature_path_y.split("/")[-1].split("_")[0]
        assert word_name_x==word_name_y
        sp_ch_ut_id_x = SSL_feature_path_x.split("/")[-1].split("_")[-1].split(".")[0]
        sp_ch_ut_id_y = SSL_feature_path_y.split("/")[-1].split("_")[-1].split(".")[0]

        # if self.check:
        #     word_features_x = torch.load(SSL_feature_path_x)
        #     word_features_y = torch.load(SSL_feature_path_y)
        # else:
        #     word_features_x = torch.load(SSL_feature_path_x, map_location=torch.device('cpu'))
        #     word_features_y = torch.load(SSL_feature_path_y, map_location=torch.device('cpu'))

        word_features_x = torch.load(SSL_feature_path_x)
        word_features_y = torch.load(SSL_feature_path_y)

        return torch.squeeze(word_features_x),torch.tensor(word_features_x.size()[1]), word_name_x, sp_ch_ut_id_x, \
            torch.squeeze(word_features_y),torch.tensor(word_features_y.size()[1]), word_name_y, sp_ch_ut_id_y


class awe_dataset_pre_computed_pre_training(Dataset):
    def __init__(self, feature_df, partition):
        self.metadata = pd.read_csv(feature_df)
        self.partition = partition
        self.metadata = self.metadata[self.metadata["partition"]==self.partition]

    def __len__(self):
        return len(self.metadata)


    def __getitem__(self, idx):
        SSL_feature_path = self.metadata.iloc[idx, 0]
        word_name = SSL_feature_path.split("/")[-1].split("_")[0]
        sp_ch_ut_id = SSL_feature_path.split("/")[-1].split("_")[-1].split(".")[0]
        # SSL_feature_path = "/home/amitmeghanani/H-drive/" + SSL_feature_path
        if not os.path.exists(SSL_feature_path):
            SSL_feature_path = "/home/amitmeghanani/M-drive/" + SSL_feature_path
        #SSL_feature_path = "/home/amitmeghanani/M-drive/" + SSL_feature_path
        word_features = torch.load(SSL_feature_path)
        # token = ['SOW'] + sp.encode(word_name.lower(), out_type=str) + ['EOW']

        token = ['SOW'] + tokenizer.encode(word_name.lower()).tokens + ['EOW']
        token_len = len(token)

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_ch_ut_id, token, token_len
