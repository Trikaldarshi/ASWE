import argparse
import time
import wandb
import datetime
import random
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import copy
from os import path
from collections import Counter
from sklearn.metrics import accuracy_score
from models.model_cae import model_cae
from models.sw_cls_cae_pretrained import sw_cls_cae_pretrained
from utility_functions.awe_dataset_class import awe_dataset_pre_computed_pre_training
from utility_functions.utils_function import (average_precision, collate_fn_pre_training)
import torch.nn.functional as F
from ast import literal_eval
import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = Whitespace()

def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing paths of wave files, words, start point, duration \
      or SSL features metadata file")
    parser.add_argument("--model_weights", type = str, help = "model weights for which inference is to be done", nargs='?' ,default = "None")

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def main():
    args = check_argv()

    torch.manual_seed(3112)
    torch.cuda.manual_seed(3112)
    torch.cuda.manual_seed_all(3112)
    np.random.seed(3112)
    random.seed(3112)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(3121)

    bidirectional=True
    batch_size=1
    # load from args
    model_weights = args.model_weights
    metadata_file = args.metadata_file
    print("location of the saved model weights:", model_weights)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is device CUDA:", device.type=="cuda")
    if device.type == "cuda":
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print("number of workers:", num_workers)
    print("pin memory status:", pin_memory)

    train_data = awe_dataset_pre_computed_pre_training(
        feature_df=metadata_file,
        partition="train")

    val_data = awe_dataset_pre_computed_pre_training(
        feature_df=metadata_file,
        partition="val"
    )
    test_data = awe_dataset_pre_computed_pre_training(
        feature_df=metadata_file,
        partition="test"
    )


#   Uncomment the following lines to use a subset of the data for debugging

    # indices = np.random.choice(range(len(train_data)), 100, replace=False)
    # train_data = torch.utils.data.Subset(train_data, indices)
    # indices = np.random.choice(range(len(val_data)), 100, replace=False)
    # val_data = torch.utils.data.Subset(val_data, indices)
    # test_data = torch.utils.data.Subset(test_data, indices)


    print("length of training data:",len(train_data))
    print("length of validation data:",len(val_data))
    print("length of test data:",len(test_data))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pre_training,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pre_training,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn_pre_training,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )


    input_dim=768
    hidden_dim=256
    embedding_dim=128
    rnn_type="GRU"
    num_layers=4
    dropout=0.2

    filename = '../checkpoints/model_subword_classification/sub_classification_01/dict_tokens.pt'
    isExist = os.path.exists(filename)
    if isExist:
        dict_tokens = torch.load(filename)
        print("loaded the dictionary of tokens")
    else:
        print("creating a new dictionary of tokens")
        token_list = []
        for ij, (x, lens_x, _,_,tokens,_) in enumerate(train_loader):
            if ij%100==0:
                print("processed:",ij)
            token_list = token_list + tokens.tolist()

        out = [item for t in token_list for item in t]
        out = ['SOW', 'EOW', 'PAD'] + out
        dict_tokens = Counter(out)
        print("total tokens:",len(dict_tokens))
        torch.save(dict_tokens, 'dict_tokens.pt')

    classes = dict_tokens.keys()
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(dict_tokens) 

    ## Define the pre-trained model
    pre_model = model_cae(input_dim, hidden_dim, embedding_dim, 
                            rnn_type, bidirectional, num_layers, 
                            dropout)

    ## Define the model

    model = sw_cls_cae_pretrained(hidden_dim, embedding_dim, rnn_type, 
                    bidirectional, num_layers, num_classes, dropout, 
                    pre_model.encoder)
    model_checkpoint = torch.load(model_weights, 
                                    map_location=torch.device(device))
    model.load_state_dict(model_checkpoint['model_state_dict'])
    model = model.to(device)
    print(model)

    def doop(l):
        return ''.join(l)

    def evaluate_accuracy(model, data_loader, device, max_length=12):
        model.eval()
        EOW_token = 1
        y_true = []
        y_pred = []
        attention_list = []
        for idx, (data,lens,word_name,_,token,token_len) in enumerate(data_loader):
            token = torch.from_numpy(np.vectorize(class_to_idx.get)(token))
            data, lens, token = data.to(device), lens.to(device), token.to(device)
            if  device.type == 'cuda()':
                word_name = word_name.cpu()

            with torch.no_grad():
                encoder_outputs, encoded_x = model.encoder(data, lens)

            decoded_subwords = []

            trg = token.T # [token len, batch size]
            # trg_len = trg.size(0) # [token_len]
            input = trg[0,:] # [batch_size,1,1]
            hidden = encoded_x # [1, batch_size,embedding_dim/hidden dim]
            mask = model.create_mask(data).to(device)
            attentions = torch.zeros(max_length, 1, lens).to(device)
            for di in range(1,max_length):
                output, hidden, attention = model.decoder(
                    input, hidden, encoder_outputs, mask)
                attentions[di] = attention
                top1 = output.argmax(1)
                if top1.item()==EOW_token:
                    decoded_subwords.append([idx_to_class[index.item()].replace("▁","").replace("EOW","").upper() for index in top1])
                    break
                else:
                    decoded_subwords.append([idx_to_class[index.item()].replace("▁","").replace("EOW","").upper() for index in top1])
                input = top1.detach()
            decoded_words = np.apply_along_axis(doop, 0, np.array(decoded_subwords))
            y_true = y_true + list(word_name)
            y_pred = y_pred + decoded_words.tolist()

        accuracy = accuracy_score(y_true,y_pred)

        return accuracy,attention_list

    def evaluate_attn(model, data, lens, word_name, token, device, max_length=11):
        model.eval()
        EOW_token = 1
        SOW_token = 0
        token = torch.from_numpy(np.vectorize(class_to_idx.get)(token))
        data, lens, token = data.to(device), lens.to(device), token.to(device)
        if  device.type == 'cuda()':
            word_name = word_name.cpu()
        with torch.no_grad():
            encoder_outputs, encoded_x = model.encoder(data, lens)
        mask = model.create_mask(data)
        trg_indexes = [SOW_token]


        hidden = encoded_x # [1, batch_size,embedding_dim/hidden dim]
        
        attentions = torch.zeros(max_length, 1, lens).to(device)

        for di in range(max_length):
            trg_tensor = torch.LongTensor([trg_indexes[-1]]).to(device)
            output, hidden, attention = model.decoderV(
                trg_tensor, hidden, encoder_outputs, mask)
            attentions[di] = attention
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)
            if pred_token == EOW_token:
                break

        trg_tokens = [idx_to_class[i] for i in trg_indexes]
            
        return trg_tokens[1:], attentions[:len(trg_tokens)-1]

    print("evaluate accuracy...................")

    train_accuracy,_ = evaluate_accuracy(model, train_loader, device)
    val_accuracy,_ = evaluate_accuracy(model, val_loader, device)
    test_accuracy,_ = evaluate_accuracy(model, test_loader, device)

    print("train_accuracy", train_accuracy)
    print("val_accuracy", val_accuracy)
    print("test_accuracy", test_accuracy)

    # # # get subword embedding for subword discrimination task
    # # Test set


    # print("get subword embedding for subword discrimination task....")

    # dict_sub_emb = {p : [] for p in range(10)} # 10 is the max length of subword
    # dict_sub_emb_id = {p : [] for p in range(10)}
    # for idx, (data,lens,word_name,_,token,token_len) in enumerate(test_loader):
    #     trg_tokens,_,sub_emb = evaluate_attn(model, data, lens, word_name, token, device)
    #     # token = token[0][1:-1] # remove start and end of word token
    #     # note: can't use token because it is not the same as the token predicted by the model, which is trg_tokens, can have differnt length
    #     for i in range(len(trg_tokens)-1):    # -1 because we don't want to include the end of word token as a subword
    #         dict_sub_emb[i].append(sub_emb[i].detach())
    #         dict_sub_emb_id[i].append(class_to_idx[trg_tokens[i]])

    # test_avg_precision = []
    # for i in range(len(dict_sub_emb)):
    #     embs = torch.cat(dict_sub_emb[i],0).to(torch.float16).cpu().numpy()
    #     ids = np.array(dict_sub_emb_id[i])
    #     avg_precision,_ = average_precision(embs, ids, "cosine", show_plot=False)
    #     test_avg_precision.append(avg_precision)
    #     print("average precision for subword in the test set at position {} is {}".format(i,avg_precision))

    # print("mean of the average precision for all subwords in the test set is {}".format(np.mean(test_avg_precision)))


    # dict_sub_emb = {p : [] for p in range(10)}
    # dict_sub_emb_id = {p : [] for p in range(10)}
    # for idx, (data,lens,word_name,_,token,token_len) in enumerate(val_loader):
    #     trg_tokens,_,sub_emb = evaluate_attn(model, data, lens, word_name, token, device)
    #     for i in range(len(trg_tokens)-1):    # -1 because we don't want to include the end of word token as a subword
    #         dict_sub_emb[i].append(sub_emb[i].detach())
    #         dict_sub_emb_id[i].append(class_to_idx[trg_tokens[i]])

    # val_avg_precision = []
    # for i in range(len(dict_sub_emb)):
    #     embs = torch.cat(dict_sub_emb[i],0).to(torch.float16).cpu().numpy()
    #     ids = np.array(dict_sub_emb_id[i])
    #     avg_precision,_ = average_precision(embs, ids, "cosine", show_plot=False)
    #     val_avg_precision.append(avg_precision)
    #     print("average precision for subword in the val set at position {} is {}".format(i,avg_precision))

    # print("mean of the average precision for all subwords in the val set is {}".format(np.mean(val_avg_precision)))


    # # AP-CW calculation
    # # create dictoinary of dictionary with position as key and value as list of subword embeddings

    # filename= os.path.join('/'.join(model_weights2.split('/')[:-1]),"dict_pos.pt")
    # if os.path.exists(filename):
    #     print("loading dict_pos")
    #     dict_pos = torch.load(filename)
    # else:
    #     print("creating and saving dict_pos")

    #     dict_pos = {p : {} for p in range(10)}
    #     dict_pos_count = {p : {} for p in range(10)}
    #     for idx, (data,lens,word_name,_,token,token_len) in enumerate(train_loader):
    #         trg_tokens,b,c = evaluate_attn(model, data, lens, word_name, token, device)
    #         trg_tokens = trg_tokens[:-1] # remove <eow> token
    #         c = c[:-1] # remove <eow> token
    #         token = token[0][1:-1] # remove <sow> token
    #         if ''.join(trg_tokens) != ''.join(token):
    #             continue
    #         else:   

    #             for i in range(len(trg_tokens)):    
    #                 if trg_tokens[i] not in dict_pos[i]:
    #                     dict_pos[i][trg_tokens[i]] = c[i]
    #                     dict_pos_count[i][trg_tokens[i]] = 1
    #                 else:
    #                     dict_pos[i][trg_tokens[i]] += c[i]
    #                     dict_pos_count[i][trg_tokens[i]] += 1

    #     # take the averratge of all the subword embeddings base on the position dictionary count
    #     for i in range(len(dict_pos)):
    #         for key in dict_pos[i]:
    #             dict_pos[i][key] = F.normalize(dict_pos[i][key]/dict_pos_count[i][key])
    #     #save the dictionary
    #     torch.save(dict_pos, filename)


    # tokenizer = Tokenizer.from_file(tokenizer_file)
    # df_test_emb = torch.load(os.path.join(emb_file,"test_emb_norm.pt"))
    # df_test = pd.read_csv(os.path.join(emb_file, "test_emb.csv"), converters={'tokenized': literal_eval})

    # sub_embeddings = df_test_emb
    # ids = df_test['ids'].values
    # test_avg_precision,_ = average_precision(sub_embeddings.cpu(),ids, "cosine",show_plot=False)
    # print("average precision on test set words - AP :", test_avg_precision)

    # shuffle_pos = "no"
    # # unique_words = df_test[df_test['tokenized'].str.len()==3]['words'].unique()
    # unique_words = df_test['words'].unique()
    # print("total unique words in test set",len(unique_words))

    # artificial_words = np.zeros(shape=(len(unique_words),128)) ## embedding dim is input_dim  here

    # if shuffle_pos =="no":
    #     flag = [True]*len(unique_words)
    #     for i, w in enumerate(unique_words):
    #         bpe_list = tokenizer.encode(w.lower()).tokens
    #         sum_emb = torch.zeros(1,128)
    #         for j in range(len(bpe_list)):
    #             if bpe_list[j] in dict_pos[j]:
    #                 sum_emb += dict_pos[j][bpe_list[j]].detach()
    #             else:
    #                 flag[i] = False
    #                 # print("not found",bpe_list[j])
    #                 break
    #         artificial_words[i] = torch.tanh(model.proj_layer(sum_emb)).detach().numpy()
    #     artificial_words = F.normalize(torch.from_numpy(artificial_words)).numpy()
    #     artificial_words = artificial_words[flag]
    #     unique_words = unique_words[flag]
    #     print("total unique words in test set",len(unique_words))
    #     print("shape of artificial words",artificial_words.shape)
    #     lista = unique_words
    #     listb = df_test["words"].values

    #     labels = []
    #     for i in lista:
    #         for j in listb:
    #             if i==j:
    #                 labels.append(True)
    #             else:
    #                 labels.append(False)
    #     ap_rw, _ = metric2(artificial_words, df_test_emb, np.array(labels), "cosine") ## calculate df_test_emb befoer hand

    #     print("Average precision for reconstructed words (AP-CW):", ap_rw)


if __name__ == '__main__':
    main()