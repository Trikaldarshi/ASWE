import torch
import numpy as np
from utility_functions.awe_dataset_class import awe_dataset_pre_computed
from utility_functions.utils_function import average_precision, collate_fn
import torch.nn as nn
import random
from utility_functions.model_cae import model_cae
import pandas as pd
import argparse
import sys
import os
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer

possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H","MFCC", "SPEC"]
# Utility function
def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model_name", type=str, help = "name of the model for example, HUBERT_BASE", nargs='?', default = "HUBERT_BASE", choices = possible_models)
    parser.add_argument("--input_dim", type = int, help = "dimension of input features", nargs='?', default=768)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing paths of wave files, words, start point, duration \
      or SSL features metadata file")
    parser.add_argument("--model_weights", type = str, help = "path of the pre-trained model which will be used as a embedding extractor")
    parser.add_argument("--tokenizer", type = str, help = "path of the pre-trained tokenizer model")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", nargs='?',default = "./output")
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--embedding_dim", type = int, help = "value of embedding dimensions",nargs='?',default = 128)
    parser.add_argument("--hidden_dim", type = int, help = "rnn hidden dimension values", default=512)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    parser.add_argument("--rnn_type", type = str, help = " type or rnn, gru or lstm?", default="LSTM", choices=["GRU","LSTM"])
    parser.add_argument("--bidirectional", type = bool, help = " bidirectional rnn or not", default = True)
    parser.add_argument("--num_layers", type = int, help = " number of layers in rnn network, input more than 1", default=2) 
    parser.add_argument("--dropout", type = float, help = "dropout applied inside rnn network", default=0.2)

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def cal_precision(model, loader, device, distance):
    embeddings, words, unique_id = [], [], []
    model = model.eval()
    with torch.no_grad():
        for idx, (data, lens, word_name, sp_ch_id) in enumerate(loader):
            lens, perm_idx = lens.sort(0, descending=True)
            data = data[perm_idx]
            word_name = word_name[perm_idx]

            data, lens  = data.to(device), lens.to(device)
            _,emb = model.encoder(data, lens)
            embeddings.append(emb)
            words.append(word_name)
            unique_id.append(sp_ch_id)
            print(idx)
    words = np.concatenate(words)
    unique_id = np.concatenate(unique_id)
    uwords = np.unique(words)
    word2id = {v: k for k, v in enumerate(uwords)}
    ids = [word2id[w] for w in words]
    embeddings, ids = torch.cat(embeddings,0).detach().cpu(), np.array(ids)
    if len(embeddings)>=10000:
        avg_precision = 0
    else:
        avg_precision,_ = average_precision(embeddings,ids, distance, show_plot=True)
    return avg_precision, embeddings, words, unique_id, ids




# MAIN Function

def main():
    # For reproducibility
    print("This is main program")
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

    args = check_argv()

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer = Tokenizer.from_file(args.tokenizer)
    # check whether the output path exists or not.

    isExist = os.path.exists(args.path_to_output)
  
    # Create a new directory for output because it does not exist 
    if not isExist:
        os.makedirs(args.path_to_output)
        print("The new directory for output is created to save embeddings!")
    
    ## Load the tokenizer:
    def get_tokens(x):
        if type(x) != str:
            print("problem in data", x)
        return tokenizer.encode(x.lower()).tokens

    input_dim =  args.input_dim         #768
    hidden_dim = args.hidden_dim        #256
    embedding_dim = args.embedding_dim  #128
    rnn_type = args.rnn_type            #"GRU"
    bidirectional = args.bidirectional       #True
    num_layers = args.num_layers        #4
    dropout = args.dropout              #0.2
    batch_size = args.batch_size        #64



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Is device CUDA?:", device.type=="cuda")
    if device.type == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    print("using pre-computed features", args.model_name)
    
    train_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="train"
    )
    val_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="val"
    )
    test_data = awe_dataset_pre_computed(
        feature_df=args.metadata_file,
        partition="test"
    )
    print("length of training data:",len(train_data))
    print("length of validation data:",len(val_data))
    print("length of test data:",len(test_data))
    
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
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
        collate_fn=collate_fn,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )


    model = model_cae(input_dim, hidden_dim, embedding_dim, rnn_type,
    bidirectional, num_layers, dropout)

    checkpoint = torch.load(args.model_weights, map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()



    print("extracting embedding for training set")
    train_avg_precision, emb_train, words, unique_id, ids = cal_precision(model, train_loader, device, args.distance)
    print("train average precision:", train_avg_precision)
    df_train = pd.DataFrame()
    df_train["unique_id"] = unique_id
    df_train["words"] = words
    df_train["ids"] = ids
    df_train['tokenized'] = df_train['words'].apply(get_tokens)
    torch.save(emb_train, os.path.join(args.path_to_output, "train_emb.pt"))
    df_train.to_csv(os.path.join(args.path_to_output,"train_emb.csv"), index=False)

    print("extracting embedding for val set")
    val_avg_precision, emb_val, words, unique_id, ids = cal_precision(model, val_loader, device, args.distance)
    print("val average precision:", val_avg_precision)
    df_val = pd.DataFrame()
    df_val["unique_id"] = unique_id
    df_val["words"] = words
    df_val["ids"] = ids
    df_val['tokenized'] = df_val['words'].apply(get_tokens)
    torch.save(emb_val, os.path.join(args.path_to_output, "val_emb.pt"))
    df_val.to_csv(os.path.join(args.path_to_output, "val_emb.csv"), index=False)

    print("extracting embedding for test set")
    test_avg_precision, emb_test, words, unique_id, ids = cal_precision(model, test_loader, device, args.distance)
    print("test average precision:", test_avg_precision)
    df_test = pd.DataFrame()
    df_test["unique_id"] = unique_id
    df_test["words"] = words
    df_test["ids"] = ids
    df_test['tokenized'] = df_test['words'].apply(get_tokens)
    torch.save(emb_test, os.path.join(args.path_to_output, "test_emb.pt"))
    df_test.to_csv(os.path.join(args.path_to_output, "test_emb.csv"), index=False)

    ## Load the embeddings, normalize and then save again!

    # Load
    train_emb = torch.load(os.path.join(args.path_to_output, "train_emb.pt"))
    val_emb = torch.load(os.path.join(args.path_to_output, "val_emb.pt"))
    test_emb = torch.load(os.path.join(args.path_to_output, "test_emb.pt"))

    # Normalize
    train_emb_unit_norm = torch.nn.functional.normalize(train_emb,dim=1)
    val_emb_unit_norm = torch.nn.functional.normalize(val_emb,dim=1)
    test_emb_unit_norm = torch.nn.functional.normalize(test_emb,dim=1)

    # Save again!
    torch.save(train_emb_unit_norm, os.path.join(args.path_to_output, "train_emb_norm.pt"))
    torch.save(val_emb_unit_norm, os.path.join(args.path_to_output, "val_emb_norm.pt"))
    torch.save(test_emb_unit_norm, os.path.join(args.path_to_output, "test_emb_norm.pt"))


if __name__ == "__main__":
    main()