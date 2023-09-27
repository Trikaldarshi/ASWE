"""
Calculation of metric AP-SD (subword discrimination task) and AP-RW (reconstructed word)

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""
import argparse
import wandb
import random
import sys
import os
import numpy as np
import pandas as pd
import torch
from collections import Counter
from os import path
from utility_functions.factorization_dataloader import dataset_embedding
from utility_functions.factorisation_model import fact_net
from utility_functions.utils_function import average_precision, metric2
from ast import literal_eval
import torch.nn.functional as F


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--input_dim", type = int, help = "dimension of input features/embedding", nargs='?', default=128)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing embedding metadata")
    parser.add_argument("--model_loc", type = str, help = "location of the saved model")
    parser.add_argument("--embedding_loc", type = str, help = "path to pre-computed embeddings for dataloader")
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--sub_embedding_dim", type = int, help = "value of subword embedding dimensions",nargs='?',default = 128)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    parser.add_argument("--proj", type = str, help = "whether to use unit norm embeddings or not", choices=["True","False"])
    parser.add_argument("--norm", type = str, help = "whether to use unit norm embeddings or not", choices=["True","False"])
    parser.add_argument("--shuffle_pos", type = str, help = "whether to swap subword position 2 and 3 or not", choices=["True","False"])

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def cal_precision(model, loader, device, class_to_idx, distance):
    sub_embeddings, sub_words, words, unique_id = [], [], [], []
    s1,s2,s3,e1,e2,e3 = [],[],[],[],[],[]
    model = model.eval()
    with torch.no_grad():
        for idx, (emb, tokens, _, word_name, sp_ch_id) in enumerate(loader):
            tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
            emb, tokens = emb.to(device), tokens.to(device)
        
            emb1, emb2, emb3 = model(emb)[0:3]
            l1 = tokens[0, :]
            l2 = tokens[1, :]
            l3 = tokens[2, :]
            # x1 is the first sub-word of the words in batch, same for x2, x3
            sub_embeddings.append(emb1)
            sub_embeddings.append(emb2)
            sub_embeddings.append(emb3)
            
            sub_words.append(l1.cpu())
            sub_words.append(l2.cpu())
            sub_words.append(l3.cpu())
            unique_id.append(sp_ch_id*3)
            words.append(word_name*3)
            s1.append(l1)
            s2.append(l2)
            s3.append(l3)
            e1.append(emb1)
            e2.append(emb2)
            e3.append(emb3)

    words = np.concatenate(words)       
    sub_words = np.concatenate(sub_words)
    s1 = np.concatenate(s1)
    s2 = np.concatenate(s2)
    s3 = np.concatenate(s3)

    unique_id = np.concatenate(unique_id)
    u_sub_words = np.unique(sub_words)
    sub_word2id = {v: k for k, v in enumerate(u_sub_words)}
    ids = [sub_word2id[w] for w in sub_words]
    sub_embeddings, ids = torch.cat(sub_embeddings,0).detach().cpu(), np.array(ids)

    e1 = torch.cat(e1,0).to(torch.float16)
    e2 = torch.cat(e2,0).to(torch.float16)
    e3 = torch.cat(e3,0).to(torch.float16)
    return s1,s2,s3,e1,e2,e3,sub_embeddings, sub_words, unique_id, ids, words

    

#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#
def main():

    # For reproducibility

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

    if args.proj=="True":
        args.proj = True
    else:
        args.proj = False

    if args.norm=="True":
        args.norm = True
    else:
        args.norm = False

    if args.shuffle_pos=="True":
        args.shuffle_pos = True
    else:
        args.shuffle_pos = False

    print("batch_size:", args.batch_size)

    # Check whether the specified text/dataframe meta file exists or not
    isExist = os.path.exists(args.metadata_file)

    if not isExist:
        print(args.metadata_file)
        print("provide the correct path for the metadata file")
        sys.exit(1)


    # Check whether the specified output path exists or not
    isExist = os.path.exists(args.model_loc)

    # Create a new directory for output if it does not exist 
    if not isExist:
        print(args.model_loc)
        print("provide the correct path for the model file")
        sys.exit(1)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size

    print("available device:",device)

    if device.type == "cuda":
        num_workers = 1
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False

    if args.norm:
        print("using normalized word embeddings")
        train_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "train_emb.csv"),
        embedding_mat = os.path.join(args.embedding_loc, "train_emb_norm.pt")
        )
        val_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "val_emb.csv"),
        embedding_mat=os.path.join(args.embedding_loc, "val_emb_norm.pt")
        )
        test_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "test_emb.csv"),
        embedding_mat = os.path.join(args.embedding_loc, "test_emb_norm.pt")
        )
    else:
        print("using un normalized word embeddings")
        train_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "train_emb.csv"),
        embedding_mat = os.path.join(args.embedding_loc, "train_emb.pt")
        )
        val_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "val_emb.csv"),
        embedding_mat=os.path.join(args.embedding_loc, "val_emb.pt")
        )
        test_data = dataset_embedding(
        metadata = os.path.join(args.metadata_file, "test_emb.csv"),
        embedding_mat = os.path.join(args.embedding_loc, "test_emb.pt")
        )

    # indices = np.random.choice(range(len(train_data)), 5000, replace=False)
    # train_data = torch.utils.data.Subset(train_data, indices)
    # indices = np.random.choice(range(len(val_data)), 5000, replace=False)
    # val_data = torch.utils.data.Subset(val_data, indices)
    # test_data = torch.utils.data.Subset(test_data, indices)

    print("length of training data:",len(train_data))
    print("length of validation data:",len(val_data))
    print("length of test data:",len(test_data))

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=None,
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
        collate_fn=None,
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
        collate_fn=None,
        drop_last = False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g
    )

    # define the class to sub-word labelled dictionary

    token_list = []
    for _, (_, tokens, _, _, _) in enumerate(train_loader):
        token_list = token_list + tokens

    out = [item for t in token_list for item in t]
    dict_tokens = Counter(out)
    print("total tokens:",len(dict_tokens))
    classes = dict_tokens.keys()
    idx_to_class = {i:j for i, j in enumerate(classes)}
    class_to_idx = {value:key for key,value in idx_to_class.items()}
    num_classes = len(dict_tokens)
    
    # Define the model
    model = fact_net(args.input_dim, args.sub_embedding_dim, num_classes, args.proj)
    model = model.to(device)       

    checkpoint = torch.load(args.model_loc, map_location=torch.device(device))

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    ## AP-SD metric
    print("calculating AP-SD.........")
    s1,s2,s3,e1,e2,e3,sub_embeddings, _, _, ids, _ = cal_precision(model, test_loader, device, class_to_idx, args.distance)
    # test_avg_precision,_ = average_precision(sub_embeddings.cpu(),ids, args.distance,show_plot=False)
    # print("average precision on test set - AP-SD :", test_avg_precision)

    test_avg_precision,_ = average_precision(e1.cpu(),s1, args.distance,show_plot=False)
    print("average precision on test set - ABX1 :", test_avg_precision)

    test_avg_precision,_ = average_precision(e2.cpu(),s2,args.distance,show_plot=False)
    print("average precision on test set - ABX2  :", test_avg_precision)

    # test_avg_precision,_ = abx(e2.cpu(),s2, s1,s3,args.distance,show_plot=False)
    # print("average precision on test set - ABX2 with different context:", test_avg_precision)

    test_avg_precision,_ = average_precision(e3.cpu(),s3, args.distance,show_plot=False)
    print("average precision on test set - ABX3 :", test_avg_precision)


    ## AP-RW metric

    print("calculating AP-RW.........")
    print("extracting subword embedding for every position from the train set....")
    s1,s2,s3,e1,e2,e3,_, sub_words, unique_id, ids, words = cal_precision(model, train_loader, device, class_to_idx, args.distance)

    df_train_sub_metadata = pd.DataFrame()
    df_train_sub_metadata["unique_id"] = unique_id
    df_train_sub_metadata["words"] = words
    df_train_sub_metadata["ids"] = ids
    df_train_sub_metadata["sub_words"] = np.vectorize(idx_to_class.get)(sub_words)
    
    df_train_sub_metadata_plus = pd.DataFrame()
    df_train_sub_metadata_plus["s1"] = np.vectorize(idx_to_class.get)(s1)
    df_train_sub_metadata_plus["s2"] = np.vectorize(idx_to_class.get)(s2)
    df_train_sub_metadata_plus["s3"] = np.vectorize(idx_to_class.get)(s3)

    if args.norm:
        df_test_emb = torch.load(os.path.join(args.embedding_loc, "test_emb_norm.pt"))
        df_test_metadata = pd.read_csv(os.path.join(args.metadata_file, "test_emb.csv"), converters={'tokenized': literal_eval})
    else:
        df_test_emb = torch.load(os.path.join(args.embedding_loc, "test_emb.pt"))
        df_test_metadata = pd.read_csv(os.path.join(args.metadata_file, "test_emb.csv"), converters={'tokenized': literal_eval})



    ## calculate average sub-word embedding from training set for all three positions

    list_sub_words1 = df_train_sub_metadata_plus['s1'].unique()
    print("list of subwords in case of first subword")
    sub_word_mat1 = np.zeros(shape=(len(list_sub_words1), args.sub_embedding_dim))
    for i, sword in enumerate(list_sub_words1):
        index_sword = df_train_sub_metadata_plus[df_train_sub_metadata_plus["s1"]==sword].index
        sub_word_mat1[i] = torch.mean(e1[index_sword], 0).numpy() 

    if args.norm:
        print("error alert")
        sub_word_mat1 = F.normalize(torch.from_numpy(sub_word_mat1),dim=1)
        print("checking norm is 1",torch.norm(sub_word_mat1[0]))
        sub_word_mat1 = sub_word_mat1.numpy()
    
    
    sub_word_dict1 = dict(zip(list_sub_words1,sub_word_mat1))
    print("total unique subwords in 1st position", len(list_sub_words1))
    print(sub_word_mat1.shape)


    list_sub_words2 = df_train_sub_metadata_plus['s2'].unique()
    print("list of subwords in case of second subword")
    sub_word_mat2 = np.zeros(shape=(len(list_sub_words2), args.sub_embedding_dim))
    for i, sword in enumerate(list_sub_words2):
        index_sword = df_train_sub_metadata_plus[df_train_sub_metadata_plus["s2"]==sword].index
        sub_word_mat2[i] = torch.mean(e2[index_sword], 0).numpy() 

    if args.norm:
        print("error alert")
        sub_word_mat2 = F.normalize(torch.from_numpy(sub_word_mat2))
        print("checking norm is 1",torch.norm(sub_word_mat2[0]))
        sub_word_mat2 = sub_word_mat2.numpy()
        
    sub_word_dict2 = dict(zip(list_sub_words2,sub_word_mat2))
    print("total unique subwords in second position", len(list_sub_words2))
    print(sub_word_mat2.shape)

    list_sub_words3 = df_train_sub_metadata_plus['s3'].unique()
    print("list of subwords in case of third subword")
    sub_word_mat3 = np.zeros(shape=(len(list_sub_words3), args.sub_embedding_dim))
    for i, sword in enumerate(list_sub_words3):
        index_sword = df_train_sub_metadata_plus[df_train_sub_metadata_plus["s3"]==sword].index
        sub_word_mat3[i] = torch.mean(e3[index_sword], 0).numpy() 

    if args.norm:
        print("error alert")
        sub_word_mat3 = F.normalize(torch.from_numpy(sub_word_mat3))
        print("checking norm is 1",torch.norm(sub_word_mat3[0]))
        sub_word_mat3 = sub_word_mat3.numpy()
        
    sub_word_dict3 = dict(zip(list_sub_words3,sub_word_mat3))
    print("total unique subwords in third position", len(list_sub_words3))
    print(sub_word_mat3.shape)


    unique_words = df_test_metadata["words"].unique()
    print("total unique words in test set",len(unique_words))

    artificial_words = np.zeros(shape=(len(unique_words), args.input_dim)) ## embedding dim is input_dim  here

    if args.shuffle_pos==False:
        flag = []
        for i, w in enumerate(unique_words):
            index = df_test_metadata[df_test_metadata["words"] == w].index[0]
            bpe_list = df_test_metadata.iloc[index,3]
            
            # check if the subwords are present in the training set or not
            if bpe_list[0] in list_sub_words1 and bpe_list[1] in list_sub_words2 and bpe_list[2] in list_sub_words3:
                sum_emb = sub_word_dict1[bpe_list[0]] + sub_word_dict2[bpe_list[1]] + sub_word_dict3[bpe_list[2]]
                flag.append(True)
            else:
                sum_emb = np.zeros((1,128))
                flag.append(False)

            if args.proj:
                sum_emb = torch.from_numpy(sum_emb)
                artificial_words[i] = torch.tanh(model.fc_projection(sum_emb.float())).detach().numpy()
            else:
                artificial_words[i] = sum_emb
        if args.norm:
            artificial_words = F.normalize(torch.from_numpy(artificial_words)).numpy()

        artificial_words = artificial_words[flag]
        unique_words = unique_words[flag]

        lista = unique_words
        listb = df_test_metadata["words"].values

        labels = []
        for i in lista:
            for j in listb:
                if i==j:
                    labels.append(True)
                else:
                    labels.append(False)
        ap_rw, _ = metric2(artificial_words, df_test_emb, np.array(labels), args.distance)

        print("Average precision for constructed words (AP-CW):", ap_rw)


    else:
        flag = []
        for i, w in enumerate(unique_words):
            index = df_test_metadata[df_test_metadata["words"] == w].index[0]
            bpe_list = df_test_metadata.iloc[index,3]

            # if bpe_list[0] in list_sub_words3 and bpe_list[2] in list_sub_words1:
            #     sum_emb = sub_word_dict1[bpe_list[2]] + sub_word_dict2[bpe_list[1]] + sub_word_dict3[bpe_list[0]]
            if bpe_list[1] in list_sub_words3 and bpe_list[2] in list_sub_words2:
                sum_emb = sub_word_dict1[bpe_list[0]] + sub_word_dict3[bpe_list[1]] + sub_word_dict2[bpe_list[2]]
                flag.append(True)
            else: 
                sum_emb = np.zeros((1,128))
                flag.append(False)

            if args.proj:
                sum_emb = torch.from_numpy(sum_emb)
                artificial_words[i] = torch.tanh(model.fc_projection(sum_emb.float())).detach().numpy()
            else:
                artificial_words[i] = sum_emb

        if args.norm:
            artificial_words = F.normalize(torch.from_numpy(artificial_words)).numpy()

        artificial_words = artificial_words[flag]
        unique_words = unique_words[flag]

        lista = unique_words
        listb = df_test_metadata["words"].values

        labels = []
        for i in lista:
            for j in listb:
                if i==j:
                    labels.append(True)
                else:
                    labels.append(False)
        test_metric2, _ = metric2(artificial_words, df_test_emb, np.array(labels), args.distance)

        print("Average precision for constructed words (AP-CW) for swapped pos 2 and 3:", test_metric2)

    
    print("Calculating AP-RW.....")
    model = model.eval()
    with torch.no_grad():
        original_embeddings = []
        reconstructed_embeddings = []
        labels = []
        for idx, (emb, tokens, _, word_name, sp_ch_id) in enumerate(test_loader):
            tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
            emb, tokens = emb.to(device), tokens.to(device)
            x_recons = model(emb)[-1]

            original_embeddings.append(emb.to(torch.float16).detach())
            reconstructed_embeddings.append(x_recons.to(torch.float16).detach())
            labels.append(word_name)
        
        labels = np.concatenate(labels)
        original_embeddings = torch.cat(original_embeddings,0).cpu()
        reconstructed_embeddings = torch.cat(reconstructed_embeddings,0).cpu()

        labels_match = []
        lista = labels
        listb = labels
        for i in lista:
            for j in listb:
                if i==j:
                    labels_match.append(True)
                else:
                    labels_match.append(False)
        print("total pairs for comparison:", len(labels_match))
        test_metric_ap_rw, _ = metric2(reconstructed_embeddings, original_embeddings, np.array(labels_match), args.distance)

        print("Average precision for reconstructed words (AP-RW):", test_metric_ap_rw)

if __name__ == "__main__":
    main()