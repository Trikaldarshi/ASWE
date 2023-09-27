"""
factorizatin model to get the sub-word embedding

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""
import argparse
import datetime
import random
import sys
import os
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch
import copy
from collections import Counter
from os import path
from utility_functions.factorization_dataloader import dataset_embedding
from utility_functions.factorisation_model import fact_net
from utility_functions.utils_function import  save_checkpoints, load_checkpoints, average_precision


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--input_dim", type = int, help = "dimension of input features/embedding", nargs='?', default=128)
    parser.add_argument("--metadata_file", type = str, help = "a text file or dataframe containing embedding metadata")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", nargs='?',default = "./output")
    parser.add_argument("--embedding_loc", type = str, help = "path to pre-computed embeddings for dataloader")
    parser.add_argument("--lr", type = float, help = "learning rate", nargs='?', default=0.001)
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?', default=2)
    parser.add_argument("--n_epochs", type = int, help = "number of epochs", nargs='?', default=10)
    parser.add_argument("--checkpoint_steps", type = int, help = "steps at which checkpoint will be saved",nargs='?',default = 2)
    parser.add_argument("--step_lr", type = int, help = "steps at which learning rate will decrease",nargs='?',default = 20)
    parser.add_argument("--sub_embedding_dim", type = int, help = "value of subword embedding dimensions",nargs='?',default = 128)
    parser.add_argument("--opt", type = str, help = "optimizer", nargs='?', default = "adam", choices=["adam","sgd"])
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",nargs='?',default = "cosine")
    parser.add_argument("--proj", type = str, help = "name of the factorization model",nargs='?')
    parser.add_argument("--norm", type = str, help = "whether to use unit norm embeddings or not", choices=["True","False"])
    parser.add_argument("--loss", type = str, help = "type of reconstruction loss",nargs='?', choices=["L1","L2"])
    parser.add_argument("--features", type = str, help = "type of features",nargs='?', choices=["MFCC","HUBERT"])
    
    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def cal_precision(model, loader, device, class_to_idx, distance):
    sub_embeddings, sub_words, words, unique_id = [], [], [], []
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

    words = np.concatenate(words)       
    sub_words = np.concatenate(sub_words)
    unique_id = np.concatenate(unique_id)
    u_sub_words = np.unique(sub_words)
    sub_word2id = {v: k for k, v in enumerate(u_sub_words)}
    ids = [sub_word2id[w] for w in sub_words]
    sub_embeddings, ids = torch.cat(sub_embeddings,0).to(torch.float16), np.array(ids)
    avg_precision,_ = average_precision(sub_embeddings.cpu(),ids, distance)
    return avg_precision, sub_embeddings, sub_words, unique_id, ids, words

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
    print("epochs:", args.n_epochs)
    print("batch_size:", args.batch_size)
    print("checkpoint steps:", args.checkpoint_steps)
    print("projection layer",args.proj)
    if args.loss=="L2":
        print("using L2 loss: (loss1 + loss2 + loss3) + recon_loss")
    else:
        print("using L1 loss: (loss1 + loss2 + loss3)")
    if args.proj=="True":
        args.proj = True
    else:
        args.proj = False

    if args.norm=="True":
        args.norm = True
    else:
        args.norm = False

    # Check whether the specified text/dataframe meta file exists or not
    isExist = os.path.exists(args.metadata_file)

    if not isExist:
        print(args.metadata_file)
        print("provide the correct path for the metadata file")
        sys.exit(1)


    # Check whether the specified output path exists or not
    isExist = os.path.exists(args.path_to_output)

    # Create a new directory for output if it does not exist 
    if not isExist:
        os.makedirs(args.path_to_output)
        print("The new directory for output is created!")

    ## create a unique output storage location based on time and date with argument files
    # now = str(datetime.datetime.now())
    # now = args.loss + '_' + str(args.proj) + '_' + 'MFCC'
    now = args.loss + '_' + str(args.proj) + '_' + args.features
    os.makedirs(path.join(args.path_to_output,now))
    args.path_to_output = path.join(args.path_to_output,now)

    with open(path.join(args.path_to_output,'config.txt'), 'w') as f:
        for key, value in vars(args).items(): 
                f.write('--%s=%s\n' % (key, value))

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
    model = fact_net(args.input_dim, args.sub_embedding_dim, num_classes,args.proj)
    model = model.to(device)        

    model_description = '_'.join(["fact", str(args.proj), str(device)])

    print(model_description)
    PATH = path.join(args.path_to_output, model_description + ".pt")
    PATH_BEST = path.join(args.path_to_output, model_description + "_BEST.pt")


    # Check whether the specified text file exists or not
    isCheckpoint = os.path.exists(PATH)



    def train_model(model, train_load, val_load, n_epochs):

        if args.opt=="sgd":
            optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.5)

        # define the loss functions to be used
        criterion_cross_ent = nn.CrossEntropyLoss().to(device)
        criterion_cos_emb = nn.CosineEmbeddingLoss().to(device)

        if isCheckpoint==True:
            print("loading the saved checkpoint for training:")
            print("best model:")
            model_best, _, _, best_epoch, history_best = load_checkpoints(model,optimizer,scheduler,PATH_BEST,device)
            print("recent checkpoint:")
            model, optimizer, scheduler, base_epoch, history = load_checkpoints(model,optimizer,scheduler,PATH,device)
            optimizer.param_groups[0]['capturable'] = True # Error: assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors.
            scheduler.step()
            base_epoch += 1
            best_model_wts = copy.deepcopy(model_best.state_dict())
            best_loss = history_best['val'][-1]
            best_ap = history_best['val_avg_precision'][-1]
        else:
            history = dict(train=[], val=[])
            base_epoch = 1
            best_model_wts = copy.deepcopy(model.state_dict())
            best_loss = 10000.0
            best_ap = 0.0

        print("training starting at epoch - ", base_epoch)
        
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(base_epoch, n_epochs + 1):
            model = model.train()

            train_losses = []
            for _, (emb, tokens, _, _, _) in enumerate(train_load):
                optimizer.zero_grad()
                tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
                emb, tokens = emb.to(device), tokens.to(device)
                emb1, emb2, emb3, y1, y2, y3, x_recons = model(emb)
                l1 = tokens[0, :]
                l2 = tokens[1, :]
                l3 = tokens[2, :]
                loss1 = criterion_cross_ent(y1, l1)
                loss2 = criterion_cross_ent(y2, l2)
                loss3 = criterion_cross_ent(y3, l3)
                
                if args.loss=="L2":
                    recon_loss = criterion_cos_emb(x_recons, emb, torch.ones(emb.size()[0]))
                    loss = (loss1 + loss2 + loss3) + recon_loss
                else:
                    loss = loss1 + loss2 + loss3

                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()
            with torch.no_grad():
                for _, (emb, tokens, _, _, _) in enumerate(val_load):
                    tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
                    emb, tokens = emb.to(device), tokens.to(device)
                    emb1, emb2, emb3, y1, y2, y3, x_recons = model(emb)
                    l1 = tokens[0, :]
                    l2 = tokens[1, :]
                    l3 = tokens[2, :]
                    loss1 = criterion_cross_ent(y1, l1)
                    loss2 = criterion_cross_ent(y2, l2)
                    loss3 = criterion_cross_ent(y3, l3)

                    if args.loss=="L2":
                        recon_loss = criterion_cos_emb(x_recons, emb, torch.ones(emb.size()[0]))
                        loss = (loss1 + loss2 + loss3) + recon_loss   
                    else:
                        loss = loss1 + loss2 + loss3
                    
                    val_losses.append(loss.item())
                

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)

            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                print("checkpoint saved for best epoch for average precision")
                save_checkpoints(epoch,model,optimizer,scheduler,history,PATH_BEST)
                best_epoch = epoch
        
            if epoch % args.checkpoint_steps == 0:
                print("checkpoint logging :")
                save_checkpoints(epoch,model,optimizer,scheduler,history,PATH)


            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')
            print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))


            scheduler.step()

        model.load_state_dict(best_model_wts)

        return model.eval(), history


    model, _ = train_model(
        model, 
        train_loader, 
        val_loader, 
        n_epochs=args.n_epochs
    )

    correct1 = 0
    total1 = 0
    correct2 = 0
    total2 = 0
    correct3 = 0
    total3 = 0
    correct = 0

    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for _, (emb, tokens, _, _, _) in enumerate(test_loader):
            tokens = torch.from_numpy(np.vectorize(class_to_idx.get)(np.array(tokens)))
            emb, tokens = emb.to(device), tokens.to(device)
            y1, y2, y3 = model(emb)[3:6]
            l1 = tokens[0, :]
            l2 = tokens[1, :]
            l3 = tokens[2, :]
            _, predicted1 = torch.max(y1.data, 1)
            total1 += l1.size(0)
            correct1 += (predicted1 == l1).sum().item()
            
            _, predicted2 = torch.max(y2.data, 1)
            total2 += l2.size(0)
            correct2 += (predicted2 == l2).sum().item()
            
            _, predicted3 = torch.max(y3.data, 1)
            total3 += l3.size(0)
            correct3 += (predicted3 == l3).sum().item()

            ## overall accuracy
            a1 = (predicted1 == l1)
            a2 = (predicted2 == l2)
            a3 = (predicted3 == l3)
            correct += ((a1 & a2) & a3).sum().item()

    print(total1, total2, total3)
    print(f'Accuracy of the network on the 1st factor: {100 * correct1 // total1} %')
    print(f'Accuracy of the network on the 2nd factor: {100 * correct2 // total2} %')
    print(f'Accuracy of the network on the 3rd factor: {100 * correct3 // total3} %')
    print(f'Accuracy of the network on the all factor together: {100 * correct // total3} %')

if __name__ == "__main__":
    main()

