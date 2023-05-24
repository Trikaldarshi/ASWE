"""
This program is used to train a subword classification model on top of a pretrained model.

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import argparse
import os
import random
import sys
import time
from collections import Counter
from distutils.util import strtobool
from os import path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from models.model_cae import model_cae
from models.sw_cls_cae_pretrained import sw_cls_cae_pretrained
from utility_functions.awe_dataset_class import awe_dataset_pre_computed_pre_training
from utility_functions.utils_function import (average_precision, collate_fn_pre_training,
                                              save_checkpoints)

possible_models = ["HUBERT_BASE","HUBERT_LARGE","HUBERT_XLARGE","WAV2VEC2_BASE","WAV2VEC2_LARGE",
                    "WAV2VEC2_LARGE_LV60K","WAV2VEC2_XLSR53","HUBERT_ASR_LARGE","HUBERT_ASR_XLARGE",
                    "WAV2VEC2_ASR_BASE_10M","WAV2VEC2_ASR_BASE_100H","WAV2VEC2_ASR_BASE_960H",
                    "WAV2VEC2_ASR_LARGE_10M","WAV2VEC2_ASR_LARGE_100H","WAV2VEC2_ASR_LARGE_960H",
                    "WAV2VEC2_ASR_LARGE_LV60K_10M","WAV2VEC2_ASR_LARGE_LV60K_100H","WAV2VEC2_ASR_LARGE_LV60K_960H","MFCC"]


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#


def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--model_name", type=str, help = "name of the model for example, HUBERT_BASE", nargs='?', 
                        default = "HUBERT_BASE", choices = possible_models)
    parser.add_argument("--input_dim", type = int, help = "dimension of input features", nargs='?', 
                        default=768)
    parser.add_argument("--metadata_file", type = str, 
                        help = "a text file or dataframe containing paths of wave files, words,  \
      start point, duration or SSL features metadata file")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored", 
                        nargs='?',default = "./output")
    parser.add_argument("--layer", type = int, help = "layer you want to extract, type mfcc for mfcc", 
                        nargs='?',default=0)
    parser.add_argument("--lr", type = float, help = "learning rate", nargs='?', 
                        default=0.001)
    parser.add_argument("--batch_size", type = int, help = "batch_size", nargs='?',
                         default=2)
    parser.add_argument("--n_epochs", type = int, help = "number of epochs", nargs='?', 
                        default=10)
    parser.add_argument("--pre_compute", type=lambda x:bool(strtobool(x)), nargs='?', const=True, 
                        default=True, help = "use pre computed features or not")
    parser.add_argument("--step_lr", type = int, help = "steps at which learning rate will decrease",
                        nargs='?',default = 20)
    parser.add_argument("--embedding_dim", type = int, help = "value of embedding dimensions",nargs='?',
                        default = 128)
    parser.add_argument("--distance", type = str, help = "type of distance to compute the similarity",
                        nargs='?',default = "cosine")
    parser.add_argument("--opt", type = str, help = "optimizer", nargs='?', default = "adam", 
                        choices=["adam","sgd"])
    parser.add_argument("--hidden_dim", type = int, help = "rnn hidden dimension values",
                         default=512)
    parser.add_argument("--rnn_type", type = str, help = " type or rnn, gru or lstm?", 
                        default="LSTM", choices=["GRU","LSTM"])
    parser.add_argument("--bidirectional", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False, 
                        help = " bidirectional rnn or not")
    parser.add_argument("--num_layers", type = int, help = " number of layers in rnn network, input more than 1",
                         default=2) 
    parser.add_argument("--dropout", type = float, help = "dropout applied inside rnn network", 
                        default=0.2)
    parser.add_argument("--wandb", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False, 
                        help = "use wandb to log your progress or not")

    parser.add_argument("--model_weights", type = str, help = " path of the pre-trained model")

    parser.add_argument("--checkpoint_model", type = str, help = "path to model checkpoints", 
                        nargs='?',default = "/saved_model")
    parser.add_argument("--pre_trained_model_weights", type =lambda x:bool(strtobool(x)), nargs='?', const=True,
                        default=True, help = "use pre trained model weights or not")
    parser.add_argument("--freeze_encoder", type = lambda x:bool(strtobool(x)), nargs='?', const=True, 
                        default=True, help = "freeze encoder or not")
    

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])


def cal_precision(model,loader,device,distance):
  embeddings, words = [], []
  model = model.eval()
  with torch.no_grad():
    for _, (data,lens,word_name,_,_,_) in enumerate(loader):

      lens, perm_idx = lens.sort(0, descending=True)
      data = data[perm_idx]
      word_name = word_name[perm_idx]
      
      data, lens  = data.to(device), lens.to(device)
      # lens = lens.to(device)

      _, emb = model.encoder(data, lens)
      embeddings.append(emb)
      words.append(word_name)
  words = np.concatenate(words)
  uwords = np.unique(words)
  word2id = {v: k for k, v in enumerate(uwords)}
  ids = [word2id[w] for w in words]
  embeddings, ids = torch.cat(embeddings,0).to(torch.float16), np.array(ids)
  avg_precision,_ = average_precision(embeddings.cpu(),ids, distance)

  return avg_precision


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


  ## read the parsed arguments

  args = check_argv()
  
  print(f"{'model_name' :<20} : {args.model_name}")
  print(f"{'input_dim' :<20} : {args.input_dim}")
  print(f"{'metadata_file' :<20} : {args.metadata_file}")
  print(f"{'path_to_output' :<20} : {args.path_to_output}")
  print(f"{'lr' :<20} : {args.lr}")
  print(f"{'layer' :<20} : {args.layer}")
  print(f"{'batch_size' :<20} : {args.batch_size}")
  print(f"{'n_epochs' :<20} : {args.n_epochs}")
  print(f"{'pre_compute' :<20} : {args.pre_compute}")
  print(f"{'step_lr' :<20} : {args.step_lr}")
  print(f"{'embedding_dim' :<20} : {args.embedding_dim}")
  print(f"{'distance' :<20} : {args.distance}")
  print(f"{'opt' :<20} : {args.opt}")
  print(f"{'hidden_dim' :<20} : {args.hidden_dim}")
  print(f"{'rnn_type' :<20} : {args.rnn_type}")
  print(f"{'bidirectional' :<20} : {args.bidirectional}")
  print(f"{'num_layers' :<20} : {args.num_layers}")
  print(f"{'dropout' :<20} : {args.dropout}")
  print(f"{'wandb' :<20} : {args.wandb}")
  print(f"{'model_weights' :<20} : {args.model_weights}")
  print(f"{'checkpoint_model' :<20} : {args.checkpoint_model}")
  print(f"{'pre_trained_model_weights' :<20} : {args.pre_trained_model_weights}")
  print(f"{'freeze_encoder' :<20} : {args.freeze_encoder}")

  
  # Check whether the specified text/dataframe metadata file exists or not
  isExist = os.path.exists(args.metadata_file)

  if not isExist:
      print(args.metadata_file)
      print("provide the correct path for the text/dataframe file having list of wave files")
      sys.exit(1)

  # Check whether the specified output path exists or not
  isExist = os.path.exists(args.path_to_output)
  
  # Create a new directory for output if it does not exist 

  if not isExist:
      os.makedirs(args.path_to_output)
      print("The new directory for output is created!")


  if args.wandb:
    wandb.init(project="word2awe", resume=True, dir=args.path_to_output)
    wandb.config.update(args)

  ## create a unique output storage location with argument files

  args.path_to_output = path.join(args.path_to_output,args.checkpoint_model)

  isExist = os.path.exists(args.path_to_output)
  if not isExist:
    os.makedirs(args.path_to_output)
    print("The new directory for saving checkpoint is created!")

    with open(path.join(args.path_to_output,'config.txt'), 'w') as f:
      for key, value in vars(args).items(): 
              f.write('--%s=%s\n' % (key, value))
  
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  batch_size = args.batch_size

  # print("available device:",device)
  print("Is device CUDA:", device.type=="cuda")
  if device.type == "cuda":
      num_workers = 4
      pin_memory = True
  else:
      num_workers = 0
      pin_memory = False

  print("number of workers:", num_workers)
  print("pin memory status:", pin_memory)


  if args.pre_compute:

    print("using pre-computed features", args.model_name)

    train_data = awe_dataset_pre_computed_pre_training(
      feature_df=args.metadata_file,
      partition="train")
  
    val_data = awe_dataset_pre_computed_pre_training(
        feature_df=args.metadata_file,
        partition="val"
    )
    test_data = awe_dataset_pre_computed_pre_training(
        feature_df=args.metadata_file,
        partition="test"
    )
  else:
    print("not compatible with on the fly computation")
    sys.exit(1)

  # Uncomment the following lines to use a subset of the data for debugging

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


  ## Code for taking acount of subword classes
  # define the class to sub-word labelled dictionary
  # check whether saved dictionary exists or not
  # if not, create a new dictionary and save it
  # if yes, load the dictionary
  filename = os.path.join(args.path_to_output,'dict_tokens.pt')
  print(filename)
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
      torch.save(dict_tokens, filename)

  classes = dict_tokens.keys()
  idx_to_class = {i:j for i, j in enumerate(classes)}
  class_to_idx = {value:key for key,value in idx_to_class.items()}
  num_classes = len(dict_tokens) 

  ## Define the pre-trained model

  pre_model = model_cae(args.input_dim, args.hidden_dim, args.embedding_dim, args.rnn_type, args.bidirectional, args.num_layers, args.dropout)
  print("Structure of the pre-trained AWE model:")
  print(pre_model)

  if args.pre_trained_model_weights:
    pre_model_checkpoint = torch.load(args.model_weights, map_location=torch.device(device))
    pre_model.load_state_dict(pre_model_checkpoint['model_state_dict'])
    pre_model = pre_model.to(device)

  if args.freeze_encoder:
    pre_model.eval()
    for param in pre_model.parameters():
        param.requires_grad = False

  ## Define the model
  model = sw_cls_cae_pretrained(args.hidden_dim, args.embedding_dim, args.rnn_type, args.bidirectional, args.num_layers, num_classes, args.dropout, pre_model.encoder)
  model = model.to(device)
  print("Structure of the model for subword classification:")
  print(model)

  model_description = '_'.join([args.model_name, str(args.embedding_dim)])


  PATH = path.join(args.path_to_output, model_description + ".pt")
  PATH_BEST = path.join(args.path_to_output, model_description + "_BEST.pt")

  isCheckpoint = os.path.exists(PATH)




  def train_model(model, train_load, val_load, n_epochs):

    if args.opt=="sgd":
      optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    else:
      optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.5)


    print("class_to_idx['PAD']", class_to_idx['PAD'])
    criterion2 = nn.CrossEntropyLoss(ignore_index=class_to_idx['PAD']).to(device)

    if isCheckpoint==True:
      print("recent checkpoint:")
      checkpoint = torch.load(PATH,map_location=torch.device(device))
      model.load_state_dict(checkpoint['model_state_dict'])
      optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
      scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
      scheduler.step()
      base_epoch = checkpoint['epoch']
      history = checkpoint['loss_history']
      best_val = history['best_val']
      best_epoch = history['best_epoch']
      optimizer.param_groups[0]['capturable'] = True # Error: assert not step_t.is_cuda, "If capturable=False, state_steps should not be CUDA tensors.
      base_epoch += 1
    else:
      history = dict(train_loss = [], val_loss = [], best_val = torch.inf, best_epoch = 0)
      base_epoch = 1
      best_val = torch.inf
      best_epoch = 0

    print("training starting at epoch - ", base_epoch)
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(base_epoch, n_epochs + 1):
      model = model.train()

      train_losses = []
      for _, (x, lens_x, _,_,token_x,token_len) in enumerate(train_load):

        optimizer.zero_grad()
        for m in range(len(token_x)):
            for subword in range(len(token_x[m])):
                token_x[m][subword] = class_to_idx[token_x[m][subword]]
            token_x[m] = torch.tensor(token_x[m])
        
        token_x = torch.nn.utils.rnn.pad_sequence(token_x, batch_first=True, padding_value=class_to_idx['PAD'])

        x, lens_x, token_x = x.to(device), lens_x.to(device), token_x.to(device)
        lens_x, perm_idx = lens_x.sort(0, descending=True)
        x = x[perm_idx]
        token_x = token_x[perm_idx]
        token_len = torch.tensor(token_len).to(device)
        token_len = token_len[perm_idx]

        decoder_output   = model(x, lens_x, token_x, token_len)
        token_x = token_x.T.contiguous()
        decoder_output = decoder_output[1:].view(-1,decoder_output.size(2))
        token_x = token_x[1:].view(-1)

        loss = criterion2(decoder_output,token_x)

        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

      val_losses =  []
      model = model.eval()
      with torch.no_grad():
        for _, (data,lens,_,_,token,token_len) in enumerate(val_load):

          for m in range(len(token)):
              for subword in range(len(token[m])):
                  token[m][subword] = class_to_idx[token[m][subword]]
              token[m] = torch.tensor(token[m])

          token = torch.nn.utils.rnn.pad_sequence(token, batch_first=True, padding_value=class_to_idx['PAD'])

          data, lens, token = data.to(device), lens.to(device), token.to(device)
          lens, perm_idx = lens.sort(0, descending=True)
          data = data[perm_idx]
          token = token[perm_idx]
          token_len = torch.tensor(token_len).to(device)
          token_len = token_len[perm_idx]

          decoder_output = model(data, lens, token, token_len)
          token = token.T.contiguous()
          decoder_output = decoder_output[1:].view(-1,decoder_output.size(2))
          token = token[1:].view(-1)  

          loss = criterion2(decoder_output,token)   
             
          val_losses.append(loss.item())


      train_loss = np.mean(train_losses)
      val_loss = np.mean(val_losses)

      history['train_loss'].append(train_loss)
      history['val_loss'].append(val_loss)


      if val_loss < best_val:
        best_val = val_loss
        print("checkpoint saved for best epoch for validation loss")
        save_checkpoints(epoch,model,optimizer,scheduler,history,PATH_BEST)
        best_epoch = epoch
        history['best_val'] = best_val
        history['best_epoch'] = best_epoch
        print("checkpoint saved for best epoch for average precision")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': history
        }, PATH_BEST)
      
      if epoch % 1 == 0:
        print("checkpoint logging :")
        torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss_history': history
        }, PATH)


      print(f'Epoch {epoch}: train loss {train_loss}; val loss {val_loss} ; best epoch {best_epoch}')
      print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

      if args.wandb:
        wandb.log({"train loss": train_loss, "val loss":val_loss, 
                   "lr_history":optimizer.param_groups[0]['lr'], "best epoch":best_epoch,})
        
      scheduler.step()

        

  train_model(
    model, 
    train_loader, 
    val_loader, 
    n_epochs=args.n_epochs
  )

  # Load the best model
  checkpoint_best = torch.load(PATH_BEST,map_location=torch.device(device))
  model.load_state_dict(checkpoint_best['model_state_dict'])
  history = checkpoint_best['loss_history']
  best_epoch = history["best_epoch"]
  best_val = history["best_val"]

  print("best loss on val set:", best_val)
  print("best epoch:", best_epoch)
  print(" We are done! Bye Bye. Have a nice day!")
  if args.wandb:
    wandb.log({"best val loss":best_val})
    wandb.finish()
 

if __name__ == "__main__":
    main()

