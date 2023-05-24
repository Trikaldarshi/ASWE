"""
Compute the mfcc features with their delta and delta-delta  and store them on local disk

Author: Amit Meghanani

Contact: ameghanani1@sheffield.ac.uk

"""

import argparse
import os
import sys
import torch
from tqdm import tqdm
import pandas as pd
import numpy as np
from os import path
import glob
import librosa


#------------------------------#
#      UTILITY FUNCTIONS       #
#------------------------------#

def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--metadata_file_path", type = str,help = "a text file or dataframe containing paths of wave files, words, start point, duration")
    parser.add_argument('--metadata_file_list', nargs="+", default=["train.csv", "val.csv","test.csv"], help = " list of metadata files")
    parser.add_argument("--path_to_output", type = str, help = "path to output folder where features will be stored")
    parser.add_argument("--path_to_libri", type = str, help = "base path to librispeech dataset")
    
    if len(sys.argv)==1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def split_string(strs):
    return strs.split(sep="/")[-2]

def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

#------------------------------#
#      MAIN FUNCTION           #
#------------------------------#

def main():
    args = check_argv()

    # Check whether the specified metadata file exists or not
    isExist = os.path.exists(args.metadata_file_path)

    if not isExist:
        print(args.metadata_file_path)
        print("provide the correct path for the text/dataframe file having list of wave files")
        sys.exit(1)

    # Check whether the specified librispeech dataset path exists or not
    isExist = os.path.exists(args.path_to_libri)

    if not isExist:
        print("provide the correct path for the librispeech dataset")
        sys.exit(1)

    # Check whether the specified output path exists or not
    for m in args.metadata_file_list[0].split(','):

        isExist = os.path.exists(path.join(args.path_to_output,m.split(".")[0]))
    
        # Create a new directory for output if it does not exist 
        if not isExist:
            os.makedirs(path.join(args.path_to_output,m.split(".")[0]))
            print("The new directory for output is created!",m.split(".")[0])

    ## Extraction of MFCCs 

    hop_length = 320 # 20 ms shift as sampling rate is 16 Khz for librispeech
    win_length = 480 # 30 ms window length

    for f_name in args.metadata_file_list[0].split(','):
        data = pd.read_csv(os.path.join(args.metadata_file_path,f_name))

        for _,row in tqdm(data.iterrows()):
            # get the path of the file

            file_path = path.join(args.path_to_libri,row["filename_path"].strip("./").strip("\n"))

            # create name to save the features

            word_description = row["word"] + "_" + str(row["start"]) + "_" + str(row["duration"]) + "_" \
            + path.splitext(path.split(row["filename_path"])[-1])[0]

            # load file

            y, sr = librosa.load(file_path,sr=None)
            magic_number = sr/hop_length
            st_v = int(np.floor(magic_number*row["start"]))
            ed_v = int(np.ceil(magic_number*row["duration"]))

            # compute mfcc

            mfcc = librosa.feature.mfcc(y=y, sr=sr, win_length=win_length, hop_length=hop_length)

            # compute delta

            mfcc_delta = librosa.feature.delta(mfcc)

            # compute delta delta

            mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

            # clip

            mfcc = mfcc[:, st_v:st_v + ed_v]
            mfcc_delta = mfcc_delta[:, st_v:st_v + ed_v]
            mfcc_delta2 = mfcc_delta2[:, st_v:st_v + ed_v]

            # # Normalization ## uncomment this if you want to normalize MFCCs

            # mfcc = scale_minmax(mfcc)
            # mfcc_delta = scale_minmax(mfcc_delta)
            # mfcc_delta2 = scale_minmax(mfcc_delta2)

            # concatenate MFCCs, delta and delta-delta features

            mfcc_dd = np.concatenate((mfcc, mfcc_delta, mfcc_delta2)) # [n_features x seq_len]
            mfcc_dd = mfcc_dd.transpose()                             # [seq_len x n_features]
            word_features  = np.expand_dims(mfcc_dd, axis=0)          # [1, seq_len, n_features]
            word_features = torch.from_numpy(word_features)

            # save features

            torch.save(word_features, path.join(args.path_to_output,f_name.split(".")[0],word_description+".pt"))


    ## save the metadata file for the extracted features
    
    PATH = args.path_to_output
    my_files = sorted(glob.glob(PATH + '*/**/*.pt',recursive=True))
    print("total calculated features files",len(my_files))
    df_metadata = pd.DataFrame(my_files,columns=["path"])
    df_metadata["partition"] = df_metadata["path"].apply(split_string)
    df_metadata.to_csv(os.path.join(PATH,"mfcc_feature_metadata.csv"),index=False)

if __name__ == "__main__":
    main()
