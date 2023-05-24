import glob
import os
import numpy as np
import pandas as pd
import argparse
import sys
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from utility_functions.sampling import sampling
from distutils.util import strtobool
import swifter


def get_spk_id_and_ch_id(df):
    return "-" + "-".join(df.split(sep = "_")[0].split(sep="-")[1:])
def match_string(target_string,df2):
    matched_string = df2.loc[df2['path'].str.contains(target_string, case=False)]
    if len(matched_string)==0:
        return np.nan
    return matched_string.values[0][0]

def check_argv():
    """ Check the command line arguments."""
    parser = argparse.ArgumentParser(add_help=True, fromfile_prefix_chars='@')
    parser.add_argument("--libri_path", type=str, help = "path for librispeech dataset")
    parser.add_argument("--vocab_size", type=int, help="size of the vocabulary")
    parser.add_argument("--libri_metadata", type=str, help = "path for librispeech metadata")
    parser.add_argument("--min_duration", type=float, help = "min duration of the word")
    parser.add_argument("--num_subwords", type=int, help="no of subwords in the word, enter -1 for no restrictions")
    parser.add_argument("--num_sampling", type=int, help = "number of unique words to be sampled", default=5000)
    parser.add_argument("--num_deletion", type=int, help = "number of words to be deleted at a time", default=1)
    parser.add_argument("--flag", type=lambda x:bool(strtobool(x)), nargs='?', const=True, default=False, 
                        help="to use unaligned data or not, default is False")

    if len(sys.argv)==1:
        parser.print_help()
        print("something is wrong")
        sys.exit(1)
    return parser.parse_args(sys.argv[1:])

def main():
    args = check_argv()
    print(f"{'libri_path' :<15} : {args.libri_path}")
    print(f"{'vocab_size' :<15} : {args.vocab_size}")
    print(f"{'libri_metadata' :<15} : {args.libri_metadata}")
    print(f"{'min_duration' :<15} : {args.min_duration}")
    print(f"{'num_subwords' :<15} : {args.num_subwords}")
    print(f"{'num_sampling' :<15} : {args.num_sampling}")
    print(f"{'num_deletion' :<15} : {args.num_deletion}")
    print(f"{'flag' :<15} : {args.flag}")

    # check for data folder, if not present create one
    if not os.path.exists("data"):
        os.makedirs("data")



    paths = ['train-clean-100',
            'train-clean-360',
            'train-other-500',
            'test-other',
            'test-clean',
            'dev-other',
        'dev-clean']
    my_files = []
    for i in paths:
        my_files = my_files + sorted(glob.glob(os.path.join(args.libri_path, i) + '*/**/*trans.txt',recursive=True))
    print(f"Number of trans.txt files: {len(my_files)}")

    df = pd.DataFrame(columns = ["filename","text"])
    filename = []
    text = []
    for i in range(len(my_files)):
        df_temp = pd.read_csv(my_files[i],header=None, usecols = [0], sep= " ",names=["filename"])
        filename = filename + list(df_temp['filename'])
        with open(my_files[i],'r') as file:
            lines = file.readlines()
            for j,line  in enumerate(lines):
                line = line.replace("'", "") ## normalization
                lines[j] = line.split(" ", 1)[1]
        text = text + lines

    df['filename'] = filename
    df['text'] = text
    df.to_csv("../data/librispeech.csv",index=False)

    with open("../data/libri_text_lowercase_normalized.txt","w") as file:
        for line in text:
            file.write(line.lower())

    if args.flag:
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    else:
        tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(vocab_size=args.vocab_size,special_tokens=["[UNK]"])
    tokenizer.pre_tokenizer = Whitespace()

    files = [f"../data/libri_text_lowercase_normalized.txt"]

    # train and save the tokenizer:

    tokenizer.train(files, trainer)
    tokenizer.save("../data/tokenizer-bpe"+str(args.vocab_size)+".json")

    # test the tokenizer
    tokenizer = Tokenizer.from_file("../data/tokenizer-bpe"+str(args.vocab_size)+".json")
    print("encoding 'greed is good'")
    output = tokenizer.encode("greed is good")
    print("printing the tokens")
    print(output.tokens)
    df_vocab = pd.DataFrame()
    df_vocab["vocab"] = tokenizer.get_vocab()
    df_vocab.to_csv("../data/vocab-bpe"+str(args.vocab_size)+".csv",index=False)


    ############# prepare the dataset ###############

    # ## utility functions to get tokens
    def get_tokens(x):
        if type(x) != str:
            print("problem in data", x)
        return tokenizer.encode(x.lower()).tokens
    # Load librispeech force aligned data
    df = pd.read_csv(args.libri_metadata, header = None, usecols = [0,1,2,3,4], sep = " ", names=["filename",
                    "flag", "start", "duration", "word"])
    
    print("shape of the dataset when intially loaded:", df.shape)
    df.dropna(inplace=True)
    print("shape of the dataframe when dropped na values loaded:",df.shape)

    # remove the word having ' from the force aligned metadata
    df = df[df["word"].str.contains("'") == False]

    df['tokenized'] = df['word'].apply(get_tokens)
    df['count_tokens'] = df['tokenized'].apply(len)
    if args.num_subwords== -1:
        df_subset = df[df['duration']>=args.min_duration]
    else:
        df_subset = df[(df['count_tokens']==args.num_subwords) & (df['duration']>=args.min_duration)]
    print(f'head of the dataset after {args.min_duration} seconds and  {args.num_subwords} tokens cirteria:',df_subset.head(5))
    print(f"shape of the dataframe after {args.min_duration} second and {args.num_subwords} tokens cirteria:",df_subset.shape)
    print("total unique words in the dataset:", len(df_subset["word"].unique()))

    # step1 completed : filtering
    print("saving the filtered data....")
    df_subset = df_subset.groupby("word").filter(lambda x: len(x) > 25)
    df_subset.to_csv("../data/dataset_prepared_" + str(args.num_subwords) + '_' + str(args.vocab_size) + '_' + str(args.num_sampling) + '_' + str(args.num_deletion) + ".csv",index=False)

    # step2 is here: sampling
    df_uniform = sampling(df_subset,args.num_sampling,args.num_deletion)

    # step3 is here: update the data to add path of corresponding librispeech file:

    filename_pd = pd.DataFrame()
    filename_pd["path"] = glob.glob(args.libri_path +'*/**/*.flac',recursive=True)
    print("correctly loaded all .flact files as the length of the loaded files is ",len(filename_pd))

    df_uniform['sp_id_and_ch_id'] = df_uniform['filename'].apply(get_spk_id_and_ch_id)

    print("progress")
    df_uniform['filename_path'] = df_uniform['sp_id_and_ch_id'].swifter.apply(match_string, df2=filename_pd)
    df_uniform.to_csv("../data/final_dataset_prepared_" + str(args.num_subwords) + '_' + str(args.vocab_size) + '_' + str(args.num_sampling) + '_' + str(args.num_deletion) + ".csv",index=False)

    print("head of the datast prepared")

    print(df_uniform.head())
    
    # check for null values:
    print("is null", df_uniform.isnull().sum())
    print("total datapoints in the dataset", len(df_uniform))

if __name__ == "__main__":
    main()
