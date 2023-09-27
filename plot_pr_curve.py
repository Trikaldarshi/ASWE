import torch
import numpy as np
import pandas as pd
import random
from utility_functions.model_cae import model_cae
from torch.utils.data import Dataset
from utility_functions.utils_function import collate_fn
from scipy.special import comb
from scipy.spatial.distance import pdist, cdist

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#h_drive = "/home/amitmeghanani/H-drive/"
h_drive = "/"

class awe_dataset_pre_computed(Dataset):
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
            word_features = torch.load(h_drive + SSL_feature_path,map_location=torch.device('cpu'))
            

        return torch.squeeze(word_features),torch.tensor(word_features.size()[1]), word_name, sp_ch_ut_id
    
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

if device == "cuda":
    num_workers = 1
    pin_memory = True
else:
    num_workers = 0
    pin_memory = False

print("using pre-computed features")

test_data1 = awe_dataset_pre_computed(
    feature_df=h_drive + "fastdata/acw21am/private/hubert_features/hubert_feature_metadata.csv",
    partition="test"
)

test_data2 = awe_dataset_pre_computed(
    feature_df=h_drive+"fastdata/acw21am/private/mfcc_features/mfcc_feature_metadata.csv",
    partition="test"
)

print(len(test_data1))
print(len(test_data2))
assert len(test_data1) == len(test_data2)


test_loader1 = torch.utils.data.DataLoader(
  test_data1,
  batch_size=32,
  shuffle=True,
  collate_fn=collate_fn,
  drop_last = False,
  num_workers=num_workers,
  pin_memory=pin_memory,
  worker_init_fn=seed_worker,
  generator=g
)
test_loader2 = torch.utils.data.DataLoader(
  test_data2,
  batch_size=32,
  shuffle=True,
  collate_fn=collate_fn,
  drop_last = False,
  num_workers=num_workers,
  pin_memory=pin_memory,
  worker_init_fn=seed_worker,
  generator=g
)

model1 = model_cae(768, 256, 128, "GRU", True,4, 0.2)
checkpoint1 = torch.load("/home/acw21am/emnlp-2023/checkpoints/model_hubert/hubert_01/HUBERT_BASE_128_BEST.pt",map_location=torch.device(device))
model1.load_state_dict(checkpoint1['model_state_dict'])
model1.eval()

model2 = model_cae(60, 256, 128, "GRU", True,4, 0.2)
checkpoint2 = torch.load("/home/acw21am/emnlp-2023/checkpoints/model_mfcc/mfcc_01/MFCC_128_BEST.pt",map_location=torch.device(device))
model2.load_state_dict(checkpoint2['model_state_dict'])
model2.eval()

def average_precision(data, labels, metric = "cosine", show_plot=False):
    """
    Calculate average precision and precision-recall breakeven, and return
    the average precision / precision-recall breakeven calculated
    using `same_dists` and `diff_dists`.
    -------------------------------------------------------------------
    returns average_precision, precision-recall break even : (float, float)
    """
    num_examples = len(labels)
    num_pairs = int(comb(num_examples, 2))


    # build up binary array of matching examples
    matches = np.zeros(num_pairs, dtype=bool)

    i = 0
    for n in range(num_examples):
        j = i + num_examples - n - 1
        matches[i:j] = (labels[n] == labels[n + 1:]).astype(np.int32)
        i = j

    num_same = np.sum(matches)

    # calculate pairwise distances and sort matches
    dists = pdist(data, metric=metric)
    matches = matches[np.argsort(dists)]


    # calculate precision, average precision, and recall
    precision = np.cumsum(matches) / np.arange(1, num_pairs + 1)
    average_precision = np.sum(precision * matches) / num_same
    recall = np.cumsum(matches) / num_same

    # multiple precisions can be at single recall point, take max
    for n in range(num_pairs - 2, -1, -1):
        precision[n] = max(precision[n], precision[n + 1])

    # calculate precision-recall breakeven
    prb_ix = np.argmin(np.abs(recall - precision))
    prb = (recall[prb_ix] + precision[prb_ix]) / 2.
    if show_plot:
        import matplotlib.pyplot as plt
        print("plot created")
        plt.plot(recall, precision)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig('12oct.pdf')
        

    return average_precision, prb, precision, recall

def cal_precision(model,loader,device,distance,show_plot=False):
    embeddings, words = [], []
    model = model.eval()
    with torch.no_grad():
        for i, (data,lens,word_name,_) in enumerate(loader):
            lens, perm_idx = lens.sort(0, descending=True)
            data = data[perm_idx]
            word_name = word_name[perm_idx]

            data, lens  = data.to(device), lens.to(device)


            _,emb = model.encoder(data, lens)
            embeddings.append(emb)
            words.append(word_name)
            print(i)
        words = np.concatenate(words)
        uwords = np.unique(words)
        word2id = {v: k for k, v in enumerate(uwords)}
        ids = [word2id[w] for w in words]

        embeddings, ids = torch.cat(embeddings,0).to(torch.float16), np.array(ids)
        avg_precision,_,p,r = average_precision(embeddings.cpu(),ids, distance,show_plot)
        return avg_precision,p,r

test_avg_precision,p_hubert,r_hubert = cal_precision(model1, test_loader1, device,"cosine",False)
print("average precision on test set:", test_avg_precision)
print(" We are done! Bye Bye. Have a nice day!")

test_avg_precision,p_mfcc,r_mfcc = cal_precision(model2, test_loader2, device,"cosine",False)
print("average precision on test set:", test_avg_precision)
print(" We are done! Bye Bye. Have a nice day!")

import matplotlib.pyplot as plt
# len(p_hubert)
# torch.save(p_hubert, "p_hubert.pt")
# torch.save(r_hubert, "r_hubert.pt")
# torch.save(p_mfcc, "p_mfcc.pt")
# torch.save(r_mfcc, "r_mfcc.pt")

# plt.plot(p_hubert, r_hubert, 'b',label="HuBERT-based CAE model")
# plt.plot(p_mfcc, r_mfcc, 'r', label="MFCC-based CAE model")
# plt.xlabel("Recall",fontweight='bold')
# plt.ylabel("Precision",fontweight='bold')
# plt.legend()
# plt.savefig('pr_curve.eps', format='eps')
# plt.show()

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10,8))

# Plot data and set labels
ax.plot(p_hubert, r_hubert, 'b', label="HuBERT-based CAE model")
ax.plot(p_mfcc, r_mfcc, 'r', label="MFCC-based CAE model")
ax.set_xlabel("Recall", fontweight='bold', fontsize=24)
ax.set_ylabel("Precision", fontweight='bold', fontsize=24)


# Set font size for axis values
ax.tick_params(axis='both', labelsize=24)

# Add legend
ax.legend(fontsize=24)

# Save plot
plt.savefig('pr_curve.eps', format='eps')