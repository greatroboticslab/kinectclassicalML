from multiprocessing import cpu_count
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim.lr_scheduler import _LRScheduler

import math
from os.path import join
from kinect_learning import * #(joints_collection, load_data, SVM, Random_Forest, AdaBoost, Gaussian_NB, Knn, Neural_Network)
import time
## Build path to file.
DATA_DIR = 'data'
FILE_NAME = 'bending.csv'
FILE_PATH = join(DATA_DIR, FILE_NAME)

left_right_col = joints_collection('left-right')
sit_stand_col = joints_collection('sit-stand')
turning_col = joints_collection('turning')
bending_col = joints_collection('bending')
up_down_col = joints_collection('up-down')
all_col = joints_collection('all')

seed = 1
np.random.seed(seed)

COLLECTION = bending_col
print("Printing scores of small collection...")
print("Collection includes", COLLECTION)
print("Printing scores of small collection with noise data...")
noise = False
X, y = load_data_multiple_dimension(FILE_PATH, COLLECTION, noise)['positions'], load_data_multiple_dimension(FILE_PATH, COLLECTION, noise)['labels']


def accuracy(output, target):
    return (output.argmax(dim=1) == target).float().mean().item()


class CyclicLR(_LRScheduler):

    def __init__(self, optimizer, schedule, last_epoch=-1):
        assert callable(schedule)
        self.schedule = schedule
        super(CyclicLR,self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [self.schedule(self.last_epoch, lr) for lr in self.base_lrs]


def cosine(t_max, eta_min=0):
        def scheduler(epoch, base_lr):
            t = epoch % t_max
            return eta_min + (base_lr - eta_min) * (1 + np.cos(np.pi * t / t_max)) / 2

        return scheduler

n = 100
sched = cosine(n)
lrs = [sched(t, 1) for t in range(n * 4)]
#plt.plot(lrs)
#plt.show()

def create_datasets(X, y, test_size=0.4):
    X = np.asarray(X, dtype=np.float32)
    y =np.asarray(y,dtype=np.int16)

    divsion = X.shape[0]%10
    actual_length =  X.shape[0] - divsion
    X = X[0:actual_length,:]
    y = y[0:actual_length]

    #X = X[:, 0]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.4)
    X_train, X_valid = [torch.tensor(arr, dtype=torch.float32) for arr in (X_train, X_valid)]
    y_train, y_valid = [torch.tensor(arr, dtype=torch.long) for arr in (y_train, y_valid)]

    #X_train = X_train.unsqueeze(-1)#.unsqueeze(1)
    #X_valid = X_valid.unsqueeze(-1)#.unsqueeze(1)
    #y_train = y_train.unsqueeze(1)
    #y_valid = y_valid.unsqueeze(1)

    train_ds = TensorDataset(X_train, y_train)
    valid_ds = TensorDataset(X_valid, y_valid)

    return train_ds, valid_ds

def create_loaders(train_ds, valid_ds, bs=512, jobs=0):
    train_dl = DataLoader(train_ds, bs, shuffle=True, num_workers=jobs)
    valid_dl = DataLoader(valid_ds, bs, shuffle=False, num_workers=jobs)
    return train_dl, valid_dl


print('Preparing datasets')
trn_ds, val_ds = create_datasets(X, y)

bs = 200
print('Creating data loaders with batch size: {}'.format(bs))
trn_dl, val_dl = create_loaders(trn_ds, val_ds, bs, jobs=cpu_count())


input_dim = 3
hidden_dim = 256
layer_dim = 5
output_dim = 2
seq_dim = 128

lr = 0.0005
n_epochs = 5000
iterations_per_epoch = len(trn_dl)
best_acc = 0
patience, trials = 100000, 0


class LSTMClassifier(nn.Module):
    """Very simple implementation of LSTM-based time-series classifier."""
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super(LSTMClassifier,self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.rnn = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        #print(type(x), x.size())
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        #print(type(x), x.size())
        h0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.layer_dim, x.size(0), self.hidden_dim)
        #t.cuda()
        return [t for t in (h0, c0)]

model = LSTMClassifier(input_dim, hidden_dim, layer_dim, output_dim)
model = model#.cuda()
criterion = nn.CrossEntropyLoss()
opt = torch.optim.RMSprop(model.parameters(), lr=lr)
sched = CyclicLR(opt, cosine(t_max=iterations_per_epoch * 2, eta_min=lr/100))

dir = '/home/iotlab/research/timeseries/timeseries/'
checkpoint = torch.load(dir + 'best.pth')
model.load_state_dict(checkpoint)

correct, total = 0, 0

time_start = time.time()
for x_val, y_val in val_dl:
    # t.cuda()
    x_val, y_val = [t for t in (x_val, y_val)]
    out = model(x_val)
    preds = F.log_softmax(out, dim=1).argmax(dim=1)
    total += y_val.size(0)
    correct += (preds == y_val).sum().item()
time_end = time.time()
duration = time_end - time_start

print(correct, total)
acc = correct*1.0/total
print("%.5f" %acc, "duration is", duration)

'''
image = Image.open(Path('C:/Users/Aeryes/PycharmProjects/simplecnn/images/pretrain_classify/rose_classify.jpg'))

input = trans(image)

input = input.view(1, 3, 32,32)

output = model(input)

prediction = int(torch.max(output.data, 1)[1].numpy())
print(prediction)

if (prediction == 0):
    print ('daisy')
if (prediction == 1):
    print ('dandelion')
if (prediction == 2):
    print ('rose')
if (prediction == 3):
    print ('sunflower')
if (prediction == 4):
    print ('tulip')

'''
