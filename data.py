import torch
import torch.nn as nn
import argparse
import pickle
import numpy as np
from tqdm import tqdm

from models import modelClass, optClass
from utils import add_shared_args


def sample_dataset(train_num, test_num, space_dim):
    sampled_numbers = torch.tensor(np.random.choice(2**space_dim, train_num+test_num, replace=False))

    binary_numbers = []
    for i in range(space_dim-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers).T

    train_input = binary_numbers[:train_num].contiguous()
    test_input = binary_numbers[train_num:].contiguous()

    train_label = (torch.rand(train_num) > 0.5).float()

    return train_input, train_label, test_input


def train(train_input, train_label, model, num_epoch, optClass, lr, device):
    loss_func = nn.BCELoss()
    optimizer = optClass(model.parameters(), lr=lr)

    model.train()
    model = model.to(device)
    train_input, train_label = train_input.to(device), train_label.to(device)

    loss_list = []
    for i in range(num_epoch):
        x = model(train_input)
        loss = loss_func(x, train_label)

        loss_list.append(loss.item())
        if loss.item() < 1e-2:
            break

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    else:
        print(loss_list[-1])

    return model, loss_list[-1]

def predict(test_input, model, device):
    model.eval()
    model = model.to(device)
    test_input = test_input.to(device)

    with torch.no_grad():
        pred = model(test_input)

    return pred


def make_data_point(args):
    train_input, train_label, test_input = sample_dataset(args.train_num, args.test_num, args.space_dim)
    model = modelClass[args.model](args.space_dim)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, final_loss = train(train_input, train_label, model, args.num_epoch, optClass[args.optimizer], args.lr, device)
    pred = predict(test_input, model, device)

    return train_input, train_label, test_input, pred, final_loss

parser = argparse.ArgumentParser()   
parser = add_shared_args(parser)    
parser.add_argument("--num_epoch", type=int, default=200)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

data_points = []
not_fitted = 0
for i in tqdm(range(args.dataset_num)):
    data_points.append(make_data_point(args))
    if data_points[-1][-1] > 1e-2:
        not_fitted += 1


model = modelClass[args.model](args.space_dim)
param_num = sum([p.numel() for p in model.parameters() if p.requires_grad==True ])
print(param_num)
print(not_fitted / args.dataset_num)

data_path = f"./datasets/data_{args.model}_{args.optimizer}_param{param_num}_dim{args.space_dim}_train{args.train_num}_test{args.test_num}_size{args.dataset_num}.pkl"

with open(data_path, "wb") as f:
    pickle.dump(data_points, f)


# train 5
    # 25537
    # 0.55,     0.03

    # 150209
    # 0.185,    0.025


    # LSTM

    # 17025
    # 0.04

    # 100097
    # 0.0

# train 10
    # 25537
    # 0.86,      0.085

    # 150209
    # 0.595,     0.16


    # LSTM

    # 17025
    # 0.165

    # 100097
    # 0.115