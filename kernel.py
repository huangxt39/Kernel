import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import math
import argparse
from tqdm import tqdm
import random

from models import modelClass
from utils import add_shared_args

class kernelTrainingDataset(Dataset):
    def __init__(self, args) -> None:
        super().__init__()

        model = modelClass[args.model](args.space_dim)
        param_num = sum([p.numel() for p in model.parameters() if p.requires_grad==True ])
        data_path = f"./datasets/data_{args.model}_{args.optimizer}_param{param_num}_dim{args.space_dim}_train{args.train_num}_test{args.test_num}_size{args.dataset_num}.pkl"

        print("loading", data_path)
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        """
        train_input(int) (train_num, space_dim)
        train_label(float) (train_num,)
        test_input(int) (test_num, space_dim)
        prediction(float) (test_num,)
        final_loss
        """
        
        self.instances = []
        # convert bits to integers
        temp = 2**torch.arange(args.space_dim-1, -1, -1).unsqueeze(0)
        for train_input, train_label, test_input, pred, final_loss in data:
            if final_loss < 1e-2:
                train_idx = (train_input * temp).sum(dim=1)
                test_idx = (test_input * temp).sum(dim=1)

                # train_num = train_idx.size(0)
                # test_idx = test_idx[:train_num].contiguous()
                # pred = pred[:train_num].contiguous()

                self.instances.append((train_idx, train_label, test_idx, pred))

        self.instances = self.instances[:32]

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
class kernelHolder(nn.Module):
    def __init__(self, space_dim, max_thr, lambda_):
        super().__init__()
        self.feature_map = nn.Parameter(torch.randn(2**space_dim, 2**space_dim))    # columns are phi(x)    /(2**(space_dim/4))
        self.max_threshold = max_thr
        self.lambda_ = lambda_


    def forward(self, train_idx, train_label, test_idx):
        phi_x = self.feature_map.T[train_idx]
        temp = torch.bmm(phi_x, phi_x.transpose(1,2))
        kernel_m_train = temp + self.lambda_ * torch.eye(train_idx.size(1), dtype=temp.dtype, device=temp.device).unsqueeze(0)

        kernel_m_train_inv = torch.inverse(kernel_m_train)

        alpha = torch.bmm(kernel_m_train_inv, train_label.unsqueeze(-1)) / math.sqrt(train_idx.size(1))

        kernel_m_test = torch.bmm(phi_x, self.feature_map.T[test_idx].transpose(1,2)) # assume all test idx not in train idx
        kernel_pred = torch.bmm(alpha.transpose(1,2), kernel_m_test) / math.sqrt(train_idx.size(1))

        return kernel_pred.squeeze(1)
    
    def clip(self):
        self.feature_map.data.clamp_(min= -self.max_threshold, max=self.max_threshold)

    def get_kernel_matrix(self):
        with torch.no_grad():
            return (torch.mm(self.feature_map.T, self.feature_map) + self.lambda_ * torch.eye(self.feature_map.size(1)))

parser = argparse.ArgumentParser()
parser = add_shared_args(parser)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_steps", type=int, default=2000)
parser.add_argument("--max_thr", type=float, default=200)
parser.add_argument("--lambda_", type=float, default=0.1)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

kernel_dataset = kernelTrainingDataset(args)
print(len(kernel_dataset))
dataloader = DataLoader(kernel_dataset, batch_size=args.batch_size, shuffle=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

kernel_holder = kernelHolder(args.space_dim, max_thr=args.max_thr, lambda_=args.lambda_).to(device)
print(kernel_holder.get_kernel_matrix().size())

optimizer = torch.optim.Adam(kernel_holder.parameters(), lr=args.lr)

loss_func = nn.MSELoss()

flag = True
loss_list = []
total_steps = 0
while flag:
    for items in dataloader:
        train_idx, train_label, test_idx, model_pred = (item.to(device) for item in items)
        
        kernel_pred = kernel_holder(train_idx, train_label, test_idx)
        loss = loss_func(kernel_pred, model_pred)

        loss_list.append(loss.item())
        if torch.rand(()) < 0.01:
            print(kernel_pred[0])
            print(model_pred[0])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_steps += 1
        if total_steps % 50 == 0:
            print(sum(loss_list)/len(loss_list))
            loss_list = []
        if total_steps >= args.num_steps:
            flag = False
            break
        
    
        
print(kernel_holder.get_kernel_matrix())

# print(kernel_holder.feature_map.data[:, 1])
# print(kernel_holder.feature_map.data[:, 5])
# print(kernel_holder.feature_map.data[:, 9])