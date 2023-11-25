import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import math
import argparse
from tqdm import tqdm
import random

torch.set_printoptions(sci_mode=False)

from utils import add_shared_args, log_toy_estimate_perf, log_estimate_perf, convert_args_to_path

from kernel import kernelTrainingDataset
    
class kernelHolder2(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_space = 2**args.space_dim
        feature_len = input_space - 2

        self.feature_map = nn.Parameter(torch.randn(feature_len, input_space))    # columns are phi(x) 

    def forward(self, train_idx, train_label, test_idx, pred):
        phi_x = self.feature_map.T[torch.hstack([train_idx, test_idx])]

        temp = torch.inverse(torch.bmm(phi_x.transpose(1,2), phi_x))
        temp = torch.bmm(phi_x, temp)
        temp = torch.bmm(temp, phi_x.transpose(1,2))
        identity = torch.eye(phi_x.size(1), dtype=temp.dtype, device=temp.device).unsqueeze(0)
        
        b = torch.hstack([train_label, pred]).unsqueeze(-1)

        loss = torch.bmm(torch.bmm(b.transpose(1,2), identity - temp), b)
        return loss.mean()

    def get_kernel_matrix(self):
        with torch.no_grad():
            return torch.mm(self.feature_map.T, self.feature_map)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = add_shared_args(parser)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_step", type=int, default=20000)
    parser.add_argument("--kernel_lr", type=float, default=1e-3)
    args = parser.parse_args()
    assert args.arch is not None


    kernel_dataset = kernelTrainingDataset(args)
    print(len(kernel_dataset))

    b = []
    for i in range(len(kernel_dataset)):
        _, train_label, _, pred = kernel_dataset[i]
        b.append(torch.hstack([train_label, pred]))
    b = torch.vstack(b)
    U, S, Vh = torch.linalg.svd(b)
    print(S)
    exit()

    dataloader = DataLoader(kernel_dataset, batch_size=args.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = 'cpu' 

    kernel_holder = kernelHolder2(args).to(device)
    print(kernel_holder.get_kernel_matrix().size())

    optimizer = torch.optim.Adam(kernel_holder.parameters(), lr=args.kernel_lr)

    flag = True
    loss_list = []
    total_steps = 0
    while flag:
        for items in dataloader:
            train_idx, train_label, test_idx, model_pred = (item.to(device) for item in items)
            
            loss = kernel_holder(train_idx, train_label, test_idx, model_pred)

            loss_list.append(loss.item())
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1
            if total_steps % 50 == 0:
                print(sum(loss_list)/len(loss_list))
                loss_list = []
            if total_steps >= args.num_step:
                flag = False
                break
        
