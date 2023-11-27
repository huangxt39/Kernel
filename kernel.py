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

class kernelTrainingDataset(Dataset):
    def __init__(self, args) -> None:
        
        data_path = convert_args_to_path(args)

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
        for item in data:
            if args.low_rank:
                train_input, train_label, test_input, pred_train, pred, final_loss, norm = item
                pred = torch.hstack([pred_train, pred])
            else:
                train_input, train_label, test_input, pred, final_loss, norm = item

            train_idx = (train_input * temp).sum(dim=1)
            test_idx = (test_input * temp).sum(dim=1)

            self.instances.append((train_idx, train_label, test_idx, pred))

        # self.instances = self.instances[:32]

    def __len__(self):
        return len(self.instances)
    
    def __getitem__(self, idx):
        return self.instances[idx]
    
class kernelHolder(nn.Module):
    def __init__(self, args):
        super().__init__()
        input_space = 2**args.space_dim
        self.feature_map = nn.Parameter(torch.randn(args.feature_len, input_space))    # columns are phi(x)    /(2**(space_dim/4))
        self.max_threshold = args.max_thr
        self.lambda_ = args.lambda_


    def forward(self, train_idx, train_label, test_idx):
        phi_x = self.feature_map.T[train_idx]
        temp = torch.bmm(phi_x, phi_x.transpose(1,2))
        kernel_m_train = temp + self.lambda_ * torch.eye(train_idx.size(1), dtype=temp.dtype, device=temp.device).unsqueeze(0)

        kernel_m_train_inv = torch.inverse(kernel_m_train)

        alpha = torch.bmm(kernel_m_train_inv, train_label.unsqueeze(-1)) # / math.sqrt(train_idx.size(1))

        kernel_m_test = torch.bmm(phi_x, self.feature_map.T[test_idx].transpose(1,2)) # assume all test idx not in train idx
        kernel_pred = torch.bmm(alpha.transpose(1,2), kernel_m_test) # / math.sqrt(train_idx.size(1))

        return kernel_pred.squeeze(1)
    
    def forward_low_rank(self, train_idx, train_label, test_idx):
        phi_x = self.feature_map.T[train_idx]
        kernel_m_train = torch.bmm(phi_x, phi_x.transpose(1,2))
        
        # kernel_m_train_inv = torch.inverse(kernel_m_train)  # may also work, may converge to something close to low rank but not low rank
        
        def check_grad(grad):
            assert grad.isnan().sum() == 0
            assert grad.isinf().sum() == 0

        kernel_m_train.register_hook(check_grad)
        kernel_m_train_inv = torch.linalg.pinv(kernel_m_train, hermitian=True)

        alpha = torch.bmm(kernel_m_train_inv, train_label.unsqueeze(-1)) # / math.sqrt(train_idx.size(1))

        train_test_idx = torch.hstack([train_idx, test_idx])
        kernel_m_train_test = torch.bmm(phi_x, self.feature_map.T[train_test_idx].transpose(1,2)) # assume all test idx not in train idx
        kernel_pred = torch.bmm(alpha.transpose(1,2), kernel_m_train_test) # / math.sqrt(train_idx.size(1))

        return kernel_pred.squeeze(1)   # prediction for both train and test
    
    def clip(self):
        self.feature_map.data.clamp_(min= -self.max_threshold, max=self.max_threshold)

    def get_kernel_matrix(self):
        with torch.no_grad():
            return (torch.mm(self.feature_map.T, self.feature_map) + 
                    self.lambda_ * torch.eye(self.feature_map.size(1), 
                                            dtype=self.feature_map.dtype, 
                                            device=self.feature_map.device))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser = add_shared_args(parser)
    parser.add_argument("--batch_size", type=int, default=64)      # 64
    parser.add_argument("--num_step", type=int, default=20000)  # 20000
    parser.add_argument("--max_thr", type=float, default=200)
    parser.add_argument("--lambda_", type=float, default=0.1)  # 0.1
    parser.add_argument("--kernel_lr", type=float, default=1e-3)
    parser.add_argument("--feature_len", type=int)
    args = parser.parse_args()
    assert args.arch is not None
    if args.feature_len is None:
        args.feature_len = 2**args.space_dim


    kernel_dataset = kernelTrainingDataset(args)
    print(len(kernel_dataset))
    dataloader = DataLoader(kernel_dataset, batch_size=args.batch_size, shuffle=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu' 

    kernel_holder = kernelHolder(args).to(device)
    print(kernel_holder.get_kernel_matrix().size())


    optimizer = torch.optim.Adam(kernel_holder.parameters(), lr=args.kernel_lr)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=20, threshold=1e-3, verbose=True)

    loss_func = nn.MSELoss()

    flag = True
    loss_list = []
    total_steps = 0
    while flag:
        for items in dataloader:
            train_idx, train_label, test_idx, model_pred = (item.to(device) for item in items)
            
            if args.low_rank:
                kernel_pred = kernel_holder.forward_low_rank(train_idx, train_label, test_idx)
            else:
                kernel_pred = kernel_holder(train_idx, train_label, test_idx)
            loss = loss_func(kernel_pred, model_pred)

            loss_list.append(loss.item())
                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_steps += 1
            if total_steps % 50 == 0:
                print(sum(loss_list)/len(loss_list))
                # scheduler.step(sum(loss_list)/len(loss_list))
                loss_list = []
            if total_steps >= args.num_step:
                flag = False
                break
            
    if args.toy_data:
        log_toy_estimate_perf(args, kernel_holder)
    else:
        log_estimate_perf(kernel_pred, model_pred)
    print(kernel_holder.get_kernel_matrix()[0])

    "===================== about h kernel ========================"

    # # compare with trained kernel
    # h_kernel = []
    # data_path = convert_args_to_path(args)

    # with open(data_path, "rb") as f:
    #     data = pickle.load(f)

    #     for item in data:
    #         h_kernel.append(item[-1])
    # h_kernel = torch.cat(list(map(lambda x: x.unsqueeze(0), h_kernel)), dim=0).mean(dim=0)
    
    # trained_kernel = kernel_holder.get_kernel_matrix()

    # h_kernel /= torch.linalg.matrix_norm(h_kernel)
    # trained_kernel /= torch.linalg.matrix_norm(trained_kernel)

    # print(h_kernel[0])
    # print(trained_kernel[0])

    # # use h kernel to match model prediction
    # loss_list = []
    # for i in tqdm(range(len(kernel_dataset))):
    #     train_idx, train_label, test_idx, model_pred = kernel_dataset[i]
        
    #     kernel_m_train = h_kernel[train_idx].T[train_idx]
    #     kernel_m_train_inv = torch.inverse(kernel_m_train)

    #     alpha = torch.mm(kernel_m_train_inv, train_label.unsqueeze(-1)) 

    #     kernel_m_test = h_kernel[test_idx].T[train_idx] # assume all test idx not in train idx
    #     kernel_pred = torch.mm(alpha.T, kernel_m_test) 

    #     loss_list.append(loss_func(kernel_pred, model_pred))
    # print(sum(loss_list) / len(loss_list))
