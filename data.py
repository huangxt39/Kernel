import torch
import torch.nn as nn
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import math

from models import modelClass, optClass
from utils import add_shared_args, convert_args_to_path

def make_toy_linear_data(args):
    train_num, test_num, space_dim = args.train_num, args.test_num, args.space_dim
    sampled_numbers = torch.tensor(np.random.choice(2**space_dim, train_num+test_num, replace=False))

    binary_numbers = []
    for i in range(space_dim-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers).T

    train_input = binary_numbers[:train_num].contiguous()
    test_input = binary_numbers[train_num:].contiguous()

    true_w = torch.rand(space_dim, 1)
    train_label = torch.mm(train_input.float(), true_w).squeeze(-1)
    
    # pred = torch.mm(test_input.float(), true_w).squeeze(-1)

    # print(pred)

    minimum_norm_w = torch.mm(torch.linalg.pinv(train_input.float()), train_label.unsqueeze(-1))
    
    #
    # y = torch.mm(train_input.float(), minimum_norm_w).squeeze(-1)
    # print("perfect match", torch.abs(y - train_label).mean())
    # print("compare norm")
    # print("true w", torch.linalg.vector_norm(true_w))
    # print("mini w", torch.linalg.vector_norm(minimum_norm_w))
    #

    pred = torch.mm(test_input.float(), minimum_norm_w).squeeze(-1)
    # print(pred)

    # exit()

    return train_input, train_label, test_input, pred, 0.0

def sample_dataset(train_num, test_num, space_dim):
    sampled_numbers = torch.tensor(np.random.choice(2**space_dim, train_num+test_num, replace=False))

    binary_numbers = []
    for i in range(space_dim-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers).T

    train_input = binary_numbers[:train_num].contiguous()
    test_input = binary_numbers[train_num:].contiguous()

    # train_label = (torch.rand(train_num) > 0.5).float()
    # train_label = torch.randn(train_num)
    train_label = torch.rand(train_num) * 2 - 1

    return train_input, train_label, test_input


def train(train_input, train_label, model, optClass, device, args):
    # loss_func = nn.BCELoss()
    loss_func = nn.MSELoss()
    optimizer = optClass(model.parameters(), lr=args.model_lr, weight_decay=args.weight_decay)

    model.train()
    model = model.to(device)
    train_input, train_label = train_input.to(device), train_label.to(device)

    loss_list = []
    for i in range(args.num_epoch):
        x = model(train_input)
        loss = loss_func(x, train_label)

        loss_list.append(loss.item())
        if loss.item() < args.threshold:
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
    model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, final_loss = train(train_input, train_label, model, optClass[args.optimizer], device, args)
    pred = predict(test_input, model, device)

    num_param = 0
    param_sqaure_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            param_sqaure_sum += (p.data**2).sum().item()
            num_param += p.numel()
    param_norm = math.sqrt(param_sqaure_sum / num_param)

    return train_input, train_label, test_input, pred, final_loss, param_norm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()   
    parser = add_shared_args(parser)    
    args = parser.parse_args()
    assert args.arch is not None

    data_points = []
    not_fitted = 0
    norms = []
    for i in tqdm(range(args.dataset_num)):
        if args.toy_data:
            data_points.append(make_toy_linear_data(args))
        else:
            data_points.append(make_data_point(args))
            if data_points[-1][-2] > args.threshold:
                not_fitted += 1
            norms.append(data_points[-1][-1])

    print("mean norm")
    print(torch.tensor(norms).mean().item())
    print("unfit rate")
    print(not_fitted / args.dataset_num)

    data_path = convert_args_to_path(args)
    with open(data_path, "wb") as f:
        pickle.dump(data_points, f)

