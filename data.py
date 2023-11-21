import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import math
import itertools
import random

from models import modelClass, optClass
from utils import add_shared_args, convert_args_to_path

def random_binary_no_repeat(power, num):
    sampled_numbers = torch.tensor(np.random.choice(2**power, num, replace=False))

    binary_numbers = []
    for i in range(power-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers).T

    return binary_numbers


def make_toy_linear_data(args):
    train_num, test_num, space_dim = args.train_num, args.test_num, args.space_dim

    binary_numbers = random_binary_no_repeat(space_dim, train_num+test_num)

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
    
    binary_numbers = random_binary_no_repeat(space_dim, train_num+test_num)

    train_input = binary_numbers[:train_num].contiguous()
    test_input = binary_numbers[train_num:].contiguous()

    # train_label = (torch.rand(train_num) > 0.5).float()
    # train_label = torch.randn(train_num)
    train_label = torch.rand(train_num) * 2 - 1

    return train_input, train_label, test_input

def sample_dataset_known_function(train_num, test_num, space_dim, max_degree, term_per_degree=3):
    # sample a function
    assert max_degree <= space_dim
    func_terms = []
    for i in range(1, max_degree+1):
        combs = list(itertools.combinations(range(space_dim), i))
        random.shuffle(combs)
        chosen_terms = combs[:term_per_degree]
        for t in chosen_terms:
            func_terms.append((random.uniform(-1.0, 1.0), t))
    
    # sample data points

    binary_numbers = random_binary_no_repeat(space_dim, train_num+test_num)

    # evaluate with function to get labels
    labels = torch.zeros(len(binary_numbers))
    one_minus_one_numbers = binary_numbers.float() * 2 - 1
    for coef, vars in func_terms:
        labels += (coef * one_minus_one_numbers[:, vars].prod(dim=1))

    train_input = binary_numbers[:train_num].contiguous()
    test_input = binary_numbers[train_num:].contiguous()
    
    train_label = labels[:train_num].contiguous()
    test_label = labels[train_num:].contiguous()

    return train_input, train_label, test_input, test_label




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

def compute_norm(model):
    num_param = 0
    param_sqaure_sum = 0
    for p in model.parameters():
        if p.requires_grad:
            param_sqaure_sum += (p.data**2).sum().item()
            num_param += p.numel()
    param_norm = math.sqrt(param_sqaure_sum / num_param)
    return param_norm

def make_data_point(args):
    train_input, train_label, test_input = sample_dataset(args.train_num, args.test_num, args.space_dim)
    model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model, final_loss = train(train_input, train_label, model, optClass[args.optimizer], device, args)
    pred = predict(test_input, model, device)

    param_norm = compute_norm(model)

    return train_input, train_label, test_input, pred, final_loss, param_norm

def make_data_point_known_function(args):
    while True:
        train_input, train_label, test_input, test_label = sample_dataset_known_function(args.train_num, args.test_num, args.space_dim, args.max_degree)
        model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        model, _ = train(train_input, train_label, model, optClass[args.optimizer], device, args)
        pred = predict(test_input, model, device)

        diff = F.mse_loss(pred, test_label)
        if diff > args.min_diff:
            break
        else:
            print("not a good training set, try again")

    param_norm = compute_norm(model)

    return train_input, train_label, test_input, pred, diff, param_norm

if __name__ == "__main__":

    parser = argparse.ArgumentParser()   
    parser = add_shared_args(parser)    
    args = parser.parse_args()
    assert args.arch is not None

    data_points = []
    sum_ = 0
    norms = []
    for i in tqdm(range(args.dataset_num)):
        if args.toy_data:
            data_points.append(make_toy_linear_data(args))
        elif args.known_func:
            data_points.append(make_data_point_known_function(args))
            sum_ += data_points[-1][-2]
            norms.append(data_points[-1][-1])
        else:
            data_points.append(make_data_point(args))
            if data_points[-1][-2] > args.threshold:
                sum_ += 1
            norms.append(data_points[-1][-1])

    print("mean norm")
    print(torch.tensor(norms).mean().item())
    if args.toy_data:
        print("ignore this")
    elif args.known_func:
        print("mean difference (bias of the model)")
    else:
        print("unfit rate")
    print(sum_ / args.dataset_num)

    data_path = convert_args_to_path(args)
    with open(data_path, "wb") as f:
        pickle.dump(data_points, f)

