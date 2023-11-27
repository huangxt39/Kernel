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
import multiprocessing
from copy import deepcopy
import os

from models import modelClass, optClass
from utils import add_shared_args, convert_args_to_path

from data import sample_dataset, sample_dataset_known_function, train, predict, compute_norm

torch.set_printoptions(sci_mode=False)

def get_h_kernel(args, model, device):
    sampled_numbers = torch.arange(2**args.space_dim)

    binary_numbers = []
    for i in range(args.space_dim-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers).T

    with torch.no_grad():
        h = []
        handle = model.head.register_forward_hook(lambda module, args, output: h.append(args[0]))
        model(binary_numbers.to(device))
        handle.remove()

        h_kernel = torch.mm(h[0], h[0].T)
    
    return h_kernel


def make_data_point(args, gpu_idx=None):
    train_input, train_label, test_input = sample_dataset(args.train_num, args.test_num, args.space_dim)
    if gpu_idx is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = f'cuda:{gpu_idx}'

    pred = []
    for i in range(args.avg_num):
        model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
        model, final_loss = train(train_input, train_label, model, optClass[args.optimizer], device, args)
        pred.append(predict(test_input, model, device))
    pred = torch.vstack(pred).mean(dim=0)

    h_kernel = get_h_kernel(args, model, device)

    return train_input, train_label, test_input, pred.to('cpu'), final_loss, h_kernel.to('cpu')

def make_data_point_known_function(args, gpu_idx=None):
    while True:
        train_input, train_label, test_input, test_label = sample_dataset_known_function(args.train_num, args.test_num, args.space_dim, args.max_degree)
        model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
        if gpu_idx is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            device = f'cuda:{gpu_idx}'

        model, _ = train(train_input, train_label, model, optClass[args.optimizer], device, args)
        pred = predict(test_input, model, device)

        diff = F.mse_loss(pred, test_label.to(device)).item()
        if diff > args.min_diff:
            if args.avg_num > 1:
                preds = [pred]
                for i in range(args.avg_num-1):
                    model = modelClass[args.model](args.space_dim, args.arch, args.shrink)
                    model, _ = train(train_input, train_label, model, optClass[args.optimizer], device, args)
                    preds.append(predict(test_input, model, device))
                pred = torch.vstack(preds).mean(dim=0) # pred is averaged pred

            break
        else:
            print("not a good training set, try again")

    h_kernel = get_h_kernel(args, model, device)

    return train_input, train_label, test_input, pred.to('cpu'), diff, h_kernel.to('cpu')

def make_data_point_child_process(args, data_num, gpu_idx, q):
    torch.manual_seed(os.getpid())  # otherwise torch generate same random numbers for all child processes
    np.random.seed(os.getpid())
    for i in range(data_num):
        if args.known_func:
            data_point = make_data_point_known_function(args, gpu_idx)
        else:
            data_point = make_data_point(args, gpu_idx)
        data_point = tuple(map(lambda x: x.numpy() if torch.is_tensor(x) else x, data_point))
        q.put(data_point)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()   
    parser = add_shared_args(parser)   
    parser.add_argument("--proc_num", type=str, help="4-2 means 4 gpus, 2 process per gpu") 

    args = parser.parse_args()
    assert args.arch is not None

    data_points = []
    sum_ = 0
    h_kernel = []

    if args.proc_num is None:
        for i in tqdm(range(args.dataset_num)):
            if args.toy_data:
                # data_points.append(make_toy_linear_data(args))
                pass
            elif args.known_func:
                data_points.append(make_data_point_known_function(args))
                sum_ += data_points[-1][-2]
                h_kernel.append(data_points[-1][-1])
            else:
                data_points.append(make_data_point(args))
                if data_points[-1][-2] > args.threshold:
                    sum_ += 1
                h_kernel.append(data_points[-1][-1])
    else:
        # to kill child processes 
        # ps aux | grep python | grep xhuang | grep -v "grep python" | awk '{print $2}' | xargs kill -9
        assert not args.toy_data
        gpu_num, proc_per_gpu = tuple(map(lambda x: int(x), args.proc_num.split("-")))
        assert gpu_num <= torch.cuda.device_count()
        proc_num = gpu_num * proc_per_gpu
        
        dataset_num_per_proc = int(args.dataset_num / proc_num)
        assert args.dataset_num == dataset_num_per_proc * proc_num

        q = multiprocessing.Queue()

        proc_list = []
        for i in range(proc_num):
            p = multiprocessing.Process(target=make_data_point_child_process, args=(args, dataset_num_per_proc, i//proc_per_gpu, q))
            proc_list.append(p)

        for p in proc_list:
            p.start()

        # k_sum = 0
        for i in tqdm(range(args.dataset_num)):
            data_point = tuple(map(lambda x: torch.tensor(x) if isinstance(x, np.ndarray) else x, q.get()))
            data_points.append(data_point)
            if args.known_func:
                sum_ += data_points[-1][-2]
                h_kernel.append(data_points[-1][-1])
            else:
                if data_points[-1][-2] > args.threshold:
                    sum_ += 1
                h_kernel.append(data_points[-1][-1])

            # k_sum = torch.cat(list(map(lambda x: x.unsqueeze(0), h_kernel)), dim=0).mean(dim=0)
            # # k_sum = h_kernel[-1]
            # k_sum = 0.95 * k_sum + 0.05 * h_kernel[-1]
            # k_sum = k_sum / torch.linalg.matrix_norm(k_sum)
            # print(k_sum[[2, 15, 37, 41, 50], [19, 20, 51, 35, 9]])

        for p in proc_list:
            p.join()

    print("num datasets: ", len(data_points))
    if args.toy_data:
        print("ignore this")
    elif args.known_func:
        print("mean difference (bias of the model)")
    else:
        print("unfit rate")
    print(sum_ / args.dataset_num)

    h_kernel = torch.cat(list(map(lambda x: x.unsqueeze(0), h_kernel)), dim=0).mean(dim=0)
    h_kernel /= torch.linalg.matrix_norm(h_kernel)
    print(h_kernel[0])

    data_path = convert_args_to_path(args)
    data_path = data_path[:-4] + "h_kernel" + data_path[-4:]
    with open(data_path, "wb") as f:
        pickle.dump(data_points, f)
    print("data dumped!")
