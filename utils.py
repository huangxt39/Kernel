import argparse
import torch

def add_shared_args(parser):
    parser.add_argument("--train_num", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=50)
    parser.add_argument("--space_dim", type=int, default=6)
    
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm", "dnn"])
    parser.add_argument("--optimizer", type=str, default="adam", choices=["sgd", "adam", "adamw"])
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--threshold", type=float, default=1e-4)

    parser.add_argument("--num_epoch", type=int, default=200)
    parser.add_argument("--model_lr", type=float, default=1e-3)
    parser.add_argument("--shrink", type=float, default=1.0)
    
    parser.add_argument("--arch", type=str) #256-6-3  hdim, nhead, nlayer

    parser.add_argument("--dataset_num", type=int, default=1024)

    parser.add_argument("--toy_data", action="store_true")

    parser.add_argument("--known_func", action="store_true")
    parser.add_argument("--max_degree", type=int)
    parser.add_argument("--min_diff", type=float, default=0.1)

    return parser

def convert_args_to_path(args):
    if args.toy_data:
        data_path = "./datasets/data_toy.pkl"
    elif args.known_func:
        data_path = f"./datasets/data_func_degree{args.max_degree}_mdiff{args.min_diff}_{args.model}-{args.arch}_{args.optimizer}_wd{args.weight_decay}_thr{args.threshold}_{args.model_lr}*{args.num_epoch}_shr{args.shrink}_dim{args.space_dim}_train{args.train_num}_test{args.test_num}_size{args.dataset_num}.pkl"
    else:
        data_path = f"./datasets/data_{args.model}-{args.arch}_{args.optimizer}_wd{args.weight_decay}_thr{args.threshold}_{args.model_lr}*{args.num_epoch}_shr{args.shrink}_dim{args.space_dim}_train{args.train_num}_test{args.test_num}_size{args.dataset_num}.pkl"
    return data_path

def log_toy_estimate_perf(args, kernel_holder):
    sampled_numbers = torch.arange(2**args.space_dim)
    binary_numbers = []
    for i in range(args.space_dim-1, -1, -1):
        binary_numbers.append( (sampled_numbers >= 2**i).long() )
        sampled_numbers = torch.where(sampled_numbers >= 2**i, sampled_numbers-2**i, sampled_numbers)

    binary_numbers = torch.vstack(binary_numbers)
    true_K = torch.mm(binary_numbers.T, binary_numbers).float()
    print("==============")
    estimate_K = kernel_holder.get_kernel_matrix()
    estimate_K = estimate_K/torch.linalg.norm(estimate_K)*torch.linalg.norm(true_K)
    print(torch.abs(true_K - estimate_K).mean())
    print("==============")
    print("diagonal")
    print(estimate_K.diag())
    print("compare first three rows")
    print(true_K[:3])
    print(estimate_K[:3])

    print(kernel_holder.feature_map[:, :4].data)


def log_estimate_perf(kernel_pred, model_pred):
        print(kernel_pred[0])
        print(model_pred[0])

        print(kernel_pred[1])
        print(model_pred[1])