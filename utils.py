import argparse

def add_shared_args(parser):
    parser.add_argument("--train_num", type=int, default=5)
    parser.add_argument("--test_num", type=int, default=50)
    parser.add_argument("--space_dim", type=int, default=6)
    
    parser.add_argument("--model", type=str, default="transformer", choices=["transformer", "lstm"])
    parser.add_argument("--optimizer", type=str, default="adam")
    
    parser.add_argument("--dataset_num", type=int, default=1024)

    return parser