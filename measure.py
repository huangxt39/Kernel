import torch
from tqdm import tqdm
import argparse

from models import modelClass, optClass
from utils import add_shared_args, convert_args_to_path
from data import sample_dataset, make_data_point_known_function, train, predict


parser = argparse.ArgumentParser()
parser = add_shared_args(parser)
parser.add_argument("--measure_num_dataset", type=int, default=50)
parser.add_argument("--measure_models_per_dataset", type=int, default=20)
args = parser.parse_args()
assert args.arch is not None

device = 'cuda' if torch.cuda.is_available() else 'cpu'
variance = []
for i in tqdm(range(args.measure_num_dataset)):

    if args.known_func:
        train_input, train_label, test_input, _, _, _ = make_data_point_known_function(args)
    else:
        train_input, train_label, test_input = sample_dataset(args.train_num, args.test_num, args.space_dim)
    predictions = []
    for j in range(args.measure_models_per_dataset):
        model = modelClass[args.model](args.space_dim, args.arch, args.shrink)

        model, _ = train(train_input, train_label, model, optClass[args.optimizer], device, args)
        pred = predict(test_input, model, device)

        predictions.append(pred)
        del model

    predictions = torch.vstack(predictions)
    variance.append(predictions.var(dim=0).mean().item())

variance = torch.tensor(variance).mean().item()

print(variance)


