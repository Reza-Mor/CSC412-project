import torch

import numpy as np
import argparse
import agents.finetune
import agents.ewc
import agents.iid
import agents.generative_replay
import random

from dataset_loader import get_dataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool
    
def main(args):
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    training_datasets, validation_datasets = get_dataset(args)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)

    if args.agent == 'finetune':
        agent = agents.finetune.Finetune(args, driving_policy)
    elif args.agent == 'iid':
        agent = agents.iid.IID(args, driving_policy)
    elif args.agent == "EWC":
        agent = agents.ewc.EWC(args, driving_policy)
    else:
        agent = agents.generative_replay.Gen_Replay(args)


    for i in range(args.runs):
        agent.train(training_datasets, validation_datasets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=30)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--agent", type=str, help="learning agent to use", default="finetune")
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--runs", type=int, help="number of runs", default=3)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset/val')
    parser.add_argument("--weights_out_folder", help="where to save the weights of the network",
                        default='weights')
    parser.add_argument("--noise_type", help="type of noise added to each task",
                        default='gaussian')
    
    args = parser.parse_args()

    main(args)

