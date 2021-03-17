import train_policy
import racer
import argparse
import os
import torch
import numpy as np
import scipy
import scipy.misc
import time
import imageio
from full_state_car_racing_env import FullStateCarRacingEnv
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from train_policy import train_discrete, test_discrete
import matplotlib.pyplot as plt

from dataset_loader import DrivingDataset
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool


def train_model(model, args, opt, data_transform,  n):

    training_dataset = DrivingDataset(root_dir= args.train_dir,
                                      categorical=True,
                                      classes=args.n_steering_classes,
                                      transform=data_transform)

    training_iterator = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    for epoch in range(args.n_epochs):

        # Train the driving policy
        train_discrete(model, training_iterator, [], opt, args)

    # Save the network weights 
    # torch.save(model.state_dict(), './weights/learner_{0}.weights'.format(n))
    torch.save(model.state_dict(), './weights/learner_{0}.weights'.format(n))


def agg_dataset(steering_network, args, run_id):
    
    env = FullStateCarRacingEnv()
    env.reset()
    
    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    total_cross_track_error = 0
    t = 0

    for i in range(args.timesteps):
        env.render()
        # execute policy
        state, expert_action, reward, done, _ = env.step(learner_action) 
        learner_action[0] = steering_network.eval(state, device=DEVICE)

        if done:
            break

        #get cross track error
        error_heading, error_dist, dest_min = env.get_cross_track_error(env.car, env.track)
        total_cross_track_error += error_dist
        t += 1

        # query expert action
        expert_steer = expert_action[0]  # [-1, 1]

        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        # update the dataset
        imageio.imwrite(os.path.join(args.train_dir, 'expert_%d_%d_%f.jpg' % (run_id, t, expert_steer)), state)

    env.close()
    return total_cross_track_error/t


'''
## Enter your DAgger code here
## Reuse functions in racer.py and train_policy.py
## Save the learner weights of the i-th DAgger iteration in ./weights/learner_i.weights where 
'''
def main(args):
    data_transform = transforms.Compose([ transforms.ToPILImage(),
                                          transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                                          transforms.RandomRotation(degrees=80),
                                          transforms.ToTensor()])
    
    validation_dataset = DrivingDataset(root_dir=args.validation_dir,
                                        categorical=True,
                                        classes=args.n_steering_classes,
                                        transform=data_transform)

    validation_iterator = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    avg_errors = []
    args.start_time = time.time()

    print('TRAINING LEARNER ON INITIAL DATASET')
    print(DEVICE)
    driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    opt = torch.optim.Adam(driving_policy.parameters(), lr=args.lr)
    train_model(driving_policy, args, opt, data_transform, 0)
    test_discrete(driving_policy, validation_iterator, opt, args)
    iters = int(args.dagger_iterations)

    for n in range(iters):
        print('GETTING EXPERT DEMONSTRATIONS')
        # execute policy, query expert actions, update the dataset, add cross track error
        avg_error = agg_dataset(driving_policy, args, n)  
        avg_errors.append(avg_error)

        # train the model on the new dataset, save the trained model, run the model on val set
        print('RETRAINING LEARNER ON AGGREGATED DATASET')
        driving_policy = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        train_model(driving_policy, args, opt, data_transform,  n)
        test_discrete(driving_policy, validation_iterator, opt, args)
        
    # save plot
    plt.plot(np.linspace(1, iters, num=iters), avg_errors, 'o', color='black')
    plt.savefig('dagger_iterations.png')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-3)
    parser.add_argument("--n_epochs", type=int, help="number of epochs", default=50)
    parser.add_argument("--batch_size", type=int, help="batch_size", default=256)
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    parser.add_argument("--train_dir", help="directory of training data", default='./dataset_1/train')
    parser.add_argument("--validation_dir", help="directory of validation data", default='./dataset_1/val')
    parser.add_argument("--weights_out_file", help="where to save the weights of the network e.g. ./weights/learner_0.weights", default='')
    parser.add_argument("--dagger_iterations", help="", default=10)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=36000)
    parser.add_argument("--weighted_loss", type=str2bool,
                        help="should you weight the labeled examples differently based on their frequency of occurence",
                        default=False)
    args = parser.parse_args()

    main(args)
    
