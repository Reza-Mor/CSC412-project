import gym
from full_state_car_racing_env import FullStateCarRacingEnv
import scipy
import scipy.misc
import numpy as np
import argparse
import torch
import os
from argparse import ArgumentParser

from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, str2bool

import matplotlib.pyplot as plt
import numpy as np



def run(steering_network, args):
    
    env = FullStateCarRacingEnv()
    env.reset()
    
    learner_action = np.array([0.0, 0.0, 0.0])
    expert_action = None
    time=[]
    steer_direction=[]
    steer_direction=[]
    
    for t in range(args.timesteps):
        env.render()
        
        state, expert_action, reward, done, _ = env.step(learner_action) 
        if done:
            break
        
        expert_steer = expert_action[0]  # [-1, 1]
        expert_gas = expert_action[1]    # [0, 1]
        expert_brake = expert_action[2]  # [0, 1]


        if args.expert_drives:
            learner_action[0] = expert_steer
            steer_direction.append(expert_steer)
        else:
            learner_action[0] = steering_network.test(state, device=DEVICE)
            steer_direction.append(learner_action[0])
            
        time.append(t)
        learner_action[1] = expert_gas
        learner_action[2] = expert_brake

        if args.save_expert_actions:
            scipy.misc.imsave(os.path.join(args.out_dir, 'expert_%d_%d_%f.jpg' % (args.run_id, t, expert_steer)), state)

    
    plt.plot(time,steer_direction)
    plt.xlabel("Time")
    plt.ylabel("Steering Direction")

    if args.expert_drives:
        filename = 'pid_expert_steering.png' 
    else:
        filename = args.learner_weights[:-4] + 'sterring_plot.png' 
    plt.savefig('steering_plots/'+ filename)
    
    env.close()
   


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", help="directory in which to save the expert's data", default='./dataset/train')
    parser.add_argument("--save_expert_actions", type=str2bool, help="save the images and expert actions in the training set",
                        default=False)
    
    parser.add_argument("--expert_drives", type=str2bool, help="should the expert steer the vehicle?", default=False)
    parser.add_argument("--run_id", type=int, help="Id for this particular data collection run (e.g. dagger iterations)", default=0)
    parser.add_argument("--timesteps", type=int, help="timesteps of simulation to run, up to one full loop of the track", default=100000)
    parser.add_argument("--learner_weights", type=str, help="filename from which to load learner weights for the steering network",
                        default='')
    parser.add_argument("--n_steering_classes", type=int, help="number of steering classes", default=20)
    
    args = parser.parse_args()
    DEVICE = torch.device('cpu')

    steering_network = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
    if args.learner_weights:
        steering_network.load_weights_from(args.learner_weights)

    run(steering_network, args)
