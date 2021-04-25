
Here is the short descriptions for files.


**agents**:  this is the directory that contains the continual learning models: iid, ewc, finetune and generative_replay

**histograms**:  this is the directory that contains histograms of accuracies for our models performed on pre-existing image test sets.

**dataset.zip**: this contains the original images used for training and testing.

**steering_plots**: the directory that contains the steering direction vs time plots for the tests on the go 

**Weights**:  the directory that contains the tuned parameters obtained from training

**racer.py**: this is the file that runs the models on the environment. We load the learned weights by giving it the path to the saved weights (see --learner_weights). 
        For expert driver (PID controller), run:  `python racer.py --timesteps 50000 --expert_drives True`
        For each of the trained models in the weights directory, go to driving policy and change the noise/filter accordingly. Then run `racer.py --timesteps         50000 --learner_weights {the pth file}`. For example, to see the result for ewc_filter_0.pth, first apply filter_0 in driving policy, then run: `python racer.py --timesteps 50000 --learner_weights ewc_filter_0.pth`

**dataset_loader.py**:  it applies filters/gaussian noises to data images to generate different data sets.  

**driving_policy.py**: this is the CNN+Feed Forward model. It gets used in full_state_car_racing_env.py.

**car_racing**: a top-down racing environment. It is used in full_state_car_racing_env.py. 

**full_state_car_racing_env.py**: the environment for cars to drive autonomously. 

**main.py**: we use this to train the models.  Run the following command: `python3 main.py --n_epochs=50 --batch_size=256 --weights_out_file=./weights/learner_0_supervised_learning.weights --train_dir=./dataset/train/ --weighted_loss=False` ***Give --weights_out_file a separate argument each time so we have access to all the trained models we experiment with.

**pid.py**: this is the PID controller used in full_state_car_racing_env.py to get optimal expert steering direction.

**results.txt**: this is where we store the accuracies of model predictions on pre-existing image test sets.

**task0/1/2_filter/gaussian.png**: these are the sample images in datasets.

**utils.py**: helpers used to implement models.

**draw_plots**:  we use this to plot graphs.
