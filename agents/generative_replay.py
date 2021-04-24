import torch
import torchvision
import numpy as np
import pyfiles.lib as lib
from driving_policy import DiscreteDrivingPolicy
from torch.utils.data import DataLoader
import sys
sys.path.append("..")
from driving_policy import DiscreteDrivingPolicy
from utils import DEVICE, accuracy, get_class_distribution
from torch import nn
import time
import numpy as np
import torch.nn.functional as F
from  draw_plots import  drawLearningCurve

class Gen_Replay():
    def __init__(self, args):
        self.args = args
        self.model = DiscreteDrivingPolicy(n_classes=args.n_steering_classes).to(DEVICE)
        self.loss_weights = torch.ones(args.n_steering_classes).to(DEVICE)
        self.trainsets = self.args.trainsets
        self.batch_size = self.args.batch_size
        self.num_noise = self.batch_size
        self.cur_task_dataset = self.args.cur_task_dataset
        self.gen = self.args.gen
        self.disc = self.args.disc
        self.pre_solver = None #self.args.pre_solver
        self.pre_gen = self.args.pre_gen
        self.solver = self.model
        self.ratio = 0.5
        self.epochs = self.args.n_epochs
        self.task_number = self.args.task_number

    def train(self, trainsets, valsets):
        self.train_generator(trainsets, valsets)
        self.train_solver(trainsets, valsets)

    def train_solver(self, trainsets, valsets):

        class_dist = get_class_distribution(self.trainsets[0], self.args)
        self.loss_weights = torch.where(class_dist == 0, class_dist, 1 / class_dist).to(DEVICE)  # get the inverse frequency

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)

        # load model 
        weights_filename = '{}/{}_{}_{}.pth'.format('weights', 'gr', self.args.noise_type, self.task_number)
        self.solver.load_weights_from(self, weights_filename)
        
        print('Training Solver for Task ', self.task_number)
        training_iterator = DataLoader(self.trainset[self.task_number], batch_size=1, shuffle=True, num_workers=0)

        recode = dict()
        recode[k] = []
        for epoch in range(self.args.n_epochs):
            print('EPOCH ', epoch)
            # Train the driving policy
            self.model.train()
            loss_hist = []
            for i_batch, batch in enumerate(training_iterator):
                # Upload the data in each batch to the GPU (if applicable)
                x = batch['image']
                x = x.to(DEVICE)
                t = batch['cmd']
                # Zero the accumulated gradient in the optimizer
                opt.zero_grad()
                # Do one pass over the data accessed by the training iterator
                out = self.model(x)
                
                t = t.to(DEVICE)
                criterion = nn.CrossEntropyLoss(weight=self.loss_weights)
                
                # generate data, label pair from old generator and solver
                if self.pre_solver is not None:
                    noise = lib.sample_noise(self.batch_size, self.num_noise)
                    g_image = self.pre_gen(noise)
                    g_label = self.pre_solver(g_image)
                    g_output = self.solver(g_image)

                # Compute the cross_entropy loss with and without weights
                loss = self.ratio * criterion(out, t) +(1 - self.ratio)* criterion(g_output, g_label) 

                # Compute the derivatives of the loss w.r.t. network parameters
                loss.backward()

                # Take a step in the approximate gradient direction using the optimizer opt
                opt.step()
                loss = loss.detach().cpu().numpy()
                loss_hist.append(loss)

                PRINT_INTERVAL = int(len(training_iterator) / 3)
                if (i_batch + 1) % PRINT_INTERVAL == 0:
                    print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t'.format(
                        i_batch, len(training_iterator),
                        i_batch / len(training_iterator) * 100,
                        np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0)
                    ))

        self.pre_solver = self.solver
        self.solver.save_weights(self.args, self.task_number)
         

        # Evaluate the driving policy on the validation set
        self.test(valsets)
    drawLearningCurve(recode)


    def train_generator(self, trainsets, valsets):

        assert (self.ratio >=0 or self.ratio <= 1)

        gen, disc = self.gen, self.disc
        #load the models

        ld = 10
        optim_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0, 0.9))
        optim_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0, 0.9))

        for k, trainset in enumerate(self.trainsets):
            print('Training Generator for task ', k)
            training_iterator = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=0)
        
            # Generator Training
            for epoch in range(self.args.n_epochs):
                gen.train()
                disc.train()

                for i_batch, batch in enumerate(training_iterator):
                    x = batch['image']
                    num_data = x.shape[0]
                    x = x.to(DEVICE)
                    t = batch['cmd']
                    noise = sample_noise(num_data, self.num_noise)
                    noise.to(DEVICE)

                    if self.pre_gen is not None:
                        # load pre_gen =
                        with torch.no_grad():
                            # append generated image & label from previous scholar
                            x_g = pre_gen(lib.sample_noise(self.batch_size, self.num_noise))
                            '''
                            gimg_min = gen_image.min(dim=1, keepdim=True)[0].min(dim=2, keepdim=True)[0]
                            gen_image = ((gen_image - gimg_min) * 256)
                            '''
                            x = torch.cat((x, x_g))                    
                            perm = torch.randperm(x.shape[0])[:num_data]
                            x = x[perm]
                        
                    #x = x.unsqueeze(1)
                    
                    ### Discriminator train
                    optim_d.zero_grad()
                    x_g = gen(noise)

                    ## Regularization term
                    eps = torch.rand(1).item()
                    x_hat = (x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)).requires_grad_(True)

                    loss_xhat = disc(x_hat)
                    fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False)
                    if torch.cuda.is_available():
                        fake = fake.cuda()
                        
                    gradients = torch.autograd.grad(outputs = loss_xhat,
                                                    inputs = x_hat,
                                                    grad_outputs=fake,
                                                    create_graph = True,
                                                    retain_graph = True,
                                                    only_inputs = True)[0]
                    gradients = gradients.view(gradients.shape[0], -1)
                    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * ld

                    p_real = disc(x)
                    p_fake = disc(x_g.detach())

                    loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
                    loss_d.backward()
                    optim_d.step()
                    
                    ### Generator Training
                    optim_g.zero_grad()
                    p_fake = disc(x_g)

                    loss_g = -torch.mean(p_fake)
                    loss_g.backward()
                    optim_g.step()

                print("[Epoch %d/%d] [D loss: %f] [G loss: %f]" % (epoch+1, self.epochs, loss_d.item(), loss_g.item()))


        

