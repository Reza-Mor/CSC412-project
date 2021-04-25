import torch
import torchvision
import numpy as np
from driving_policy import DiscreteDrivingPolicy
from torch.utils.data import DataLoader
import sys
import torch.nn.functional as F
from utils import DEVICE, accuracy, get_class_distribution
from agents.Generator import Generator_Conv, Discriminator_Conv, Solver

def sample_noise(batch_size, N_noise):
    """
    Returns
    """
    return torch.randn(batch_size, N_noise).to(DEVICE)

def model_grad_switch(net, requires_grad):
    for params in net.parameters():
        params.requires_grad_(requires_grad)

def init_params(model):
    for p in model.parameters():
        if(p.dim() > 1):
            torch.nn.init.xavier_normal_(p)
        else:
            torch.nn.init.uniform_(p, 0.1, 0.2)


def solver_evaluate(gen, solver, valsets, batch_size):
    gen.eval()
    solver.eval()
    with torch.no_grad():
        for k, valset in enumerate(valsets):
            validation_iterator = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)
            acc_hist = []
            for i_batch, batch in enumerate(validation_iterator):
                x = batch['image']
                y = batch['cmd']
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                logits = solver(x)
                y_pred = F.softmax(logits, 1)
                acc = accuracy(y_pred, y)
                acc = acc.detach().cpu().numpy()
                acc_hist.append(acc)
            avg_acc = np.asarray(acc_hist).mean()
            print('\tTask {} val acc: {}'.format(
                k,
                avg_acc,
            ))


class Gen_Replay():
    def __init__(self, args):
        self.args = args

    def train(self, trainsets, valsets):
        class_dist = get_class_distribution(trainsets[0], self.args)
        self.loss_weights = torch.where(class_dist == 0, class_dist, 1 / class_dist).to(
            DEVICE)  # get the inverse frequency
        pre_gen = None
        pre_solver = None

        for k, trainset in enumerate(trainsets):

            ratio = 1 / (k + 1)
            print('Training Task ', k)
            training_iterator = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
            if k > 0:
                pre_gen = gen
                pre_solver = solver
                model_grad_switch(pre_gen, False)
                model_grad_switch(pre_solver, False)
            gen = Generator_Conv(input_node_size=self.args.batch_size, output_shape=(3, 28, 28)).to(DEVICE)
            disc = Discriminator_Conv(input_shape=(3, 28, 28)).to(DEVICE)
            solver = Solver(k + 1).to(DEVICE)

            init_params(gen)
            init_params(disc)
            init_params(solver)

            optim_g = torch.optim.Adam(gen.parameters(), lr=0.001, betas=(0, 0.9))
            optim_d = torch.optim.Adam(disc.parameters(), lr=0.001, betas=(0, 0.9))
            optim_s = torch.optim.Adam(solver.parameters(), lr=0.001)
            for epoch in range(self.args.n_epochs):
                print('EPOCH ', epoch)
                gen.train()
                disc.train()
                try:
                    for i_batch, batch in enumerate(training_iterator):
                        x = batch['image'].to(DEVICE)
                        num_data = x.shape[0]
                        noise = sample_noise(num_data, self.args.batch_size)


                        if pre_gen is not None:
                            with torch.no_grad():
                                # append generated image & label from previous scholar
                                x_g = pre_gen(sample_noise(self.args.batch_size, self.args.batch_size))
                                x = torch.cat((x, x_g))
                                perm = torch.randperm(x.shape[0])[:num_data]
                                x = x[perm]

                        ### Discriminator train
                        optim_d.zero_grad()
                        disc.zero_grad()
                        x_g = gen(noise)

                        ## Regularization term

                        eps = torch.rand(1).item()
                        x_hat = x.detach().clone() * eps + x_g.detach().clone() * (1 - eps)
                        x_hat.requires_grad = True

                        loss_xhat = disc(x_hat)
                        fake = torch.ones(loss_xhat.shape[0], 1).requires_grad_(False).to(DEVICE)

                        gradients = torch.autograd.grad(outputs=loss_xhat,
                                                        inputs=x_hat,
                                                        grad_outputs=fake,
                                                        create_graph=True,
                                                        retain_graph=True,
                                                        only_inputs=True)[0]
                        gradients = gradients.view(gradients.shape[0], -1)
                        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

                        p_real = disc(x)
                        p_fake = disc(x_g.detach())

                        loss_d = torch.mean(p_fake) - torch.mean(p_real) + gp
                        loss_d.backward()
                        optim_d.step()

                        ### Generator Training
                        if i_batch % 5 == 4:
                            gen.zero_grad()
                            optim_g.zero_grad()
                            p_fake = disc(x_g)

                            loss_g = -torch.mean(p_fake)
                            loss_g.backward()
                            optim_g.step()

                    for i_batch, batch in enumerate(training_iterator):
                        celoss = torch.nn.CrossEntropyLoss().to(DEVICE)
                        image = batch['image'].to(DEVICE)
                        label = batch['cmd'].to(DEVICE)

                        solver.zero_grad()
                        optim_s.zero_grad()

                        output = solver(image)
                        loss = celoss(output, label) * ratio
                        loss.backward()
                        optim_s.step()

                        if pre_solver is not None:
                            solver.zero_grad()
                            optim_s.zero_grad()

                            noise = sample_noise(self.args.batch_size, self.args.batch_size)
                            g_image = pre_gen(noise)
                            g_label = pre_solver(g_image).max(dim=1)[1]
                            g_output = solver(g_image)
                            loss = celoss(g_output, g_label) * (1 - ratio)

                            loss.backward()
                            optim_s.step()

                    if (epoch + 1) % 3 == 0:
                        solver_evaluate(gen, solver, valsets, self.args.batch_size)
                except Exception as e:
                    print(e)





