import torch
from utils import DEVICE, accuracy, get_class_distribution
from torch.utils.data import DataLoader
from torch import nn
import numpy as np
import torch.nn.functional as F
from  draw_plots import  drawLearningCurve

class EWC():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.prev_params = {}
        self.loss_weights = torch.ones(args.n_steering_classes).to(DEVICE)
        self.weights = {n: p for n, p in self.model.named_parameters() if p.requires_grad}
        self.running_fisher = self.init_fisher()
        self.tmp_fisher = self.init_fisher()
        self.normalized_fisher = self.init_fisher()
        self.criterion = nn.CrossEntropyLoss()
        self.lamb = 100
        self.alpha = 0.9
        self.fisher_update_interval = 50

    def train(self, trainsets, valsets):
        class_dist = get_class_distribution(trainsets[0], self.args)
        self.loss_weights = torch.where(class_dist == 0, class_dist, 1 / class_dist).to(DEVICE)  # get the inverse frequency
        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss(weight=self.loss_weights)
        for k, trainset in enumerate(trainsets):
            print('Training Task ', k)
            training_iterator = DataLoader(trainset, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
            recode = dict()
            recode[k] = []
            for epoch in range(self.args.n_epochs):
                print('EPOCH ', epoch)
                # Train the driving policy
                self.model.train()
                loss_hist = []
                for i, batch in enumerate(training_iterator):
                    # Upload the data in each batch to the GPU (if applicable)
                    x = batch['image'].to(DEVICE)
                    t = batch['cmd'].to(DEVICE)

                    # update the running fisher
                    if (epoch * len(training_iterator) + i + 1) % self.fisher_update_interval == 0:
                        self.update_running_fisher()

                    out = self.model(x)
                    loss = self.total_loss(out, t)

                    # backward
                    opt.zero_grad()
                    loss.backward()

                    # accumulate the fisher of current batch
                    self.accum_fisher()
                    opt.step()
                    loss_hist.append(loss.detach().cpu().numpy())

                    PRINT_INTERVAL = int(len(training_iterator) / 3)
                    if (i + 1) % PRINT_INTERVAL == 0:
                        print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t'.format(
                            i, len(training_iterator),
                            i / len(training_iterator) * 100,
                            np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0)
                        ))
            # save params for current task
            for n, p in self.weights.items():
                self.prev_params[n] = p.clone().detach()

            # update normalized fisher of current task
            max_fisher = max([torch.max(m) for m in self.running_fisher.values()])
            min_fisher = min([torch.min(m) for m in self.running_fisher.values()])
            for n, p in self.running_fisher.items():
                self.normalized_fisher[n] = (p - min_fisher) / (max_fisher - min_fisher + 1e-32)

            # Evaluate the driving policy on the validation set
            self.test(valsets)
        drawLearningCurve(recode)

    def total_loss(self, inputs, targets):
        # cross entropy loss
        loss = self.criterion(inputs, targets)
        if len(self.prev_params) > 0:
            # add regularization loss
            reg_loss = 0
            for n, p in self.weights.items():
                reg_loss += (self.normalized_fisher[n] * (p - self.prev_params[n]) ** 2).sum()
            loss += self.lamb * reg_loss
        return loss

    def init_fisher(self):
        return {n: p.clone().detach().fill_(0) for n, p in self.model.named_parameters() if p.requires_grad}

    def update_running_fisher(self):
        for n, p in self.running_fisher.items():
            self.running_fisher[n] = (1. - self.alpha) * p \
                                     + 1. / self.fisher_update_interval * self.alpha * self.tmp_fisher[n]
        # reset the accumulated fisher
        self.tmp_fisher = self.init_fisher()

    def accum_fisher(self):
        for n, p in self.tmp_fisher.items():
            p += self.weights[n].grad ** 2

    def test(self, valsets):
        self.model.eval()
        with torch.no_grad():
            for k, valset in enumerate(valsets):
                validation_iterator = DataLoader(valset, batch_size=self.args.batch_size, shuffle=False, num_workers=0)
                acc_hist = []
                for i_batch, batch in enumerate(validation_iterator):
                    x = batch['image']
                    y = batch['cmd']
                    x = x.to(DEVICE)
                    y = y.to(DEVICE)
                    logits = self.model(x)
                    y_pred = F.softmax(logits, 1)
                    acc = accuracy(y_pred, y)
                    acc = acc.detach().cpu().numpy()
                    acc_hist.append(acc)
                avg_acc = np.asarray(acc_hist).mean()
                print('\tTask {} val acc: {}'.format(
                    k,
                    avg_acc,
                ))