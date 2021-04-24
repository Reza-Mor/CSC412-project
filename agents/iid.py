from torch.utils.data import DataLoader
import torch
from utils import DEVICE, accuracy, get_class_distribution
from torch import nn
import time
import numpy as np
import torch.nn.functional as F
from  draw_plots import  drawLearningCurve

class IID():
    def __init__(self, args, model):
        self.args = args
        self.model = model
        self.loss_weights = torch.ones(args.n_steering_classes).to(DEVICE)

    def train(self, trainsets, valsets):
        class_dist = get_class_distribution(trainsets[0], self.args)
        self.loss_weights = torch.where(class_dist == 0, class_dist, 1 / class_dist).to(DEVICE)  # get the inverse frequency

        opt = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
        learnCurve = list()
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
                for i_batch, batch in enumerate(training_iterator):
                    # Upload the data in each batch to the GPU (if applicable)
                    x = batch['image']
                    x = x.to(DEVICE)
                    t = batch['cmd']
                    # Zero the accumulated gradient in the optimizer
                    opt.zero_grad()
                    # Do one pass over the data accessed by the training iterator
                    out = self.model(x)
                    # Compute the cross_entropy loss with and without weights
                    t = t.to(DEVICE)
                    criterion = nn.CrossEntropyLoss(weight=self.loss_weights)
                    loss = criterion(out, t)
                    # Compute the derivatives of the loss w.r.t. network parameters
                    loss.backward()
                    # Take a step in the approximate gradient direction using the optimizer opt
                    opt.step()
                    loss = loss.detach().cpu().numpy()
                    loss_hist.append(loss)
                    recode[k].append(loss.item())
                    '''
                    PRINT_INTERVAL = int(len(training_iterator) / 3)
                    if (i_batch + 1) % PRINT_INTERVAL == 0:
                        print('\tIter [{}/{} ({:.0f}%)]\tLoss: {}\t'.format(
                            i_batch, len(training_iterator),
                            i_batch / len(training_iterator) * 100,
                            np.asarray(loss_hist)[-PRINT_INTERVAL:].mean(0)
                        ))
                    '''
                # Evaluate the driving policy on the validation set
                if (epoch + 1) % 3 == 0:
                    self.test(valsets)
            self.model.save_weights(self.args, k)

        #drawLearningCurve(recode)

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