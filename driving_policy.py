import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import datasets, transforms

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        try:
            nn.init.xavier_uniform_(m.weight.data)
            m.bias.data.fill_(0)
        except AttributeError:
            print("Skipping initialization of ", classname)

            
class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    

class DiscreteDrivingPolicy(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

        
        self.features = nn.Sequential(
            #24 conv, relu, 36 conv, relu, 48 conv, relu, 64 conv, relu, flatten
            #kernel/window of size 4, stride of size 2, and 1 pixel padding
            nn.Conv2d(3, 24, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(24, 36, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(36, 48, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(48, 64, 4, 2, 1),
            nn.ReLU(),
            Flatten(),
        )
        
        self.classifier = nn.Sequential(
            # fc 128, relu, fc 64, relu, fc n classes, relu
            nn.Linear(4096,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,self.n_classes),
            nn.ReLU(),
        )  
        
        self.apply(weights_init)

    def forward(self, x):
        f = self.features(x)
        logits = self.classifier(f)
        return logits
    
    def test(self, state, device):
        state = state.astype(np.float32)
        state = np.ascontiguousarray(np.transpose(state, (2, 0, 1)))
        state = torch.tensor(state).to(torch.device(device))
        state = state.unsqueeze(0)
        logits = self.forward(state)
        
        y_pred = logits.view(-1, self.n_classes) 
        y_probs_pred = F.softmax(y_pred, 1)

        _, steering_class = torch.max(y_probs_pred, dim=1)
        steering_class = steering_class.detach().cpu().numpy()
        
        steering_cmd = (steering_class / (self.n_classes - 1.0))*2.0 - 1.0
        
        return steering_cmd
    
    def load_weights_from(self, weights_filename):
        weights = torch.load(weights_filename)
        self.load_state_dict( {k:v for k,v in weights.items()}, strict=True)

    


