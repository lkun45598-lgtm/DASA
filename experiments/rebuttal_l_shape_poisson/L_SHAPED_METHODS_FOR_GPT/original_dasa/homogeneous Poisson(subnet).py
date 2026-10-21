#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# In[2]:


import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import time
from pyDOE import lhs

import matplotlib.pyplot as plt
import shutil
import scipy.io
import random
from math import pi
from scipy.interpolate import griddata

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

import warnings
warnings.filterwarnings("ignore")


# In[3]:


import argparse
class Options_poisson(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--no_cuda', action='store_true', default=False, help='disable CUDA or not')
        parser.add_argument('--dim_hidden', type=int, default=128, help='neurons in hidden layers')
        parser.add_argument('--hidden_layers', type=int, default=6, help='number of hidden layers')
        parser.add_argument('--lr', type=float, default=1e-3, help='initial learning rate')
        parser.add_argument('--epochs_Adam', type=int, default=10000, help='epochs for Adam optimizer')
        parser.add_argument('--epochs_LBFGS', type=int, default=1000, help='epochs for LBFGS optimizer')
        parser.add_argument('--newton_iter', type=int, default=100, help='newton_iter for LBFGS optimizer')
        parser.add_argument('--step_size', type=int, default=1000, help='step size in lr_scheduler for Adam optimizer')
        parser.add_argument('--gamma', type=float, default=0.7, help='gamma in lr_scheduler for Adam optimizer')
        
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        # arg.cuda = False
        # arg.device = torch.device('cpu')
        arg.cuda = not arg.no_cuda and torch.cuda.is_available()
        arg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        return arg

args = Options_poisson().parse()
print(args.hidden_layers)

def save_model(state, is_best=None, save_dir=None):
    last_model = os.path.join(save_dir, 'last_model.pth')
    torch.save(state, last_model)
    if is_best:
        best_model = os.path.join(save_dir, 'best_model.pth')
        shutil.copyfile(last_model, best_model)


# In[4]:


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)

def grad(outputs, inputs):
    """ compute the derivative of outputs associated with inputs

    Params
    ======
    outputs: (N, 1) tensor
    inputs: (N, D) tensor
    """
    return torch.autograd.grad(outputs, inputs,
                               grad_outputs=torch.ones_like(outputs),
                               create_graph=True)

def activation(name):
    if name in ['tanh', 'Tanh']:
        return nn.Tanh()
    elif name in ['relu', 'ReLU']:
        return nn.ReLU(inplace=True)
    elif name in ['leaky_relu', 'LeakyReLU']:
        return nn.LeakyReLU(inplace=True)
    elif name in ['sigmoid', 'Sigmoid']:
        return nn.Sigmoid()
    elif name in ['softplus', 'Softplus']:
        return nn.Softplus()
    else:
        raise ValueError(f'unknown activation function: {name}')


# ## Backbone architectre

# In[5]:


class Modified_MLP(nn.Module):
    """Modified MLP"""
    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers,
                 act_name='tanh', init_name='xavier_normal'):
        super().__init__()
        self.hidden_layers = hidden_layers

        encoder_U = nn.Sequential()
        encoder_U.add_module('fc', nn.Linear(dim_in, dim_hidden, bias=True))
        encoder_U.add_module('act', activation(act_name))
        self.encoder_U = encoder_U

        encoder_V = nn.Sequential()
        encoder_V.add_module('fc', nn.Linear(dim_in, dim_hidden, bias=True))
        encoder_V.add_module('act', activation(act_name))
        self.encoder_V = encoder_V

        model = nn.Sequential()
        model.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=True))
        model.add_module('act0', activation(act_name))
        for i in range(1, hidden_layers):
            model.add_module(f'fc{i}', nn.Linear(dim_hidden, dim_hidden, bias=True))
            model.add_module(f'act{i}', activation(act_name))
        model.add_module(f'fc{hidden_layers}', nn.Linear(dim_hidden, dim_out, bias=True))

        self.model = model
        if init_name is not None:
            self.init_weight(init_name)

        self.size = self.model_size()

    def init_weight(self, name):
        """parameters initialization"""

        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init = nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)
    
    def forward(self, x):
        U = self.encoder_U(x)
        V = self.encoder_V(x)
        for i in range(self.hidden_layers):
            x = self.model[2 * i](x)      # linear layer
            x = self.model[2 * i + 1](x)  # activation layer
            x = (1 - x) * U + x * V       # Fusion of features
        x = self.model[-1](x)             # the last linear layer
        return x

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params


# In[6]:


class DNN(nn.Module):
    """Deep Neural Network"""

    def __init__(self, dim_in, dim_out, dim_hidden, hidden_layers,
                 act_name='tanh', init_name=None):
        super().__init__()
        model = nn.Sequential()

        model.add_module('fc0', nn.Linear(dim_in, dim_hidden, bias=True))
        model.add_module('act0', activation(act_name))
        for i in range(1, hidden_layers):
            model.add_module(f'fc{i}', nn.Linear(dim_hidden, dim_hidden, bias=True))
            model.add_module(f'act{i}', activation(act_name))
        model.add_module(f'fc{hidden_layers}', nn.Linear(dim_hidden, dim_out, bias=True))

        self.model = model
        if init_name is not None:
            self.init_weight(init_name)

    def init_weight(self, name):
        if name == 'xavier_normal':
            nn_init = nn.init.xavier_normal_
        elif name == 'xavier_uniform':
            nn_init = nn.init.xavier_uniform_
        elif name == 'kaiming_normal':
            nn_init = nn.init.kaiming_normal_
        elif name == 'kaiming_uniform':
            nn_init = nn.init.kaiming_uniform_
        else:
            raise ValueError(f'unknown initialization function: {name}')

        for param in self.parameters():
            if len(param.shape) > 1:
                nn_init(param)

    def forward(self, x):
        return self.model(x)

    def forward_test(self, x):
        print(f"{'input':<20}{str(x.shape):<40}")
        for name, module in self.model._modules.items():
            x = module(x)
            print(f"{name:<20}{str(x.shape):<40}")
        return x

    def model_size(self):
        n_params = 0
        for param in self.parameters():
            n_params += param.numel()
        return n_params


# In[7]:


model = Modified_MLP(2, 1, dim_hidden=args.dim_hidden, hidden_layers=args.hidden_layers) # We use modified MLP here.
model.cuda()
args.model=model


# ## Data Generation

# In[9]:


class Trainset_poisson():
    def __init__(self, *args):
        self.args = args
        self.shape = (self.args[1], self.args[0])
        
    def __call__(self):
        return self.data()
    
    def data(self):
        Nsd = self.args[0]
        n_b = self.args[1]
        
        # interior
        lb = np.array([-1.0, -1.0])
        ub = np.array([1.0, 1.0])
        xy = (ub-lb)*lhs(2, Nsd)+lb
        xy = xy[~((xy[:,0]>=0) * (xy[:,1]>=0))]
        
        # boundary
        bound_1 = lhs(1, n_b)-1 # -1 -> 0
        bound_1 = bound_1.reshape(-1,1)
        
        bound_2 = lhs(1, n_b) # 0 -> 1
        bound_2 = bound_2.reshape(-1,1)
        
        xy_b_1 = np.hstack((bound_1, bound_1*0-1))
        xy_b_2 = np.hstack((bound_2, bound_2*0-1))
        xy_b_3 = np.hstack((bound_1*0-1, bound_1))
        xy_b_4 = np.hstack((bound_1*0-1, bound_2))
        xy_b_5 = np.hstack((bound_1, bound_1*0+1))
        xy_b_6 = np.hstack((bound_2, bound_1*0))
        xy_b_7 = np.hstack((bound_1*0+1, bound_1))
        xy_b_8 = np.hstack((bound_1*0, bound_2))        
        xy_b = np.vstack([xy_b_1, xy_b_2, xy_b_3, xy_b_4, xy_b_5, xy_b_6, xy_b_7, xy_b_8])
         
        xy = torch.from_numpy(xy).float().cuda()
        xy_b = torch.from_numpy(xy_b).float().cuda()

        return xy, xy_b


# In[10]:


trainset = Trainset_poisson(10000, 128)
args.trainset = trainset
xy, xy_b = trainset()


# ## Training

# In[11]:


class subnet(DNN):
    """
    Sub-network for predicting weights for residual points
    """
    def __init__(self, dim_in=2, dim_out=1, dim_hidden=10, hidden_layers=5,
                 act_name='sigmoid', init_name=None, out_range=30, out_min=30):
        super().__init__(dim_in, dim_out, dim_hidden, hidden_layers,
                         act_name=act_name, init_name=init_name)
        self.out_range = out_range
        self.out_min = out_min
        self.model.fc5.bias.requires_grad = False
    def minmaxscaler(self, data):
        mmin = torch.min(data,dim=0)[0]
        mmax = torch.max(data,dim=0)[0]
        data = (data - mmin)/(mmax-mmin)
        output = data/torch.sum(data) * len(data) + 1
        
        return output
    def forward(self, x):
        u = super().forward(x)
        u = self.minmaxscaler(u)
        return u


# In[12]:


class Trainer_poisson(object):
    def __init__(self, args):
        self.model = args.model
        self.lr = args.lr
        self.gamma = args.gamma

        self.newton_iter = args.newton_iter
        self.step_size = args.step_size
        
        self.model_name = self.model.__class__.__name__
        self.model_path = self._model_path()
        
        self.epochs_Adam = args.epochs_Adam
        self.epochs_LBFGS = args.epochs_LBFGS
        self.optimizer_Adam = optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.optimizer_LBFGS = optim.LBFGS(self.model.parameters(),
                                            lr=0.8,
                                            max_iter=self.newton_iter,
                                            tolerance_grad=1.e-5,
                                            tolerance_change=1.e-9)
        self.scheduler = lr_scheduler.ExponentialLR(self.optimizer_Adam, gamma=self.gamma, verbose=True)
        
        # subnet(subnet)
        self.subnet = args.subnet
        self.subnet.cuda()
        self.subnet.zero_grad()
        self.optimizer_Adam_subnet = optim.Adam(filter(lambda p: p.requires_grad==True, self.subnet.parameters()), lr=args.lr)     
        
        # data
        self.xy, self.xy_b = args.trainset()
 
        # Logger
        self.loss_log = []
        self.loss_b_log = []
        self.loss_r_log = []
        self.epoch_log = []
        
        self.loss_lbfgs_log = []
        self.loss_b_lbfgs_log = []
        self.loss_r_lbfgs_log = []
        self.epoch_lbfgs_log = []
        
        
    def _model_path(self):
        """Path to save the model"""
        if not os.path.exists('checkpoints'):
            os.mkdir('checkpoints')
        
        path = os.path.join('checkpoints', self.model_name)
        if not os.path.exists(path):
            os.mkdir(path)
        
        return path
        
        
    # governing equation
    def net_r(self, xy):
        xy.requires_grad_(True)
        
        u = self.model(xy)
        grad_u = grad(u, xy)[0]
        u_x = grad_u[:,[0]]
        u_y = grad_u[:,[1]]
        u_xx = grad(u_x, xy)[0][:,[0]]
        u_yy = grad(u_y, xy)[0][:,[1]]

        residual = -(u_xx + u_yy) - 1
        return residual

    # output of PINN
    def net_u(self, xy):
        u = self.model(xy)
        return u
    
    
    def loss(self, use_ad=False):
        self.r_pred = self.net_r(self.xy)
        self.g_pred = self.net_u(self.xy_b)
        
        if use_ad:
            self.r_weights = self.subnet(self.xy)
            loss_r = torch.mean((self.r_weights * self.r_pred)**2)
            loss_b = torch.mean(self.g_pred**2)
            loss = loss_r + 100*loss_b
            
        else:
            loss_r = torch.mean(self.r_pred**2)
            loss_b = torch.mean(self.g_pred**2)
            loss = loss_r + 100*loss_b            
        
        return loss, loss_r, loss_b
    
    
    def train_Adam(self, mode='0'):
        """
        mode='0'：Stage1: Train the backbone network.
        mode='1'：Stage2: Train the sub network.
        mode='2'：Stage3: Adversarial training for backbone network and sub network.
        mode='3'：Stage4: Finetune the backbone network.
        """
        if mode == '0':
            self.optimizer_Adam.zero_grad()

            loss_value, loss_r_value, loss_b_value = self.loss(use_ad=True)
            loss_value.backward()
            self.optimizer_Adam.step()

            return loss_value.item(), loss_r_value.item(), loss_b_value.item()
        
        elif mode == '1':
            self.optimizer_Adam_subnet.zero_grad()
            loss_value, loss_r_value, loss_b_value = self.loss(use_ad=True)
            loss_value.backward()

            for name, parms in self.subnet.named_parameters():
                if parms.requires_grad==True:
                    parms.grad *= -1

            self.optimizer_Adam_subnet.step()        

            return loss_value.item(), loss_r_value.item(), loss_b_value.item()

        elif mode == '2':
            self.optimizer_Adam.zero_grad()
            self.optimizer_Adam_subnet.zero_grad()

            loss_value, loss_r_value, loss_b_value = self.loss(use_ad=True)
            loss_value.backward()
            self.optimizer_Adam.step()

            for name, parms in self.subnet.named_parameters():
                if parms.requires_grad==True:
                    parms.grad *= -1

            self.optimizer_Adam_subnet.step()

            return loss_value.item(), loss_r_value.item(), loss_b_value.item()

        else:
            self.optimizer_Adam.zero_grad()

            loss_value, loss_r_value, loss_b_value = self.loss(use_ad=False)
            loss_value.backward()
            self.optimizer_Adam.step()

            return loss_value.item(), loss_r_value.item(), loss_b_value.item()
            
    
    def train_LBFGS(self):
        loss_value, loss_r_value, loss_b_value = self.loss()
        
        def closure():
            loss_value, loss_r_value, loss_b_value = self.loss()

            self.optimizer_LBFGS.zero_grad()
            loss_value.backward()

            return loss_value

        self.optimizer_LBFGS.step(closure)
        loss_value = closure()
        
        return loss_value.item(), loss_r_value.item(), loss_b_value.item()
    
    
    def validate(self, epoch):
        self.model.eval()
        loss_value, loss_r_value, loss_b_value = self.loss()
        
        infos = 'Valid   ' +                 f'Loss:{loss_value:.4e}  ' +                 f'Loss_r:{loss_r_value:.4e} '
        print(infos)
        self.model.train()
        return loss_value.item()
    
    
    def train(self):
        start = time.time()
        best_loss = 1.e10
        
        # Stage1: Train the backbone network
        for epoch in range(5000):
            loss_value, loss_r_value, loss_b_value = self.train_Adam(mode='3')

            self.loss_log.append(loss_value)
            self.loss_r_log.append(loss_r_value)
            self.loss_b_log.append(loss_b_value)
            self.epoch_log.append(epoch)
            
            if (epoch+1) % 100 == 0:
                running_time = time.time() - start
                start = time.time()
                
                print(f'Epoch #  {epoch+1}/{self.epochs_Adam + self.epochs_LBFGS}' + f'    time:{running_time:.2f}' + '\n' +                       f'loss:{loss_value:.2e}, loss_r:{loss_r_value:.2e}, loss_b:{loss_b_value:.2e},')  
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
                
            if (epoch+1) % 1000 == 0:                
                # data
                self.xy, self.xy_b = args.trainset()
                
                
        # Stage2: Train the sub network   
        for epoch in range(2000):
            loss_value, loss_r_value, loss_b_value = self.train_Adam(mode='1')
            
            self.loss_log.append(loss_value)
            self.loss_r_log.append(loss_r_value)
            self.loss_b_log.append(loss_b_value)
            self.epoch_log.append(epoch)
            
            if (epoch+1) % 100 == 0:
                running_time = time.time() - start
                start = time.time()
                
                print(f'Epoch #  {epoch+1}/{self.epochs_Adam + self.epochs_LBFGS}' + f'    time:{running_time:.2f}' + '\n' +                       f'loss:{loss_value:.2e}, loss_r:{loss_r_value:.2e}, loss_b:{loss_b_value:.2e},')  
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
                
            if (epoch+1) % 1000 == 0:                
                # data
                self.xy, self.xy_b = args.trainset()
                
                
        # Stage3: Adversarial training for backbone network and sub network.
        for epoch in range(self.epochs_Adam):
            loss_value, loss_r_value, loss_b_value = self.train_Adam(mode='2')
            
            self.loss_log.append(loss_value)
            self.loss_r_log.append(loss_r_value)
            self.loss_b_log.append(loss_b_value)
            self.epoch_log.append(epoch)
            
            if (epoch+1) % self.step_size == 0:
                self.scheduler.step()            
            
            if (epoch+1) % 100 == 0:
                running_time = time.time() - start
                start = time.time()
                
                print(f'Epoch #  {epoch+1}/{self.epochs_Adam + self.epochs_LBFGS}' + f'    time:{running_time:.2f}' + '\n' +                       f'loss:{loss_value:.2e}, loss_r:{loss_r_value:.2e}, loss_b:{loss_b_value:.2e},')  
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
                
            if (epoch+1) % 1000 == 0:                
                # data
                self.xy, self.xy_b = args.trainset()
                
           
        # Stage4: Finetune the backbone network.
        for epoch in range(self.epochs_Adam):
            loss_value, loss_r_value, loss_b_value = self.train_Adam(mode='3')

            self.loss_log.append(loss_value)
            self.loss_r_log.append(loss_r_value)
            self.loss_b_log.append(loss_b_value)
            self.epoch_log.append(epoch)
            
            if (epoch+1) % 100 == 0:
                running_time = time.time() - start
                start = time.time()
                
                print(f'Epoch #  {epoch+1}/{self.epochs_Adam + self.epochs_LBFGS}' + f'    time:{running_time:.2f}' + '\n' +                       f'loss:{loss_value:.2e}, loss_r:{loss_r_value:.2e}, loss_b:{loss_b_value:.2e},')  
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
                
            if (epoch+1) % 1000 == 0:                
                # data
                self.xy, self.xy_b = args.trainset()
        
        # L-BFGS for backbone network      
        for epoch in range(self.epochs_Adam, self.epochs_Adam + self.epochs_LBFGS):
            loss_value, loss_r_value, loss_b_value = self.train_LBFGS()
            
            self.loss_log.append(loss_value)
            self.loss_r_log.append(loss_r_value)
            self.loss_b_log.append(loss_b_value)
            self.epoch_log.append(epoch)
            
            if (epoch+1) % 100 == 0:
                running_time = time.time() - start
                start = time.time()
                
                print(f'Epoch #  {epoch+1}/{self.epochs_Adam + self.epochs_LBFGS}' + f'    time:{running_time:.2f}' + '\n' +                       f'loss:{loss_value:.2e}, loss_r:{loss_r_value:.2e}, loss_b:{loss_b_value:.2e},')  
                
                valid_loss = self.validate(epoch)
                is_best = valid_loss < best_loss
                state = {
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'best_loss': best_loss
                }
                save_model(state, is_best, save_dir=self.model_path)
                
            if (epoch+1) % 1000 == 0:                
                # data
                self.xy, self.xy_b = args.trainset()


# In[13]:


args.subnet = subnet(dim_in=2)
trainer = Trainer_poisson(args)
trainer.train()


# In[14]:


state_dict = torch.load(f'{trainer.model_path}/best_model.pth')
model.load_state_dict(state_dict['state_dict'])


# In[15]:


file = f'checkpoints/subnet,{args.epochs_Adam}Adam,{args.epochs_LBFGS}LBFGS'
if os.path.exists(file)==False:
    os.mkdir(file)


# In[16]:


np.save(f'{file}/loss_r_log.npy',np.array(trainer.loss_r_log))
np.save(f'{file}/loss_b_log.npy',np.array(trainer.loss_b_log))


# In[17]:


trainer.loss_r_log = np.load(f'{file}/loss_r_log.npy')
trainer.loss_b_log = np.load(f'{file}/loss_b_log.npy')


# In[18]:


import seaborn as sns
with sns.axes_style("darkgrid"):
    fig = plt.figure(figsize=(6, 5))
    plt.rcParams.update({'font.size':16})
    plt.plot(np.arange(len(trainer.loss_r_log)), trainer.loss_r_log, label='$ \mathcal{L}_r$')
    plt.plot(np.arange(len(trainer.loss_b_log)), trainer.loss_b_log, label='$ \mathcal{L}_b$')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f'{file}/loss.jpg', dpi=750, bbox_inches = 'tight')


# ## Prediction

# In[19]:


model.cpu();
trainer.subnet.cpu();


# In[20]:


# load the data
data = scipy.io.loadmat('Lpoission_more.mat')

x = data['x']
y = data['y']
u_star = data['z']

ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_position(('data', -1))
ax.spines['left'].set_position(('data', -1))
plt.scatter(x, y, c=u_star, s=1, marker='.', cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'Reference $u(x,y)$')
plt.tight_layout()


# In[21]:


xy = np.concatenate([x.reshape(-1,1), y.reshape(-1,1)], axis=1)
xy_torch = torch.from_numpy(xy).float()
u_pred = model(xy_torch)
u_pred = u_pred.detach().numpy()
error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)

print('Error u: %e' % (error_u))


# In[22]:


plt.rcParams.update({'font.size':18})

fig, ax = plt.subplots(3, figsize=(24, 5))

ax1 = plt.subplot(1, 4, 1)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)
ax1.spines['left'].set_visible(False)
ax1.spines['bottom'].set_position(('data', -1))
ax1.spines['left'].set_position(('data', -1))
plt.scatter(x, y, c=u_star, s=1, marker='.', cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'Reference $u(x,y)$')
plt.tight_layout()
ax1.set_aspect(1./ax1.get_data_ratio())

ax2 = plt.subplot(1, 4, 2)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['bottom'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_position(('data', -1))
ax2.spines['left'].set_position(('data', -1))
plt.scatter(x, y, c=u_pred, s=1, marker='.', cmap='jet')
plt.colorbar()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'Predicted $u(x,y)$')
plt.tight_layout()
ax2.set_aspect(1./ax2.get_data_ratio())

ax3 = plt.subplot(1, 4, 3)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['bottom'].set_visible(False)
ax3.spines['left'].set_visible(False)
ax3.spines['bottom'].set_position(('data', -1))
ax3.spines['left'].set_position(('data', -1))
plt.scatter(x, y, c=np.abs(u_star-u_pred), s=1, marker='.', cmap='jet')
cbar = plt.colorbar()
cbar.formatter.set_powerlimits((0,0))
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.title(r'Absolute error')
plt.tight_layout()
ax3.set_aspect(1./ax3.get_data_ratio())


with sns.axes_style("darkgrid"):
    ax4 = plt.subplot(1, 4, 4)
    plt.plot(np.arange(len(trainer.loss_r_log)), trainer.loss_r_log, label='$ \mathcal{L}_r$')
    plt.plot(np.arange(len(trainer.loss_b_log)), trainer.loss_b_log, label='$ \mathcal{L}_b$')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    ax4.set_aspect(1./ax4.get_data_ratio())


plt.savefig(f'{file}/results_Lshape_poisson.png', dpi=400, bbox_inches = 'tight')
plt.show()

