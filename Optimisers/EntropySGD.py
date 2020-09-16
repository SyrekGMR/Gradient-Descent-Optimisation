import math
import torch
from torch.optim import Optimizer


class EntropySGD(Optimizer):


    def __init__(self, params, model, lr=1e-3, inner_step=0.1, scope=1E-4, L=20, alpha=0.5):

        defaults = dict(lr=lr, inner_step=inner_step, scope=scope, L=L, alpha=alpha)
        super(EntropySGD, self).__init__(params, defaults)
        
        self.state['in_lr'] = inner_step
        self.state['l'] = 1

        self.state['model'] = model.state_dict()
        
        for key in model.state_dict().keys():
            if "running" in str(key) or "batches_tracked" in str(key):
                self.state['model'].pop(key)

        self.state['mu'] = self.state['model']
        self.update_req = False
        
        
    def comp(self):
        print(type(self.param_groups))
    
    @torch.no_grad()
    def step(self, model, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        
        if self.update_req:
            self.state['model'] = model.state_dict()
            self.state['mu'] = model.state_dict()
        
        
        if self.state['l'] % self.defaults['L'] == 0:
            for group, model_group, mu_group in zip(self.param_groups, self.state['model'], self.state['mu']):
                for p in group['params']:
                    p = self.state['model'][model_group].add(-(self.state['model'][model_group]), \
                                                             alpha=self.defaults['scope']*self.defaults['lr'])
            
            self.update_req = True
            self.L = 1
            del self.state['model']
            del self.state['mu']
            
                    
        else:
            for group in self.param_groups:
                for p, model_group, mu_group in zip(group['params'], self.state['model'], self.state['mu']):
                    # dx' update

                    p.add_(-(p.grad.add(-(self.state['model'][model_group].add(-p)))),alpha=self.defaults['inner_step'])

                    # mu update
                    self.state['mu'][mu_group] = (1 - self.defaults['alpha']) * self.state['mu'][mu_group]\
                    .add(p, alpha=self.defaults['alpha'])
            


        #return loss