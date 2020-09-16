import math
import torch
from torch.optim.optimizer import Optimizer, required

class SARAH(Optimizer):

    def __init__(self, params, device='cuda', innerloop=20, innerlr=1e-3 ):

        defaults = dict(lr=lr)
        super(RAdam, self).__init__(params, defaults)
        self.innerloop = innerloop
        self.device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        self.count = 0
        self.inner_data = []

    def __setstate__(self, state):
        super(RAdam, self).__setstate__(state)

    def _evaluate(self, img, lbl):
        out = model()

    def step(self, loader, closure=None):

        loss = None

        if self.count == 0:
            self.grad_tally = copy.deepcopy(self.param_groups)

        for g1, g2 in zip(self.param_groups, self.grad_tally):

            for p, g in zip(g1['params'], g2['params']):
                if p.grad is None:
                    continue
                
                # Pre-Inner loop
                
                if self.count == loader_length-2:
                    p.add_(g/len(loader), alpha=-lr)
                
                # Full gradient

                else:
                if self.count == 0:
                    self.g = p.grad
                else:
                    self.g += p.grad


        # Full inner loop
        if self.count == len(loader) - 2:
            inner_count = 0
            inner_loop_length = random.randint(0, self.innerloop)
            for 

        return loss