from torch.optim.optimizer import Optimizer, required
import torch
import pdb
import pickle
import math
import logging
import scipy
import scipy.stats
import scipy.stats.mstats

class SVRG(Optimizer):
    r"""Implements the standard SVRG method
    """

    def __init__(self, params, nbatches, lr=0.01):
        self.nbatches = nbatches
        self.batches_processed = 0
        self.epoch = 0


        self.recalibration_i = 0
        self.running_interp = 0.9
        self.denom_epsilon = 1e-7 # avoid divide by zero

        defaults = dict(lr=lr, momentum=0.9, weight_decay=0.0001)
        super(SVRG, self).__init__(params, defaults)
        
        self.gradient_variances = []
        self.vr_step_variances = []
        self.batch_indices = []
        self.iterate_distances = []

        self.inrun_iterate_distances = []
        self.inrun_grad_distances = []

        self.initialize()



    #def __setstate__(self, state):
    #    super(SVRG, self).__setstate__(state)


    def initialize(self):
        m = self.nbatches

        for group in self.param_groups:

            for p in group['params']:
                momentum = group['momentum']

                gsize = p.data.size()
                gtbl_size = torch.Size([m] + list(gsize))

                param_state = self.state[p]

                if 'gktbl' not in param_state:
                    param_state['gktbl'] = torch.zeros(gtbl_size)

                if 'tilde_x' not in param_state:
                    param_state['tilde_x'] = p.data.clone()
                    param_state['running_x'] = p.data.clone()

                if 'gavg' not in param_state:
                    param_state['gavg'] = p.data.clone().double().zero_()

                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.clone().zero_()


    def step(self, batch_id):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        #loss = closure()
        dist_sq_acum = 0.0
        grad_dist_sq_acum = 0.0

        #print("step loss: ", loss)

        for group in self.param_groups:
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            learning_rate = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                gk = p.grad.data

                param_state = self.state[p]

                gktbl = param_state['gktbl']
                gavg = param_state['gavg'].type_as(p.data)
                tilde_x = param_state['tilde_x']

                if momentum != 0:
                    buf = param_state['momentum_buffer']

                #########
                gi = gktbl[batch_id, :].cuda()

                vr_gradient = gk.clone().sub_(gi - gavg)

                # Some diagnostics
                iterate_diff = p.data - tilde_x
                #pdb.set_trace()
                dist_sq_acum += iterate_diff.norm()**2 #torch.dot(iterate_diff,iterate_diff)
                grad_diff = gi - gk
                grad_dist_sq_acum += grad_diff.norm()**2 #torch.dot(grad_diff,grad_diff)

                if weight_decay != 0:
                    vr_gradient.add_(weight_decay, p.data)

                if momentum != 0:
                    dampening = 0.0
                    vr_gradient = buf.mul_(momentum).add_(1 - dampening, vr_gradient)

                # Take step.
                p.data.add_(-learning_rate, vr_gradient)

                # Update running iterate mean:
                param_state['running_x'].mul_(self.running_interp).add_(1-self.running_interp, p.data)

        # track number of minibatches seen
        self.batches_processed += 1

        dist = math.sqrt(dist_sq_acum)
        grad_dist = math.sqrt(grad_dist_sq_acum)

        self.inrun_iterate_distances.append(dist)
        self.inrun_grad_distances.append(grad_dist)

        #return loss