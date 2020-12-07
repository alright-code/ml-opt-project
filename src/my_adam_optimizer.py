import numpy as np
import torch


class MyAdamOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0):
        defaults = {'lr': lr, 'beta1': beta1,
                    'beta2': beta2, 'eps': eps, 'weight_decay': 0}
        super(MyAdamOptimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr = group['lr']
            beta1 = group['beta1']
            beta2 = group['beta2']
            eps = group['eps']
            weight_decay = group['weight_decay']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, lr=weight_decay)
                param_state = self.state[p]

                # Fill in initial values
                if 'm' not in param_state:
                    param_state['m'] = torch.zeros(p.data.shape).cuda()   # 1st moment vector
                if 'v' not in param_state:
                    param_state['v'] = torch.zeros(p.data.shape).cuda()   # 2nd moment vector
                if 't' not in param_state:
                    param_state['t'] = 0                                  # timestep

                param_state['t'] += 1
                param_state['m'] = torch.add(torch.mul(beta1, param_state['m']),
                                             torch.mul(1 - beta1, d_p))
                param_state['v'] = torch.add(torch.mul(beta2, param_state['v']),
                                             torch.mul(1 - beta2, d_p**2))

                bias_correct_m = param_state['m']/(1 - beta1**param_state['t'])
                bias_correct_v = param_state['v']/(1 - beta2**param_state['t'])
                mul_term = bias_correct_m/(torch.sqrt(bias_correct_v) + eps)

                p.data = p.data - torch.mul(lr, mul_term)
