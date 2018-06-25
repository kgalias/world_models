from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.nn import functional as F


class Controller(nn.Module):
    def __init__(self, v_dim, action_dim, m_dim=0):
        super(Controller, self).__init__()
        self.out = nn.Linear(v_dim + m_dim, action_dim)

    def forward(self, obs, mem):
        if mem:
            obs = torch.cat((obs, mem), dim=-1)

        out = F.tanh(self.out(obs))

        # Scale the last two values to [0, 1]. # TODO: is there a nicer way to do this?
        out[:, -2] = (out[:, -2] + 1) / 2
        out[:, -1] = (out[:, -1] + 1) / 2

        return out


def main():
    pass

if __name__ == '__main__':
    main()
