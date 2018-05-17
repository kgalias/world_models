from __future__ import absolute_import, division, print_function

import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


# TODO: add temperature
class MDN(nn.Module):
    def __init__(self, input_dim, output_dim, n_gaussians):
        super(MDN, self).__init__()
        self.output_dim = output_dim
        self.n_gaussians = n_gaussians
        self.pi = nn.Linear(input_dim, n_gaussians)
        self.mu = nn.Linear(input_dim, output_dim*n_gaussians)
        self.sigma = nn.Linear(input_dim, output_dim*n_gaussians)

    def forward(self, x):
        pi = F.softmax(self.pi(x), dim=1)
        mu = self.mu(x)
        mu = mu.view(-1, self.n_gaussians, self.output_dim)
        sigma = torch.exp(self.sigma(x))
        sigma = sigma.view(-1, self.n_gaussians, self.output_dim)
        return pi, sigma, mu

    def sample(self, pi, mu, sigma):
        cat = Categorical(pi)
        pi_idxs = cat.sample().numpy()
        batchsize = pi.size(0)
        sample = torch.Tensor((batchsize, self.output_dim))
        for i, idx in enumerate(pi_idxs):
            sample[i] = torch.normal(mu[i][idx], sigma[i][idx])
        return sample


class MDNRNN(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=32, n_gaussians=5):
        super(MDNRNN, self).__init__(a)
        self.rnn = nn.LSTM(input_size=action_dim+latent_dim, hidden_dim=hidden_dim)
        self.mdn = MDN(input_dim=hidden_dim, output_dim=latent_dim, n_gaussians=n_gaussians)

    def forward(self, action, state, hidden_state):
        rnn_input = torch.cat((action, state), dim=-1)
        output, hidden_state = self.rnn(rnn_input, hidden_state)
        # reshape output ?
        pi, sigma, mu = self.mdn(output)
        return self.mdn.sample(pi, sigma, mu), hidden_state
