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
        return pi, mu, sigma

    def sample(self, pi, mu, sigma):
        cat = Categorical(pi)
        pi_idxs = cat.sample().cpu().numpy()
        batchsize = pi.size(0)
        sample = torch.empty(batchsize, self.output_dim)
        for i, idx in enumerate(pi_idxs):
            sample[i] = torch.normal(mu[i][idx], sigma[i][idx])
        return sample


class MDNRNN(nn.Module):
    def __init__(self, action_dim, hidden_dim=256, latent_dim=32, n_gaussians=5):
        super(MDNRNN, self).__init__()
        # input, (h_0, c_0)
        # input of shape (seq_len, batch, input_size)
        # h_0 of shape (num_layers * num_directions, batch, hidden_size)
        # c_0 of shape (num_layers * num_directions, batch, hidden_size)
        self.hidden_dim = hidden_dim
        self.rnn = nn.LSTM(input_size=action_dim+latent_dim, hidden_size=hidden_dim)
        self.mdn = MDN(input_dim=hidden_dim, output_dim=latent_dim, n_gaussians=n_gaussians)

    def forward(self, action, state, hidden_state=None):
        # batch_size =
        # if hidden_state is None:
        #     # use new so that we do not need to know the tensor type explicitly.
        #     hidden_state = (Variable(inpt.data.new(1, batch_size, self.hidden_size)),
        #                     Variable(inpt.data.new(1, batch_size, self.hidden_size)))

        rnn_input = torch.cat((action, state), dim=-1)
        output, hidden_state = self.rnn(rnn_input, hidden_state)
        output = output.view(-1, self.hidden_dim)
        pi, mu, sigma = self.mdn(output)
        return pi, mu, sigma, hidden_state


def nll_gmm_loss(x, pi, mu, sigma):
    x = x.expand(mu.size())
    log_pi = pi.log()
    