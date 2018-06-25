from __future__ import absolute_import, division, print_function

import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical


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

    # TODO: add temperature.
    def sample(self, pi, mu, sigma):
        cat = Categorical(pi)
        pi_idxs = cat.sample().cpu().numpy()
        batch_size = pi.size(0)
        sample = torch.empty(batch_size, self.output_dim)
        for i, idx in enumerate(pi_idxs):
            sample[i] = torch.normal(mu[i][idx], sigma[i][idx])
        return sample


class MDNRNN(nn.Module):
    def __init__(self, action_dim, hidden_dim, latent_dim, n_gaussians):
        super(MDNRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_gaussians = n_gaussians
        self.rnn = nn.LSTM(input_size=action_dim+latent_dim, hidden_size=hidden_dim, batch_first=True)
        self.mdn = MDN(input_dim=hidden_dim, output_dim=latent_dim, n_gaussians=n_gaussians)

    def forward(self, action, state, hidden_state=None):
        seq_len = action.size(1) + 1
        rnn_input = torch.cat((action, state), dim=-1)  # Concatenate action and state.
        # TODO: verify whether to use both parts of hidden_state.
        output, hidden_state = self.rnn(rnn_input, hidden_state)
        output = output.contiguous()
        output = output.view(-1, self.hidden_dim)
        pi, mu, sigma = self.mdn(output)
        # Reshape to (batch_size, seq_len, ..).
        pi = pi.view(-1, seq_len-1, self.n_gaussians)
        mu = mu.view(-1, seq_len-1, self.n_gaussians, self.latent_dim)
        sigma = sigma.view(-1, seq_len-1, self.n_gaussians, self.latent_dim)
        return pi, mu, sigma, hidden_state


def nll_gmm_loss(x, pi, mu, sigma, size_average=True):
    x = x.unsqueeze(2)  # For broadcasting on the pi dimension.
    output_dim = mu.size(-1)
    log_pi = pi.log()
    log_pdf = -1 / 2 * (sigma.prod(dim=-1).log() + (x - mu).pow(2).mul(sigma.reciprocal()).sum(dim=-1))
    log_pdf += -output_dim / 2 * torch.ones_like(log_pdf) * math.log(2 * math.pi)
    if size_average:
        ll = logsumexp(log_pi + log_pdf, dim=-1).mean()
    else:
        ll = logsumexp(log_pi + log_pdf, dim=-1).sum()
    return -1 * ll


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.
       from https://github.com/pytorch/pytorch/issues/2591#issuecomment-364474328

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs
