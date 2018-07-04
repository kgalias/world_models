from __future__ import absolute_import, division, print_function

import os
import argparse
import datetime
from functools import partial

import gym
import numpy as np
import torch
import torch.multiprocessing as mp

from src import DATA_DIR, RESULTS_DIR
from src.controller import ControllerAgent
from src.es import CMAES
from src.rnn import MDNRNN
from src.utils import suppress_stdout, param_count
from src.vae import VAE


def evaluate(controller_params, env_name, vae, v_dim, action_dim, rnn, m_dim, n_rollouts):
    if env_name == 'CarRacing-v0':
        gym.logger.setLevel(30)  # For suppressing the env creation message. TODO: can remove suppress_stdout() now?
        env = gym.make(env_name)
    else:
        raise NotImplementedError('Environment not supported: ' + env_name)

    agent = ControllerAgent(vae, v_dim, action_dim, rnn, m_dim, controller_params)
    rewards = []
    for rollout_num in range(1, n_rollouts+1):
        ep_reward = 0
        with suppress_stdout():  # Suppress track generation message.
            obs = env.reset()
        env.env.viewer.window.dispatch_events()  # CarRacing-v0 is bugged and corrupts obs without this.
        done = False
        reward = 0
        if rnn is not None:  # TODO: fix clunky problems like this with supporting both obs+mem and just obs.
            mem = agent.rnn.init_hidden(1)
        else:
            mem = None
        while not done:
            action, mem = agent.act(obs, reward, done, mem)
            obs, reward, done, _ = env.step(action)
            ep_reward += reward
        rewards.append(ep_reward)
    rewards = np.array(rewards).mean(axis=-1)  # Average over episodes.

    return rewards


def main():
    parser = argparse.ArgumentParser(description='Evolutionary training of controller')
    parser.add_argument('--env_name', nargs='?', default='CarRacing-v0',
                        help='Environment to use (default=CarRacing-v0)')
    parser.add_argument('--n_rollouts', type=int, default=1,
                        help='How many rollouts to perform when evaluating (default=1)')
    parser.add_argument('--n_generations', type=int, default=300,
                        help='Number of generations to train (default=300)')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Dimension of latent space (default=32)')
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Length of sequences for learning (default=10)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Dimension of action space (default=3)')
    parser.add_argument('--rnn_hidden_dim', nargs='?', type=int, default=256,
                        help='Dimension of RNN hidden state (default=256)')
    parser.add_argument('--n_gaussians', type=int, default=5,
                        help='Number of gaussians for the Mixture Density Network (default=5)')
    parser.add_argument('--pop_size', type=int, default=64,
                        help='Population size for evolutionary search (default=64)')
    parser.add_argument('--n_workers', type=int, default=32,
                        help='Number of workers for parallel processing (default=32)')
    parser.add_argument('--vae_fname',
                        help='VAE model file name')
    parser.add_argument('--rnn_fname', nargs='?',
                        help='RNN model file name')
    parser.add_argument('--eval_interval', nargs='?', default=15, type=int,
                        help='After how many generation to evaluate best params (default=15)')
    args = parser.parse_args()

    device = torch.device('cpu')

    # Load the VAE model from file.
    vae = VAE(latent_dim=args.latent_dim)
    vae.load_state_dict(torch.load(os.path.join(DATA_DIR, 'vae', args.vae_fname),
                                   map_location={'cuda:0': 'cpu'}))  # Previously trained on GPU.
    vae.to(device)

    # TODO: add identity/None RNN for dealing with the below?
    if args.rnn_fname is not None:  # Use memory module.
        # Load the MDNRNN model from file.
        mdnrnn = MDNRNN(action_dim=args.action_dim,
                        hidden_dim=args.rnn_hidden_dim,
                        latent_dim=args.latent_dim,
                        n_gaussians=args.n_gaussians)
        mdnrnn.load_state_dict(torch.load(os.path.join(DATA_DIR, 'rnn', args.rnn_fname),
                                          map_location={'cuda:0': 'cpu'}))  # Previously trained on GPU.
        mdnrnn.to(device)
    else:  # TODO: hacky, but dunno how to have default value for dim and pass it later without too many ifs. Fix?
        args.rnn_hidden_dim = 0
        mdnrnn = None

    # Set up controller model.
    agent = ControllerAgent(vae=vae,
                            v_dim=args.latent_dim,
                            action_dim=args.action_dim,
                            rnn=mdnrnn,
                            m_dim=args.rnn_hidden_dim)

    # Set up evolutionary strategy optimizer.
    with suppress_stdout():  # Suppress evolutionary strategy optimizer creation message.
        es = CMAES(num_params=param_count(agent.controller),
                   sigma_init=0.1,  # initial standard deviation
                   popsize=args.pop_size)

    # Set up multiprocessing.
    pool = mp.Pool(processes=args.n_workers)

    # Create results folder.
    dir_name = datetime.datetime.today().isoformat() + '_' + str(args.rnn_hidden_dim)
    os.makedirs(os.path.join(RESULTS_DIR, 'controller', dir_name))

    # TODO: add antithetic?
    for i in range(1, args.n_generations+1):
        start_time = datetime.datetime.now()

        # Create a set of candidate specimens.
        specimens = es.ask()

        # Evaluate the fitness of candidate specimens.
        func = partial(evaluate,
                       env_name=args.env_name,
                       vae=vae,
                       rnn=mdnrnn,
                       v_dim=args.latent_dim,
                       action_dim=args.action_dim,
                       m_dim=args.rnn_hidden_dim,
                       n_rollouts=args.n_rollouts)
        fitness_list = np.array(pool.map(func, specimens))

        # Give list of fitness results back to ES.
        es.tell(fitness_list)

        # get best parameter, fitness from ES
        es_solution = es.result()
        duration = datetime.datetime.now() - start_time

        history = {'best_params': es_solution[0],  # Best historical parameters.
                   'best_fitness': es_solution[1],  # Best historical reward.
                   'curr_best_fitness': es_solution[2],  # Best fitness of current generation.
                   'mean_fitness': fitness_list.mean(),  # Mean fitness of current generation.
                   'std_fitness': fitness_list.std()  # Std of fitness of current generation.
                   }
        np.savez(os.path.join(RESULTS_DIR, 'controller', dir_name, str(i)),
                 **history)

        print('Gen: {0:}\t| Best fit of gen: {1:.2f}\t| Best fit historical: {2:.2f}\t|'
              ' Mean fit: {3:.2f}\t| Std of fit: {4:.2f}\t| Time: {5:}m {6:}s'
              .format(i, es_solution[2], es_solution[1], fitness_list.mean(), fitness_list.std(),
                      *divmod(int(duration.total_seconds()), 60)))

        if i % args.eval_interval == 0:
            start_time = datetime.datetime.now()
            eval_fitness_list = np.array(pool.map(func,
                                                  np.broadcast_to(es_solution[0],
                                                                  (args.n_workers,) + es_solution[0].shape)))
            duration = datetime.datetime.now() - start_time
            print('{0:}-worker average fit of best params after gen {1:}: {2:.2f}. Time: {3:}m {4:}s.'.format(
                  args.n_workers, i, eval_fitness_list.mean(), *divmod(int(duration.total_seconds()), 60)))

            np.savez(os.path.join(RESULTS_DIR, 'controller', dir_name, str(i) + '_eval'),
                     eval_fitness_list)
if __name__ == '__main__':
    main()
