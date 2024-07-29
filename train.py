from ur.env import MultiValleyMountainCarEnv
from ur import utils
import torch
from ur.net import TripleNet
import numpy as np
import argparse


def train(config):

    # Parameters Initialization
    niters = config["niters"]
    batch_size = config["batch_size"]
    ent_coef = config["ent_coef"]
    logg_iters = config["logg_iters"]
    path_to_save = config["path_to_save"]

    # User notification
    print("Training started")
    print()
    print("Number of training steps: ",niters)
    print("Batch size: ", batch_size)
    print("Entropy coefficient: ", ent_coef)
    print("Logging period: ", logg_iters)
    print("Model will be saved as: ", path_to_save)
    print()


    # Initialization of model, environment and device (cpu or gpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ",device.type)
    print()
    env = MultiValleyMountainCarEnv(force=0.001)
    net = TripleNet(2, 2, 3, lr_v=1e-5, lr_p=1e-5, lr_pi=1e-5, weight_decay_v=1e-4, weight_decay_p=5e-4,
                    weight_decay_pi=5e-6, reflecting_representation=env.reflecting_representation).to(device)
    net.train()


    # Constants initialization
    N = 201
    dx = 2 / (N - 1)
    dv = 2 * 0.07 / (N - 1)
    X, V = np.mgrid[-1:1 + dx:dx, -0.07:0.07 + dv:dv]
    shift, scale = torch.tensor([[0.99, 0.07]]).to(device), torch.tensor([[1.98, 0.14]]).to(device)
    log_gamma = np.log(0.95)

    # Network update
    for i in range(niters):
        net.train()
        state = torch.rand(batch_size, 2, device=device) * scale - shift # Sample a set of state
        action, value, value_prime, log_prob, log_prob_prime, log_pi, log_pi_prime = net(state) # Sample actions and NN estimates
        p0, r, v, div_v = env.rate(state, action, device=device) # Evaluate the environment: p(s,0), reward, \dot(s), and \grad_s \cdot \dot(s)

        p = torch.exp(log_prob)
        ent_reg = -(log_prob + log_pi).detach()
        # Advantage and Growth rate calculation
        with torch.no_grad():
            advantage = r + (v * value_prime).sum(-1).detach() + log_gamma * value + ent_coef * ent_reg # See Eq. (11)
            divergence = utils.divergence(p, log_prob_prime, log_pi_prime, v, div_v) #See Sec. F of TA
            growth_rate = log_gamma * (p - p0) - divergence # See Eq. (16)

        # Loss function for the gradient estimates calculation
        value_loss = - value * advantage
        state_loss = - log_prob * growth_rate
        policy_loss = - log_pi * advantage

        loss = value_loss + policy_loss + state_loss

        # NN update step
        net.zero_grad()
        loss.mean().backward()
        net.step()

        # Logging
        iter = i + 1
        if (iter == 1) or (iter % logg_iters == 0) or (iter == niters):
            net.eval()
            S = np.concatenate((X[:, :, np.newaxis], V[:, :, np.newaxis]), axis=2)
            value, p, pi = net.evaluate(torch.Tensor(S).to(device))
            value = value.detach().cpu().numpy()
            print("Iteration: ", iter)
            print("Average Value: ", value.mean())

            mean, std = utils.evaluate_policy(net, env, np.exp(log_gamma), num_steps=1000, num_episodes=50)
            print("Return on Test:")
            print("Mean: ", mean, "Deviation: ", std)
            print()
            torch.save(net.state_dict(), path_to_save)

    print("Done")


def main():

    parser = argparse.ArgumentParser(description="Umbrella Reinforce",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--niters", help="Number of training steps",default=1_200_000,type = int)
    parser.add_argument("--batch_size", help="Batch size",default=10_000,type = int)
    parser.add_argument("--ent_coef", help="Entropy coefficient",default=0.01 ,type = float)
    parser.add_argument("--logg_iters", help="Logging period: policy evaluation, saving" ,default=10_000,type = int)
    parser.add_argument("--path_to_save", help="Path to save NN" ,default="out/net.pth",type = str)

    args = parser.parse_args()
    config = vars(args)
    train(config)


if __name__ == "__main__":
    main()
