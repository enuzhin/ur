from lib.env import MultiValleyMountainCarEnv
from lib.net import TripleNet
from lib import utils
import torch
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib.colors as colors
# from scipy.ndimage import gaussian_filter


def estimate_density(config):
    # Parameters Initialization
    dt = config["dt"]
    niters = config["niters"]
    path_to_load = config["path_to_load"]
    path_to_save_data = config["path_to_save_data"]
    path_to_save_fig = config["path_to_save_fig"]
    logg_iters = config["logg_iters"]

    # User notification
    print("Density estimation started")
    print()
    print("Number of density time updates: ",niters)
    print("NN model: ", path_to_load)
    print("Timestep in density simulation: ", dt)
    print("Logging period: ", logg_iters)
    print("Data will be saved as: ", path_to_save_data)
    print("Figure will be saved as: ", path_to_save_fig)
    print()

    # Initialization of model, environment and device (cpu or gpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ",device.type)
    print()
    env = MultiValleyMountainCarEnv(force=0.001)
    model = TripleNet(2, 2, 3, lr_v=1e-5, lr_p=1e-5, lr_pi=1e-5, weight_decay_v=1e-4, weight_decay_p=5e-4,
                    weight_decay_pi=5e-6, reflecting_representation=env.reflecting_representation).to(device)
    model.load_state_dict(torch.load(path_to_load,weights_only=True))
    model.eval()


    # Constants initialization
    N = 1001
    dx = 1.98 / (N - 1)
    dv = 0.14 / (N - 1)
    X, V = np.mgrid[-0.99:0.99:dx, -0.07:0.07 + dv:dv]
    S = np.concatenate((X[:, :, np.newaxis], V[:, :, np.newaxis]), axis=2)
    S = torch.Tensor(S).to(device)
    p0 = env.start_prob(S,device)
    p0 = torch.Tensor(p0.cpu().numpy()).to(device)
    # p0 = torch.Tensor(gaussian_filter(p0.cpu().numpy(), sigma=3)).to(device)
    p0 = utils.ensured_probability_density(p0, (dx, dv))
    p = p0.clone()


    # Density update
    t = 0
    for i in range(niters):
        t += dt
        with torch.no_grad():
            action_logits = model._fc_policy.forward(S)
            pi = torch.softmax(action_logits, axis=-1)
        A_mean = pi[:, :, 1]
        v_mean = env.state_dot(S, A_mean)
        p = utils.update_state_distribution(p, v_mean, p0, t, dx, dv, dt, N)

        # Logging
        iter = i+1
        if (iter == 1) or (iter % logg_iters == 0) or (iter == niters):
            print("Iteration: ", iter)
            print("Time: ", t)
            print()
            save_fig(X, V, np.clip(p.cpu().numpy(),0,100),title="time: " + str(t), path = path_to_save_fig, cmap="gray", norm_gamma=0.4, dpi=600,ticks =[0,20,40,60,80,100] ,ticklabels=[0,20,40,60,80,str(100)+" and more"])
            np.savetxt(path_to_save_data, p.cpu().numpy())

    print("Done")


def save_fig(X,V,data,title, path, x_label = "Position", y_lebel = "Velocity", cmap = 'viridis', norm_gamma = 1,
              use_cbar = True, ticks = None,ticklabels = None, vmin = None, vmax = None,dpi = 600):
    norm=colors.PowerNorm(gamma=norm_gamma,vmin=vmin,vmax=vmax)
    plt.title(title)
    plt.imshow(data[:,::-1].T, cmap = cmap, norm = norm,extent = (X.min(),X.max(),V.min(),V.max()),aspect = 'auto')
    plt.xlabel(x_label)
    plt.ylabel(y_lebel)
    if use_cbar:
        cbar = plt.colorbar(ticks=ticks)
    if ticklabels is not None:
        cbar.ax.set_yticklabels(ticklabels)
    plt.savefig(path,dpi = dpi)
    plt.close()


def main():

    parser = argparse.ArgumentParser(description="Umbrella Reinforce",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--niters", help="Number of density time updates",default=1_300_000,type = int)
    parser.add_argument("--dt", help="Timestep in density simulation", default=0.00005, type=float)
    parser.add_argument("--logg_iters", help="Logging period: density visualization and saving",default=1_000,type = int)
    parser.add_argument("--path_to_save_data", help="Path to save the array, representing density",default="out/pdf.out",type = str)
    parser.add_argument("--path_to_save_fig", help="Path to save the figure, representing density",default="out/pdf.pdf", type=str)
    parser.add_argument("--path_to_save", help="Path to save density", default="out/pdf.out", type=str)
    parser.add_argument("--path_to_load", help="Path to upload NN", default="out/net.pth", type=str)

    args = parser.parse_args()
    config = vars(args)
    estimate_density(config)


if __name__ == "__main__":
    main()
