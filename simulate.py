from lib.env import MultiValleyMountainCarEnv,StandUpEnv
from lib.net import TripleNet
import torch
import numpy as np
import argparse
from time import sleep
import matplotlib.pyplot as plt
from matplotlib import animation


def simulate(config):
    # Parameters Initialization
    niters = config["niters"]
    path_to_save = config["path_to_save"]
    path_to_load = config["path_to_load"]
    env_name = config["env"].lower()

    # User notification
    print("Simulation started")
    print()
    if env_name == "mvmc":
        print("Environment: Multi Valley Mountain Car")
        env = MultiValleyMountainCarEnv()
    elif env_name == "standup":
        print("Environment: Stand Up")
        env = StandUpEnv()
    else:
        print("Wrong environment name")
        return
    print("Number of simulation steps: ",niters)
    print("NN model: ", path_to_load)
    print("GIF will be saved as: ", path_to_save)
    print()

    # Initialization of model, environment and device (cpu or gpu)
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    print("Device: ",device.type)
    print()
    if env_name == "mvmc":
        model = TripleNet(2, 2, 3, lr_v=1e-5, lr_p=1e-5, lr_pi=1e-5, weight_decay_v=1e-4, weight_decay_p=5e-4,
                        weight_decay_pi=5e-6, reflecting_representation=env.reflecting_representation).to(device)
    elif env_name == "standup":
        model = TripleNet(2, 4, hidden_dims=[128, 128], lr_v=1e-6, lr_p=1e-7, lr_pi=1e-6, weight_decay_v=1e-5,
                        weight_decay_p=5e-4, weight_decay_pi=5e-5,
                        reflecting_representation=env.reflecting_representation).to(device)
    model.load_state_dict(torch.load(path_to_load,weights_only=True))
    model.eval()


    # Visualization update
    state = tuple(env.reset())
    frames = []
    for i in range(niters):
        action = model.predict(torch.tensor(state).to(device), deterministic=True)
        state, reward, done, _ = env.step(action)
        frame = env.render(mode = "rgb_array")
        sleep(0.01)
        frames.append(frame)
    env.close()

    # Simulation saving
    print("Please wait.. gif saving..")
    save_frames_as_gif(np.array(frames), path_to_save)
    print("Done")


def save_frames_as_gif(frames, path):
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path)


def main():
    parser = argparse.ArgumentParser(description="Umbrella Reinforce",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--niters", help="Number of simulation steps",default=1_000,type = int)
    parser.add_argument("--path_to_save", help="Path to save gif",default="out/sim.gif",type = str)
    parser.add_argument("--path_to_load", help="Path to upload NN", default="out/net.pth", type=str)
    parser.add_argument("--env", help="Environment: MVMC or StandUp", default="mvmc", choices=['mvmc', 'standup'])

    args = parser.parse_args()
    config = vars(args)
    simulate(config)


if __name__ == "__main__":
    main()
