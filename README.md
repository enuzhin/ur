# Umbrella Reinforce

**Umbrella Reinforce** is an advanced **Reinforcement Learning (RL) algorithm** inspired by the **Umbrella Sampling** technique from computational physics. It is specifically designed to address key challenges in RL, including:

- **Long-delayed rewards**, which hinder learning efficiency.
- **State traps**, where conventional algorithms struggle to escape local optima.
- **Lack of a terminal state**, making policy optimization difficult.

Unlike conventional RL methods that rely on sequential trajectory simulations, **Umbrella Reinforce** employs **random state sampling**, ensuring **invariance to discretization time steps** and **independence from fixed episode lengths**. This results in more efficient learning across a diverse range of environments.

## Features
- **Robust Exploration**: Effectively navigates environments with sparse rewards and complex dynamics.
- **Stable Convergence**: Invariant to simulation time-step selection, ensuring numerical consistency.
- **Modular and Extensible**: Easily integrates into **custom RL environments** and allows for modifications.
- **OpenAI Gym-Compatible**: Environments adhere to the **Gym API**, enabling seamless integration.
- **Optimized Implementation**: Utilizes **PyTorch** for automatic differentiation and computational efficiency.

## Applications
Umbrella Reinforce has demonstrated superior performance in tackling complex RL problems, including:
- **Multi-Valley Mountain Car**: A variant of the Mountain Car problem featuring multiple valleys, requiring strategic long-term planning and precise control.
- **StandUp Problem**: A robotic learning task in which an articulated arm must achieve and maintain balance in an upright position without predefined termination states.

## Installation & Usage

Follow these steps to set up and run Umbrella Reinforce:

### 1. Clone the Repository
```bash
git clone https://github.com/enuzhin/ur.git
cd ur
```

### 2. Install Dependencies
Ensure you have Python 3 installed, then install required packages:
```bash
pip install -r requirements.txt
```

### 3. Train the Model
Run the training script to train the reinforcement learning agent:
```bash
python train.py
```

### 4. Run a Policy Simulation
Once training is complete, test the trained policy using:
```bash
python simulate.py
```

### 5. Evaluate Agents' Distribution
Analyze the distribution of agents in the environment with:
```bash
python mvmc_pdf.py
```

### 6. View Available Parameters
For a list of available parameters and their default values, run:
```bash
python train.py -h
```
Replace `train.py` with `simulate.py` or `mvmc_pdf.py` for script-specific options.

## Examples
### Simulation and Agents' Distribution

| **Simulation** | **Agents' Distribution** |
|:-------------:|:------------------------:|
| ![GIF](out/mvmc.gif) | [📄 View PDF](out/mvmc.pdf) |

## Dependencies
```bash
Python 3
PyTorch
NumPy
Gymnasium
Matplotlib
```

## References
- **[Umbrella Reinforcement Learning – Computationally Efficient Tool for Hard Non-Linear Problems](https://doi.org/10.1016/j.cnsns.2024.108583)**
- **[Umbrella Sampling (Wikipedia)](https://en.wikipedia.org/wiki/Umbrella_sampling)**
- **[Reinforcement Learning (Wikipedia)](https://en.wikipedia.org/wiki/Reinforcement_learning)**
- **[PyTorch](https://pytorch.org)**
- **[Gymnasium](https://gymnasium.farama.org/)**

