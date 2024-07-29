
import torch
from torch import nn
from torch.distributions import Categorical


class Exp(nn.Module):
    def forward(self,x):
        return torch.exp(x)


class TripleNet(nn.Module):
    def __init__(
            self,
            state_dim,
            action_dim,
            rep_dim = None,
            reflecting_representation = None,
            hidden_dims=[128, 128],
            lr_v=1e-3,
            lr_p=1e-3,
            lr_pi=1e-3,
            weight_decay_v=0,
            weight_decay_p=0,
            weight_decay_pi=0,
    ):
        super(TripleNet, self).__init__()
        self.state_dim = state_dim
        if rep_dim is None:
            rep_dim = state_dim
        self.rep_dim = rep_dim

        self._fc_policy = self.create_nn([state_dim,] + hidden_dims + [action_dim,], nn.Tanh)
        self._fc_log_prob = self.create_nn([rep_dim, ] + hidden_dims + [1,], nn.ELU)
        self._fc_value = self.create_nn([rep_dim, ] + hidden_dims + [1,], nn.ELU)


        self.reflecting_representation = reflecting_representation

        self.optimizer_v = torch.optim.Adam(self._fc_value.parameters(), lr=lr_v, weight_decay=weight_decay_v)
        self.optimizer_p = torch.optim.Adam(self._fc_log_prob.parameters(), lr=lr_p, weight_decay=weight_decay_p)
        self.optimizer_pi = torch.optim.Adam(self._fc_policy.parameters(), lr=lr_pi, weight_decay=weight_decay_pi)


    def create_nn(self, dims, activation):
        layers = []
        for input,output in zip(dims[:-1],dims[1:]):
            layers.append(nn.Linear(input,output))
            layers.append(activation())
        layers = layers[:-1]
        return nn.Sequential(*layers)


    def forward_base(self,state):
        if self.reflecting_representation is None:
            state_representation = state
        else:
            state_representation = self.reflecting_representation(state)

        log_prob = self._fc_log_prob.forward(state_representation).squeeze(-1)
        value = self._fc_value.forward(state_representation).squeeze(-1)
        action_logits = self._fc_policy.forward(state)

        return action_logits, log_prob, value

    def sample_action(self,action_logits,sample_shape):
        policy = Categorical(logits = action_logits)
        action = policy.sample(sample_shape)
        log_pi = policy.log_prob(action)
        return action, log_pi

    def forward(self, state, create_prime_graph = False,sample_shape = torch.Size()):
        def grad(output, input, create_prime_graph):
            return torch.autograd.grad(output, input, torch.ones_like(output), retain_graph=True, create_graph=create_prime_graph)[0]

        state.requires_grad = True

        action_logits, log_prob, value = self.forward_base(state)
        action, log_pi = self.sample_action(action_logits,sample_shape)

        value_prime = grad(value, state, create_prime_graph)
        log_prob_prime = grad(log_prob, state, create_prime_graph)
        log_pi_prime = grad(log_pi, state, create_prime_graph)

        state.requires_grad = False
        return  action, value, value_prime, log_prob, log_prob_prime, log_pi, log_pi_prime


    def evaluate(self,state):
        action_logits, log_prob, value = self.forward_base(state)
        p = torch.exp(log_prob)
        pi = torch.softmax(action_logits,axis = -1)
        return value, p, pi


    def predict(
        self,
        observation,
        deterministic = False,
        numpy = True
    ):
        required_squeeze = False
        if observation.ndim == 1:
            observation = observation.reshape((-1,) + (self.state_dim,))
            required_squeeze = True

        if not torch.is_tensor(observation):
            observation = torch.tensor(observation)

        with torch.no_grad():
            action_logits = self._fc_policy.forward(observation)
            distribution = Categorical(logits = action_logits)
            if deterministic:
                action = torch.argmax(distribution.probs, dim=-1)
            else:
                action = distribution.sample()

        if required_squeeze:
            action = action.squeeze()
        if numpy:
            action = action.cpu().numpy()
        return action


    def zero_grad(self):
        self.optimizer_v.zero_grad()
        self.optimizer_p.zero_grad()
        self.optimizer_pi.zero_grad()

    def step(self):
        self.optimizer_v.step()
        self.optimizer_p.step()
        self.optimizer_pi.step()




