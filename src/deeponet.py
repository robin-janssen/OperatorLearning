import torch
import torch.nn as nn


class BranchNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(BranchNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TrunkNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super(TrunkNet, self).__init__()
        layers = [nn.Linear(input_size, hidden_size), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class DeepONet(nn.Module):
    def __init__(
        self, inputs_b, hidden_b, layers_b, inputs_t, hidden_t, layers_t, outputs
    ):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=1)


class MultiONet(nn.Module):
    def __init__(
        self, inputs_b, hidden_b, layers_b, inputs_t, hidden_t, layers_t, outputs, N
    ):
        super(MultiONet, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for b_split, t_split in zip(branch_splits, trunk_splits):
            result.append(torch.sum(b_split * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1)

    def _calculate_split_sizes(self, total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits

        sizes = [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]
        return sizes


class MultiONetB(nn.Module):
    def __init__(
        self, inputs_b, hidden_b, layers_b, inputs_t, hidden_t, layers_t, outputs, N
    ):
        super(MultiONetB, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        trunk_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, trunk_outputs, layers_t)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        branch_splits = torch.split(branch_output, split_sizes, dim=1)

        result = []
        for b_split in branch_splits:
            result.append(torch.sum(b_split * trunk_output, dim=1, keepdim=True))

        return torch.cat(result, dim=1)

    def _calculate_split_sizes(self, total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits

        sizes = [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]
        return sizes


class MultiONetT(nn.Module):
    def __init__(
        self, inputs_b, hidden_b, layers_b, inputs_t, hidden_t, layers_t, outputs, N
    ):
        super(MultiONetT, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        branch_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, branch_outputs, layers_b)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)

        # Splitting the outputs for multiple output values
        split_sizes = self._calculate_split_sizes(self.outputs, self.N)
        trunk_splits = torch.split(trunk_output, split_sizes, dim=1)

        result = []
        for t_split in trunk_splits:
            result.append(torch.sum(branch_output * t_split, dim=1, keepdim=True))

        return torch.cat(result, dim=1)

    def _calculate_split_sizes(self, total_neurons, num_splits):
        """Helper function to calculate split sizes for even distribution"""
        base_size = total_neurons // num_splits
        remainder = total_neurons % num_splits

        sizes = [
            base_size + 1 if i < remainder else base_size for i in range(num_splits)
        ]
        return sizes
