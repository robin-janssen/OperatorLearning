import torch
import torch.nn as nn


# # TODO Use inheritance to constrain the definition of operator networks and to enable proper type hinting.
# class OperatorNetwork(nn.Module):
#     def __new__(cls, *args, **kwargs):
#         if cls is OperatorNetwork:
#             raise TypeError("OperatorNetwork class may not be instantiated")
#         # Ensure that the subclass has a forward method
#         if not hasattr(cls, "forward") or not callable(cls.forward):
#             raise TypeError("OperatorNetwork subclass must implement a forward method")
#         # Ensure that the subclass has a __init__ method
#         if not hasattr(cls, "__init__") or not callable(cls.__init__):
#             raise TypeError(
#                 "OperatorNetwork subclass must implement an __init__ method"
#             )
#         # Ensure that the subclass has branch and trunk networks
#         if not hasattr(cls, "branch_net") or not hasattr(cls, "trunk_net"):
#             raise TypeError(
#                 "OperatorNetwork subclass must have branch_net and trunk_net attributes"
#             )
#         return super().__new__(cls)

#     def __init__(self):
#         super(OperatorNetwork, self).__init__()


class OperatorNetwork(nn.Module):
    def __init__(self):
        super(OperatorNetwork, self).__init__()

    def post_init_check(self):
        if not hasattr(self, "branch_net") or not hasattr(self, "trunk_net"):
            raise NotImplementedError(
                "Child classes must initialize a branch_net and trunk_net."
            )
        if not hasattr(self, "forward") or not callable(self.forward):
            raise NotImplementedError("Child classes must implement a forward method.")

    def forward(self, branch_input, trunk_input):
        # Define a generic forward pass or raise an error to enforce child class implementation
        raise NotImplementedError("Forward method must be implemented by subclasses.")


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


class DeepONet(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        device,
    ):
        super(DeepONet, self).__init__()
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=1)


class MultiONet(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONet, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

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


class MultiONetB(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONetB, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        trunk_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b).to(device)
        self.trunk_net = TrunkNet(inputs_t, hidden_t, trunk_outputs, layers_t).to(
            device
        )

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


class MultiONetT(OperatorNetwork):
    def __init__(
        self,
        inputs_b,
        hidden_b,
        layers_b,
        inputs_t,
        hidden_t,
        layers_t,
        outputs,
        N,
        device,
    ):
        super(MultiONetT, self).__init__()
        self.N = N  # Number of outputs
        self.outputs = outputs  # Number of neurons in the last layer
        branch_outputs = outputs // N

        # Initialize branch and trunk networks
        self.branch_net = BranchNet(inputs_b, hidden_b, branch_outputs, layers_b).to(
            device
        )
        self.trunk_net = TrunkNet(inputs_t, hidden_t, outputs, layers_t).to(device)

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


class MultiONetTest(OperatorNetwork):
    def __init__(
        self, inputs_b, hidden_b, layers_b, inputs_t, hidden_t, layers_t, outputs
    ):
        super(MultiONetTest, self).__init__()
        self.branch_net = BranchNet(inputs_b, hidden_b, outputs, layers_b)

    def forward(self, branch_input, trunk_input):
        branch_output = self.branch_net(branch_input)
        trunk_output = self.trunk_net(trunk_input)
        return torch.sum(branch_output * trunk_output, dim=1)
