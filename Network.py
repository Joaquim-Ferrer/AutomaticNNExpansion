import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import randomly_split_list
import math
import copy

#device = torch.device("cuda")

INPUT_CHANNELS = 3
MAX_POOL_SIZE = 2
INITIAL_IMAGE_SIZE = 32
MIN_OUTPUT_FOR_EXPANTION = 5


def copy_weights(old_layer, new_layer, indexes):
    for i in range(len(old_layer)):
        original_module = old_layer[i]
        new_module = new_layer[i]
        if isinstance(original_module, nn.BatchNorm2d) or isinstance(original_module, nn.Linear) or isinstance(original_module, nn.Conv2d):
            new_module.weight = nn.Parameter(
                original_module.weight[indexes, ...].data)
            new_module.bias = nn.Parameter(
                original_module.bias[indexes, ...].data)


def parse_conv_layers(used_layers, last_used=None, with_final_layer=False):
    # List of tuples (module, number of output channels, prev_layers, after_layers), after_layers and prev_layers being lists_of_modules
    layers = []
    cur_vol_size = INITIAL_IMAGE_SIZE

    first = True
    if last_used != None:
        first = False
    else:
        last_used = INPUT_CHANNELS
    pooling = 0
    for l in used_layers:
        if l == "M":
            cur_vol_size = int(cur_vol_size / 2)
            pooling += 1
        else:
            prev_layers = [] if pooling == 0 else [nn.MaxPool2d(
                MAX_POOL_SIZE, stride=MAX_POOL_SIZE) for i in range(pooling)]
            after_layers = [nn.BatchNorm2d(l), nn.ReLU()]
            layers.append((nn.Conv2d(last_used, l, 3, padding=1),
                           l, prev_layers, after_layers, cur_vol_size))
            pooling = False
            last_used = l
    return layers


def parse_linear_layers(used_layers, last_used, with_final_layer=True):
    # last_used is the number of neurons in the last layer.
    # List of tuples (module, number of output channels, prev_layers, after_layers), after_layers and prev_layers being lists_of_modules
    layers = []
    for i, l in enumerate(used_layers):
        # Always add Dropout before a Linear layer
        prev_layers = [nn.Flatten()]
        after_layers = []
        if not with_final_layer or i != len(used_layers) - 1:
            after_layers = [nn.ReLU()]
        layers.append((nn.Linear(last_used, l), l, prev_layers, after_layers))
        # If the last layer is not final we add RELU to all the layers
        last_used = l
    return layers


def get_n_input(self, prev_res, next_res, n_max_pools):
    if next_res.is_conv or not prev_res.is_conv:
        return prev_res.size
    elif prev_res.is_conv:
        return int(((INITIAL_IMAGE_SIZE / (MAX_POOL_SIZE**n_max_pools))**2) * prev_res.size)


def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear) or isinstance(m, nn.BatchNorm2d):
        m.reset_parameters()


class NetworkGraph(nn.Module):
    def __init__(self, initial_config_conv, initial_config_linear, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.input_output_res = [None, None]
        self.results = []  # List of results
        self.layers = nn.ModuleList()

        conv_modules = parse_conv_layers(initial_config_conv)
        last_output = INPUT_CHANNELS
        if len(conv_modules) != 0:
            self.input_output_res[0] = Result(
                INPUT_CHANNELS, True, batch_size, INITIAL_IMAGE_SIZE)
        else:
            self.input_output_res[0] = Result(
                INITIAL_IMAGE_SIZE ** 2, False, batch_size)
        self.results.append(self.input_output_res[0])
        for i, (module, n_output_neurons, prev_layers, after_layers, volume_size) in enumerate(conv_modules):
            new_layer = self.add_to_end(
                module, last_output, n_output_neurons, prev_layers, after_layers, volume_size=volume_size)
            last_output = n_output_neurons

        n_max_pool = initial_config_conv.count("M")
        last_output = last_n_neurons = int(
            ((INITIAL_IMAGE_SIZE / (MAX_POOL_SIZE**n_max_pool))**2) * last_output)
        linear_modules = parse_linear_layers(
            initial_config_linear, last_used=last_output, with_final_layer=True)
        for module, n_output_neurons, prev_layers, after_layers in linear_modules:
            new_layer = self.add_to_end(
                module, last_output, n_output_neurons, prev_layers, after_layers)
            last_output = n_output_neurons

    def add_to_end(self, module, n_input_neurons, n_output_neurons, prev_layers, after_layers, volume_size=None):
        new_output_res = Result(n_output_neurons, isinstance(
            module, nn.Conv2d), self.batch_size, volume_size)
        prev_res = None
        if self.input_output_res[1] == None:
            # No layers yet
            prev_res = self.input_output_res[0]
        else:
            prev_res = self.input_output_res[1]
        new_layer = self.new_layer(module, n_input_neurons, n_output_neurons, prev_layers,
                                   after_layers, prev_res, new_output_res, np.array([i for i in range(n_output_neurons)]))

        self.results.append(new_output_res)
        self.layers.append(new_layer)
        self.input_output_res[1] = new_output_res
        return new_layer

    def new_layer(self, module, n_input_neurons, n_output_neurons, pre_modules, pos_modules, prev_res, next_res, indices, copy_prev=False):
        if copy_prev:
            pre_modules = [copy.deepcopy(mod) for mod in pre_modules]
        new_layer = Layer(module, n_input_neurons, n_output_neurons,
                          pre_modules, pos_modules, prev_res, next_res, indices)
        prev_res.append_next_layer(new_layer)
        return new_layer

    def forward(self, x):
        if not self.input_output_res[0].is_conv:
            x = x.view(x.shape[0], -1)
        self.reset_results()
        #print(self.input_output_res[0].value.size(), x.size())
        self.input_output_res[0].set(
            x, [i for i in range(self.input_output_res[0].size)])
        results_ready = [self.input_output_res[0]]
        while results_ready:
            res = results_ready.pop()
            for l in res.next_layers:
                new_value = l(res.value)
                # print(new_value.size())
                if l.get_next_res().is_ready():
                    results_ready.append(l.get_next_res())
        if not self.input_output_res[1].is_ready():
            raise Exception("OUTPUT NOT READY IN TIME")
        return self.input_output_res[1].value

    def reset_results(self):
        for res in self.results:
            res.reset()

    def expand(self, layer_to_expand, choosing_strategy, ratio_of_neurons_chosen, ratio_of_expantion):
        prev_res, next_res = layer_to_expand.get_prev_res(), layer_to_expand.get_next_res()
        si_a, si_b = layer_to_expand.expand(
            choosing_strategy, ratio_of_neurons_chosen, ratio_of_expantion)
        n_neurons_expanded = len(si_b)
        if len(si_b) == 0:
            print("Cant expand because layer is too small")
            return
        n_expantion = int(ratio_of_expantion * n_neurons_expanded)
        new_layers = [n_expantion, n_neurons_expanded]
        if next_res.is_conv:
            new_layers = parse_conv_layers(
                new_layers, last_used=layer_to_expand.n_input_neurons, with_final_layer=len(next_res.next_layers) == 0)
        else:
            new_layers = parse_linear_layers(
                new_layers, last_used=layer_to_expand.n_input_neurons, with_final_layer=len(next_res.next_layers) == 0)

        # First expanded layer
        module1, n_output_neurons1, prev_layers1, after_layers1 = new_layers[0][0:4]
        res1 = Result(n_output_neurons1, isinstance(
            module1, nn.Conv2d), self.batch_size, next_res.volume_size)
        self.results.append(res1)
        layer_expanded_1 = self.new_layer(module1, layer_to_expand.n_input_neurons,
                                          n_output_neurons1, layer_to_expand.prev_modules, after_layers1, prev_res, res1,
                                          np.array(
                                              [i for i in range(n_output_neurons1)]), copy_prev=True)
        # Second expanded layer
        module2, n_output_neurons2, prev_layers2, after_layers2 = new_layers[1][0:4]
        layer_expanded_2 = self.new_layer(module2, n_output_neurons1,
                                          n_output_neurons2, prev_layers2, after_layers2, res1, next_res,
                                          layer_to_expand.res_indices[si_b]
                                          )
        # Non expanded layer
        layer_to_expand.remove_weights(si_a)
        self.layers.extend([layer_expanded_1, layer_expanded_2])

        for l in next_res.next_layers:
            l.reset()

    def save_grads(self):
        for l in self.layers:
            l.save_grads()

    # comparison is a funtion that gets two arguments and says wether the first is "better" than the other
    # get_metric is a function that returs the metric of the layer
    def most_something_layer(self, comparison):
        best_layer = self.layers[0]
        for l in self.layers:
            if comparison(l, best_layer):
                best_layer = l
        return best_layer

    def get_lowest_mean_grad_layer(self):
        def metric(layer): return np.mean(layer.grads_memory)
        def comparison(layer_1, layer_2): return True if metric(
            layer_1) < metric(layer_2) else False
        return self.most_something_layer(comparison)

    def get_highest_grad_layer(self):
        def metric(layer): return np.mean(layer.grads_memory)
        def comparison(layer_1, layer_2): return True if metric(
            layer_1) > metric(layer_2) else False
        return self.most_something_layer(comparison)

    def prepare_to_save(self):
        for r in self.results:
            r.make_save_ready()


class Result():
    def __init__(self, size, is_conv, batch_size, volume_size=None):
        self.is_conv = is_conv
        self.values = []
        self.next_layers = []
        self.completed = 0
        self.size = size
        self.batch_size = batch_size
        self.volume_size = volume_size
        self.done = np.array([False for i in range(self.size)])

        self.reset()

    def append_next_layer(self, next_layer):
        self.next_layers.append(next_layer)

    def remove_next_layer(self, layer):
        self.next_layers.remove(layer)

    def set(self, tensor, indexes):
        self.value[:, indexes, ...] = tensor
        self.done[indexes] = True
        self.completed += len(indexes)

    def is_ready(self):
        if self.size == self.completed:
            if not np.all(self.done):
                raise Exception("RESULT SAYS IT'S READY BUT IT'S NOT")
            else:
                return True
        else:
            return False

    def reset(self):
        if self.is_conv:
            self.value = torch.empty(
                (self.batch_size, self.size, self.volume_size, self.volume_size)).to("cuda")
        else:
            self.value = torch.empty((self.batch_size, self.size)).to("cuda")
        self.done = np.array([False for i in range(self.size)])
        self.completed = 0

    def make_save_ready(self):
        self.value = self.value.to("cpu")


class Layer(nn.Module):
    def __init__(self, module, n_input_neurons, n_output_neurons, prev_modules, after_modules, prev_res, next_res, res_indices, n_grads_memory=1000):
        # Post modules is a Sequential Module
        super().__init__()
        self.prev_next_res = (prev_res, next_res)
        self.res_indices = res_indices
        self.module = module
        self.prev_modules = nn.Sequential(*prev_modules)
        self.after_modules = nn.Sequential(*after_modules)
        self.n_output_neurons = n_output_neurons
        self.n_input_neurons = n_input_neurons
        # Let's implement a running average instead of calculating the average in the end
        self.grads_memory = None
        self.n_grads_memory = n_grads_memory

    def forward(self, x):
        x = self.prev_modules(x)
        self.y = self.module(x)
        self.y.retain_grad()
        y = self.after_modules(self.y)
        self.get_next_res().set(y, self.res_indices)
        return y

    def save_grads(self):
        grads = self.y.grad.cpu().numpy()
        batch_size = grads.shape[0]
        grads = np.mean(grads, axis=0)
        if len(grads.shape) > 2:
            # It means that it's a convolution. We only need gradients across channels so we average them out (we sum them which is the same thing as they will only compare them with themselves)
            grads = np.sum(grads, axis=(1, 2))

        if self.grads_memory is None:
            self.grads_memory = grads
        else:
            # Running average
            new_contribution = batch_size / self.n_grads_memory
            self.grads_memory = min(
                1, new_contribution)*grads + max(0, 1-new_contribution)*self.grads_memory

    def expand(self, choosing_strategy, ratio_of_neurons_chosen, ratio_of_expantion):
        n_neurons_expanded = int(
            self.n_output_neurons * ratio_of_neurons_chosen)

        grads_abs = np.abs(self.grads_memory)
        sorted_indices = grads_abs.argsort()
        sorted_grads = grads_abs[sorted_indices]

        si_a, si_b = None, None
        if choosing_strategy == "high":
            si_a = sorted_indices[0:self.n_output_neurons-n_neurons_expanded]
            si_b = sorted_indices[self.n_output_neurons -
                                  n_neurons_expanded:self.n_output_neurons]
        elif choosing_strategy == "low":
            si_a = sorted_indices[n_neurons_expanded:self.n_output_neurons]
            si_b = sorted_indices[0:n_neurons_expanded]
        elif choosing_strategy == "random":
            si_a, si_b = randomly_split_list(
                np.array([i for i in range(self.n_output_neurons)]),
                self.n_output_neurons-n_neurons_expanded
            )
        else:
            raise Exception("Derivative choosing method " +
                            choosing_strategy + " doesn't exist. Exiting.")

        self.si_a = si_a
        self.si_b = si_b
        return si_a, si_b

    def remove_weights(self, indices):
        new_layers = None
        last_used = None

        if isinstance(self.module, nn.Conv2d):
            new_layers = parse_conv_layers([len(indices)], last_used=self.n_input_neurons, with_final_layer=len(
                self.prev_next_res[1].next_layers) == 0)
        else:
            new_layers = parse_linear_layers([len(indices)], last_used=self.n_input_neurons, with_final_layer=len(
                self.prev_next_res[1].next_layers) == 0)

        new_module, new_n_output_neurons, new_prev_modules, new_after_modules = new_layers[
            0][0:4]
        copy_weights(self.after_modules, new_after_modules, indices)
        copy_weights([self.module], [new_module], indices)

        self.module = new_module
        self.after_modules = nn.Sequential(*new_after_modules)
        self.res_indices = self.res_indices[indices]
        self.n_output_neurons = len(indices)
        self.grads_memory = None

    def reset(self):
        self.grads_memory = None
        # Reset modules
        self.apply(weight_reset)

    def get_prev_res(self):
        return self.prev_next_res[0]

    def get_next_res(self):
        return self.prev_next_res[1]
