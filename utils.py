import numpy as np
import os
import torch


class ConfigsDict(object):
    '''Serves as a dictionary in the form of an object.'''

    def __init__(self, adict):
        '''Construct a bunch
        Args:
            adict (dict): The dictionary to build the object from
        '''
        self.adict = adict
        self.__dict__.update(adict)

    def __str__(self):
        '''Get the string representation for the bunch (inherit from dict..)
        Returns:
            The string representation
        '''
        return self.adict.__str__()


"""
configs file has the form:
n_epochs: <int>
n_experiments: <int>
initial_config_conv: [<int>/"M",*]
initial_config_linear: [<int>* ]
neuron_choosing_strategy: random/high/low
expantion_schedule: [(<int>, str),(<int>,str),...] #str can be highest_mean_gradient/lowest_mean_gradient 
ratio_of_neurons_chosen: <float>
ratio_of_expantion: <float>
seeds: <str> #experiment name to use as seed
"""


def read_configs(configs_file):
    configs = dict()
    default_values = {
        "n_epochs": 80,
        "n_experiments": 10,
        "initial_config_conv": [64, "M", 64, "M", 256, "M", 256, "M", 512, "M", 512],
        "initial_config_linear": [4096, 100],
        "neuron_choosing_strategy": "low",
        "expantion_schedule": [(5, "highest_mean_gradient"), (10, "highest_mean_gradient")],
        "ratio_of_neurons_chosen": 0.2,
        "ratio_of_expantion": 3,
        "seeds_experiment": None,
    }

    parsing_dict = {
        "n_epochs": int,
        "n_experiments": int,
        "initial_config_conv": eval,
        "initial_config_linear": eval,
        "neuron_choosing_strategy": str,
        "expantion_schedule": eval,
        "ratio_of_neurons_chosen": float,
        "ratio_of_expantion": float,
        "seeds_experiment": str,
    }

    with open(configs_file, "r") as f:
        for line in f:
            line = line.split(": ")
            configs[line[0]] = parsing_dict[line[0]](line[1].strip())

    # Set defaults
    for key, value in default_values.items():
        if key not in configs:
            configs[key] = value

    if not check_rules(configs):
        raise Exception("Rules of configs not obeyed")
    return ConfigsDict(configs)


def check_rules(configs):
    return True


def randomly_split_list(l, size):
    choice = np.random.choice(range(l.shape[0]), size=(size,), replace=False)
    ind = np.zeros(l.shape[0], dtype=bool)
    ind[choice] = True
    rest = ~ind
    return l[ind], l[rest]


def write_res(experiments_folder, res):
    try:
        os.makedirs(os.path.join(experiments_folder, "res"))
        os.makedirs(os.path.join(experiments_folder, "models"))
    except:
        print("CHILL The results folders already existed already there.")
    for i, r in enumerate(res):
        np.savetxt(os.path.join(experiments_folder, "res",
                                "train_loss_%d" % i), np.array(r[0]))
        np.savetxt(os.path.join(experiments_folder, "res",
                                "test_loss_%d" % i), np.array(r[1]))
        np.savetxt(os.path.join(experiments_folder, "res",
                                "train_accuracy_%d" % i), np.array(r[2]))
        np.savetxt(os.path.join(experiments_folder, "res",
                                "test_accuracy_%d" % i), np.array(r[3]))
        torch.save(r[4], os.path.join(
            experiments_folder, "models", "model_%d" % i))
        #np.savetxt(os.path.join(experiments_folder, "res", "grads_history_%d"%i), np.array(r[3]))


# CODE BY github/stefanonardo
# https://gist.github.com/stefanonardo/693d96ceb2f531fa05db530f3e21517d
class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)
        self.already_stopped = False

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            self.already_stopped = True
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.already_stopped = True
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                    best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                    best * min_delta / 100)


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp
