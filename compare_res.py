import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from utils import randomly_split_list, write_res, read_configs


def get_average_std(res_path):
    res_train = []
    res_test = []
    max_n_epochs = 0
    for run in os.listdir(res_path):
        file_path = os.path.join(res_path, run)
        r_exp = list(np.loadtxt(file_path))
        if run.startswith("test_accuracy"):
            res_test.append(r_exp)
            max_n_epochs = max(max_n_epochs, len(r_exp))
        elif run.startswith("train_accuracy"):
            res_train.append(r_exp)
        else:
            print("FILE NAME NOT RECOGNIZED", file_path)
    for i_run in range(len(res_train)):
        for run in [res_train[i_run], res_test[i_run]]:
            run.extend([run[-1] for i in range(max_n_epochs-len(run))])

    res_train = np.array(res_train)
    res_test = np.array(res_test)

    mean_train = np.mean(res_train, axis=0)
    mean_test = np.mean(res_test, axis=0)

    std_train = np.std(res_train, axis=0)
    std_test = np.std(res_test, axis=0)

    max_train = np.max(res_train, axis=0)
    max_test = np.max(res_test, axis=0)
    min_train = np.min(res_train, axis=0)
    min_test = np.min(res_test, axis=0)
    print(res_path + ": " + str(res_train.shape[0]) + " runs")
    return mean_train, mean_test, std_train, std_test


experiments_folder = sys.argv[1]
experiments = sys.argv[2:]
plt.figure()

for experiment in experiments:
    experiment_path = os.path.join(experiments_folder, experiment)
    configs = read_configs(os.path.join(experiment_path, "configs.txt"))
    exp_path = os.path.join(experiment_path, "res")
    mean_train, mean_test, std_train, std_test = get_average_std(exp_path)
    # if configs.seeds_folder is not None:
    #     seeds_res_path = os.path.join(experiments_folder, configs.seeds_folder, "res")
    #     seeds_mean_train, seeds_mean_test, seeds_std_train, seeds_std_test = get_average_std(seeds_res_path)
    #     mean_test = np.concatenate((seeds_mean_test, mean_test))
    #     std_test = np.concatenate((seeds_std_test, std_test))
    #     mean_train = np.concatenate((seeds_mean_train, mean_train))
    #     std_train = np.concatenate((seeds_std_train, std_train))

    plt.subplot(2, 1, 1)
    plt.title("Test loss")
    plt.errorbar(range(len(mean_test)), mean_test,
                 yerr=std_test, label=experiment)
    plt.subplot(2, 1, 2)
    plt.title("Train loss")
    plt.errorbar(range(len(mean_train)), mean_train,
                 yerr=std_train, label=experiment)
plt.legend()
plt.show()
