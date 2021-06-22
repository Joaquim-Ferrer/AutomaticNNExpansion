import os
import sys
import time
import torch
from torch import nn, optim
from torchvision import datasets, transforms
from Network import NetworkGraph, Layer, Result
from utils import read_configs, write_res, EarlyStopping, get_n_params
from torch.multiprocessing import Process, Manager, Pool


def _train(trainloader, epochs, model, criterion, optimizer, testloader, device, es=None, i_epoch=0):
    loss_list = []
    test_loss_list = []
    accuray_list = []
    test_accuracy_list = []
    print(get_n_params(model), model)

    if es is not None and es.already_stopped:
        return loss_list, test_loss_list

    for e in range(epochs):
        time_start = time.process_time()
        running_loss = 0
        n_total = 0
        n_correct = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            if images.size()[0] != model.batch_size:
                continue
            y = model(images)
            loss = criterion(y, labels)
            # print("\n\n\n\HERE", y.size(), labels.size(), "\n\n\n", flush=True)

            n_total += images.size(0)
            n_correct += (torch.max(y, 1)[1] == labels).sum().item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model.save_grads()
            running_loss += loss.item()
        print("Epoch: %d;  Training loss: %0.4f; Accuracy: %0.3f; Duration:%0.1f s" % (
            e+i_epoch, running_loss / len(trainloader), n_correct/n_total, time.process_time() - time_start), flush=True)
        loss_list.append(running_loss / len(trainloader))
        accuray_list.append(n_correct/n_total)
        if testloader != None:
            test_loss, test_accuracy = _test(
                testloader, model, criterion, device)
            print("Test loss: %0.4f; Test accuracy: %0.3f\n" %
                  (test_loss, test_accuracy))
            test_loss_list.append(test_loss)
            test_accuracy_list.append(test_accuracy)
            if es is not None and es.step(test_loss):
                break
    return loss_list, test_loss_list, accuray_list, test_accuracy_list


def _test(testloader, model, criterion, device):
    model.eval()
    test_loss = 0
    n_total = 0
    n_correct = 0
    with torch.no_grad():
        for images, labels in testloader:
            if images.size()[0] != model.batch_size:
                continue
            images, labels = images.to(device), labels.to(device)
            y = model(images)

            n_total += images.size(0)
            n_correct += (torch.max(y, 1)[1] == labels).sum().item()
            loss = criterion(y, labels).item()
            test_loss += loss
    test_loss /= len(testloader)
    model.train()
    return test_loss, n_correct / n_total


def train(trainloader, testloader, epochs, neuron_choosing_strategy, expantion_schedule, ratio_of_neurons_chosen, ratio_of_expantion, model, device):
    lr = 0.003
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    es = EarlyStopping()

    loss_list = []
    test_loss_list = []
    accuracy_list = []
    test_accuracy_list = []

    last_time_step = -1
    for expantion_time, layer_choosing_strategy in expantion_schedule:
        new_loss_list, new_test_loss_list, new_accuracy_list, news_test_accuracy_list = _train(
            trainloader, expantion_time-last_time_step, model, criterion, optimizer, testloader, device, es, i_epoch=len(loss_list))
        loss_list.extend(new_loss_list)
        test_loss_list.extend(new_test_loss_list)
        accuracy_list.extend(new_accuracy_list)
        test_accuracy_list.extend(news_test_accuracy_list)

        layer_to_expand = None
        if layer_choosing_strategy == "lowest_mean_gradient":
            layer_to_expand = model.get_lowest_mean_grad_layer()
        elif layer_choosing_strategy == "highest_mean_gradient":
            layer_to_expand = model.get_highest_grad_layer()
        elif layer_choosing_strategy.isdigit():
            layer_to_expand = model.layers[int(layer_choosing_strategy)]
        else:
            raise Exception("Layer choosing strategy not recognized")
        model.expand(layer_to_expand, neuron_choosing_strategy,
                     ratio_of_neurons_chosen, ratio_of_expantion)
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        last_time_step = expantion_time

    new_loss_list, new_test_loss_list, new_accuracy_list, news_test_accuracy_list = _train(
        trainloader, epochs-last_time_step, model, criterion, optimizer, testloader, device, es, i_epoch=len(loss_list))
    loss_list.extend(new_loss_list)
    test_loss_list.extend(new_test_loss_list)
    accuracy_list.extend(new_accuracy_list)
    test_accuracy_list.extend(news_test_accuracy_list)
    return loss_list, test_loss_list, accuracy_list, test_accuracy_list, model


def run_experiment(epochs, neuron_choosing_strategy, expantion_schedule,
                   ratio_of_neurons_chosen, ratio_of_expantion, initial_conv, initial_linear, seed_experiment, name):
    batch_size = 256
    try:
        if seed_experiment is None:
            model = NetworkGraph(initial_conv, initial_linear, batch_size)
        else:
            model = torch.load(seed_experiment)
        print(name)
        device = torch.device("cuda")
        model = model.to("cuda")
        # Removed some dataset expantion transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(
            './.pytorch/CIFAR_10/', download=True, train=True, transform=transform)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=256, shuffle=True)
        # Download and load the test data
        testset = datasets.CIFAR10('./.pytorch/CIFAR_10/', download=True, train=False, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=256, shuffle=True)
        loss_list, test_loss_list, accuracy_list, test_accuracy_list, model = train(trainloader, testloader, epochs,
                                                                                    neuron_choosing_strategy,
                                                                                    expantion_schedule, ratio_of_neurons_chosen, ratio_of_expantion, model, device)
        cpu_model = model.to("cpu")
        # model.prepare_to_save()
        return loss_list, test_loss_list, accuracy_list, test_accuracy_list, model.state_dict()
    except KeyboardInterrupt:
        return


def normal_experiments(experiments_path):
    configs_path = os.path.join(experiments_path, "configs.txt")
    configs = read_configs(configs_path)
    seeds_path = os.path.join(configs.seeds_experiment, "models",
                              "model_%d") if configs.seeds_experiment is not None else None
    print(configs)
    res = []
    with Pool(processes=2) as pool:
        iterable_arguments = [(
            configs.n_epochs,
            configs.neuron_choosing_strategy,
            configs.expantion_schedule,
            configs.ratio_of_neurons_chosen,
            configs.ratio_of_expantion,
            configs.initial_config_conv,
            configs.initial_config_linear,
            seeds_path % i if configs.seeds_experiment is not None else None,
            str(i)
        ) for i in range(configs.n_experiments)]
        res = pool.starmap(run_experiment, iterable_arguments)
        write_res(experiments_path, res)


def main(experiments_folder, experiments_name):
    experiment_path = os.path.join(experiments_folder, experiments_name)
    print("Running " + experiments_name)
    normal_experiments(experiment_path)


if __name__ == "__main__":
    experiments_folder = str(sys.argv[1])
    experiment = str(sys.argv[2])
    main(experiments_folder, experiment)
