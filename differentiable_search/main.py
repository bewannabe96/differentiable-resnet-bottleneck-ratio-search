import os
import time

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

from drbrs import DRBRS

LOGPATH = None
TRAIN_CSVPATH = None
VAL_CSVPATH = None
ARCH_CSVPATH = None
MODELPATH = None

SPLIT_PORTION = 0.8
TRAIN_SET_LOADER = None
VAL_SET_LOADER = None

BATCH_SIZE = 256

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500
LR = 0.1
MT = 0.9
WD = 5e-4

def write_log(log=''):
	logfile = open(LOGPATH, "a")
	logfile.write(log)
	logfile.write('\n')
	logfile.close()

def write_csv(path, row):
	logfile = open(path, "a")
	logfile.write(','.join(map(lambda x:str(x), row)))
	logfile.write('\n')
	logfile.close()

def prepareDataset(batch_size=BATCH_SIZE):
    global TRAIN_SET_LOADER, VAL_SET_LOADER

    transform = transforms.Compose([
	    transforms.ToTensor(),
	    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    train_set = torchvision.datasets.CIFAR10('../dataset/cifar10', train=True, transform=transform, download=False)

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(SPLIT_PORTION * num_train))
    
    TRAIN_SET_LOADER = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    )
    VAL_SET_LOADER = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]),
    )

def summarize_model(model):
    summary = "[MODEL SUMMARY]"
    summary += "\nPARAMS: %d" % (sum(p.numel() for p in model.parameters()))
    return summary

def flatten_architecture(model):
    flat_alphas = []
    alphas = model.arch_parameters()
    for alpha in alphas:
        flat_alphas.extend(F.softmax(alpha, dim=-1).tolist())
    return flat_alphas

def train(epoch, model, search_optimizer, train_optimizer, criterion):
    loss_sum = 0
    batches = 0
    total = 0
    match = 0
    search_time = 0
    train_time = 0
    for batch_idx, (train_inputs, train_targets) in enumerate(TRAIN_SET_LOADER):
        search_inputs, search_targets = next(iter(VAL_SET_LOADER))

        model.train()

        search_inputs, search_targets = search_inputs.to(DEVICE), search_targets.to(DEVICE)
        train_inputs, train_targets = train_inputs.to(DEVICE), train_targets.to(DEVICE)

        # architecture search
        search_time -= time.time()
        search_optimizer.zero_grad()
        search_outputs = model(search_inputs)
        loss = criterion(search_outputs, search_targets)
        loss.backward()
        search_optimizer.step()
        search_time += time.time()

        # parameter train
        train_time -= time.time()
        train_optimizer.zero_grad()
        train_outputs = model(train_inputs)
        loss = criterion(train_outputs, train_targets)
        loss.backward()
        train_optimizer.step()
        train_time += time.time()

        total += train_targets.size(0)
        match += train_outputs.argmax(1).eq(train_targets).sum().item()
        
        accuracy = 100. * match / total
        loss_sum += loss.item()
        batches = batch_idx+1

    loss = loss_sum / batches

    ca = model.current_arch()
    cac = model.current_arch_confidence()
    write_log(
        '[EPOCH%3d] [SEARCH]\t\tTIME: %7.3f | C1-R%d(%6.3f%%) | C2-R%d(%6.3f%%) | C3-R%d(%6.3f%%) | C4-R%d(%6.3f%%)'
            % (epoch+1, search_time, ca[0], cac[0], ca[1], cac[1], ca[2], cac[2], ca[3], cac[3])
    )
    write_csv(ARCH_CSVPATH, [search_time, *model.flatten_arch_parameters()])

    write_log(
        '[EPOCH%3d] [TRAIN]\t\tTIME: %7.3f | LOSS: %.3f | ACC: %7.3f%% (%d/%d)'
            % (epoch+1, train_time, loss, accuracy, match, total)
    )
    write_csv(TRAIN_CSVPATH, [epoch+1, train_time, loss, accuracy, match, total])

def validation(epoch, model, criterion):
    cum_time = -time.time()

    model.eval()

    loss_sum = 0
    batches = 0
    total = 0
    match = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets)in enumerate(VAL_SET_LOADER):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            total += targets.size(0)
            match += outputs.argmax(1).eq(targets).sum().item()

            accuracy = 100. * match / total
            loss_sum += loss.item()
            batches = batch_idx+1

    loss = loss_sum / batches

    cum_time = cum_time + time.time()

    write_log(
        '[EPOCH%3d] [VAL]\t\tTIME: %7.3f | LOSS: %.3f | ACC: %7.3f%% (%d/%d)'
            % (epoch+1, cum_time, loss, accuracy, match, total)
    )
    write_csv(VAL_CSVPATH, [epoch+1, cum_time, loss, accuracy, match, total])

def evaluate():
    global LOGPATH, TRAIN_CSVPATH, VAL_CSVPATH, ARCH_CSVPATH, MODELPATH
    LOGPATH = 'log_differentiable'
    TRAIN_CSVPATH = 'csv_differentiable.train.csv'
    VAL_CSVPATH = 'csv_differentiable.val.csv'
    ARCH_CSVPATH = 'architecture.csv'
    MODELPATH = 'model_differentiable.pt'

    write_log('=====================================================================')
    write_log('Evaluating DRBRS')
    write_log()

    write_log('==> GENERATING MODELS')
    model = DRBRS().to(DEVICE)
    search_optimizer = optim.Adam(model.arch_parameters(), lr=LR, weight_decay=WD)
    train_optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MT, weight_decay=WD)
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    write_log(summarize_model(model))
    write_log()

    write_log('==> EVALUATING')
    epoch = 0
    while(True):
        # Load Model
        try:
            checkpoint = torch.load(MODELPATH)
            epoch = checkpoint['epoch'] + 1
            model.load_state_dict(checkpoint['model_state_dict'])
            search_optimizer.load_state_dict(checkpoint['search_optimizer_state_dict'])
            train_optimizer.load_state_dict(checkpoint['train_optimizer_state_dict'])
        except FileNotFoundError:
            epoch = 0

        if(epoch >= EPOCHS):
            break

        train(epoch, model, search_optimizer, train_optimizer, criterion)
        validation(epoch, model, criterion)
        write_log()

        # Save Model
        try:
            os.remove(MODELPATH)
        except FileNotFoundError:
            pass
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'search_optimizer_state_dict': search_optimizer.state_dict(),
            'train_optimizer_state_dict': train_optimizer.state_dict(),
        }, MODELPATH)
    write_log()

    write_log('=====================================================================')

def main():
    # Prepare Data
    prepareDataset()

    # Start Evaluation
    evaluate()

main()