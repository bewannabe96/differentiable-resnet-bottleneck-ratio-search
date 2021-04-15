import os
import time
import random

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from resnet import ResNet

LOGPATH = None
TRAIN_CSVPATH = None
VAL_CSVPATH = None
MODELPATH = None

SPLIT_PORTION = 0.8
TRAIN_SET_LOADER = None
VAL_SET_LOADER = None

BATCH_SIZE = 256

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 100
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

def train(epoch, model, optimizer, criterion):
    cum_time = -time.time()

    model.train()

    loss_sum = 0
    batches = 0
    total = 0
    match = 0
    for batch_idx, (inputs, targets) in enumerate(TRAIN_SET_LOADER):
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total += targets.size(0)
        match += outputs.argmax(1).eq(targets).sum().item()
        
        accuracy = 100. * match / total
        loss_sum += loss.item()
        batches = batch_idx+1

    loss = loss_sum / batches

    cum_time = cum_time + time.time()

    write_log(
        '[EPOCH%3d] [TRAIN]\t\tTIME: %7.3f | LOSS: %.3f | ACC: %7.3f%% (%d/%d)'
            % (epoch+1, cum_time, loss, accuracy, match, total)
    )
    write_csv(TRAIN_CSVPATH, [epoch+1, cum_time, loss, accuracy, match, total])

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

def evaluate(model_id, ratio):
    global LOGPATH, TRAIN_CSVPATH, VAL_CSVPATH, MODELPATH
    LOGPATH = 'log_' + model_id
    TRAIN_CSVPATH = 'csv_' + model_id + '.train.csv'
    VAL_CSVPATH = 'csv_' + model_id + '.val.csv'
    MODELPATH = 'model_' + model_id + '.pt'

    write_log('=====================================================================')
    write_log('Evaluating ResNet18 [%s]' % (model_id))
    write_log()

    write_log('==> GENERATING MODELS')
    model = ResNet(ratio=ratio).to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MT, weight_decay=WD)
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except FileNotFoundError:
            epoch = 0

        if(epoch >= EPOCHS):
            break

        train(epoch, model, optimizer, criterion)
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
            'optimizer_state_dict': optimizer.state_dict(),
        }, MODELPATH)
    write_log()

    write_log('==> DELETING MODEL')
    # Delete Model
    try:
        os.remove(MODELPATH)
    except FileNotFoundError:
        pass

    write_log('=====================================================================')

def main():
    # Prepare Data
    prepareDataset()

    # Start Evaluation
    # evaluate('searched', [1, 1, 4, 8])

    OPTIM_TRIAL = 1
    for trial in range(OPTIM_TRIAL):
        r1 = random.randint(1, 8)
        r2 = random.randint(1, 8)
        r3 = random.randint(1, 8)
        r4 = random.randint(1, 8)

        id = 'trial%d_r%d-%d-%d-%d' % (trial, r1, r2, r3, r4)
        evaluate(id, [r1, r2, r3, r4])

main()