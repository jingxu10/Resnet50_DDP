#!/usr/bin/env python
# encoding: utf-8

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.utils.data as Data
import torchvision
import torch_ccl

EPOCH = 1
BATCH_SIZE = 256
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

def train(train_loader, net, criterion, optimizer, epoch):
    net.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0 or len(data) < BATCH_SIZE:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    t1 = time.time()
    print('time elapsed: {:.2f}s'.format(t1-t0))

def test(test_loader, net, criterion, optimizer):
    net.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
    test_loss /= count
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, count, 100. * correct / count))

def main():
    rank = 0
    world_size = 1

    if 'PMI_RANK' in os.environ:
        os.environ['RANK'] = os.environ.get('PMI_RANK', -1)
        rank = int(os.environ['RANK'])
    if 'PMI_SIZE' in os.environ:
        os.environ['WORLD_SIZE'] = os.environ.get('PMI_SIZE', -1)
        world_size = int(os.environ['WORLD_SIZE'])

    torch.manual_seed(10)

    if world_size > 1:
        if not 'MASTER_ADDR' in os.environ:
            os.environ['MASTER_ADDR'] = '127.0.0.1'
        if not 'MASTER_PORT' in os.environ:
            os.environ['MASTER_PORT'] = '29500'
        os.environ['RANK'] = str(rank)
        os.environ['WORLD_SIZE'] = str(world_size)
        print('rank: {}/{}'.format(rank, world_size))
        dist.init_process_group(
                backend='ccl'
        )

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    # train_dataset = torchvision.datasets.ImageFolder(
    #         root='{}/train'.format(DATA),
    #         transform=transform
    # )
    train_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=True,
            transform=transform,
            download=DOWNLOAD,
    )
    sampler_train = None
    if world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler_train
    )
    # test_dataset = torchvision.datasets.ImageFolder(
    #         root='{}/val'.format(DATA),
    #         transform=transform
    # )
    test_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=False,
            transform=transform,
            download=DOWNLOAD,
    )
    sampler_test = None
    if world_size > 1:
        sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler_test
    )

    net = torchvision.models.resnet50()
    if world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9)

    for epoch in range(EPOCH):
        train(train_loader, net, criterion, optimizer, epoch)
        test(test_loader, net, criterion, optimizer)

if __name__ == '__main__':
    main()
