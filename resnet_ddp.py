#!/usr/bin/env python
# encoding: utf-8

import argparse
import time
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

EPOCH = 1
BATCH_SIZE = 64
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

def train(train_loader, net, criterion, optimizer, epoch):
    net.train()
    td=0
    for batch_idx, (data, target) in enumerate(train_loader):
        t0 = time.time()
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        t1 = time.time()
        td = td + t1 - t0
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader), loss.item()))
    print('time elasped: {:.2f}s'.format(td))

def test(test_loader, net, criterion, optimizer):
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = net(data)
            test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(test_loss, correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset)))

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    args = parser.parse_args()

    torch.manual_seed(10)

    if args.world_size > 1:
        print('rank: {}/{}'.format(args.local_rank+1, args.world_size))
        torch.distributed.init_process_group(
                backend='gloo',
                init_method='file:///home/u31238/resnet_ddp/tmpfile',
                rank=args.local_rank,
                world_size=args.world_size
        )

    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=True,
            transform=transform,
            download=DOWNLOAD,
    )
    sampler_train = None
    if args.world_size > 1:
        sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
            root=DATA,
            train=False,
            transform=transform,
            download=DOWNLOAD,
    )
    sampler_test = None
    if args.world_size > 1:
        sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler_test
    )

    net = torchvision.models.resnet50()
    if args.world_size > 1:
        net = torch.nn.parallel.DistributedDataParallel(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr = LR, momentum=0.9)

    for epoch in range(EPOCH):
        train(train_loader, net, criterion, optimizer, epoch)
        test(test_loader, net, criterion, optimizer)

if __name__ == '__main__':
    main()
