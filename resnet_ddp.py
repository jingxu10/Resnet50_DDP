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

is_hvd_enabled = False
try:
    import horovod.torch as hvd
    is_hvd_enabled = True
except:
    pass

EPOCH = 100
LR = 0.001
DOWNLOAD = True
DATA = 'datasets/cifar10/'

def train(train_loader, net, criterion, optimizer, epoch, args):
    if is_hvd_enabled:
        hvd.broadcast_parameters(net.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
    net.train()
    t0 = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(args.device, non_blocking=True)
        if args.ipex and args.device == 'cpu':
            data = data.to(memory_format=torch.channels_last)
        target = target.to(args.device, non_blocking=True)
        t00 = time.time()
        optimizer.zero_grad()
        with torch.cpu.amp.autocast(enabled=args.bf16):
            output = net(data)
            loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        t01 = time.time()
        if batch_idx % 1 == 0 or len(data) < args.batch_size:
            print('[{}] Train Epoch: {} [{:5d}/{} ({:6.2f}%)]\tLoss: {:.6f}\tdur: {:.5f}ms'.format(args.rank, epoch, args.world_size * batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / float(len(train_loader)), loss.item(), (t01-t00)*1000))
    t1 = time.time()
    print('time elapsed: {:.2f}s'.format(t1-t0))

def test(test_loader, net, criterion, optimizer, args):
    net.eval()
    test_loss = 0
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(args.device, non_blocking=True)
            if args.ipex and args.device == 'cpu':
                data = data.to(memory_format=torch.channels_last)
            target = target.to(args.device, non_blocking=True)
            with torch.cpu.amp.autocast(enabled=args.bf16):
                output = net(data)
            test_loss += criterion(output, target).item() * len(data) # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += len(data)
    test_loss /= count
    print('[{}] Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(args.rank, test_loss, correct, count, 100. * correct / count))
    return test_loss

def main(args):
    torch.manual_seed(10)

    if args.world_size > 1:
        if is_hvd_enabled:
            print('Distributed training with Horovod')
        else:
            print('Distributed training with DDP')
            if not 'MASTER_ADDR' in os.environ:
                os.environ['MASTER_ADDR'] = args.master_addr
            if not 'MASTER_PORT' in os.environ:
                os.environ['MASTER_PORT'] = args.port
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            dist.init_process_group(
                    backend=args.backend,
                    init_method='env://'
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
        if is_hvd_enabled:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            sampler_train = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = Data.DataLoader(
            dataset=train_dataset,
            batch_size=args.batch_size,
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
        if is_hvd_enabled:
            sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
        else:
            sampler_test = torch.utils.data.distributed.DistributedSampler(test_dataset)
    test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=args.batch_size,
            sampler=sampler_test
    )

    net = torchvision.models.resnet50()
    net = net.to(args.device)
    lr_scaler = 1
    if is_hvd_enabled:
        lr_scaler = hvd.size()
    optimizer = torch.optim.SGD(net.parameters(), lr = LR * lr_scaler, momentum=0.9)
    if is_hvd_enabled:
        optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=net.named_parameters())

    net.train()
    if args.ipex:
        if args.device == 'cpu':
            if args.bf16 and args.backend == 'ccl':
                net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=torch.bfloat16, level="O1")
            else:
                net, optimizer = ipex.optimize(net, optimizer=optimizer, dtype=torch.float32, level="O1")

    if args.world_size > 1 and not is_hvd_enabled:
        device_ids = None
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=device_ids)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(args.device)

    for epoch in range(EPOCH):
        train(train_loader, net, criterion, optimizer, epoch, args)
        if rank == 0 and (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                }, 'checkpoint_{}.pth'.format(epoch))
        loss = test(test_loader, net, criterion, optimizer, args)
        if loss <= 0.000001:
            break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch DistributedDataParallel Training')
    parser.add_argument('--local_rank', default=0, type=int, help='local rank')
    parser.add_argument('--rank', default=0, type=int, help='rank')
    parser.add_argument('--local_world_size', default=1, type=int, help='local world size')
    parser.add_argument('--world_size', default=1, type=int, help='world size')
    parser.add_argument('--device', default='cpu', type=str, help='Device to run on, default to cpu')
    parser.add_argument('--ipex', action='store_true', help='with Intel(R) Extension for PyTorch*')
    parser.add_argument('--bf16', action='store_true', help='Train with BFloat16')
    parser.add_argument('--backend', default='gloo', type=str, help='DDP backend, default to gloo')
    parser.add_argument('--master_addr', default='127.0.0.1', type=str, help='Master Addr')
    parser.add_argument('--port', default='29500', type=str, help='Port')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    args = parser.parse_args()

    local_rank = args.local_rank
    rank = args.rank
    local_world_size = args.local_world_size
    world_size = args.world_size
    if is_hvd_enabled:
        hvd.init()
        if hvd.size() > 1:
            local_rank = hvd.local_rank()
            rank = hvd.rank()
            local_world_size = hvd.local_size()
            world_size = hvd.size()
        else:
            is_hvd_enabled = False
    if not is_hvd_enabled:
        if 'PMI_RANK' in os.environ and \
           'PMI_SIZE' in os.environ:
            rank = int(os.environ.get('PMI_RANK', args.local_rank))
            local_rank = rank
            world_size = int(os.environ.get('PMI_SIZE', args.world_size))
            local_world_size = world_size
        if 'LOCAL_RANK' in os.environ and \
           'RANK' in os.environ and \
           'LOCAL_WORLD_SIZE' in os.environ and \
           'WORLD_SIZE' in os.environ:
            local_rank = int(os.environ.get('LOCAL_RANK', args.local_rank))
            rank = int(os.environ.get('RANK', args.rank))
            local_world_size = int(os.environ.get('LOCAL_WORLD_SIZE', args.local_world_size))
            world_size = int(os.environ.get('WORLD_SIZE', args.world_size))
    args.local_rank = local_rank
    args.rank = rank
    args.local_world_size = local_world_size
    args.world_size = world_size
    print('local | global: {}/{} | {}/{}'.format(local_rank, local_world_size, rank, world_size))

    if args.device == 'cuda':
        args.backend = 'nccl'
        if args.local_world_size > 0:
            args.device = 'cuda:{}'.format(args.local_rank)
            # torch.cuda.set_device(args.local_rank)
    if args.backend == 'ccl':
        import torch_ccl
    if args.ipex:
        if args.device == 'cpu':
            try:
                import intel_extension_for_pytorch as ipex
                print('Successfully loaded intel_extension_for_pytorch')
            except Exception as e:
                print('Failed to load intel_extension_for_pytorch: {}'.format(e))
                args.ipex = False
        else:
            args.ipex = False
    print('Device: {}'.format(args.device))
    if args.world_size == 1:
        print('Train with single instance')
    else:
        print('Backend: {}'.format(args.backend))

    main(args)
