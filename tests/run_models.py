from models.mobilenet import mobilenet_v2
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from torch import nn

import math

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()
    return model

def train(train_loader, model, criterion, optimizer, epoch):
    loss = AverageMeter()
    acc = AverageMeter()

    model.train()

    for i, sample_batched in enumerate(train_loader):
        input, target = sample_batched[0], sample_batched[1]

        input = input.cuda()
        target = target.cuda()

        output = model(input)
        ep_loss = criterion(output, target)

        prec1 = accuracy(output.data, target)

        loss.update(ep_loss.item(), input.size(0))
        acc.update(prec1[0], input.size(0))

        optimizer.zero_grad()
        ep_loss.backward()
        optimizer.step()

    print('Epoch: [{0}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {acc.val:.3f} ({acc.avg:.3f})'.format(
                   epoch, loss=loss, acc=acc))


model = mobilenet_v2(pretrained=True, num_classes=2)
model = initialize_weights(model).cuda()
ds = ImageFolder(root='/home/stasysp/Envs/Datasets/debug_classification/small',
                 transform=transforms.Compose([
                     transforms.Resize((224, 224)),
                     transforms.ToTensor()])
                 )
train_loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=8, pin_memory=True)
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

for epoch in range(10):
        train(train_loader, model, criterion, optimizer, epoch)
