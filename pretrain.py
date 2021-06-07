import torch
import torch.nn as nn
import torch.optim as optim
import os

from data.dataloader import load_data
from data.ClassAwareSampler import get_sampler
from models.model import GIST


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

data_root = '/change/to/your/path'
data_root = '/gpu7_ssd/liubo/ImageNet'
sampler_dic = {'type': 'ClassAwareSampler', 'def_file': './data/ClassAwareSampler.py', 'num_samples_cls': 4, 'sampler':get_sampler()}
train_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=128, sampler_dic=sampler_dic, num_workers=16, test_open=False, shuffle=True)
train_plain_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=128, sampler_dic=None, num_workers=16, test_open=False, shuffle=True)

device = 'cuda:0'

model = GIST(n_struct=0)
model = nn.DataParallel(model)
model.to(device)

loss_lambda = 0.5
n_epoch = 60
ce_loss = nn.CrossEntropyLoss()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)

enum_train_loader = enumerate(train_loader, 0)
for epoch in range(n_epoch):
    model.train()
    loss_all = 0.0
    for i, (images, labels, paths) in enumerate(train_plain_loader, 0):
        images, labels = images.to(device), labels.to(device)
        j, (images_CB, labels_CB, paths_CB) = next(enum_train_loader)
        if j == len(train_loader) - 1:
            enum_train_loader = enumerate(train_loader)
        images_CB, labels_CB = images_CB.to(device), labels_CB.to(device)
        M = images_CB.size(0)
        images_all = torch.cat((images_CB, images), dim=0)
        output_c, output_r = model(images_all)
        output_c = output_c[:M, :]
        output_r = output_r[M:, :]
        loss1 = ce_loss(output_r, labels)
        loss2 = ce_loss(output_c, labels_CB)

        loss = loss1 + loss_lambda * loss2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_all += loss.item()
    scheduler.step(epoch)
    loss_all /= len(train_plain_loader)
    print('Epoch[{}, {}] loss: {:.4f}'.format(epoch, n_epoch, loss_all))

torch.save(model.state_dict(), 'pretrain.pth')
