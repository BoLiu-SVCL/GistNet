import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from data.dataloader import load_data
from utils import mic_acc_cal, shot_acc
from models.model import GIST


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

data_root = '/change/to/your/path'
data_root = '/gpu7_ssd/liubo/ImageNet'
train_plain_loader = load_data(data_root, dataset='ImageNet_LT', phase='train', batch_size=128, sampler_dic=None, num_workers=16, test_open=False, shuffle=True)
test_loader = load_data(data_root, dataset='ImageNet_LT', phase='test', batch_size=256, sampler_dic=None, num_workers=16, test_open=False, shuffle=False)

device = 'cuda:0'

model = GIST(n_struct=4)
model = nn.DataParallel(model)
weights = torch.load('GIST.pth')
model.load_state_dict(weights)
model.to(device)

model.eval()

total_logits = torch.empty((0, 1000)).to(device)
total_labels = torch.empty(0, dtype=torch.long).to(device)

with torch.no_grad():
    for i, (images, labels, paths) in enumerate(test_loader, 0):
        images, labels = images.to(device), labels.to(device)
        logits, _ = model(images)

        total_logits = torch.cat((total_logits, logits))
        total_labels = torch.cat((total_labels, labels))

probs, preds = F.softmax(total_logits.detach(), dim=1).max(dim=1)

eval_acc_mic_top1 = mic_acc_cal(preds[total_labels != -1], total_labels[total_labels != -1])
many_acc_top1, median_acc_top1, low_acc_top1 = shot_acc(preds[total_labels != -1], total_labels[total_labels != -1], train_plain_loader)
# Top-1 accuracy and additional string
print_str = ['Evaluation_accuracy_micro_top1: %.3f'
             % (eval_acc_mic_top1),
             '\n',
             'Many_shot_accuracy_top1: %.3f'
             % (many_acc_top1),
             'Median_shot_accuracy_top1: %.3f'
             % (median_acc_top1),
             'Low_shot_accuracy_top1: %.3f'
             % (low_acc_top1),
             '\n']
print(print_str)
