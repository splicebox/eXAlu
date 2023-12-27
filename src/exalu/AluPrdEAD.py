from six import with_metaclass
import torch
import numpy as np
import os
import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, confusion_matrix
import time
import random
import matplotlib.pyplot as plt

from .AluDatasetEAD import AluDatasetEAD
from .models.Model_CNN_v10_5 import CNet_v10_5
from .encoding import *
from .load_data import *


class AluPrdEAD(object):
    def __init__(self, datasets, model_name='cnet', num_epochs=140, lr=5e-4, run_mode=0):
        self.step_count_th = 10
        self.tb_loss_writer = SummaryWriter(comment='_loss', flush_secs=5)
        self.tb_val_writer = SummaryWriter(comment='_val', flush_secs=5)
        self.tb_test_writer = SummaryWriter(comment='_test', flush_secs=5)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.batch_size = 2048
        if model_name == 'cnet':
            self.model = CNet_v10_5(self.batch_size)
        # elif model_name == 'fcn':
        #     self.model = FCN_v1(self.batch_size)
        self.model = self.model.to(self.device)
        # self.warmup_epoch_th = 15
        self.warmup_epoch_th = 15
        self.optimizer_warmup = torch.optim.Adam(self.model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
        # self.optimizer_warmup = torch.optim.NAdam(self.model.parameters(), lr=lr) # lr 3e-4
        self.optimizer_main = torch.optim.SGD(self.model.parameters(), lr=lr)
        # self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=[0.8, 0.999], eps=0.1)
        # self.optimizer = torch.optim.AdamW(self.model.parameters())
        # self.optimizer = torch.optim.Adadelta(self.model.parameters())
        print(self.model)
        if run_mode == 0 or run_mode == 2:
            # mode 0: only train
            # mode 2: train+infer
            self.num_epochs = num_epochs
            train_dataset = AluDatasetEAD(datasets['train'])
            val_dataset =   AluDatasetEAD(datasets['val'])
            test_dataset =  AluDatasetEAD(datasets['test'])
            train2_dataset = AluDatasetEAD(datasets['train'])
            self.dataset_sizes = {'train': len(train_dataset), 'val': len(val_dataset), 'test': len(test_dataset), 'train2': len(train2_dataset)}
            self.dataloaders = {'train': DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True),
                                'val':   DataLoader(val_dataset,   batch_size=self.batch_size, shuffle=True),
                                'test':  DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=True),
                                'train2':DataLoader(train2_dataset,batch_size=self.batch_size, shuffle=False)}
            if run_mode == 2:
                infer_dataset = AluDatasetEAD(datasets['infer'])
                self.dataset_sizes['infer'] = len(infer_dataset)
                self.dataloaders['infer'] = DataLoader(infer_dataset, batch_size=self.batch_size, shuffle=False)
            # classification
            self.criterion = nn.BCEWithLogitsLoss(reduction='sum')
            # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=200, gamma=0.95)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_warmup, milestones=[120], gamma=0.2)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_main, milestones=[60, 120], gamma=0.2)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_main, milestones=[30, 40, 50, 60], gamma=0.2)
            # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[200], gamma=0.5)
            self.records = []
            self.losses = []
        elif run_mode in [1, 3, 4, 5, 6, 8]:
            # mode 1: infer (with label) pair with mode 0
            # mode 3: infer (with label) pair with mode 2, the infer set is from mode 2
            # mode 4: simple infer (without label)
            # mode 6: simple infer particularly for gencode
            # mode 8: simple infer (with label)
            infer_dataset = AluDatasetEAD(datasets['infer'])
            self.dataloaders = {'infer': DataLoader(infer_dataset, batch_size=self.batch_size, shuffle=False)}
            self.dataset_sizes = {'infer': len(infer_dataset)}
        elif run_mode == 7:
            # model 7: backward gradients
            test_dataset =  AluDatasetEAD(datasets['test'])
            self.dataloaders = {'test': DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)}
            self.dataset_sizes = {'test': len(test_dataset)}
            self.criterion = nn.BCEWithLogitsLoss()
        print(self.dataset_sizes)

    def compute_precision_recall_curve(self, prd_y, y, epoch, img_dir):
        precision_lst = []
        recall_lst = []
        records = {}
        for thr in range(0, 100, 5):
            thr = thr / 100.0
            prd_y_bool = torch.stack([torch.as_tensor(1) if v >= thr \
                                else torch.as_tensor(0) for v in prd_y])
            precision = precision_score(y, prd_y_bool)
            recall = recall_score(y, prd_y_bool)
            precision_lst.append(precision)
            recall_lst.append(recall)
            records[f'{thr}_precision'] = precision
            records[f'{thr}_recall'] = recall
        plt.figure()
        plt.scatter(recall_lst, precision_lst)
        plt.title(f'Epoch {epoch} precision-recall curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.savefig(os.path.join(img_dir, f'epoch_{epoch}.png'), dpi=300)
        plt.close()
        return records

    def compute_metirx(self, prd_y, y, name):
        # prd_y is a floating prob number
        # prd_y_bool is a 0/1 th by 0.5
        records = {}
        # Compute ROC curve and area the curve
        prd_y_bool = torch.stack([torch.as_tensor(1) if v >= 0.5 \
                                  else torch.as_tensor(0) for v in prd_y])
        fpr, tpr, thresholds = roc_curve(y, prd_y)
        roc_auc = auc(fpr, tpr)
        tn, fp, fn, tp = confusion_matrix(y, prd_y_bool).ravel()
        precision = precision_score(y, prd_y_bool)
        recall = recall_score(y, prd_y_bool)
        # recall2 = tp / (tp + fn)
        specificity = tn / (tn + fp)
        f1 = f1_score(y, prd_y_bool)
        # Compute the accuracy
        prd_corrects = 0
        prd_corrects += torch.sum(prd_y_bool == y)
        acc = prd_corrects.double() / y.shape[0]
        # add records and print
        records[f'F1_{name}'] = f1
        records[f'Precision_{name}'] = precision
        records[f'Recall_{name}'] = recall 
        records[f'Specificity_{name}'] = specificity
        records[f'AUC_{name}'] = roc_auc
        records[f'Accuracy_{name}'] = acc.item()
        print('{}\tF1: {:.4f}\tPrecision: {:.4f}\tRecall: {:.4f}\tSpecificity: {:.4f}\tAUC: {:.4f}\tAccuracy: {:4f}'.format(name, f1, precision, recall, specificity, roc_auc, acc.item()))
        return records 
    
    def check_gradient(self, dataset=None):
        self.model.train()
        # self.model.eval()
        grad_sum = torch.Tensor().to(self.device)
        for x, y, id_line in self.dataloaders['test']:
            # print(x.shape)
            # print(y.shape)
            x = x.to(self.device)
            y = y.to(self.device)
            x.requires_grad = True
            self.optimizer.zero_grad()
            prd_y = self.model(x)
            prd_y = torch.squeeze(prd_y)
            loss = self.criterion(prd_y, y)
            loss.backward()
            # batch_grad_sum = torch.sum(x.grad, dim=0)
            batch_grad_sum = x.grad
            torch.set_printoptions(profile="full")
            print(batch_grad_sum.shape)
            print(batch_grad_sum)
            print(x.grad.shape)
            print('<><><><><><><>')


    def evaluate(self, dataset='test'):
        self.model.eval()
        prd_y_test = torch.Tensor().to(self.device)
        y_test = torch.Tensor()
        id_line_lst = []
        for x, y, id_line in self.dataloaders[dataset]:
            id_line_lst += list(id_line)
            y_test = torch.cat((y_test, y), dim=0)
            x = x.to(self.device)
            with torch.set_grad_enabled(False):
                self.optimizer_warmup.zero_grad()
                # self.optimizer_main.zero_grad()
                prd_y = self.model(x)
                if prd_y.shape[0] > 1:
                    prd_y = torch.squeeze(prd_y)
                else:
                    prd_y = prd_y[:,0] 
                prd_y_test = torch.cat((prd_y_test, nn.Sigmoid()(prd_y)), dim=0)
        return prd_y_test.cpu(), y_test, id_line_lst

    def train(self, run_mode=0, work_dir=None, infer_set=None, model_dst_dir=None):
        # save train/test metrics
        since = time.time()
        best_model_wts = deepcopy(self.model.state_dict())
        best_stat = 0.0
        best_records = None
        perf_results = []
        for epoch in range(1, self.num_epochs + 1):
            if epoch <= self.warmup_epoch_th:
                self.optimizer = self.optimizer_warmup
            else:
                self.optimizer = self.optimizer_warmup
            print('Epoch {}/{}\tlr: {}'.format(epoch, self.num_epochs, self.optimizer.param_groups[0]['lr']))
            print('-' * 15)
            records_dict = {}
            prd_y_dict = {}
            y_dict = {}
            # prd_y_val = torch.Tensor().to(self.device)
            # y_val = torch.Tensor().to(self.device)
            display_loss_bce = defaultdict(float)
            display_loss_all = defaultdict(float)
            # train_loss = 0.0
            # val_loss = 0.0
            for phase in ['train', 'train2', 'val', 'test']:
                prd_y_dict[phase] = torch.Tensor()
                y_dict[phase] = torch.Tensor()
                step_count = 0
                if phase == 'train':
                    self.model.train()  # Set model to training mode
                else:
                    self.model.eval()   # Set model to evaluate mode
                # for x, y, id_line, ss_str, ss_score, l in self.dataloaders[phase]:
                for x, y, id_line in self.dataloaders[phase]:
                    step_count += 1
                    # print(x.shape)
                    # print(y.shape)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    # ss_str = ss_str.to(self.device)
                    # ss_score = ss_score.to(self.device, dtype=float)
                    # l = l.to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        # prd_y = self.model(x, ss_str, ss_score, l)
                        prd_y = self.model(x)
                        prd_y = torch.squeeze(prd_y)
                        loss = self.criterion(prd_y, y)
                        display_loss_bce[phase] += loss
                        l1_lambda, l2_lambda = 1e-3, 1e-2
                        l1_lambda, l2_lambda = 1e-3, 0
                        l1_norm = sum(p.abs().sum() for p in self.model.parameters())
                        l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                        loss = loss + l1_norm * l1_lambda + l2_norm * l2_lambda
                        display_loss_all[phase] += loss
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                        else:
                            prd_y_dict[phase] = torch.cat((prd_y_dict[phase], nn.Sigmoid()(prd_y).cpu()), dim=0)
                            y_dict[phase] = torch.cat((y_dict[phase], y.cpu()), dim=0)
                display_loss_bce[phase] /= self.dataset_sizes[phase]
                display_loss_all[phase] /= self.dataset_sizes[phase]
                if phase == 'train':
                    self.scheduler.step()
                if phase != 'train':
                    records = self.compute_metirx(prd_y_dict[phase], y_dict[phase], phase)
                    records_dict[phase] = records
                    if phase == 'test':
                        img_dir = os.path.join(model_dst_dir, 'precision_recall_imgs')
                        os.makedirs(os.path.join(model_dst_dir, 'precision_recall_imgs'), exist_ok=True)
                        precision_recall_records = self.compute_precision_recall_curve(prd_y_dict[phase], y_dict[phase], epoch, img_dir)
                        records_dict['precision_recall'] = precision_recall_records
                    if phase == 'val' and records['F1_val'] > best_stat:
                        # model selection
                        best_records = records
                        best_stat = records['F1_val']
                        best_model_wts = deepcopy(self.model.state_dict())
            # output loss to the tensorboard
            self.tb_loss_writer.add_scalars('loss_bce', display_loss_bce, epoch)
            self.tb_loss_writer.add_scalars('loss_all', display_loss_all, epoch)
            print('bce train loss: {:.8f}\ttrain2 loss: {:.8f}\tval loss: {:.8f}\ttest loss: {:.8f}'.format(display_loss_bce['train'], display_loss_bce['train2'], display_loss_bce['val'], display_loss_bce['test']))
            print('all train loss: {:.8f}\ttrain2 loss: {:.8f}\tval loss: {:.8f}\ttest loss: {:.8f}'.format(display_loss_all['train'], display_loss_all['train2'], display_loss_all['val'], display_loss_all['test']))
            # save each epoch's wts
            if model_dst_dir != None: 
                torch.save(self.model.state_dict(), os.path.join(model_dst_dir, f'epoch_{epoch}.pt'))
            # write other metric
            for evaluation_metric in ['F1', 'Precision', 'Recall', 'Specificity', 'AUC', 'Accuracy']:
                self.tb_val_writer.add_scalar(evaluation_metric, records_dict['val'][f'{evaluation_metric}_val'], epoch)
                self.tb_test_writer.add_scalar(evaluation_metric, records_dict['test'][f'{evaluation_metric}_test'], epoch)
            # run infer
            if run_mode == 2:
                record_loss = {}
                for k, v in display_loss_bce.items():
                    record_loss[k + '_loss'] = v.item()
                perf_results.append(records_dict['train2'] | records_dict['val'] | records_dict['test'] | records_dict['precision_recall'] | record_loss)
                # print(perf_results)
                continue    
                prd_y_infer, y_infer, id_line_infer = self.evaluate('infer')
                prd_y_infer = prd_y_infer.tolist()
                y_infer = y_infer.tolist()
                assert(len(prd_y_infer) == len(id_line_infer) == len(y_infer))
                prd_file = work_dir + '/prd_y.txt'
                with open(prd_file, 'w') as write_fh:
                    for i in range(len(id_line_infer)):
                        write_fh.write(f'{prd_y_infer[i]}\t{y_infer[i]}\t{id_line_infer[i]}' + '\n')
                infer_records = draw('infer', epoch, dataset=infer_set, work_dir=work_dir,log_fh=None)
                perf_results.append(records_dict['val'] | records_dict['test'] | infer_records)
            # else:
            print('')
        # write to csv
        import csv 
        perf_keys = perf_results[0].keys()
        with open(os.path.join(model_dst_dir, 'performance.csv'), 'w', newline='') as csv_fh:
            dict_writer = csv.DictWriter(csv_fh, perf_keys, delimiter='\t')
            dict_writer.writeheader()
            dict_writer.writerows(perf_results)
        # load best model weights
        self.model.load_state_dict(best_model_wts)
        # self.records.append(self.evaluate())
        self.tb_val_writer.flush()
        self.tb_test_writer.close()

        return best_model_wts, self.records