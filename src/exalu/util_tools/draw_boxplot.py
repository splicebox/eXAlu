import os
from scipy.sparse import data
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score

def compute_metirx(prd_y, prd_y_bin, y, log_fh=None):
    records = {}
    # Compute ROC curve and area the curve

    fpr, tpr, thresholds = roc_curve(y, prd_y)
    roc_auc = auc(fpr, tpr)
    records['AUC_infer'] = roc_auc
    precision = precision_score(y, prd_y_bin)
    recall = recall_score(y, prd_y_bin)
    f1 = f1_score(y, prd_y_bin)
    records['F1_infer'] = f1
    # Compute the accuracy
    prd_corrects = 0
    prd_corrects += np.sum(prd_y_bin == y)
    # print('y.shape', y.shape)
    acc = float(prd_corrects) / y.shape[0]
    records['Accuracy_infer'] = acc
    print('infer\tPrecision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}\tAUC: {:.4f}\tAccuracy: {:4f}'.format(precision, recall, f1, roc_auc, acc))
    if log_fh != None:
        log_fh.write('infer\tPrecision: {:.4f}\tRecall: {:.4f}\tF1: {:.4f}\tAUC: {:.4f}\tAccuracy: {:4f}'.format(precision, recall, f1, roc_auc, acc) + '\n')
    return records 

def draw(name, epoch, dataset=None, work_dir=None, log_fh=None, label_file='labels.txt', prd_file='prd_y.txt'):
    id_lst = []
    label_lst = []
    prd_y_lst = []
    with open(os.path.join(work_dir, label_file), 'r') as label_fh,\
         open(os.path.join(work_dir, prd_file), 'r') as prd_y_fh:
        for line in label_fh.readlines():
            line_lst = line.rstrip().split('\t')
            if len(line_lst) == 0:
                break
            label_lst.append(int(float(line_lst[1])))
        for line in prd_y_fh.readlines():
            line_lst = line.rstrip().split('\t')
            if len(line_lst) == 0:
                break
            prd_y_lst.append(float(line_lst[0]))    
            id_lst.append(line_lst[2])
    save_df = pd.DataFrame({'ID': id_lst, 'Class': label_lst, 'Score': prd_y_lst})
    print(save_df)
    save_df.to_csv(open(work_dir + '/prd_records.txt', 'w'), sep='\t', index=None)

    df = pd.DataFrame({'Class': label_lst, 'Score': prd_y_lst})
    # df.loc[df['Class'] == 0, 'Class_bin'] = 0
    # if dataset == 'MOAT':
        # df.loc[df['Class'] == 1, 'Class_bin'] = 1
    # elif dataset == 'OtherSpecies' or dataset == 'Gencode':
        # df.loc[df['Class'] == 2, 'Class_bin'] = 1 
    # df.loc[df['Score'] >= 0.5, 'Prd_bin'] = 1
    # df.loc[df['Score'] < 0.5, 'Prd_bin'] = 0
    # df = df[df['Class_bin'].notna()]
    # records = compute_metirx(df['Score'], df['Prd_bin'], df['Class_bin'], log_fh=log_fh)
    print(df)
    imgs_dir = os.path.join(work_dir, 'imgs')
    os.makedirs(imgs_dir, exist_ok=True)
    ax = sns.histplot(x='Score', hue='Class', data=df, palette='Set1', bins=20, common_norm=False, stat='probability', multiple='dodge')
    # plt.savefig(os.getcwd() + f'/data/{dataset}/imgs/hist_{name}_e{epoch}.png')
    plt.savefig(os.path.join(imgs_dir, f'hist_{name}_e{epoch}.png'))
    plt.close()

    ax = sns.boxplot(x='Class', y='Score', data=df, palette='Set1')
    # plt.savefig(os.getcwd() + f'/data/{dataset}/imgs/box_{name}_e{epoch}.png')
    plt.savefig(os.path.join(imgs_dir, f'box_{name}_e{epoch}.png'))
    plt.close()
    # ax = sns.kdeplot(x="Score", hue="Class",data=df, cumulative=True, palette=['r','g','b'])
    ax = sns.kdeplot(x='Score', hue='Class',data=df, palette='Set1', common_norm=False)
    # plt.savefig(os.getcwd() + f'/data/{dataset}/imgs/kde_{name}_e{epoch}.png')
    plt.savefig(os.path.join(imgs_dir, f'kde_{name}_e{epoch}.png'))
    plt.close()
    return
    # return records

# def draw_gencode(name, epoch):
#     id_lst = []
#     label_lst = []
#     prd_y_lst = []
#     with open(os.getcwd() + '/data/Gencode/labels.txt', 'r') as label_fh,\
#          open(os.getcwd() + '/data/Gencode/prd_y.txt',  'r') as prd_y_fh:
#         for line in label_fh.readlines():
#             line_lst = line.rstrip().split('\t')
#             if len(line_lst) == 0:
#                 break
#             id_lst.append(line_lst[0])
#             label_lst.append(int(line_lst[1]))
#         for line in prd_y_fh.readlines():
#             line = line.rstrip()
#             if line == '':
#                 break
#             prd_y_lst.append(float(line))    
#     # print(len(label_lst))
#     # print(len(prd_y_lst))
#     save_df = pd.DataFrame({'ID': id_lst, 'Class': label_lst, 'Score': prd_y_lst})
#     save_df.to_csv(open(os.getcwd() + '/data/Gencode/prd_records.txt',  'w'), sep='\t', index=None)

#     df = pd.DataFrame({'Class': label_lst, 'Score': prd_y_lst})
#     df.loc[df['Class'] == 0, 'Class_bin'] = 0
#     df.loc[df['Class'] == 2, 'Class_bin'] = 1
#     df.loc[df['Score'] >= 0.5, 'Prd_bin'] = 1
#     df.loc[df['Score'] < 0.5, 'Prd_bin'] = 0
#     df = df[df['Class_bin'].notna()]
#     records = compute_metirx(df['Score'], df['Prd_bin'], df['Class_bin'])
#     # print(df)
#     # tips = sns.load_dataset("tips")
#     # g = sns.PairGrid(data=df)
#     # g.map_diag(sns.histplot(x='Score', hue='Class', data=df, palette='Set1', bins=20, common_norm=False))
#     # g.map_diag(sns.boxplot(x='Class', y='Score', data=df, palette='Set1'))
#     # g.map_diag(sns.kdeplot(x='Score', hue='Class',data=df, palette='Set1', common_norm=False))
#     ax = sns.histplot(x='Score', hue='Class', data=df, palette='Set1', bins=20, common_norm=False, stat='probability', multiple='dodge')
#     plt.savefig(os.getcwd() + '/data/Gencode/imgs/hist_{}_e{}.png'.format(name, epoch))
#     plt.close()

#     ax = sns.boxplot(x='Class', y='Score', data=df, palette='Set1')
#     plt.savefig(os.getcwd() + '/data/Gencode/imgs/box_{}_e{}.png'.format(name, epoch))
#     plt.close()
#     # ax = sns.kdeplot(x="Score", hue="Class",data=df, cumulative=True, palette=['r','g','b'])
#     ax = sns.kdeplot(x='Score', hue='Class',data=df, palette='Set1', common_norm=False)
#     plt.savefig(os.getcwd() + '/data/Gencode/imgs/kde_{}_e{}.png'.format(name, epoch))
#     plt.close()
#     return records


if __name__ == '__main__':
    pass