import torch
from torch.utils.data import Dataset, DataLoader
from torch import from_numpy, as_tensor


class AluDatasetEAD(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        '''
        load data format
        result_dataset.append([d[2], ENCODE_FN(d[1]), flag])

        id_line, seq_one_hot, flag
        '''
        # return self.dataset[idx][1], self.dataset[idx][2], self.dataset[idx][0], self.dataset[idx][3], self.dataset[idx][4], self.dataset[idx][5]
        return self.dataset[idx][1], self.dataset[idx][2], self.dataset[idx][0]

def prepare_data_for_model(dataset_lst, cv=0):
    dataset = {'train': [], 'val': []}
    dataset['val'] = dataset_lst[cv]
    for idx, ds in enumerate(dataset_lst):
        if idx == cv:
            continue
        dataset['train'] += ds
    # # dataset only contains train and val dataset, exclude eval dataset
    # dataset = {'train': [], 'val': []}
    # # train val csv->df->dataset
    # df_train_val = pd.read_csv(work_dir + train_val_file, sep='\t')
    # df_train_val = df_train_val[(df_train_val['mhc'] == allele) & \
    #                     (df_train_val['peptide_length'] == pep_len)]
    # if df_train_val.empty:
    #     return 1
    # # format species mhc peptide_length cv sequence inequality meas
    # train_binder_count = 0
    # for idx, row in df_train_val.iterrows():
    #     if row['meas'] <= 500:
    #         train_binder_count += 1
        # pep_ary_tensor = torch.from_numpy(pep_encoding(row['sequence'])).float()
        # pep_ba  = torch.as_tensor(ic50_to_ba(row['meas'])).float()
    #     # if row['cv'] == 1:
    #         # for now, not used, only a placeholder
    #     dataset['val'].append((pep_ary_tensor, pep_ba))
    #     # else: # for now, train dataset have all data from train_val data file
    #     dataset['train'].append((pep_ary_tensor, pep_ba))
    # if train_binder_count == 0:
    #     return 2
    # eval_binder_count = 0
    # # evaluate csv->df->dataset
    # df_eval = pd.read_csv(work_dir + eval_file, sep='\t')
    # df_eval = df_eval[(df_eval['mhc'] == allele) & (df_eval['peptide_length'] == pep_len)]
    # if df_eval.empty:
    #     return 3
    # eval_x, eval_y = [], []
    # # format species mhc peptide_length cv sequence inequality meas
    # for idx, row in df_eval.iterrows():
    #     if row['meas'] <= 500:
    #         eval_binder_count += 1
    #     pep_ary_tensor = torch.from_numpy(pep_encoding(row['sequence'])).float()
    #     pep_ba  = torch.as_tensor(ic50_to_ba(row['meas'])).float()
    #     eval_x.append(pep_ary_tensor)
    #     eval_y.append(pep_ba)
    # if eval_binder_count == 0:
    #     return 4
    # self.records['TrainSize'] = self.dataset_sizes['train']
    # self.records['TrainBinder'] = train_binder_count
    # self.records['EvaluateSize'] = self.eval_dataset_size
    # self.records['EvaluateBinder'] = eval_binder_count
    # print(self.records)
    # return 0