from torch.utils.data import Dataset


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
        return self.dataset[idx][1], self.dataset[idx][2], self.dataset[idx][0]

def prepare_data_for_model(dataset_lst, cv=0):
    dataset = {'train': [], 'val': []}
    dataset['val'] = dataset_lst[cv]
    for idx, ds in enumerate(dataset_lst):
        if idx == cv:
            continue
        dataset['train'] += ds