import os
import time
import torch
import pickle
from .AluPrdEAD import AluPrdEAD
from .load_data import *


def run_ead(strand, model_name='cnet',
            context_mode='none', single_side_pad_len=0,
            run_mode=0, work_dir=None, data_dir=None, infer_set=None,
            infer_file=None, prd_file='prd_y.txt',
            model_wts_name=None,
            run_comment=None,
            model_dst_dir=None,
            dataset_save_file=None,
            test_chrs=['chr2', 'chr5'],
            infer_bed_file=None,
            gencode_neg_dataset_file=None):
    # infer_file is a fasta file
    # while infer_bed_file is a bed file
    since = time.time()
    print('working dir', work_dir)
    # save path config
    if run_mode in [0, 1, 2, 4, 5, 6, 8]:
        if context_mode == 'bothfix':
            max_seq_len = 350 + single_side_pad_len * 2
        else:
            max_seq_len = 0 # not support yet
        version_name = f'ead_v1.4t_{infer_set}'
        dataset_dst_dir = os.path.join(work_dir, f'dataset_pickles')
        dataset_save_file = os.path.join(dataset_dst_dir, f'{version_name}_runmode_{run_mode}_padding_{context_mode}_{single_side_pad_len}.pickle') # no dc here
        print('dataset save file', dataset_save_file)
        if run_comment:
            save_name_prefix = f'{version_name}_runmode_{run_mode}_padding_{context_mode}_{single_side_pad_len}_dc_{run_comment}'
        else:
            save_name_prefix = f'{version_name}_runmode_{run_mode}_padding_{context_mode}_{single_side_pad_len}'
        time_str = time.strftime('%Y%m%d_%H%M%S')
        timed_save_name_prefix = f'{save_name_prefix}_{model_name}_{time_str}'
        print('timed save name prefix', timed_save_name_prefix)
        if run_mode in [0, 2]:
            model_dst_dir = os.path.join(work_dir, 'model_pts', timed_save_name_prefix)
            os.makedirs(model_dst_dir, exist_ok='True')

    print('dataset pickle file:', dataset_save_file)

    if run_mode in [4, 5, 6, 8]:
        # mode 4
        # just the simplist infer
        if run_mode == 4:
            datasets = load_data_ead_simpleinfer(infer_file, max_seq_len=max_seq_len)
            os.makedirs(dataset_dst_dir, exist_ok='True')
            # protect dataset; disable dump 
            with open(dataset_save_file, 'wb') as fh:
                pickle.dump(datasets, fh)
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        if torch.cuda.is_available():
            model_wts = torch.load(model_wts_name)
        else:
            model_wts = torch.load(model_wts_name, map_location=torch.device('cpu'))
        alu_prd.model.load_state_dict(model_wts)
        prd_y, y, id_line= alu_prd.evaluate('infer')
        prd_y = prd_y.tolist()
        y = y.tolist()
        assert(len(prd_y) == len(y) == len(id_line))
        with open(work_dir + '/' + prd_file, 'w') as write_fh:
            for i in range(len(id_line)):
                write_fh.write(f'{prd_y[i]}\t{y[i]}\t{id_line[i]}' + '\n')
    # time stamp
    time_elapsed = time.time() - since
    print('This run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return dataset_save_file, model_dst_dir
