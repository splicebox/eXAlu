import os
import time
import torch
import pickle
import re
from .AluPrdEAD import AluPrdEAD
from .load_data import *
from .data_preprocess.curate_data import curate_data


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
    # only run mode 6 requires infer_bed_file
    since = time.time()
    print('working dir', work_dir)
    print('test_chrs', test_chrs)
    # save path config
    if run_mode in [0, 1, 2, 4, 5, 6, 8]:
        # run_mode 0, 2 will use their own dataset/model
        # run_mode 1 will only use 0's model, however, it will use its own dataset
        # run_mode 3 will use 2's dataset/model
        if context_mode == 'bothfix':
            max_seq_len = 350 + single_side_pad_len * 2
        else:
            max_seq_len = 0 # not support yet
        print(infer_set)
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
    if run_mode == 0:
        # mode 0: only train
        neg_alu_fa = os.path.join(work_dir, 'neg_alu.fa')
        pos_alu_fa_file = os.path.join(
            work_dir, 'whole_pos_alu.fa')
        # load datasets
        if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        else:
            print('dataset pickle file DOES NOT exist')
            curate_data(infer_set=infer_set, strand=strand, mode=context_mode, single_side_pad_len=single_side_pad_len, work_dir=work_dir, data_dir=data_dir)
            # datasets = load_data_ead_alu_chr_woinfer(alu_file=pos_alu_fa_file, neg_file=neg_alu_fa)
            datasets = load_data_ead_alu_chr_train_unique_duplicate(alu_file=pos_alu_fa_file, neg_file=neg_alu_fa, duplicate_times=10, test_chrs=test_chrs, max_seq_len=max_seq_len)
            os.makedirs(dataset_dst_dir, exist_ok='True')
            with open(dataset_save_file, 'wb') as fh:
                pickle.dump(datasets, fh)
        # train
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        best_model_wts, model_records = alu_prd.train(run_mode, work_dir, infer_set, model_dst_dir)
        # save model
        torch.save(best_model_wts, os.path.join(model_dst_dir, 'best.pt'))
    if run_mode == 1:
        # mode 1: infer, use with mode 0, don't consider duplicates
        # the infer set is read by this mode self
        neg_alu_fa = os.path.join(work_dir, 'neg_alu.fa')
        if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        else:
            print('dataset pickle file DOES NOT exist')
            datasets = load_data_ead_alu_chr_inferonly(strand, None, work_dir, infer_set, max_seq_len=max_seq_len)
            os.makedirs(dataset_dst_dir, exist_ok='True')
            with open(dataset_save_file, 'wb') as fh:
                pickle.dump(datasets, fh)
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        model_wts = torch.load(os.path.join(
            model_dst_dir, 'best.pt'))
        alu_prd.model.load_state_dict(model_wts)
        prd_y, y, id_line= alu_prd.evaluate('infer')
        prd_y = prd_y.tolist()
        y = y.tolist()
        assert(len(prd_y) == len(y) == len(id_line))
        with open(work_dir + '/' + prd_file, 'w') as write_fh:
            for i in range(len(id_line)):
                write_fh.write(f'{prd_y[i]}\t{y[i]}\t{id_line[i]}' + '\n')
    if run_mode == 2:
        # mode 2: train and infer every epoch
        neg_alu_fa = os.path.join(work_dir, 'neg_alu.fa')
        pos_alu_fa_file = os.path.join(
            work_dir, 'whole_pos_alu.fa')
        # neg_alu_fa = '/home/zhe29/Projects/eXAlu/data/curated_data_MOAT/dataset_structure/neg_alu_300_rnafold.out'
        # pos_alu_fa_file = '/home/zhe29/Projects/eXAlu/data/curated_data_MOAT/dataset_structure/pos_alu_300_rnafold.out'
        # load datasets
        if 0:
        # if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        else:
            print('dataset pickle file DOES NOT exist')
            curate_data(infer_set=infer_set, strand=strand, mode=context_mode, single_side_pad_len=single_side_pad_len, work_dir=work_dir, data_dir=data_dir)
            os.makedirs(dataset_dst_dir, exist_ok='True')
            datasets = load_data_ead_alu_chr_withinfer2(
                                            strand=strand,
                                            alu_file=pos_alu_fa_file,
                                            neg_file=neg_alu_fa,
                                            work_dir=work_dir,
                                            infer_set=infer_set,
                                            test_chrs=test_chrs,
                                            duplicate_times=10,
                                            max_seq_len=max_seq_len
                                            )
            with open(dataset_save_file, 'wb') as fh:
                pickle.dump(datasets, fh)
        # train
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        best_model_wts, model_records = alu_prd.train(run_mode, work_dir, infer_set, model_dst_dir)
        # save model
        torch.save(best_model_wts, os.path.join(
            model_dst_dir, 'best.pt'))
    if run_mode == 3:
        # mode 3: infer, use with mode 2, consider duplicates, the infer set is from mode 2
        # load datasets
        if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        model_wts = torch.load(os.path.join(
            model_dst_dir, 'best.pt'))
        alu_prd.model.load_state_dict(model_wts)
        prd_y, y, id_line= alu_prd.evaluate('infer')
        prd_y = prd_y.tolist()
        y = y.tolist()
        assert(len(prd_y) == len(y) == len(id_line))
        with open(work_dir + '/' + prd_file, 'w') as write_fh:
            for i in range(len(id_line)):
                write_fh.write(f'{prd_y[i]}\t{y[i]}\t{id_line[i]}' + '\n')

    if run_mode in [4, 5, 6, 8]:
        # mode 4
        # just the simplist infer
        # mode 5
        # just the simplist infer, but will remove the duplicates
        # mode 6
        # particularly for gencode analysis, due to three-label issue
        # mode 8
        # infer with label, e.g. to infer on test set
        # load datasets
        if 0:
        # if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        else:
            print('dataset pickle file DOES NOT exist')
            if run_mode == 4:
                datasets = load_data_ead_simpleinfer(infer_file, max_seq_len=max_seq_len)
            elif run_mode == 5:
                datasets = load_data_ead_simpleinfer_rmdup(infer_file, max_seq_len=max_seq_len)
            elif run_mode == 8:
                datasets = load_data_ead_simpleinfer_flag(infer_file, max_seq_len=max_seq_len)
            elif run_mode == 6:
                datasets = load_data_ead_infer_gencode(strand=strand,
                                                       infer_fa_file=infer_file, 
                                                       infer_bed_file=infer_bed_file,
                                                       infer_unique=True,
                                                       max_seq_len=max_seq_len)
                # load training datasets for neg data
                with open(gencode_neg_dataset_file, 'rb') as fh:
                    # this neg datasets is the training set
                    # we only use the neg data in this analysis
                    training_datasets = pickle.load(fh)
                    neg_dataset = training_datasets['train'][int(len(training_datasets['train'])/2):]
                    random.shuffle(neg_dataset)
                    infer_set_len = len(datasets['infer'])
                    for i, d in enumerate(neg_dataset):
                        if i >= infer_set_len:
                            break
                        datasets['infer'].append(d)
                    print('read neg data from training set.. done...')
            os.makedirs(dataset_dst_dir, exist_ok='True')
            # protect dataset; disable dump 
            with open(dataset_save_file, 'wb') as fh:
                pickle.dump(datasets, fh)
        alu_prd = AluPrdEAD(datasets, model_name=model_name, run_mode=run_mode)
        print(model_wts_name)
        if torch.cuda.is_available():
            model_wts = torch.load(model_wts_name)
        else:
            model_wts = torch.load(model_wts_name, map_location=torch.device('cpu'))
        alu_prd.model.load_state_dict(model_wts)
        prd_y, y, id_line= alu_prd.evaluate('infer')
        prd_y = prd_y.tolist()
        y = y.tolist()
        assert(len(prd_y) == len(y) == len(id_line))
        if run_mode == 8:
            with open(work_dir + '/' + prd_file, 'w') as write_fh:
                for i in range(len(id_line)):
                    # id_line[i] = h38_mk_AluJb_855_229_817_bothfix_0_0_NA::chr9:34216640-34216919(-)
                    # bed: chr10	15102160	15102520	h38_mk_AluJb_11_310_11_bothfix_0_0_NA	0	-
                    alu_name = id_line[i].split('::')[0]
                    _id = id_line[i].split('::')[-1]
                    id_lst = re.split('[:()]', _id)
                    id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]
                    if prd_y[i] >= 0.5:
                        if y[i] == 1.0:
                            cm = 'TP'
                        else:
                            cm = 'FP'
                    else:
                        if y[i] == 1.0:
                            cm = 'FN'
                        else:
                            cm = 'TN'
                    write_fh.write(f'{id_lst[0]}\t{id_lst[1]}\t{id_lst[2]}\t{cm}::{prd_y[i]}::{y[i]}::{alu_name}\t0\t{id_lst[3]}' + '\n')
        else:
            with open(work_dir + '/' + prd_file, 'w') as write_fh:
                for i in range(len(id_line)):
                    write_fh.write(f'{prd_y[i]}\t{y[i]}\t{id_line[i]}' + '\n')
    
    if run_mode == 7:
        # backward calculate gradients
        # load datasets
        if os.path.isfile(dataset_save_file):
            print('dataset pickle file exists')
            with open(dataset_save_file, 'rb') as fh:
                datasets = pickle.load(fh)
        alu_prd = AluPrdEAD(datasets=datasets, model_name=model_name, run_mode=run_mode)
        model_wts = torch.load(os.path.join(
            model_dst_dir, 'best.pt'))
        alu_prd.model.load_state_dict(model_wts)
        alu_prd.check_gradient()
    # time stamp
    time_elapsed = time.time() - since
    print('This run complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    
    return dataset_save_file, model_dst_dir
