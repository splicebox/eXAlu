import time
from collections import defaultdict
import os
from .encoding import *

# TODO
data_dir = os.getcwd() + '/data'

# ENCODE_FN = seq_encoding_onehot
ENCODE_FN = seq_encoding_onehot_padN


def get_dataset_from_seq_ead(dataset, flag, max_seq_len):
    flag = torch.tensor(flag)
    result_dataset = []
    for d in dataset:
        # d: (_id, seq_line, id_line[1:])
        # or (seq_line, id_line[1:])
        result_dataset.append([d[-1], ENCODE_FN(d[-2], max_seq_len), flag])
    return result_dataset

def get_dataset_from_seq_ead_flag_variable(dataset, flag_lst, max_seq_len):
    result_dataset = []
    for d, f in zip(dataset, flag_lst):
        # d: (_id, seq_line, id_line[1:])
        # or (seq_line, id_line[1:])
        flag = torch.tensor(f)
        result_dataset.append([d[-1], ENCODE_FN(d[-2], max_seq_len), flag])
    return result_dataset

def load_data_ead_simpleinfer(fa_file, max_seq_len):
    datasets = defaultdict(list)
    st = time.time()
    print('loading... simpleinfer data')
    # infer_dict = defaultdict(list)
    infer_lst = []
    with open(fa_file, 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            # id_line: >h38_mk_AluJb_5_bothfix_0_0_NA::chr1:192937823-192938210(+)::BASELINE
            # _id is useless here
            # _id = id_line.split('::')[1]
            # chr = _id.split(':')[0]
            # assert(chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            # infer_dict[chr].append((seq_line, id_line[1:]))
            infer_lst.append((seq_line, id_line[1:]))
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    # for chr in infer_dict.keys():
    datasets['infer'] = get_dataset_from_seq_ead_fn(infer_lst, 3., max_seq_len)
    print('load_data_ead_simpleinfer time usage: {}'.format(time.time() - st))
    return datasets

def load_data_ead_simpleinfer_flag(fa_file, max_seq_len):
    datasets = defaultdict(list)
    st = time.time()
    print('loading... simpleinfer data')
    # infer_dict = defaultdict(list)
    infer_lst = []
    flag_lst = []
    with open(fa_file, 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            # id_line: >h38_mk_AluJb_5_bothfix_0_0_NA::chr1:192937823-192938210(+)::BASELINE
            # _id is useless here
            # _id = id_line.split('::')[1]
            # chr = _id.split(':')[0]
            # assert(chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            # infer_dict[chr].append((seq_line, id_line[1:]))
            infer_lst.append((seq_line, id_line[1:]))
            c = id_line[1:].split('_')[0]
            flag_lst.append(float(c))
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead_flag_variable
    # for chr in infer_dict.keys():
    datasets['infer'] = get_dataset_from_seq_ead_fn(infer_lst, flag_lst, max_seq_len)
    print('load_data_ead_simpleinfer time usage: {}'.format(time.time() - st))
    return datasets


def load_data_ead_simpleinfer_rmdup(fa_file, max_seq_len):
    datasets = defaultdict(list)
    st = time.time()
    print('loading... simpleinfer data')
    infer_dict = defaultdict(list)
    _id_set = set()
    with open(fa_file, 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            # id_line: >h38_mk_AluJb_5_bothfix_0_0_NA::chr1:192937823-192938210(+)::BASELINE
            # _id is useless here
            _id = id_line.split('::')[1]
            if _id in _id_set:
                src_fa_fh.readline()
                continue
            _id_set.add(_id)
            chr = _id.split(':')[0]
            assert (chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_dict[chr].append((_id, seq_line, id_line[1:]))
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    for chr in infer_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(infer_dict[chr], 3., max_seq_len)
    print('load_data_ead_simpleinfer time usage: {}'.format(time.time() - st))
    return datasets
