import random
import time
from collections import defaultdict
import os
from multiprocessing import Pool
import csv
from .encoding import *

# TODO
data_dir = os.getcwd() + '/data'

# ENCODE_FN = seq_encoding_onehot
ENCODE_FN = seq_encoding_onehot_padN


def get_dataset_from_seq_ead_multi(dataset, flag):
    # no pad setting!
    p = Pool(16)
    print(dataset[0])
    result_dataset = list(p.map(ENCODE_FN, dataset))
    p.close()
    p.join()
    return [[r, torch.tensor(flag)] for r in result_dataset]
    # return [[r, flag] for r in result_dataset]


def get_dataset_from_seq_ead_fold(dataset, flag, max_seq_len):
    flag = torch.tensor(flag)
    result_dataset = []
    for d in dataset:
        # d: (_id, seq_line, id_line[1:])
        # or (seq_line, id_line[1:])
        result_dataset.append([d[2], ENCODE_FN(d[1], max_seq_len), flag, ENCODE_FN(d[3]), d[4], len(d[1])])
    return result_dataset

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

def get_dataset_from_seq_ead_gencode(dataset, flag_dict, max_seq_len):
    # no pad setting
    result_dataset = []
    for d in dataset:
        flag = torch.tensor(flag_dict[d[0]])
        # ((_id, seq_line, id_line[1:])
        result_dataset.append([d[2], ENCODE_FN(d[1], max_seq_len), flag])
    return result_dataset


def get_dataset_from_seq_dp(data, l, r):
    # no pad setting
    # flag = np.array(pos_neg)
    result_dataset = []
    st = time.time()
    for i in data:
        seq_ary = ENCODE_FN(i.upper())
        label_ary = np.zeros(len(i) - 100, dtype=np.int64)
        label_ary[l - 50] = 1  # starter
        label_ary[r - 50] = 2  # ender
        result_dataset.append([seq_ary, label_ary])
    print('get_dataset_from_seq time usage: {}'.format(time.time() - st))
    return result_dataset


def load_data_dp(alu_file, exon_file, neg_file, pos_ratio=0.5):
    datasets = {}
    pos_data = []
    neg_data = []
    # with open(alu_file, 'r') as alu_fh:
    with open(exon_file, 'r') as exon_fh:
        exon_line = exon_fh.readline()
        while (exon_line):
            loc_lst = exon_line.split('::')[0].split('_')[-2:]
            l = int(loc_lst[0])
            r = int(loc_lst[1])
            exon_line = exon_fh.readline().rstrip()
            pos_data.append(exon_line)
            exon_line = exon_fh.readline().rstrip()
            if exon_line == '':
                break
    pos_dataset = get_dataset_from_seq_dp(pos_data, l, r)
    datasets['train'] = pos_dataset
    datasets['test'] = []
    return datasets


def load_data_ead_alu(alu_file, neg_file, pos_ratio=0.5):
    # load all alu only
    # TODO: maybe deprecate
    datasets = {}
    pos_data = []
    neg_data = []
    with open(alu_file, 'r') as alu_fh:
        alu_line = alu_fh.readline()
        while (alu_line):
            alu_line = alu_fh.readline().rstrip()
            if alu_line == '':
                break
            pos_data.append(alu_line)
            alu_fh.readline()
    with open(neg_file, 'r') as neg_fh:
        neg_line = neg_fh.readline()
        while (neg_line):
            neg_line = neg_fh.readline().rstrip()
            if neg_line == '':
                break
            neg_fh.readline()
            neg_data.append(neg_line)

    selected_neg_data = random.sample(neg_data, len(
        pos_data) * int((1 - pos_ratio) / pos_ratio))
    pos_dataset = get_dataset_from_seq_ead(pos_data, 1)
    neg_dataset = get_dataset_from_seq_ead(selected_neg_data, 0)

    datasets['train'] = pos_dataset + neg_dataset
    datasets['test'] = []
    return datasets


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


def load_data_ead_infer_gencode(strand, infer_bed_file, infer_fa_file, infer_unique=True, max_seq_len=0):
    # read infer only, the difference between load_data_ead_simpleinfer and this function is
    # this function will store gencode's types
    st = time.time()
    # read infer
    infer_bed_fh = open(infer_bed_file, 'r')
    src_bed_reader = csv.reader(infer_bed_fh, delimiter='\t')
    infer_id_label_dict = defaultdict(float)
    for idx, row in enumerate(src_bed_reader):
        print(row)
        l = int(row[1])
        r = int(row[2])
        if strand == True:
            _id = f'{row[0]}:{l}-{r}({row[5]})'
        else:
            _id = f'{row[0]}:{l}-{r}'
        assert (row[4] in ['1', '2', '3'])
        if infer_id_label_dict[_id] == 0.0:
            if row[4] in ['1', '2']:
                # Class 1: code 1, 2, terminal exons only
                infer_id_label_dict[_id] = 1.0
            else:
                # Class 2: code 3, internal exons only
                infer_id_label_dict[_id] = 2.0
        # Class 3: can be both Class 1 and Class 2
        elif infer_id_label_dict[_id] == 1.0:
            if row[4] == '3':
                infer_id_label_dict[_id] = 3.0
        elif infer_id_label_dict[_id] == 2.0:
            if row[4] in ['1', '2']:
                infer_id_label_dict[_id] = 3.0
    infer_bed_fh.close()

    # read infer alu file
    # label_file = work_dir + '/labels.txt'
    print('loading... infer positive data ')
    # infer_label_fh = open(label_file, 'w')
    infer_pos_dict = defaultdict(list)
    # TODO: add test_pos_id_set too
    infer_pos_id_set = set()
    with open(infer_fa_file, 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in infer_pos_id_set and infer_unique:
                # remove duplicate Gencode
                src_fa_fh.readline()
                continue
            infer_pos_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_pos_dict[chr].append((_id, seq_line, id_line[1:]))
            # infer_label_fh.write(_id + '\t' + str(infer_id_label_dict[_id]) + '\n')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead_gencode
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in infer_pos_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_pos_dict[chr], infer_id_label_dict, max_seq_len)
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    return datasets

def load_data_ead_alu_chr_withinfer2(strand, alu_file, neg_file, work_dir, infer_set,
                                     neg_ratio=1.0, val_ratio=0.1, test_chrs=['chr2', 'chr5'],
                                     train_unique=True, test_unique=True, infer_unique=True, duplicate_times=5, test_ratio=0.1, max_seq_len=0):
    # read infer set first
    # if train file's seqs appear in the infer set, then skip
    st = time.time()
    # read infer
    # map Gencode class to 0 1 2(the real positive, like other's 1)
    class_1_counter = 0  # 1,2 term->1,
    class_2_counter = 0  # 3 in->2
    infer_bed_fh = open(work_dir + f'/{infer_set}' + '_pos_alu.bed', 'r')
    src_bed_reader = csv.reader(infer_bed_fh, delimiter='\t')
    infer_id_label_dict = {}
    for idx, row in enumerate(src_bed_reader):
        l = int(row[1])
        r = int(row[2])
        if strand == True:
            _id = f'{row[0]}:{l}-{r}({row[5]})'
        else:
            _id = f'{row[0]}:{l}-{r}'
        # 1, 2, 3 are for gencode
        # 0 is for others
        if row[4] == '1' or row[4] == '2':
            infer_id_label_dict[_id] = '1'
            class_1_counter += 1
        elif row[4] == '3':
            infer_id_label_dict[_id] = '2'
            class_2_counter += 1
        elif row[4] == '0':
            infer_id_label_dict[_id] = '1'
        else:
            print(row[4])
            raise ValueError('unrecognizable label')
    infer_bed_fh.close()
    print(f'infer class 1:{class_1_counter} infer class 2:{class_2_counter}')

    # read infer alu file
    label_file = work_dir + '/labels.txt'
    print('loading... infer positive data: ' + label_file)
    infer_label_fh = open(label_file, 'w')
    infer_pos_dict = defaultdict(list)
    # TODO: add test_pos_id_set too
    infer_pos_id_set = set()
    with open(work_dir + f'/{infer_set}' + '_pos_alu.fa', 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in infer_pos_id_set and infer_unique:
                # remove duplicate Gencode
                src_fa_fh.readline()
                continue
            infer_pos_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_pos_dict[chr].append((_id, seq_line, id_line[1:]))
            infer_label_fh.write(_id + '\t' + infer_id_label_dict[_id] + '\n')

    # read train and test
    train_pos_dict = defaultdict(list)
    test_pos_dict = defaultdict(list)
    train_pos_all_n, test_pos_all_n = 0, 0
    train_pos_id_set = set()
    test_pos_id_set = set()
    print('loading... train/test positive data')
    with open(alu_file, 'r') as alu_fh:
        while (id_line := alu_fh.readline().rstrip()):
            # >h38_mk_AluJ_38::chr2:96241692-96242002(-)
            _id = id_line.split('::')[-1]
            chr = id_line.split(':')[2]
            if _id in infer_pos_id_set:
                # skip
                alu_fh.readline()
                continue
            assert (chr[0:3] == 'chr')
            if test_chrs and (chr in test_chrs):
                if _id in test_pos_id_set and test_unique:
                    # duplicate, skip
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                test_pos_id_set.add(_id)
                test_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                test_pos_all_n += 1
            else:
                if _id in train_pos_id_set and train_unique:
                    # train_unique should be alaways 1 to make sure this is duplicate unique version
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                train_pos_id_set.add(_id)
                train_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                train_pos_all_n += 1
    print('train len:', train_pos_all_n)
    print('test len:', test_pos_all_n)
    train_pos_all_n = 0
    val_pos_all_n = 0
    # duplicate the train set

    val_pos_dict = defaultdict(list)
    for chr, lst in train_pos_dict.items():
        random.shuffle(lst)
        if test_chrs:
            val_pos_dict[chr] = lst[:int(len(lst) * val_ratio)] * duplicate_times
            train_pos_dict[chr] = lst[int(len(lst) * val_ratio):] * duplicate_times
        else:
            test_pos_dict[chr] = lst[:int(len(lst) * test_ratio)]
            # val_pos_dict[chr] = lst[int(len(lst) * test_ratio) : int(len(lst) * test_ratio) + int(len(lst) * val_ratio)] * duplicate_times
            val_pos_dict[chr] = lst[int(len(lst) * test_ratio) : int(len(lst) * test_ratio) + int(len(lst) * val_ratio)] 
            train_pos_dict[chr] = lst[int(len(lst) * test_ratio) + int(len(lst) * val_ratio):] * duplicate_times
            test_pos_all_n += len(test_pos_dict[chr])
        val_pos_all_n += len(val_pos_dict[chr])
        train_pos_all_n += len(train_pos_dict[chr])
    print('train len after duplicating:', train_pos_all_n)
    print('val len after duplicating:', val_pos_all_n)

    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            neg_dict[chr].append((_id, seq_line, id_line[1:]))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for train/test/infer set')
    infer_neg_dict = defaultdict(list)
    train_neg_dict = defaultdict(list)
    val_neg_dict = defaultdict(list)
    test_neg_dict = defaultdict(list)
    neg_id_set = set()
    for chr, id_seq_lst in neg_dict.items():
        train_neg_chr_n = int(len(train_pos_dict[chr]) * neg_ratio)
        val_neg_chr_n = int(len(val_pos_dict[chr]) * neg_ratio)
        test_neg_chr_n = int(len(test_pos_dict[chr]) * neg_ratio)
        infer_neg_chr_n = int(len(infer_pos_dict[chr]) * neg_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if id_seq[0] not in neg_id_set:
                neg_id_set.add(id_seq[0])
                if len(infer_neg_dict[chr]) < infer_neg_chr_n:
                    # infer_neg not enough, continue add
                    # TODO: if those neg_chr_n conditions are <=, which will result in that each chr has (#pos + 1) neg
                    infer_neg_dict[chr].append(id_seq)
                    infer_label_fh.write(id_seq[0] + '\t' + '0' + '\n')
                else:
                    # for test, train, and val
                    if test_chrs:
                        if chr in test_chrs:
                            # test
                            if len(test_neg_dict[chr]) < test_neg_chr_n:
                                test_neg_dict[chr].append(id_seq)
                        else:
                            # train, val
                            if len(train_neg_dict[chr]) < train_neg_chr_n:
                                # train
                                train_neg_dict[chr].append(id_seq)
                            elif len(val_neg_dict[chr]) < val_neg_chr_n:
                                # val
                                val_neg_dict[chr].append(id_seq)
                    else:
                            if len(test_neg_dict[chr]) < test_neg_chr_n:
                                # test
                                test_neg_dict[chr].append(id_seq)
                            elif len(train_neg_dict[chr]) < train_neg_chr_n:
                                # train
                                train_neg_dict[chr].append(id_seq)
                            elif len(val_neg_dict[chr]) < val_neg_chr_n:
                                # val
                                val_neg_dict[chr].append(id_seq)

        print(
            f'{chr} train\tpos: {len(train_pos_dict[chr])}\t neg: {len(train_neg_dict[chr])}')
        print(
            f'{chr} val\tpos: {len(val_pos_dict[chr])}\t neg: {len(val_neg_dict[chr])}')
        print(
            f'{chr} test\tpos: {len(test_pos_dict[chr])}\t neg: {len(test_neg_dict[chr])}')
        print(
            f'{chr} infer\tpos: {len(infer_pos_dict[chr])}\t neg: {len(infer_neg_dict[chr])}')
        print('')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in train_pos_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_pos_dict[chr], 1., max_seq_len)
    for chr in train_neg_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_neg_dict[chr], 0., max_seq_len)
    for chr in val_pos_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_pos_dict[chr], 1., max_seq_len)
    for chr in val_neg_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_neg_dict[chr], 0., max_seq_len)
    for chr in test_pos_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_pos_dict[chr], 1., max_seq_len)
    for chr in test_neg_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_neg_dict[chr], 0., max_seq_len)
    for chr in infer_pos_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_pos_dict[chr], 1., max_seq_len)
    for chr in infer_neg_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_neg_dict[chr], 0., max_seq_len)
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    # for i in train_pos_id_set:
    #     print('train', i)
    # for i in test_pos_id_set:
    #     print('test', i)
    # for i in infer_pos_id_set:
    #     print('infer', i)
    return datasets

def load_data_ead_alu_chr_withinfer2_fold(strand, alu_file, neg_file, work_dir, infer_set,
                                     neg_ratio=1.0, val_ratio=0.1, test_chrs=['chr2', 'chr5'],
                                     train_unique=True, test_unique=True, infer_unique=True, duplicate_times=5, max_seq_len=0):
    # read infer set first
    # if train file's seqs appear in the infer set, then skip
    st = time.time()
    # read infer
    # map Gencode class to 0 1 2(the real positive, like other's 1)
    class_1_counter = 0  # 1,2 term->1,
    class_2_counter = 0  # 3 in->2
    infer_bed_fh = open(work_dir + f'/{infer_set}' + '_pos_alu.bed', 'r')
    src_bed_reader = csv.reader(infer_bed_fh, delimiter='\t')
    infer_id_label_dict = {}
    for idx, row in enumerate(src_bed_reader):
        l = int(row[1])
        r = int(row[2])
        if strand == True:
            _id = f'{row[0]}:{l}-{r}({row[5]})'
        else:
            _id = f'{row[0]}:{l}-{r}'
        # 1, 2, 3 are for gencode
        # 0 is for others
        if row[4] == '1' or row[4] == '2':
            infer_id_label_dict[_id] = '1'
            class_1_counter += 1
        elif row[4] == '3':
            infer_id_label_dict[_id] = '2'
            class_2_counter += 1
        elif row[4] == '0':
            infer_id_label_dict[_id] = '1'
        else:
            print(row[4])
            raise ValueError('unrecognizable label')
    infer_bed_fh.close()
    print(f'infer class 1:{class_1_counter} infer class 2:{class_2_counter}')

    # read infer alu file
    label_file = work_dir + '/labels.txt'
    print('loading... infer positive data: ' + label_file)
    infer_label_fh = open(label_file, 'w')
    infer_pos_dict = defaultdict(list)
    # TODO: add test_pos_id_set too
    infer_pos_id_set = set()
    with open(work_dir + f'/{infer_set}' + '_pos_alu.fa', 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in infer_pos_id_set and infer_unique:
                # remove duplicate Gencode
                src_fa_fh.readline()
                continue
            infer_pos_id_set.add(_id)
            _chr = id_line.split(':')[2]
            assert (_chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_pos_dict[_chr].append((_id, seq_line, id_line[1:]))
            infer_label_fh.write(_id + '\t' + infer_id_label_dict[_id] + '\n')

    # read train and test
    train_pos_dict = defaultdict(list)
    test_pos_dict = defaultdict(list)
    train_pos_all_n, test_pos_all_n = 0, 0
    train_pos_id_set = set()
    test_pos_id_set = set()
    print('loading... train/test positive data')
    with open(alu_file, 'r') as alu_fh:
        while (id_line := alu_fh.readline().rstrip()):
            # >h38_mk_AluJ_38::chr2:96241692-96242002(-)
            _id = id_line.split('::')[-1]
            _chr = id_line.split(':')[2]
            if _id in infer_pos_id_set:
                # skip
                alu_fh.readline()
                alu_fh.readline()
                continue
            assert (_chr[0:3] == 'chr')
            if _chr in test_chrs:
                if _id in test_pos_id_set and test_unique:
                    # duplicate, skip
                    alu_fh.readline()
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                ss_line = alu_fh.readline().rstrip()
                ss_str, ss_score = ss_line.split(' ')
                ss_score = float(ss_score[1:-1])
                test_pos_id_set.add(_id)
                test_pos_dict[_chr].append((_id, seq_line, id_line[1:], ss_str, ss_score))
                test_pos_all_n += 1
            else:
                if _id in train_pos_id_set and train_unique:
                    # train_unique should be alaways 1 to make sure this is duplicate unique version
                    alu_fh.readline()
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                ss_line = alu_fh.readline().rstrip()
                ss_str, ss_score = ss_line.split(' ')
                ss_score = float(ss_score[1:-1])
                train_pos_id_set.add(_id)
                train_pos_dict[_chr].append((_id, seq_line, id_line[1:], ss_str, ss_score))
                train_pos_all_n += 1
    print('train len:', train_pos_all_n)
    print('test len:', test_pos_all_n)
    train_pos_all_n = 0
    val_pos_all_n = 0
    # duplicate the train set

    val_pos_dict = defaultdict(list)
    for _chr, lst in train_pos_dict.items():
        random.shuffle(lst)
        val_pos_dict[_chr] = lst[:int(len(lst) * val_ratio)] * duplicate_times
        train_pos_dict[_chr] = lst[int(len(lst) * val_ratio):] * duplicate_times
        val_pos_all_n += len(val_pos_dict[_chr])
        train_pos_all_n += len(train_pos_dict[_chr])
    print('train len after duplicating:', train_pos_all_n)
    print('val len after duplicating:', val_pos_all_n)

    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            _chr = id_line.split(':')[2]
            assert (_chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            ss_line = neg_fh.readline().rstrip()
            ss_str, ss_score = ss_line.split(' ')
            ss_score = float(ss_score[1:-1])
            neg_dict[_chr].append((_id, seq_line, id_line[1:], ss_str, ss_score))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for train/test/infer set')
    infer_neg_dict = defaultdict(list)
    train_neg_dict = defaultdict(list)
    val_neg_dict = defaultdict(list)
    test_neg_dict = defaultdict(list)
    neg_id_set = set()
    for _chr, id_seq_lst in neg_dict.items():
        train_neg_chr_n = int(len(train_pos_dict[_chr]) * neg_ratio)
        val_neg_chr_n = int(len(val_pos_dict[_chr]) * neg_ratio)
        test_neg_chr_n = int(len(test_pos_dict[_chr]) * neg_ratio)
        infer_neg_chr_n = int(len(infer_pos_dict[_chr]) * neg_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if id_seq[0] not in neg_id_set:
                neg_id_set.add(id_seq[0])
                if len(infer_neg_dict[_chr]) < infer_neg_chr_n:
                    # infer_neg not enough, continue add
                    # TODO: if those neg_chr_n conditions are <=, which will result in that each chr has (#pos + 1) neg
                    infer_neg_dict[_chr].append(id_seq)
                    infer_label_fh.write(id_seq[0] + '\t' + '0' + '\n')
                else:
                    # for test, train, and val
                    if _chr in test_chrs:
                        # test
                        if len(test_neg_dict[_chr]) < test_neg_chr_n:
                            test_neg_dict[_chr].append(id_seq)
                    else:
                        # train, val
                        if len(train_neg_dict[_chr]) < train_neg_chr_n:
                            # train
                            train_neg_dict[_chr].append(id_seq)
                        elif len(val_neg_dict[_chr]) < val_neg_chr_n:
                            # val
                            val_neg_dict[_chr].append(id_seq)
        print(
            f'{_chr} train\tpos: {len(train_pos_dict[_chr])}\t neg: {len(train_neg_dict[_chr])}')
        print(
            f'{_chr} val\tpos: {len(val_pos_dict[_chr])}\t neg: {len(val_neg_dict[_chr])}')
        print(
            f'{_chr} test\tpos: {len(test_pos_dict[_chr])}\t neg: {len(test_neg_dict[_chr])}')
        print(
            f'{_chr} infer\tpos: {len(infer_pos_dict[_chr])}\t neg: {len(infer_neg_dict[_chr])}')
        print('')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead_fold
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for _chr in train_pos_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_pos_dict[_chr], 1., max_seq_len)
    for _chr in train_neg_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_neg_dict[_chr], 0., max_seq_len)
    for _chr in val_pos_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_pos_dict[_chr], 1., max_seq_len)
    for _chr in val_neg_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_neg_dict[_chr], 0., max_seq_len)
    for _chr in test_pos_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_pos_dict[_chr], 1., max_seq_len)
    for _chr in test_neg_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_neg_dict[_chr], 0., max_seq_len)
    for _chr in infer_pos_dict.keys():
        # datasets['infer'] += get_dataset_from_seq_ead_fn(
            # infer_pos_dict[_chr], 1.)
        datasets['infer'] += []
    for _chr in infer_neg_dict.keys():
        # datasets['infer'] += get_dataset_from_seq_ead_fn(
            # infer_neg_dict[_chr], 0.)
        datasets['infer'] += []
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    # for i in train_pos_id_set:
    #     print('train', i)
    # for i in test_pos_id_set:
    #     print('test', i)
    # for i in infer_pos_id_set:
    #     print('infer', i)
    return datasets


def load_data_ead_alu_chr_withinfer(strand, alu_file, neg_file, work_dir, infer_set,
                                    pos_ratio=0.5, test_chrs=['chr2', 'chr5'],
                                    train_unique=False, test_unique=True, infer_unique=True, max_seq_len=0):
    # read train first
    # if seq in infer set appears in train, then skip
    # if distinct is True, then test set only has distinct data
    # if not, then test set keep original
    st = time.time()
    # read train set alu
    train_pos_dict = defaultdict(list)
    test_pos_dict = defaultdict(list)
    train_pos_all_n, test_pos_all_n = 0, 0
    train_pos_id_set = set()
    test_pos_id_set = set()
    print('loading... train/test positive data')
    with open(alu_file, 'r') as alu_fh:
        while (id_line := alu_fh.readline().rstrip()):
            # >h38_mk_AluJ_38::chr2:96241692-96242002(-)
            _id = id_line.split('::')[-1]
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            if chr in test_chrs:
                if _id in test_pos_id_set and test_unique:
                    # duplicate, skip
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                test_pos_id_set.add(_id)
                test_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                test_pos_all_n += 1
            else:
                if _id in train_pos_id_set and train_unique:
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                train_pos_id_set.add(_id)
                train_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                train_pos_all_n += 1
    print('train len:', train_pos_all_n)
    print('test len:', test_pos_all_n)

    # generate infer labels
    class_1_counter = 0  # 1,2 term->1,
    class_2_counter = 0  # 3 in->2
    infer_bed_fh = open(work_dir + f'/{infer_set}' + '_pos_alu.bed', 'r')
    src_bed_reader = csv.reader(infer_bed_fh, delimiter='\t')
    infer_id_label_dict = {}
    for idx, row in enumerate(src_bed_reader):
        l = int(row[1])
        r = int(row[2])
        if strand == True:
            _id = f'{row[0]}:{l}-{r}({row[5]})'
        else:
            _id = f'{row[0]}:{l}-{r}'
        # 1, 2, 3 are for gencode
        # 0 is for others
        if row[4] == '1' or row[4] == '2':
            infer_id_label_dict[_id] = '1'
            class_1_counter += 1
        elif row[4] == '3':
            infer_id_label_dict[_id] = '2'
            class_2_counter += 1
        elif row[4] == '0':
            infer_id_label_dict[_id] = '1'
        else:
            print(row[4])
            raise ValueError('unrecognizable label')
    infer_bed_fh.close()
    print(f'infer class 1:{class_1_counter} infer class 2:{class_2_counter}')

    # read infer alu file
    label_file = work_dir + '/labels.txt'
    print('loading... infer positive data: ' + label_file)
    infer_label_fh = open(label_file, 'w')
    infer_pos_dict = defaultdict(list)
    infer_pos_id_set = set()
    # TODO: add test_pos_id_set too
    with open(work_dir + f'/{infer_set}' + '_pos_alu.fa', 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in train_pos_id_set:
                # this _id has appeared in training set
                # skip
                src_fa_fh.readline()
                continue
            elif _id in infer_pos_id_set and infer_unique:
                # unique mode and _id has appeared in training set
                # skip
                src_fa_fh.readline()
                continue
            infer_pos_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_pos_dict[chr].append((_id, seq_line, id_line[1:]))
            infer_label_fh.write(_id + '\t' + infer_id_label_dict[_id] + '\n')

    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            neg_dict[chr].append((_id, seq_line, id_line[1:]))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for train/test/infer set')
    infer_neg_id_set = set()
    infer_neg_dict = defaultdict(list)
    train_neg_dict = defaultdict(list)
    test_neg_id_set = set()
    test_neg_dict = defaultdict(list)
    for chr, id_seq_lst in neg_dict.items():
        train_neg_chr_n = len(
            train_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        test_neg_chr_n = len(
            test_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        infer_neg_chr_n = len(
            infer_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if id_seq[0] not in infer_neg_id_set:
                if len(infer_neg_dict[chr]) < infer_neg_chr_n:
                    # infer_neg not enough, continue add
                    infer_neg_id_set.add(id_seq[0])
                    infer_neg_dict[chr].append(id_seq)
                    infer_label_fh.write(id_seq[0] + '\t' + '0' + '\n')
                else:
                    if chr in test_chrs:
                        # for test
                        if id_seq[0] not in test_neg_id_set:
                            if len(test_neg_dict[chr]) < test_neg_chr_n:
                                test_neg_id_set.add(id_seq[0])
                                test_neg_dict[chr].append(id_seq)
                    else:
                        # for train
                        if len(train_neg_dict[chr]) < train_neg_chr_n:
                            train_neg_dict[chr].append(id_seq)
        print(
            f'{chr} train pos: {len(train_pos_dict[chr])}\t neg: {len(train_neg_dict[chr])}')
        print(
            f'{chr} test pos: {len(test_pos_dict[chr])}\t neg: {len(test_neg_dict[chr])}')
        print(
            f'{chr} infer pos: {len(infer_pos_dict[chr])}\t neg: {len(infer_neg_dict[chr])}')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in train_pos_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_pos_dict[chr], 1., max_seq_len)
    for chr in train_neg_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_neg_dict[chr], 0., max_seq_len)
    for chr in test_pos_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_pos_dict[chr], 1., max_seq_len)
    for chr in test_neg_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_neg_dict[chr], 0., max_seq_len)
    for chr in infer_pos_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_pos_dict[chr], 1., max_seq_len)
    for chr in infer_neg_dict.keys():
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_neg_dict[chr], 0., max_seq_len)
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    return datasets


def load_data_ead_alu_chr_train_unique_duplicate(alu_file, neg_file,
                                                 neg_ratio=1.0, val_ratio=0.1, test_chrs=['chr2', 'chr5'],
                                                 test_unique=True, duplicate_times=5, max_seq_len=0):
    # read train set and only keep the unique alus
    # then duplicate the unique alus multiple times
    st = time.time()
    # read train set alu
    train_pos_dict = defaultdict(list)
    test_pos_dict = defaultdict(list)
    train_pos_all_n, test_pos_all_n = 0, 0
    train_pos_id_set = set()
    test_pos_id_set = set()
    print('loading... train/test positive data')
    with open(alu_file, 'r') as alu_fh:
        while (id_line := alu_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            if chr in test_chrs:
                if _id in test_pos_id_set and test_unique:
                    # duplicate, skip
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                test_pos_id_set.add(_id)
                test_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                test_pos_all_n += 1
            else:
                if _id in train_pos_id_set:
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                train_pos_id_set.add(_id)
                train_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                train_pos_all_n += 1
    print('train len:', train_pos_all_n)
    print('test len:', test_pos_all_n)
    train_pos_all_n = 0
    # duplicate the train set
    val_pos_dict = defaultdict(list)
    for chr, lst in train_pos_dict.items():
        random.shuffle(lst)
        val_pos_dict[chr] = lst[:int(len(lst) * val_ratio)] * duplicate_times
        train_pos_dict[chr] = lst[int(len(lst) * val_ratio):] * duplicate_times
        val_pos_all_n += len(val_pos_dict[chr])
        train_pos_all_n += len(train_pos_dict[chr])
    print('train len after duplicating:', train_pos_all_n)
    print('val len after duplicating:', val_pos_all_n)
    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            neg_dict[chr].append((_id, seq_line, id_line[1:]))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for train/test set')
    train_neg_dict = defaultdict(list)
    val_neg_dict = defaultdict(list)
    test_neg_dict = defaultdict(list)
    neg_id_set = set()
    for chr, id_seq_lst in neg_dict.items():
        train_neg_chr_n = int(len(train_pos_dict[chr]) * neg_ratio)
        val_neg_chr_n = int(len(val_pos_dict[chr]) * neg_ratio)
        test_neg_chr_n = int(len(test_pos_dict[chr]) * neg_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if id_seq[0] not in neg_id_set:
                neg_id_set.add(id_seq[0])
                # for test, train, and val
                if chr in test_chrs:
                    # test
                    if len(test_neg_dict[chr]) <= test_neg_chr_n:
                        test_neg_dict[chr].append(id_seq)
                else:
                    # train, val
                    if len(train_neg_dict[chr]) <= train_neg_chr_n:
                        # train
                        train_neg_dict[chr].append(id_seq)
                    elif len(val_neg_dict[chr]) <= val_neg_chr_n:
                        # val
                        val_neg_dict[chr].append(id_seq)
        print(
            f'{chr} train pos: {len(train_pos_dict[chr])}\t neg: {len(train_neg_dict[chr])}')
        print(
            f'{chr} val\tpos: {len(val_pos_dict[chr])}\t neg: {len(val_neg_dict[chr])}')
        print(
            f'{chr} test pos: {len(test_pos_dict[chr])}\t neg: {len(test_neg_dict[chr])}')
        print('')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in train_pos_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_pos_dict[chr], 1., max_seq_len)
    for chr in train_neg_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_neg_dict[chr], 0., max_seq_len)
    for chr in val_pos_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_pos_dict[chr], 1., max_seq_len)
    for chr in val_neg_dict.keys():
        datasets['val'] += get_dataset_from_seq_ead_fn(val_neg_dict[chr], 0., max_seq_len)
    for chr in test_pos_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_pos_dict[chr], 1., max_seq_len)
    for chr in test_neg_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_neg_dict[chr], 0., max_seq_len)
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    return datasets


def load_data_ead_alu_chr_woinfer(alu_file, neg_file,
                                  pos_ratio=0.5, test_chrs=['chr2', 'chr5'],
                                  train_unique=False, test_unique=True, max_seq_len=0):
    # load train/test alu and infer alu
    # split each set using chr#
    # if distinct is True, then test set only has distinct data
    # if not, then test set keep original
    st = time.time()
    # read train set alu
    train_pos_dict = defaultdict(list)
    test_pos_dict = defaultdict(list)
    train_pos_all_n, test_pos_all_n = 0, 0
    train_pos_id_set = set()
    test_pos_id_set = set()
    print('loading... train/test positive data')
    with open(alu_file, 'r') as alu_fh:
        while (id_line := alu_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            if chr in test_chrs:
                if _id in test_pos_id_set and test_unique:
                    # duplicate, skip
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                test_pos_id_set.add(_id)
                test_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                test_pos_all_n += 1
            else:
                if _id in train_pos_id_set and train_unique:
                    alu_fh.readline()
                    continue
                seq_line = alu_fh.readline().rstrip()
                train_pos_id_set.add(_id)
                train_pos_dict[chr].append((_id, seq_line, id_line[1:]))
                train_pos_all_n += 1
    print('train len:', train_pos_all_n)
    print('test len:', test_pos_all_n)

    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            neg_dict[chr].append((_id, seq_line, id_line[1:]))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for train/test set')
    train_neg_dict = defaultdict(list)
    test_neg_id_set = set()
    test_neg_dict = defaultdict(list)
    for chr, id_seq_lst in neg_dict.items():
        train_neg_chr_n = len(
            train_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        test_neg_chr_n = len(
            test_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if chr in test_chrs:
                # for test
                if id_seq[0] not in test_neg_id_set:
                    if len(test_neg_dict[chr]) < test_neg_chr_n:
                        test_neg_id_set.add(id_seq[0])
                        test_neg_dict[chr].append(id_seq)
            else:
                # for train
                if len(train_neg_dict[chr]) < train_neg_chr_n:
                    train_neg_dict[chr].append(id_seq)
        print(
            f'{chr} train pos: {len(train_pos_dict[chr])}\t neg: {len(train_neg_dict[chr])}')
        print(
            f'{chr} test pos: {len(test_pos_dict[chr])}\t neg: {len(test_neg_dict[chr])}')

    # write several for-loop due to keys may have small difference
    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in train_pos_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_pos_dict[chr], 1., max_seq_len)
    for chr in train_neg_dict.keys():
        datasets['train'] += get_dataset_from_seq_ead_fn(
            train_neg_dict[chr], 0., max_seq_len)
    for chr in test_pos_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_pos_dict[chr], 1., max_seq_len)
    for chr in test_neg_dict.keys():
        datasets['test'] += get_dataset_from_seq_ead_fn(test_neg_dict[chr], 0., max_seq_len)
    print('load_data_ead_alu_chr time usage: {}'.format(time.time() - st))
    return datasets


def load_data_ead_alu_chr_inferonly(strand, infer_neg_file, work_dir=None, infer_set=None,
                                    pos_ratio=0.5,
                                    infer_unique=True, max_seq_len=0):
    # load train/test alu and infer alu
    # split each set using chr#
    # if distinct is True, then test set only has distinct data
    # if not, then test set keep original
    st = time.time()
    # read train set alu
    class_1_counter = 0  # 1,2 term->1,
    class_2_counter = 0  # 3 in->2
    # gencode_bed_file = data_dir + '/Gencode/GENCODE.v36.ALUs.all.overlap.filtered.bed'
    # gencode_bed_file = data_dir + '/Gencode/gencode_alu.bed'

    infer_bed_fh = open(work_dir + f'/{infer_set}' + '_pos_alu.bed', 'r')
    src_bed_reader = csv.reader(infer_bed_fh, delimiter='\t')
    infer_id_label_dict = {}
    for idx, row in enumerate(src_bed_reader):
        l = int(row[1])
        r = int(row[2])
        if strand == True:
            _id = f'{row[0]}:{l}-{r}({row[5]})'
        else:
            _id = f'{row[0]}:{l}-{r}'
        # 1, 2, 3 are for gencode
        # 0 is for others
        if row[4] == '1' or row[4] == '2':
            infer_id_label_dict[_id] = '1'
            class_1_counter += 1
        elif row[4] == '3':
            infer_id_label_dict[_id] = '2'
            class_2_counter += 1
        elif row[4] == '0':
            infer_id_label_dict[_id] = '1'
        else:
            print(row[4])
            raise ValueError('unrecognizable label')
    infer_bed_fh.close()
    print(f'infer class 1:{class_1_counter} infer class 2:{class_2_counter}')

    # read infer alu file
    print('loading... infer positive data')
    label_file = work_dir + '/labels.txt'
    infer_label_fh = open(label_file, 'w')
    infer_pos_dict = defaultdict(list)
    infer_pos_id_set = set()
    with open(work_dir + f'/{infer_set}' + '_pos_alu.fa', 'r') as src_fa_fh:
        while (id_line := src_fa_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in infer_pos_id_set and infer_unique:
                src_fa_fh.readline()
                continue
            infer_pos_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = src_fa_fh.readline().rstrip()
            infer_pos_dict[chr].append((_id, seq_line, id_line[1:]))
            infer_label_fh.write(_id + '\t' + infer_id_label_dict[_id] + '\n')

    get_dataset_from_seq_ead_fn = get_dataset_from_seq_ead
    if infer_neg_file == None:
        datasets = defaultdict(list)
        for chr in infer_pos_dict.keys():
            if len(infer_pos_dict[chr]) == 0:
                print('0 pos', chr)
                continue
            else:
                print(len(infer_pos_dict[chr]), chr)
            datasets['infer'] += get_dataset_from_seq_ead_fn(
                infer_pos_dict[chr], 1., max_seq_len)
        return datasets
    # read neg alu file
    print('loading... negative data')
    neg_dict = defaultdict(list)
    neg_id_set = set()
    with open(infer_neg_file, 'r') as neg_fh:
        while (id_line := neg_fh.readline().rstrip()):
            _id = id_line.split('::')[-1]
            if _id in neg_id_set:
                neg_fh.readline()
                continue
            neg_id_set.add(_id)
            chr = id_line.split(':')[2]
            assert (chr[0:3] == 'chr')
            seq_line = neg_fh.readline().rstrip()
            neg_dict[chr].append((_id, seq_line, id_line[1:]))

    # get distinct neg data for infer set, and train/test set
    print('spliting... negtive data for infer set')
    infer_neg_id_set = set()
    infer_neg_dict = defaultdict(list)
    for chr, id_seq_lst in neg_dict.items():
        if chr not in infer_pos_dict.keys():
            continue
        infer_neg_chr_n = len(
            infer_pos_dict[chr]) * int((1 - pos_ratio) / pos_ratio)
        random.shuffle(id_seq_lst)
        for id_seq in id_seq_lst:
            if id_seq[0] not in infer_neg_id_set:
                if len(infer_neg_dict[chr]) < infer_neg_chr_n:
                    # infer_neg not enough, continue add
                    infer_neg_id_set.add(id_seq[0])
                    infer_neg_dict[chr].append(id_seq)
                    infer_label_fh.write(id_seq[0] + '\t' + '0' + '\n')
        # print(f'{chr} infer pos: {len(infer_pos_dict[chr])}\t neg: {len(infer_neg_dict[chr])}')

    # write several for-loop due to keys may have small difference
    print('converting... to labeled data')
    datasets = defaultdict(list)
    for chr in infer_pos_dict.keys():
        if len(infer_pos_dict[chr]) == 0:
            print('0 pos', chr)
            continue
        else:
            print(len(infer_pos_dict[chr]), chr)
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_pos_dict[chr], 1., max_seq_len)
    for chr in infer_neg_dict.keys():
        if len(infer_neg_dict[chr]) == 0:
            print('0 neg', chr)
            continue
        else:
            print(len(infer_pos_dict[chr]), chr)
        datasets['infer'] += get_dataset_from_seq_ead_fn(
            infer_neg_dict[chr], 0., max_seq_len)
    print('load_data_ead_alu_chr_inferonly time usage: {}'.format(time.time() - st))
    return datasets


def load_data_two_tissue(first_alu_file, second_alu_file, first_exon_file=None, second_exon_file=None, ratio=0.1, max_seq_len=0):
    datasets = {}
    first_data = []
    second_data = []
    with open(first_alu_file, 'r') as alu_fh:
        #  open(first_exon_file, 'r') as exon_fh:
        # exon_line = exon_fh.readline()
        alu_line = alu_fh.readline()
        while (alu_line):
            # exon_line = exon_fh.readline().rstrip()
            # exon_fh.readline()
            alu_line = alu_fh.readline().rstrip()
            alu_fh.readline()
            first_data.append(alu_line)
    with open(second_alu_file, 'r') as alu_fh:
        #  open(second_exon_file, 'r') as exon_fh:
        # exon_line = exon_fh.readline()
        alu_line = alu_fh.readline()
        while (alu_line):
            # exon_line = exon_fh.readline().rstrip()
            # exon_fh.readline()
            alu_line = alu_fh.readline().rstrip()
            alu_fh.readline()
            second_data.append(alu_line)

    print('{} {}'.format(first_alu_file, len(first_data)))
    print('{} {}'.format(second_alu_file, len(second_data)))
    pos_dataset = get_dataset_from_seq_ead(first_data, 1., max_seq_len)
    neg_dataset = get_dataset_from_seq_ead(second_data, 0., max_seq_len)

    datasets['train'] = pos_dataset + neg_dataset
    datasets['test'] = []
    return datasets
