# this file is for loading SNP mutated seqs
import gzip
import csv
import re
import os
from collections import defaultdict
import pybedtools
from exalu.data_preprocess.read_tissue_alu import add_context, split
import pandas as pd
from exalu.data_preprocess.curate_data import curate_data, curate_simpleinfer
from exalu.run_model import run_ead
import matplotlib.pyplot as plt
import numpy as np
import subprocess
import seaborn
import pickle


def read_snp_per_seq(chr, start, end, snp_fh):
    '''
    snp vcf sample:
    1       11012   rs544419019     C   G
    '''
    _chr = chr[3:]
    snp_lst = []
    start = int(start)
    end = int(end)
    for line in snp_fh:
        if line[0] == '#':
            continue
        line_lst = line.rstrip().split('\t')
        if line_lst[0] != _chr:
            continue
        loc = int(line_lst[1])
        if start <= loc and loc <= end:
            if len(line_lst[3]) != 1 or len(line_lst[4]) != 1:
                # not single SNP
                if ',' in line_lst[4]:
                    continue
            snp_lst.append((i for i in line_lst))
        if loc > end:
            break
    return snp_lst

def read_snp(bed_file, snp_file):
    '''
    TODO: this function is outdated!!! camparing it's old pair read_fa()
    read bed file, output dict[key: each seq's id in bed, value: the seq's snps]
    
    bed example:
    chr15	41297596	41297996	h38_mk_AluSx1_4_both_11_12_3	3	-
    '''
    # open snp file
    if snp_file[-2:] == 'gz':
        snp_fh = gzip.open(snp_file, 'rt')
    else:
        snp_fh = open(snp_file, 'r')
    # read bed file 
    bed_fh = open(bed_file, 'r')
    bed_reader = csv.reader(bed_fh, delimiter='\t')
    seq_snp_dict = {}
    for idx, row in enumerate(bed_reader):
        seq_id = (row[0], row[1], row[2], row[3])
        seq_snp_dict[seq_id] = read_snp_per_seq(seq_id[0], seq_id[1], seq_id[2], snp_fh)
    bed_fh.close()
    snp_fh.close()
    return seq_snp_dict

def read_fa(bed_file, fa_file, genome, strand):
    '''
    read bed file, output dict [key: each seq's id in bed, value: the seq's fa seq]
    
    fa example: 
    >h38_mk_AluJb_2_both_19_20_3::chr11:3361721-3362121(+)
    tacatgctcttaggagattaccagtga......
    '''
    ref_fa = pybedtools.example_filename(genome)
    bed = pybedtools.BedTool(bed_file)
    bed.sequence(fi=ref_fa, fo=fa_file, name=True, s=strand)
    seq_fa_dict = {}
    with open(fa_file, 'r') as fa_fh:
        while(id_line := fa_fh.readline().rstrip()):
            id_from_fa = id_line.split('::')[-1]
            # id_lst = re.split(r'\:|\-|\(|\)', id_from_fa)
            if strand == True:
                id_lst = re.split('[:()]', id_from_fa)
                id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]
                _id = (id_lst[0], id_lst[1], id_lst[2], id_lst[3])
            else:
                id_lst = re.split('[:-]', id_from_fa)
                _id = (id_lst[0], id_lst[1], id_lst[2])
            assert(id_lst[0][0:3] == 'chr')
            seq_line = fa_fh.readline().rstrip()
            seq_fa_dict[_id] = (seq_line, id_line)
    # with open(fa_file, 'w') as fa_fh:
        # for _id, seq in seq_fa_dict.items():
            # ori_id_str = '_'.join(_id)
            # id_line = f'>{ori_id_str}::BASELINE\n'
            # fa_fh.write(id_line)
            # fa_fh.write(seq + '\n')
    return seq_fa_dict

def read_splice_sites(alu_bed_file, exon_bed_file):
    # read splice sites base on exon file
    alu_bed_fh = open(alu_bed_file, 'r')
    exon_bed_fh = open(exon_bed_file, 'r')
    alu_bed_reader = csv.reader(alu_bed_fh, delimiter='\t')
    exon_bed_reader = csv.reader(exon_bed_fh, delimiter='\t')
    splice_sites_dict = {}
    for alu_row, exon_row in zip(alu_bed_reader, exon_bed_reader):
        _id = (alu_row[0], alu_row[1], alu_row[2], alu_row[5])
        if exon_row[-1] == '+':
            exon_info = [(exon_row[1], 'acceptor'), (exon_row[2], 'donor')]
        else:
            exon_info = [(exon_row[1], 'donor'), (exon_row[2], 'acceptor')]
        splice_sites_dict[_id] = exon_info
    alu_bed_fh.close()
    exon_bed_fh.close()
    return splice_sites_dict

def mutate_seq_per_seq(seq, _id, snp_lst):
    '''
    deprecated;
    '''
    # mutate the input seq in the input bed file and output to the output bed file
    # modify the bed ID col
    _chr, start, end, name= _id
    # at this time, seq has been RCed if the strand is -
    mutated_seq_ids = []
    for mutation in snp_lst:
        m_chr, m_loc, m_name, m_ref, m_alt = mutation
        assert(m_chr == _chr[3:])
        i_in_seq = int(m_loc) - int(start) - 1
        # print(seq[:i_in_seq], seq[i_in_seq], m_ref, m_alt, seq[i_in_seq + 1:])
        # print(m_loc, start, end, i_in_seq)
        assert(m_ref.casefold() == seq[i_in_seq].casefold())
        mutated_seq_ids.append((seq[:i_in_seq] + m_alt + seq[i_in_seq + 1:], (str(i_in_seq), m_loc, m_name, m_ref, m_alt)))
    return mutated_seq_ids

def mutate_seq(seq_snp_dict, seq_fa_dict, mutated_fa):
    '''
    deprecated;
    mutate seq based on src like dbSNP or ClinVar
    '''
    mutated_fa_fh = open(mutated_fa, 'w') 
    for _id, snp_lst in seq_snp_dict.items():
        mutated_seq_ids = mutate_seq_per_seq(seq_fa_dict[_id], _id, snp_lst)
        for m_seq, m_id in mutated_seq_ids:
            mutated_id_str = '_'.join(m_id)
            ori_id_str = '_'.join(_id)
            id_line = f'>{ori_id_str}::{mutated_id_str}\n'
            mutated_fa_fh.write(id_line)
            mutated_fa_fh.write(m_seq + '\n')

def mutate_seq_per_seq_all(seq):
    for i in range(len(seq)):
        for c in 'ACGT':
            if seq[i].upper() == c:
                continue
            yield i, seq[i], c, seq[:i] + c + seq[i + 1:]

def mutate_seq_all(seq_fa_dict, mutated_fa):
    '''
    mutate every base in the seq
    '''
    mutated_fa_fh = open(mutated_fa, 'w')
    for _id in seq_fa_dict:
        for pos, ref_base, alt_base, mutated_seq in mutate_seq_per_seq_all(seq_fa_dict[_id][0]):
            id_line = f'{seq_fa_dict[_id][1]}::{pos}_{ref_base}_{alt_base}\n'
            mutated_fa_fh.write(id_line)
            mutated_fa_fh.write(mutated_seq + '\n')

def get_mutated_fa(src_bed, contexted_bed, unmutated_fa, mutated_fa, mode, single_side_pad_len, genome, strand, snp_file=None):
    '''
    the bed_file is UN-contexted

    unmutated_fa_file is dst_bed_file's pair fa
    '''
    # # add context
    # add_context(src_bed, contexted_bed, mode, single_side_pad_len)
    # # bed -> seq_snp_dict
    # seq_snp_dict = read_snp(contexted_bed, snp_file)
    # # bed -> seq_fa_dict
    # seq_fa_dict = read_fa(contexted_bed, unmutated_fa, genome)
    # # mutate by dbSNP/ClinVar
    # mutate_seq(seq_snp_dict, seq_fa_dict, mutated_fa)

    # split bed file
    alu_bed = src_bed[:-4] + '_alu.bed'
    exon_bed = src_bed[:-4] + '_exon.bed'
    split(src_bed, exon_bed, alu_bed)
    # add context for alu bed file
    add_context(alu_bed, contexted_bed, mode, single_side_pad_len)
    # bed -> seq_fa_dict
    seq_fa_dict = read_fa(contexted_bed, unmutated_fa, genome, strand)
    print('unmutated num:', len(seq_fa_dict.keys()))
    # mutate at each base
    mutate_seq_all(seq_fa_dict, mutated_fa)
    return

def gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, save_table_dir=None):
    save_path = os.path.join(work_dir, 'muta.pickle')
    if os.path.exists(save_path):
        print(save_path, 'already exists!')
        return
    else:
        print(save_path, 'doesnt exist... saving...')
    all_mutated_score_dict = {}
    for _id in mutated_rslt.keys():
        # _id -> chr1:123-456(-)
        id_lst = re.split('[:()]', _id)
        id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]
        ori_alu_id_str = f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})'
        if save_table_dir:
            print((save_table_dir))
            os.makedirs(save_table_dir, exist_ok=True)
            save_table_fh = open(os.path.join(save_table_dir, ori_alu_id_str + '.tsv'), 'w')
            save_table_fh.write('AluID\tExtAluID\tOffset\tPOS\tREF\tALT\tMutatedScore\tBaselineScore\tChange\n')
        char_result_dict = {}
        for mutation, score in mutated_rslt[_id]:
            pos, ref, alt = mutation.split('_')
            change = score - baseline_rslt[_id]
            if save_table_dir:
                # pos here are from 0 to 1050 (max), the contexts are 350bp for each
                save_table_fh.write(f'{ori_alu_id_str}\t{_id}\t{pos}\t{int(id_lst[1]) + int(pos)}\t{ref}\t{alt}\t{score}\t{baseline_rslt[_id]}\t{change}\n')
            if alt not in char_result_dict:
                char_result_dict[alt] = [[], []]
            if 325 <= int(pos) < 700:
                # new_pos here are those pos from 325 to 700 (max) actually
                # the contexts are 25bp now, truncate from above original pos, in order to show only 25 bp context instead of 350 bp.
                new_pos = int(pos) - 325
                char_result_dict[alt][0].append(new_pos)
                char_result_dict[alt][1].append(change)
        if save_table_dir:
            save_table_fh.close()
        all_mutated_score_dict[tuple(id_lst)] = char_result_dict
    pickle.dump(all_mutated_score_dict, open(save_path, 'wb'))

def peak_detect(work_dir):
    # this function can be a standalone func to print peak info, or
    # used for draw_plot to draw peak on the satruation mutagenesis images
    all_mutated_score_dict = pickle.load(open(os.path.join(work_dir, 'muta.pickle'), 'rb')) 
    peak_dict = {}
    for _id, d in all_mutated_score_dict.items():
        # if _id != ('chr10', '102452009', '102452377', '-'):
            # continue
        # if idx == 5:
            # return
        seq_len = int(_id[2]) - int(_id[1])
        print(_id, seq_len)
        Y = []
        for char_i, char in enumerate('ACGT'):
            Y.append([0.0] * seq_len)
            X_char, Y_char = d[char]
            for x_i, x in enumerate(X_char):
                Y[char_i][x] = Y_char[x_i]
        Y = np.array(Y).T

        var_glo = Y.var()
        mean_glo = Y.mean()
        D_glo = var_glo / mean_glo
        print('>>> Global <<<')
        print(f'var: {var_glo}, mean: {mean_glo}, D: {D_glo}')
        if (abs(D_glo) > 0.5):
            print('Class Variance\n')

        win_size = 10
        stride = 5

        lvic_size = 20
        rvic_size = 20

        peak_win_i_lst = []
        win_i_mean_lst = []
        for win_i in range(0, seq_len - win_size, stride):
            # window
            Y_win = Y[win_i : win_i + win_size]
            var_win = Y_win.var()
            mean_win = Y_win.mean()
            win_i_mean_lst.append(mean_win)
            D_win = var_win / mean_win
            # vicinities
            # use vicinities to control the filter-out
            lvic_i = win_i - lvic_size
            rvic_i = win_i + win_size
            if lvic_i < 0:
                vic_win_left = 0
            else:
                vic_win_left = lvic_i
            if rvic_i + rvic_size >= seq_len:
                vic_win_right = seq_len
            else:
                vic_win_right = rvic_i + rvic_size
            Y_vic_win = Y[vic_win_left : vic_win_right]
            var_vic_win = Y_vic_win.var()
            mean_vic_win = Y_vic_win.mean()
            D_vic_win = var_vic_win / mean_vic_win
            if var_vic_win > var_win * 0.5:
                # print('skip because high viccinities variance!')
                continue
            
            # if abs(D_glo) < abs(D_win):
            if var_win > var_glo * 2.2:
                peak_win_i_lst.append(win_i)
                print('peak', f'[{win_i}]\tvar: {var_win}\tmean: {mean_win}\tD: {D_win}')

        peak_lst = []
        peak_mean_lst = []
        # new one to have median
        if len(peak_win_i_lst) == 0:
            print('Empty')
        elif len(peak_win_i_lst) == 1:
            peak_lst.append((peak_win_i_lst[0], peak_win_i_lst[0] + win_size))
        else:
            i = 0
            mean_lst = [win_i_mean_lst[i]//stride]
            for j in range(1, len(peak_win_i_lst)):
                if peak_win_i_lst[j - 1] + stride == peak_win_i_lst[j]:
                    pass
                elif peak_win_i_lst[j - 1] + win_size == peak_win_i_lst[j]:
                    pass
                else:
                    peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[j - 1] + win_size))
                    i = j
            if i < j:
                peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[-1] + win_size))
            if peak_win_i_lst[-2] + stride != peak_win_i_lst[-1] and peak_win_i_lst[-2] + win_size != peak_win_i_lst[-1]:
                # it's acctually peak_win_i_lst[-1] > peak_win_i_lst[-2] + win_size
                peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[-1] + win_size))
        if peak_lst and peak_lst[-1][-1] >= seq_len - 10:
            poped = peak_lst.pop()
            peak_lst.append((poped[0], seq_len - 11))
        
        for peak in peak_lst:
            # print(len(win_i_mean_lst), peak[0], peak[0]//stride, peak[1], peak[1]//stride)
            mean_lst = [win_i_mean_lst[pos//stride] for pos in range(peak[0], peak[1], stride)]
            # print(mean_lst)
            peak_mean = sum(mean_lst)/len(mean_lst)
            peak_mean_lst.append(peak_mean)
        print(peak_lst)
        print(peak_mean_lst)
        peak_dict[_id] = (peak_lst, peak_mean_lst)
    return peak_dict

def peak_detect_store():
    work_dir = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat'
    overlap_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/MOAT_v3_Alu.r.overlap.filtered.bed'
    splice_sites_dict = read_splice_sites(overlap_bed[:-4] + '_alu.bed', overlap_bed[:-4] + '_exon.bed')
    print(splice_sites_dict)
    prd_file = 'baseline_prd_y.txt'
    baseline_rslt = {}
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            baseline_rslt[line_lst[2].split('::')[1]] = float(line_lst[0])
    # run mutated
    mutated_rslt = defaultdict(list) 
    prd_file= 'mutated_prd_y.txt'
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            # 0.984420120716095	3.0	chr10_101017466_101017851_+_h38_mk_AluSq2_1_335_1_bothfix_0_0_NA::3_t_C
            # id line: h38_mk_AluSq2_2_329_2_bothfix_0_0_NA::chr11:112229347-112229726(-)::378_A_T
            mutated_rslt[line_lst[2].split('::')[1]].append((line_lst[2].split('::')[-1], float(line_lst[0]))) # {id:(mutation, score)}
    # generate mutated score dict
    # gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt)
    gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
    # peak detection
    peak_dict = peak_detect(work_dir)
    peak_all_bed = work_dir + '/peaks/' + f'peaks_all.bed'
    peak_all_fa = work_dir + '/peaks/' + f'peaks_all.fa'
    peak_all_fh = open(peak_all_bed, 'w') 
    # peak store
    for _chr in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']:
        peak_bed = work_dir + '/peaks/' + f'peaks_{_chr}.bed'
        peak_fa = work_dir + '/peaks/' + f'peaks_{_chr}.fa'
        genome = '/home/zhe29/Projects/eXAlu/data/shared/hg38/hg38c.fa'
        peak_fh = open(peak_bed, 'w') 
        for _id in peak_dict:
            # _id is ext id, 350 bp context
            # but those peaks are calculated on the truncated 25 bp context instead
            # k,v: ('chr10', '101017141', '101018176', '+') [(165, 180), (310, 325)]
            if _id[0] != _chr:
                continue
            ori_alu_id = (_id[0], str(int(_id[1]) + 350), str(int(_id[2]) - 350), _id[3])
            for i in range(len(peak_dict[_id][0])):
                peak = peak_dict[_id][0][i]
                if _id[3] == '+':
                    peak_start = int(ori_alu_id[1]) + (peak[0] - 25) - 2
                    peak_end = int(ori_alu_id[1]) + (peak[1] - 25) + 2
                else:
                    peak_start = int(ori_alu_id[2]) - (peak[1] - 25) - 2
                    peak_end = int(ori_alu_id[2]) - (peak[0] - 25) + 2
                exon_left = min(int(splice_sites_dict[ori_alu_id][0][0]), int(splice_sites_dict[ori_alu_id][1][0]))
                exon_right = max(int(splice_sites_dict[ori_alu_id][0][0]), int(splice_sites_dict[ori_alu_id][1][0]))
                # if exon_left < peak_start and peak_end < exon_right:
                pos_neg = 'Pos' if peak_dict[_id][1][i] >= 0 else 'Neg'
                peak_id = '_'.join(ori_alu_id) + '_' + str(peak[0] - 25) + '_' + str(peak[1] - 25) + '_' + str(pos_neg) # + '_' + '_'.join([str(exon_left), str(exon_right)])
                # we need seq on the opposite strand (exon)
                if _id[3] == '+':
                    writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t-\n'
                else:
                    writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t+\n'
                peak_fh.write(writing_line)
                peak_all_fh.write(writing_line)
        peak_fh.close()
        alu_bed = pybedtools.BedTool(peak_bed)
        alu_bed.sequence(fi=genome, fo=peak_fa, name=True, s=True)
    peak_all_fh.close()
    alu_bed = pybedtools.BedTool(peak_all_bed)
    alu_bed.sequence(fi=genome, fo=peak_all_fa, name=True, s=True)


def peak_detect_store_filter_by_spliceai():
    work_dir = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat'
    overlap_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/MOAT_v3_Alu.r.overlap.filtered.bed'
    spliceai_dir = '/home/zhe29/Projects/eXAlu/data/spliceai/moat/output'
    splice_sites_dict = read_splice_sites(overlap_bed[:-4] + '_alu.bed', overlap_bed[:-4] + '_exon.bed')
    prd_file = 'baseline_prd_y.txt'
    baseline_rslt = {}
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            baseline_rslt[line_lst[2].split('::')[1]] = float(line_lst[0])
    # run mutated
    mutated_rslt = defaultdict(list) 
    prd_file= 'mutated_prd_y.txt'
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            # 0.984420120716095	3.0	chr10_101017466_101017851_+_h38_mk_AluSq2_1_335_1_bothfix_0_0_NA::3_t_C
            # id line: h38_mk_AluSq2_2_329_2_bothfix_0_0_NA::chr11:112229347-112229726(-)::378_A_T
            mutated_rslt[line_lst[2].split('::')[1]].append((line_lst[2].split('::')[-1], float(line_lst[0]))) # {id:(mutation, score)}
    # generate mutated score dict
    # gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt)
    gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
    # peak detection
    peak_dict = peak_detect(work_dir)
    peak_all_bed = work_dir + '/peaks/' + f'peaks_all.bed'
    peak_all_fa = work_dir + '/peaks/' + f'peaks_all.fa'
    peak_all_fh = open(peak_all_bed, 'w') 
    # peak store
    for _chr in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']:
        peak_bed = work_dir + '/peaks/' + f'peaks_{_chr}.bed'
        peak_fa = work_dir + '/peaks/' + f'peaks_{_chr}.fa'
        genome = '/home/zhe29/Projects/eXAlu/data/shared/hg38/hg38c.fa'
        peak_fh = open(peak_bed, 'w') 
        for _id in peak_dict:
            # _id is ext id, 350 bp context
            # but those peaks are calculated on the truncated 25 bp context instead
            # k,v: ('chr10', '101017141', '101018176', '+') [(165, 180), (310, 325)]
            if _id[0] != _chr:
                continue
            ori_alu_id = (_id[0], str(int(_id[1]) + 350), str(int(_id[2]) - 350), _id[3])
            for i in range(len(peak_dict[_id][0])):
                peak = peak_dict[_id][0][i]
                if _id[3] == '+':
                    peak_start = int(ori_alu_id[1]) + (peak[0] - 25) - 2
                    peak_end = int(ori_alu_id[1]) + (peak[1] - 25) + 2
                else:
                    peak_start = int(ori_alu_id[2]) - (peak[1] - 25) - 2
                    peak_end = int(ori_alu_id[2]) - (peak[0] - 25) + 2
                exon_left = min(int(splice_sites_dict[ori_alu_id][0][0]), int(splice_sites_dict[ori_alu_id][1][0]))
                exon_right = max(int(splice_sites_dict[ori_alu_id][0][0]), int(splice_sites_dict[ori_alu_id][1][0]))
                if exon_left < peak_start and peak_end < exon_right:
                    pos_neg = 'Pos' if peak_dict[_id][1][i] >= 0 else 'Neg'
                    if ori_alu_id[3] == '+':
                        spliceai_output_file = f'{ori_alu_id[0]}:{int(ori_alu_id[1]) - 5350}-{int(ori_alu_id[2]) + 5350}(-).tsv'
                    else:
                        spliceai_output_file = f'{ori_alu_id[0]}:{int(ori_alu_id[1]) - 5350}-{int(ori_alu_id[2]) + 5350}(+).tsv'
                    with open(os.path.join(spliceai_dir, spliceai_output_file), 'r') as spliceai_fh:
                        pos_score = {}
                        for line in spliceai_fh.readlines():
                            line_lst = line.rstrip().split('\t')
                            pos_score[int(line_lst[2])] = (float(line_lst[3]), float(line_lst[4]))
                        skip_this_peak = False
                        for peak_pos_i in range(peak_start, peak_end + 1):
                            print(peak_pos_i, peak_start, peak_end)
                            print(pos_score[peak_pos_i])
                            th = 0.003
                            if pos_score[peak_pos_i][0] >= th or pos_score[peak_pos_i][1] >= th:
                                skip_this_peak = True
                                break
                        if skip_this_peak:
                            continue
                    peak_id = '_'.join(ori_alu_id) + '_' + str(peak[0] - 25) + '_' + str(peak[1] - 25) + '_' + str(pos_neg) # + '_' + '_'.join([str(exon_left), str(exon_right)])
                    # we need seq on the opposite strand (exon)
                    if _id[3] == '+':
                        writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t-\n'
                    else:
                        writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t+\n'
                    peak_fh.write(writing_line)
                    peak_all_fh.write(writing_line)
        peak_fh.close()
        alu_bed = pybedtools.BedTool(peak_bed)
        alu_bed.sequence(fi=genome, fo=peak_fa, name=True, s=True)
    peak_all_fh.close()
    alu_bed = pybedtools.BedTool(peak_all_bed)
    alu_bed.sequence(fi=genome, fo=peak_all_fa, name=True, s=True)

def draw_plot(work_dir, splice_sites_dict, plot_mode='fixed', strand=True, has_peaks=False, peak_dict=None):
    # has_peaks controls if we draw peaks range on the satruation mutagenesis imgaes
    # if yes, then draw 
    # plot mode fixed means: fix the range of y-axis from -0.3 to 0.3
    # adaptive means
    # for _id in baseline_rslt.keys():
    char_color_dict = {'A': 'green', 'C': 'blue', 'G': 'orange', 'T': 'red'}
    all_mutated_score_dict = pickle.load(open(os.path.join(work_dir, 'muta.pickle'), 'rb')) 
    for id_lst, char_result_dict in all_mutated_score_dict.items():
        if plot_mode == 'adaptive':
            max_change_abs = 0.0
            for char, char_result in char_result_dict.items():
                for change in char_result[1]:
                    if abs(change) > max_change_abs:
                        max_change_abs = abs(change)
            max_change_abs *= 1.1
        # 25: context displayed
        # 350: actual context
        # peaks are calculated on 25bp displayed context 
        alu_len = int(id_lst[2]) - int(id_lst[1]) - 325 * 2
        plt.figure(figsize=(40, 4.8), dpi=150)
        plt.hlines([0], xmin=0, xmax=alu_len + 1, colors=['gray'], linestyles='dashed')
        # plt.vlines([0], ymin=-1, ymax=1, colors=['gray'], linestyles='dashed')
        # alu boundary lines
        plt.axvline(25, color='gray', linestyle='dashed') 
        if strand:
            if plot_mode == 'adaptive':
                plt.text(25, max_change_abs*1.1, f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})', ha='center')
            elif plot_mode == 'fixed':
                plt.text(25, 0.31, f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})', ha='center')
        plt.axvline(alu_len - 25, color='gray', linestyle='dashed')
        # exon lines
        ori_alu_id = (id_lst[0], str(int(id_lst[1]) + 350), str(int(id_lst[2]) - 350), id_lst[3])
        exon_info = splice_sites_dict[ori_alu_id]
        for loc, s in exon_info:
            if s == 'donor':
                t = '--I--Don.--E--'
            else:
                t = '--E--Acc.--I--'
            if id_lst[3] == '+':
                plt.axvline(int(loc) - int(id_lst[1]) - 325, color='red', linestyle='dashed')
                if plot_mode == 'adaptive':
                    plt.text(int(loc) - int(id_lst[1]) - 325 + 0.2, max_change_abs*1.1, t, ha='center')
                elif plot_mode == 'fixed':
                    plt.text(int(loc) - int(id_lst[1]) - 325 + 0.2, 0.31, t, ha='center')
            else:
                plt.axvline(alu_len - (int(loc) - int(id_lst[1]) - 325), color='red', linestyle='dashed')
                if plot_mode == 'adaptive':
                    plt.text(alu_len - (int(loc) - int(id_lst[1]) - 325) + 0.2, max_change_abs*1.1, t, ha='center')
                elif plot_mode == 'fixed':
                    plt.text(alu_len - (int(loc) - int(id_lst[1]) - 325) + 0.2, 0.31, t, ha='center') 
        if has_peaks:
            for peak in peak_dict[id_lst][0]:
                if plot_mode == 'adaptive':
                    plt.arrow(peak[0], -max_change_abs*1.1, peak[1] - peak[0], 0, shape='full')
                    # plt.axvline(int(loc) - int(id_lst[1]), color='red', linestyle='dashed')
                elif plot_mode == 'fixed':
                    plt.annotate(text='', xy=(peak[0], -0.29), xytext=(peak[1], -0.29), arrowprops=dict(arrowstyle='<->', color='purple', shrinkA=0, shrinkB=0))
                    # plt.text(peak[0], 0.31, 'peak', ha='center')
        for char in 'ACGT':
            if char not in char_result_dict.keys():
                continue
            X, Y = char_result_dict[char]
            plt.scatter(X, Y, c=char_color_dict[char], s=20, label=char)
        if plot_mode == 'adaptive':
            plt.ylim(-max_change_abs, +max_change_abs)
        elif plot_mode == 'fixed':
            plt.ylim(-0.3, +0.3)
        plt.xlim(0, alu_len + 1)
        plt.xticks(np.arange(0, alu_len, 5))
        current_ticks = plt.xticks()[0]
        new_ticks = current_ticks - 25
        plt.xticks(current_ticks, new_ticks)
        plt.legend()

        img_dir = os.path.join(work_dir, 'imgs')
        os.makedirs(img_dir, exist_ok=True)
        id_str = '_'.join(ori_alu_id)
        plt.savefig(os.path.join(img_dir, f'{id_str}.png'))
        plt.close()


def gen_saturation_mutagenesis_graphs():
    # run the model
    for context_mode, single_side_pad_len, strand in [\
        ('bothfix', 350, True)]:
        print('>>>>>>>>>>>', context_mode, single_side_pad_len, '<<<<<<<<<<<<<')
        infer_set = 'simpleinfer'
        model_wts_name = '/home/zhe29/Projects/eXAlu/data/curated_data_MOAT/padding_bothfix_350/model_pts/ead_v1.4t_MOAT_runmode_2_padding_bothfix_350_dc_115_cnet_20231106_205116/epoch_116.pt'
        work_dir = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat'
        # work_dir = os.path.join('/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/IFNAR2/')
        src_bed = os.path.join(work_dir, 'MOAT_v3_Alu.r.overlap.filtered.bed')
        # src_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/three.bed'
        # src_bed = os.path.join(work_dir, 'IFNAR2.overlap.bed')
        contexted_bed = os.path.join(work_dir, 'contexted_test.bed')
        unmutated_fa = os.path.join(work_dir, 'unmutated_test.fa')
        mutated_fa = os.path.join(work_dir, 'mutated_test.fa')
        # snp_file = '/home/zhe29/Projects/eXAlu/data/SNP/SNPs_shrink/common_all_20180418.vcf'
        genome = '/home/zhe29/Projects/eXAlu/data/shared/hg38/hg38c.fa'
        # generate files needed for this experiment
        get_mutated_fa(src_bed, contexted_bed, unmutated_fa, mutated_fa, context_mode, single_side_pad_len, genome, strand)
        # read exon info
        splice_sites_dict = read_splice_sites(src_bed[:-4] + '_alu.bed', src_bed[:-4] + '_exon.bed')
        # run baseline
        infer_file = os.path.join(work_dir, 'unmutated_test.fa')
        prd_file = 'baseline_prd_y.txt'
        run_ead(strand,
                model_name='cnet',
                run_mode=4,
                work_dir=work_dir,
                infer_set='baseline', # removed dataset_comment, so use infer set to label baseline and mutated
                context_mode=context_mode,
                single_side_pad_len=single_side_pad_len,
                infer_file=infer_file,
                prd_file=prd_file,
                model_wts_name=model_wts_name
                )
        baseline_rslt = {}
        with open(work_dir + '/' + prd_file, 'r') as prd_fh:
            for line in prd_fh.readlines():
                line_lst = line.rstrip()[1:].split('\t')
                baseline_rslt[line_lst[2].split('::')[1]] = float(line_lst[0])
        # run mutated
        mutated_rslt = defaultdict(list) 
        infer_file = os.path.join(work_dir, 'mutated_test.fa')
        prd_file= 'mutated_prd_y.txt'
        run_ead(strand,
                model_name='cnet',
                run_mode=4,
                work_dir=work_dir,
                infer_set='mutated', # removed dataset_comment, so use infer set to label baseline and mutated
                context_mode=context_mode,
                single_side_pad_len=single_side_pad_len,
                infer_file=infer_file,
                prd_file=prd_file,
                model_wts_name=model_wts_name
        )
        with open(work_dir + '/' + prd_file, 'r') as prd_fh:
            for line in prd_fh.readlines():
                line_lst = line.rstrip()[1:].split('\t')
                # 0.984420120716095	3.0	chr10_101017466_101017851_+_h38_mk_AluSq2_1_335_1_bothfix_0_0_NA::3_t_C
                # id line: h38_mk_AluSq2_2_329_2_bothfix_0_0_NA::chr11:112229347-112229726(-)::378_A_T
                mutated_rslt[line_lst[2].split('::')[1]].append((line_lst[2].split('::')[-1], float(line_lst[0]))) # {id:(mutation, score)}
        # generate mutated score dict
        # gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt)
        gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
        # peak detection
        peak_dict = peak_detect(work_dir)
        # draw
        draw_plot(work_dir, splice_sites_dict, 'fixed', strand, has_peaks=True, peak_dict=peak_dict)



def gen_genesplicer_input(ext_exon_seq_dict, dst_dir):
    os.makedirs(dst_dir, exist_ok=True)
    for _id, ext_exon in ext_exon_seq_dict.items():
        with open(os.path.join(dst_dir, '_'.join(_id) + '.txt'), 'w') as dst_fh:
            dst_fh.write(ext_exon[2] + '\n')
            dst_fh.write(ext_exon[1] + '\n')

def run_genesplicer(classes_dict, ext_exon_seq_dict, genesplicer_input_dir, result_file):
    result_fh = open(result_file, 'w')
    result_fh.write('alu_id\texon_id\tdl_alu_id\text_exon_id\tdonor_selected\tacceptor_selected\tdonor_sites\tacceptor_sites\n') 
    for _id, class_bin in classes_dict.items():
        ext_exon_id = ext_exon_seq_dict[_id][0]
        donor_dict = defaultdict(list)
        acceptor_dict = defaultdict(list)
        genesplicer_fa = os.path.join(genesplicer_input_dir, '_'.join(_id) + '.txt')
        if not os.path.exists(genesplicer_fa):
            continue
        cmd = f'/home/zhe29/Software/GeneSplicer/bin/linux/genesplicer {genesplicer_fa} /home/zhe29/Software/GeneSplicer/human -a -1000 -d -1000 -i -1 -e -1'
        result = subprocess.check_output(cmd, shell=True).rstrip()
        for line in result.decode('UTF-8').split('\n'):
            if line == '':
                continue
            line_lst = line.split(' ')
            end5 = int(line_lst[0]) - 1
            end3 = int(line_lst[1]) - 1
            if end5 < end3:
                end3 += 1
            else:
                continue
            end5 = str(end5)
            end3 = str(end3)

            if line_lst[4] == 'donor':
                donor_dict[_id].append(end5 + '_' + end3 + '_' + line_lst[2])
            elif line_lst[4] == 'acceptor':
                acceptor_dict[_id].append(end5 + '_' + end3 + '_' + line_lst[2])
        if len(donor_dict[_id]) == 0:
            donor_str = 'Empty'
        else:
            donor_str = ','.join(donor_dict[_id])
        if len(acceptor_dict[_id]) == 0:
            acceptor_str = 'Empty'
        else:
            acceptor_str = ','.join(acceptor_dict[_id])
        _id_str = '_'.join(_id)
        exon_id_str = '_'.join((ext_exon_id[0], str(int(ext_exon_id[1]) + 430), str(int(ext_exon_id[2]) - 430), ext_exon_id[3]))
        dl_alu_id_str = '_'.join((_id[0], str(int(_id[1]) - 350), str(int(_id[2]) + 350), _id[3]))
        ext_exon_id_str = '_'.join(ext_exon_id)
        donor_selected, acceptor_selected = class_bin
        result_fh.write(f'{_id_str}\t{exon_id_str}\t{dl_alu_id_str}\t{ext_exon_id_str}\t{donor_selected}\t{acceptor_selected}\t{donor_str}\t{acceptor_str}\n') 
    result_fh.close()

def read_ext_exon_seq(alu_bed_file, ext_exon_bed_file, ext_exon_fa_file, genome):
    # read splice sites base on exon file
    alu_bed_fh = open(alu_bed_file, 'r')
    ext_exon_bed_fh = open(ext_exon_bed_file, 'r')
    alu_bed_reader = csv.reader(alu_bed_fh, delimiter='\t')
    ext_exon_bed_reader = csv.reader(ext_exon_bed_fh, delimiter='\t')
    ext_exon_seq_dict = {}
    for alu_row, ext_exon_row in zip(alu_bed_reader, ext_exon_bed_reader):
        alu_id = (alu_row[0], alu_row[1], alu_row[2], alu_row[5])
        ext_exon_id = (ext_exon_row[0], ext_exon_row[1], ext_exon_row[2], ext_exon_row[5])
        ext_exon_seq_dict[alu_id] = ext_exon_id
    alu_bed_fh.close()
    ext_exon_bed_fh.close()

    ext_exon_seq_fa_dict = read_fa(ext_exon_bed_file, ext_exon_fa_file, genome, strand=True)
    print(ext_exon_seq_fa_dict)
    
    for alu_id, ext_exon_id in ext_exon_seq_dict.items():
        ext_exon_seq_dict[alu_id] = (ext_exon_id, ext_exon_seq_fa_dict[ext_exon_id][0], ext_exon_seq_fa_dict[ext_exon_id][1])

    return ext_exon_seq_dict

def check_peak_splice_sites_matches(splice_sites_dict, peak_dict):
    class_dict = {}
    # TODO integrate this load func
    for ext_alu_id, peak_lst in peak_dict.items():
        ori_alu_id = (ext_alu_id[0], str(int(ext_alu_id[1]) + 350), str(int(ext_alu_id[2]) - 350), ext_alu_id[3])
        exon_info = splice_sites_dict[ori_alu_id]
        class_three = [0, 0]
        # class_three's elems meaning: 
        # index
        # 0: left
        # 1: right
        # value
        # 0: splice site not show
        # 1: splice show but no peak match
        # 2: splice show and peak match
        offset = 10
        for loc, s in exon_info:
            exon_local_i = int(loc) - int(ext_alu_id[1])
            if s == 'donor':
                if int(ori_alu_id[1]) <= int(loc) and int(loc) <= int(ori_alu_id[2]):
                    class_three[0] = 1
                    for peak in peak_lst:
                        if peak[0] - offset <= exon_local_i and exon_local_i <= peak[1] + offset:
                            class_three[0] = 2
            else:
                if int(ori_alu_id[1]) <= int(loc) and int(loc) <= int(ori_alu_id[2]):
                    class_three[1] = 1
                    for peak in peak_lst:
                        if peak[0] - offset <= exon_local_i and exon_local_i <= peak[1] + offset:
                            class_three[1] = 2 
        class_dict[ori_alu_id] = tuple(class_three)
    return class_dict

def read_classes_file(classes_dir):
    # deprecated
    # classes_dict: {ori_alu_id:(Donor, Acceptor)}
    classes_dict = {}
    for f in os.listdir(classes_dir):
        class_str = f[:-4]
        fh = open(os.path.join(classes_dir, f), 'r')
        for line in fh.readlines():
            id_str = line.split('.')[0]
            id_lst = re.split('[:()]', id_str)
            id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]
            # above id is dl_alu_id
            _id = (id_lst[0], str(int(id_lst[1]) + 350), str(int(id_lst[2]) - 350), id_lst[3])
            if class_str == 'Both':
                class_bin = (True, True)
            elif class_str == 'None':
                class_bin = (False, False)
            elif class_str == 'Left':
                class_bin = (True, False)
            elif class_str == 'Right':
                class_bin = (False, True)
            if _id in classes_dict.keys() and class_bin != classes_dict[_id]:
                print('repeated', _id, classes_dict[_id], class_bin)
                raise(ValueError('classes confilict!'))
            classes_dict[_id] = class_bin
    return classes_dict

def draw_histograms(result_file, histogram_dir):
    color_dict = {'Unselected': seaborn.color_palette('Set1')[0], 'Selected':seaborn.color_palette('Set1')[1]}
    read_fh = open(result_file, 'r')
    reader = csv.reader(read_fh, delimiter='\t')
    next(reader)
    # donor
    alu_id_lst, ext_id_lst, label_lst, score_lst = [], [], [], []
    for row in reader:
        alu_coord = [int(v) for v in row[0].split('_')[1:3]]
        exon_coord = [int(v) for v in row[1].split('_')[1:3]]
        ext_exon_coord = [int(v) for v in row[3].split('_')[1:3]]
        strand = row[3].split('_')[-1]
        if exon_coord[1] < alu_coord[0] or exon_coord[1] > alu_coord[1] and strand == '+':
            continue
        if exon_coord[0] < alu_coord[0] or exon_coord[0] > alu_coord[1] and strand == '-':
            continue
        # if exon_coord[1] < alu_coord[0] or exon_coord[1] > alu_coord[1]:
        #     continue
        if strand == '+':
            site_start = ext_exon_coord[1] - ext_exon_coord[0] - 430
        else:
            site_start = ext_exon_coord[1] - ext_exon_coord[0] - 429
        for site in row[6].split(','):
            site_lst = site.split('_')
            if int(site_lst[0]) == site_start:
                if row[4] == '0':
                    break
                elif row[4] == '1':
                    label_lst.append('Unselected')
                    # break
                else:
                    label_lst.append('Selected')
                score_lst.append(float(site_lst[2]))
                alu_id_lst.append(row[0])
                ext_id_lst.append(row[3])
                break
    print(len(score_lst), len(alu_id_lst), len(ext_id_lst), len(label_lst))
    df = pd.DataFrame({'alu_id': alu_id_lst, 'ext_id':ext_id_lst, 'Selection': label_lst, 'Score': score_lst})
    print(df)
    # df = df.loc[df['Selection'] == 'Selected']
    # print(df)
    ax = seaborn.histplot(x='Score', hue='Selection', data=df, palette=color_dict, bins=50, common_norm=False, stat='probability', multiple='dodge')
    plt.savefig(histogram_dir + 'donor_right_auto_10.png')
    plt.close()
    read_fh.close()

    read_fh = open(result_file, 'r')
    reader = csv.reader(read_fh, delimiter='\t')
    next(reader)
    # acceptor
    alu_id_lst, ext_id_lst, label_lst, score_lst = [], [], [], []
    for row in reader:
        alu_coord = [int(v) for v in row[0].split('_')[1:3]]
        exon_coord = [int(v) for v in row[1].split('_')[1:3]]
        ext_exon_coord = [int(v) for v in row[3].split('_')[1:3]]
        strand = row[3].split('_')[-1]
        # if exon_coord[0] < alu_coord[0] or exon_coord[0] > alu_coord[1]:
            # continue
        if exon_coord[1] < alu_coord[0] or exon_coord[1] > alu_coord[1] and strand == '-':
            continue
        if exon_coord[0] < alu_coord[0] or exon_coord[0] > alu_coord[1] and strand == '+':
            continue
        if strand == '+':
            site_start = 427
        else:
            site_start = 428
        for site in row[7].split(','):
            site_lst = site.split('_')
            if int(site_lst[0]) == site_start:
                if row[5] == '0':
                    break
                elif row[5] == '1':
                    label_lst.append('Unselected')
                    # break
                else:
                    label_lst.append('Selected')
                score_lst.append(float(site_lst[2]))
                alu_id_lst.append(row[0])
                ext_id_lst.append(row[3])
                break
    print(len(score_lst), len(alu_id_lst), len(ext_id_lst), len(label_lst))
    df = pd.DataFrame({'alu_id': alu_id_lst, 'ext_id':ext_id_lst, 'Selection': label_lst, 'Score': score_lst})
    print(df)
    # df = df.loc[(df['Selection'] == 'Selected')]
    # print(df)
    ax = seaborn.histplot(x='Score', hue='Selection', data=df, palette=color_dict, bins=50, common_norm=False, stat='probability', multiple='dodge')
    plt.savefig(histogram_dir + 'acceptor_left_auto_10.png')
    plt.close()
    return

def copy_into_classes_dir(classes_dir, src_imgs):
    import shutil
    for f in os.listdir(classes_dir):
        class_str = f[:-4]
        fh = open(os.path.join(classes_dir, f), 'r')
        os.makedirs(os.path.join(classes_dir, f[:-4]), exist_ok=True)
        for line in fh.readlines():
            id_str = line.split('.')[0]
            src_img_path = os.path.join(src_imgs, id_str + '.png')
            if os.path.isfile(src_img_path):
                shutil.copyfile(src_img_path, os.path.join(classes_dir, f[:-4], id_str + '.png'))
            id_lst = re.split('[:()]', id_str)
            id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]


def gen_genesplicer_histograms(manual_classes=False):
    genome = '/home/zhe29/Projects/eXAlu/data/shared/hg38/hg38c.fa'
    work_dir = '/home/zhe29/Projects/eXAlu/data_analysis/moat_analysis_350/genesplicer_histogram/'
    src_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/three.bed'
    histogram_dir = work_dir + 'imgs/'
    genesplicer_input_dir = work_dir + 'genesplicer_input'
    alu_bed = work_dir + 'MOAT_v3_Alu.r.overlap.filtered_alu.bed'
    exon_bed = work_dir + 'MOAT_v3_Alu.r.overlap.filtered_exon.bed'
    ext_exon_bed = work_dir + 'MOAT_v3_Alu.r.overlap.filtered_exon_ext430.bed'
    ext_exon_fa = work_dir + 'MOAT_v3_Alu.r.overlap.filtered_exon_ext430.fa'
    classes_dir = '/home/zhe29/Projects/eXAlu/data_analysis/moat_analysis/genesplicer_histogram/classes'
    result_file = work_dir + 'result.tsv'
    split(src_bed, exon_bed, alu_bed)
    add_context(exon_bed, ext_exon_bed, mode='bothfix', single_side_pad_len=430, len_check=False)
    ext_exon_seq_dict = read_ext_exon_seq(alu_bed, ext_exon_bed, ext_exon_fa, genome)
    if manual_classes:
        classes_dict = read_classes_file(classes_dir)
    else:
        peak_dict = peak_detect('/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/MOAT_cnet_v55/')
        splice_sites_dict = read_splice_sites(alu_bed_file=alu_bed, exon_bed_file=exon_bed) 
        classes_dict = check_peak_splice_sites_matches(splice_sites_dict, peak_dict)
    # gen_genesplicer_input(ext_exon_seq_dict, genesplicer_input_dir)
    # print(ext_exon_seq_dict.keys())
    run_genesplicer(classes_dict, ext_exon_seq_dict, genesplicer_input_dir, result_file)
    draw_histograms(result_file, histogram_dir)


def cp_moat_imgs_spliceai():
    import shutil
    peak_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/peaks_inside_exon_and_spliceai/peaks_all.bed'
    # Define the source and destination file paths
    imgs_src = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/imgs'
    imgs_dst = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/imgs_interest'

    with open(peak_bed, 'r') as fh:
        for line in fh.readlines():
            line_lst = line.rstrip().split('\t')
            _id = line_lst[3].split('_')[0:4]
            file_name = f'{_id[0]}_{int(_id[1])}_{int(_id[2])}_{_id[3]}.png'
            shutil.copy(os.path.join(imgs_src, file_name), os.path.join(imgs_dst, file_name))

def cp_moat_imgs_polya():
    import shutil
    peak_bed = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat_del/stats_test/Right_Acc'
    # Define the source and destination file paths
    imgs_src = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/imgs'
    imgs_dst = '/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/muta_moat/imgs_polya_mightbe'

    with open(peak_bed, 'r') as fh:
        for line in fh.readlines():
            line_lst = line.rstrip().split('\t')
            if line_lst[0] == 'Alu_ID':
                continue
            if line_lst[2] == line_lst[3] == line_lst[4] == 'Empty':
                continue
            if line_lst[5] == line_lst[6] == line_lst[7] == 'Empty':
                continue
            _id = line_lst[0]
            file_name = _id + '.png'
            shutil.copy(os.path.join(imgs_src, file_name), os.path.join(imgs_dst, file_name))
    
def main():
    # gen_saturation_mutagenesis_graphs()
    peak_detect_store()
    # peak_detect_store_filter_by_spliceai()
    # cp_moat_imgs_spliceai()
    # cp_moat_imgs_polya()

    # gen_genesplicer_histograms()
    # copy_into_classes_dir('/home/zhe29/Projects/eXAlu/data_analysis/moat_analysis/genesplicer_histogram/classes','/home/zhe29/Projects/eXAlu/data/curated_data_simpleinfer/padding_bothfix_350/MOAT_cnet_v55/imgs_bk')



main()
