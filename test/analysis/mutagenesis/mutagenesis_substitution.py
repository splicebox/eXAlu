import os
from collections import defaultdict
from exalu.data_preprocess.read_tissue_alu import add_context
from exalu.run_model import run_ead
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import gen_mutated_score_dict, read_fa


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

def get_mutated_fa(unmutated_fa, mutated_fa, genome, strand=True, src_bed=None, contexted_bed=None, mode='bothfix', single_side_pad_len=350):
    '''
    src_bed and contexted_bed should be passed together, or None together
    unmutated_fa_file is dst_bed_file's pair fa
    '''
    if src_bed:
        # add context for alu bed file
        add_context(src_bed, contexted_bed, mode, single_side_pad_len)
    # bed -> seq_fa_dict
    seq_fa_dict = read_fa(unmutated_fa, genome, strand, bed_file=contexted_bed)
    print('unmutated num:', len(seq_fa_dict.keys()))
    mutate_seq_all(seq_fa_dict, mutated_fa)

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
        print(_id, seq_len)
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
        peak_dict[_id] = (peak_lst, peak_mean_lst)
    return peak_dict

def peak_detect_store(work_dir):
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
    gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
    # peak detection
    peak_dict = peak_detect(work_dir)
    os.makedirs(work_dir + '/peaks/', exist_ok=True)
    peak_all_bed = work_dir + '/peaks/' + f'peaks_all.bed'
    peak_all_fh = open(peak_all_bed, 'w') 
    # peak store
    for _chr in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']:
        peak_bed = work_dir + '/peaks/' + f'peaks_{_chr}.bed'
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
    peak_all_fh.close()
    return peak_dict

def draw_plot(work_dir, plot_mode='fixed', strand=True, has_peaks=False, peak_dict=None, no_alu_boundaries=False):
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
        if not no_alu_boundaries:
            plt.axvline(25, color='gray', linestyle='dashed') 
            plt.axvline(alu_len - 25, color='gray', linestyle='dashed')
        if strand:
            if plot_mode == 'adaptive':
                plt.text(25, max_change_abs*1.1, f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})', ha='center')
            elif plot_mode == 'fixed':
                plt.text(25, 0.31, f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})', ha='center')
        ori_alu_id = (id_lst[0], str(int(id_lst[1]) + 350), str(int(id_lst[2]) - 350), id_lst[3])
        if has_peaks:
            for peak in peak_dict[id_lst][0]:
                if plot_mode == 'adaptive':
                    # plt.arrow(peak[0], -max_change_abs*1.1, peak[1] - peak[0], 0, shape='full')
                    plt.annotate(text='', xy=(peak[0], -max_change_abs), xytext=(peak[1], -max_change_abs), arrowprops=dict(arrowstyle='<->', color='purple', shrinkA=0, shrinkB=0))
                    # plt.axvline(int(loc) - int(id_lst[1]), color='red', linestyle='dashed')
                elif plot_mode == 'fixed':
                    plt.annotate(text='', xy=(peak[0], -0.29), xytext=(peak[1], -0.29), arrowprops=dict(arrowstyle='<->', color='purple', shrinkA=0, shrinkB=0))
                    # plt.text(peak[0], 0.31, 'peak', ha='center')
        for char in 'ACGT':
            if char not in char_result_dict.keys():
                continue
            X, Y = char_result_dict[char]
            # if id_lst[0] == 'chr12':
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


def gen_saturation_mutagenesis_graphs_substitution(work_dir, model_wts_name, genome=None, alu_bed_file=None, alu_fa_file=None, plot_mode='fixed', has_peaks=True, no_alu_boundaries=False):
    os.makedirs(work_dir, exist_ok=True)
    if alu_bed_file:
        src_bed = alu_bed_file
        contexted_bed = os.path.join(work_dir, 'contexted_test.bed')
        unmutated_fa = os.path.join(work_dir, 'unmutated_test.fa')
        mutated_fa = os.path.join(work_dir, 'mutated_test.fa')
        get_mutated_fa(unmutated_fa, mutated_fa, genome, strand=True, src_bed=src_bed, contexted_bed=contexted_bed, mode='bothfix', single_side_pad_len=350)
    elif alu_fa_file:
        unmutated_fa = alu_fa_file
        mutated_fa = os.path.join(work_dir, 'mutated_test.fa')
        get_mutated_fa(unmutated_fa, mutated_fa, genome, strand=True, src_bed=None, contexted_bed=None, mode='bothfix', single_side_pad_len=350)
    infer_file = unmutated_fa
    prd_file = 'baseline_prd_y.txt'
    run_ead(strand=True, run_mode=4, work_dir=work_dir, infer_set='simpleinfer', context_mode='bothfix', single_side_pad_len=350,
            infer_file=infer_file, prd_file=prd_file, model_wts_name=model_wts_name)
    baseline_rslt = {}
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            baseline_rslt[line_lst[2].split('::')[1]] = float(line_lst[0])
    # run mutated
    mutated_rslt = defaultdict(list) 
    infer_file = mutated_fa
    prd_file= 'mutated_prd_y.txt'
    run_ead(strand=True, run_mode=4, work_dir=work_dir, infer_set='simpleinfer', context_mode='bothfix', single_side_pad_len=350,
            infer_file=infer_file, prd_file=prd_file, model_wts_name=model_wts_name)
    with open(work_dir + '/' + prd_file, 'r') as prd_fh:
        for line in prd_fh.readlines():
            line_lst = line.rstrip()[1:].split('\t')
            # 0.984420120716095	3.0	chr10_101017466_101017851_+_h38_mk_AluSq2_1_335_1_bothfix_0_0_NA::3_t_C
            # id line: h38_mk_AluSq2_2_329_2_bothfix_0_0_NA::chr11:112229347-112229726(-)::378_A_T
            mutated_rslt[line_lst[2].split('::')[1]].append((line_lst[2].split('::')[-1], float(line_lst[0]))) # {id:(mutation, score)}
    # generate mutated score dict
    gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
    if has_peaks:
        peak_dict = peak_detect_store(work_dir)
        draw_plot(work_dir, plot_mode, strand=True, has_peaks=has_peaks, peak_dict=peak_dict, no_alu_boundaries=no_alu_boundaries)
    else:
        draw_plot(work_dir, plot_mode, strand=True, has_peaks=has_peaks, no_alu_boundaries=no_alu_boundaries)