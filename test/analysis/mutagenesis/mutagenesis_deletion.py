import re
import os
from collections import defaultdict
from exalu.data_preprocess.read_tissue_alu import add_context, split
from exalu.run_model import run_ead
import matplotlib.pyplot as plt
import numpy as np
import pickle
from utils import gen_mutated_score_dict, read_fa


def mutate_seq_per_seq_all(seq, ks):
    for k in ks:
        for i in range(len(seq) - k):
            yield i, seq[i], seq[i:i+k], seq[:i] + seq[i+k:]

def mutate_seq_all(seq_fa_dict, mutated_fa, ks):
    '''
    mutate every base in the seq
    '''
    mutated_fa_fh = open(mutated_fa, 'w')
    for _id in seq_fa_dict:
        for pos, ref_base, alt_base, mutated_seq in mutate_seq_per_seq_all(seq_fa_dict[_id][0], ks):
            id_line = f'{seq_fa_dict[_id][1]}::{pos}_{ref_base}_{alt_base}\n'
            mutated_fa_fh.write(id_line)
            mutated_fa_fh.write(mutated_seq + '\n')

def get_mutated_fa(unmutated_fa, mutated_fa, genome, ks, strand=True, src_bed=None, contexted_bed=None, mode='bothfix', single_side_pad_len=350):
    '''
    the bed_file is UN-contexted

    unmutated_fa_file is dst_bed_file's pair fa
    '''
    if src_bed:
        # add context for alu bed file
        add_context(src_bed, contexted_bed, mode, single_side_pad_len)
    # bed -> seq_fa_dict
    seq_fa_dict = read_fa(unmutated_fa, genome, strand, bed_file=contexted_bed)
    print('unmutated num:', len(seq_fa_dict.keys()))
    mutate_seq_all(seq_fa_dict, mutated_fa, ks)

def gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, save_table_dir=None):
    save_path = os.path.join(work_dir, 'muta.pickle')
    # if os.path.exists(save_path):
    #     print(save_path, 'already exists!')
    #     return
    # else:
    #     print(save_path, 'doesnt exist... saving...')
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
        len_result_dict = {}
        for mutation, score in mutated_rslt[_id]:
            pos, ref, alt = mutation.split('_')
            change = score - baseline_rslt[_id]
            if save_table_dir:
                # pos here are from 0 to 1050 (max), the contexts are 350bp for each
                save_table_fh.write(f'{ori_alu_id_str}\t{_id}\t{pos}\t{int(id_lst[1]) + int(pos)}\t{ref}\t{alt}\t{score}\t{baseline_rslt[_id]}\t{change}\n')
            alt_len = len(alt)
            if alt_len not in len_result_dict:
                len_result_dict[alt_len] = [[], []]
            if 325 <= int(pos) < int(id_lst[2]) - int(id_lst[1]) - 325:
                # new_pos here are those pos from 325 to 700 (max) actually
                # the contexts are 25bp now, truncate from above original pos, in order to show only 25 bp context instead of 350 bp.
                new_pos = int(pos) - 325
                len_result_dict[alt_len][0].append(new_pos)
                len_result_dict[alt_len][1].append(change)
        if save_table_dir:
            save_table_fh.close()
        all_mutated_score_dict[tuple(id_lst)] = len_result_dict
    pickle.dump(all_mutated_score_dict, open(save_path, 'wb'))

def peak_detect(work_dir, ks):
    # this function can be a standalone func to print peak info, or
    # used for draw_plot to draw peak on the satruation mutagenesis images
    all_mutated_score_dict = pickle.load(open(os.path.join(work_dir, 'muta.pickle'), 'rb')) 
    k_peak_dict = {}
    for k in ks:
        output_fh = open(os.path.join(work_dir, f'peak_global_values_{k}.tsv'), 'w')
        output_fh.write('Alu_ID\tk\tVar\tMean\tD\n')
        peak_dict = {}
        for _id, d in all_mutated_score_dict.items():
            seq_len = int(_id[2]) - int(_id[1]) - 325 * 2
            print('')
            print(_id, seq_len)
            Y = []
            Y.append([0.0] * seq_len)
            X_char, Y_char = d[k]
            for x_i in range(seq_len):
                Y[0][X_char[x_i]] = Y_char[x_i]
            Y = np.array(Y).T
            # print(Y.shape, k)
            var_glo = Y.var()
            mean_glo = Y.mean()
            D_glo = var_glo / mean_glo
            print('>>> Global <<<')
            print(f'var: {var_glo}, mean: {mean_glo}, D: {D_glo}')
            ori_alu_id_str = f'{_id[0]}:{int(_id[1]) + 350}-{int(_id[2]) - 350}({_id[3]})'
            output_fh.write(f'{ori_alu_id_str}\t{k}\t{var_glo}\t{mean_glo}\t{D_glo}\n')
            if (abs(D_glo) > 0.5):
                print('Class Variance')

            win_size = 10
            stride = 1
            lvic_size = 40
            rvic_size = 40

            print('>>> Sliding Window <<<')
            peak_win_i_lst = []
            win_i_mean_lst = []
            print(seq_len)
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
                print('win\t', f'[{win_i}]\tvar: {var_win}\tmean: {mean_win}\tD: {D_win}')
                # when k = 5, 10, 15, those parameters are selected manually
                if var_vic_win > var_win * 1.0 and k == 5:
                    continue
                if var_vic_win > var_win * 2.5 and k == 10:
                    continue
                if var_vic_win > var_win * 4 and k == 15:
                    continue
                if k not in [5, 10, 15]:
                    # approximate based on above 5 10 15
                    if var_vic_win > var_win * 0.3 * k:
                        continue
                
                # when k = 5, 10, 15, those parameters are selected manually
                if abs(mean_win) > abs(mean_glo * 2.0) and k == 5:
                    peak_win_i_lst.append(win_i)
                if abs(mean_win) > abs(mean_glo * 1.7) and k == 10:
                    peak_win_i_lst.append(win_i)
                if abs(mean_win) > abs(mean_glo * 1.5) and k == 15:
                    peak_win_i_lst.append(win_i)
                if k not in [5, 10, 15]:
                    # approximate based on above 5 10 15
                    if abs(mean_win) > abs(mean_glo * (-0.05 * k + 2.2333)):
                        peak_win_i_lst.append(win_i)
                

            # calculating the dist
            mean_i_lst = []
            for win_i in range(0, seq_len - win_size, 1):
                # window
                Y_win = Y[win_i : win_i + win_size]
                mean_win = Y_win.mean()
                mean_i_lst.append(abs(mean_win))

            mean_i_lst.sort(reverse=True)
            top_percent = 0.20
            top_i = int(len(mean_i_lst) * top_percent)
            top_th = mean_i_lst[top_i]
            print(top_i, top_th)

            peak_lst = []
            peak_mean_lst = []
            print(peak_win_i_lst)
            # new one to have median
            if len(peak_win_i_lst) == 0:
                print('Empty')
            elif len(peak_win_i_lst) == 1:
                peak_lst.append((peak_win_i_lst[0], peak_win_i_lst[0] + win_size))
                # peak_mean_lst.append((win_i_mean_lst[peak_win_i_lst[0]//5] + win_i_mean_lst[peak_win_i_lst[0]//5 + 1])/2)
            else:
                i = 0
                mean_lst = [win_i_mean_lst[peak_win_i_lst[i]//stride]]
                for j in range(1, len(peak_win_i_lst)):
                    if peak_win_i_lst[j - 1] + stride == peak_win_i_lst[j]:
                        pass
                    elif peak_win_i_lst[j - 1] + win_size >= peak_win_i_lst[j]:
                        pass
                    else:
                        peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[j - 1] + win_size))
                        i = j
                if i < j:
                        peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[-1] + win_size))
                if peak_win_i_lst[-2] + win_size < peak_win_i_lst[-1]:
                    peak_lst.append((peak_win_i_lst[i], peak_win_i_lst[-1] + win_size))
            if peak_lst and peak_lst[-1][-1] >= seq_len - 10:
                poped = peak_lst.pop()
                peak_lst.append((poped[0], seq_len - 11))
            
            final_peak_lst = []
            print('peak_lst', peak_lst)
            for peak in peak_lst:
                print(len(win_i_mean_lst), peak[0],  peak[1])
                mean_lst = [win_i_mean_lst[pos//stride] for pos in range(peak[0], peak[1], stride)]
                print(mean_lst)
                if not mean_lst:
                    continue
                peak_mean = sum(mean_lst)/len(mean_lst)
                print(peak_mean, abs(peak_mean), top_th)
                if abs(peak_mean) > top_th:
                    final_peak_lst.append(peak)
                    peak_mean_lst.append(peak_mean)
            # print(final_peak_lst)
            # print(peak_mean_lst)
            peak_dict[_id] = (final_peak_lst, peak_mean_lst)
        k_peak_dict[k] = peak_dict
        output_fh.close()
    return k_peak_dict

def peak_detect_store(work_dir, ks):
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
    k_peak_dict = peak_detect(work_dir, ks)
    os.makedirs(work_dir + '/peaks/', exist_ok=True)
    peak_all_bed = work_dir + '/peaks/' + f'peaks_all.bed'
    peak_all_fh = open(peak_all_bed, 'w') 
    # peak store
    for _chr in ['chr1', 'chr2', 'chr3', 'chr4', 'chr5', 'chr6', 'chr7', 'chr8', 'chr9', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY']:
        peak_bed = work_dir + '/peaks/' + f'peaks_{_chr}.bed'
        peak_fh = open(peak_bed, 'w') 
        for k in ks:
            peak_dict = k_peak_dict[k]
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
                    peak_id = '_'.join(ori_alu_id) + '_' + str(peak[0] - 25) + '_' + str(peak[1] - 25) + '_' + str(pos_neg) + '_'  + str(peak_dict[_id][1][i]) + '_' + str(k)
                    # we need seq on the opposite strand (exon)
                    if _id[3] == '+':
                        writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t-\n'
                    else:
                        writing_line = f'{_id[0]}\t{peak_start}\t{peak_end}\t{peak_id}\t0\t+\n'
                    peak_fh.write(writing_line)
                    peak_all_fh.write(writing_line)
        peak_fh.close()
    peak_all_fh.close()
    return k_peak_dict


def draw_plot(work_dir, ks, plot_mode='fixed', strand=True, has_peaks=True, k_peak_dict=None):
    # has_peaks controls if we draw peaks range on the satruation mutagenesis imgaes
    # if yes, then draw 
    # plot mode fixed means: fix the range of y-axis from -0.3 to 0.3
    # adaptive means
    # for _id in baseline_rslt.keys():
    color_names = ['green', 'blue', 'red', 'orange']
    char_color_dict = {}
    for i, k in enumerate(ks):
        char_color_dict[k] = color_names[i]
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
        if has_peaks:
            for k in ks:
                peak_dict = k_peak_dict[k]
                for peak in peak_dict[id_lst][0]:
                    if plot_mode == 'adaptive':
                        plt.annotate(text='', xy=(peak[0], -max_change_abs + 0.0015 * (k - 5)), xytext=(peak[1], -max_change_abs + 0.0015 * (k - 5)), arrowprops=dict(arrowstyle='<->', color=char_color_dict[k], shrinkA=0, shrinkB=0))
                    elif plot_mode == 'fixed':
                        plt.annotate(text='', xy=(peak[0], -0.29 + 0.0015 * (k - 5)), xytext=(peak[1], -0.29 + 0.0015 * (k - 5)), arrowprops=dict(arrowstyle='<->', color=char_color_dict[k], shrinkA=0, shrinkB=0))
        for char in ks:
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


def gen_saturation_mutagenesis_graphs_deletion(work_dir, model_wts_name, ks, genome=None, alu_bed_file=None, alu_fa_file=None, plot_mode='fixed', has_peaks=True):
    os.makedirs(work_dir, exist_ok=True)
    if alu_bed_file:
        src_bed = alu_bed_file
        contexted_bed = os.path.join(work_dir, 'contexted_test.bed')
        unmutated_fa = os.path.join(work_dir, 'unmutated_test.fa')
        mutated_fa = os.path.join(work_dir, 'mutated_test.fa')
        get_mutated_fa(unmutated_fa, mutated_fa, genome, ks, strand=True, src_bed=src_bed, contexted_bed=contexted_bed, mode='bothfix', single_side_pad_len=350)
    elif alu_fa_file:
        unmutated_fa = alu_fa_file
        mutated_fa = os.path.join(work_dir, 'mutated_test.fa')
        get_mutated_fa(unmutated_fa, mutated_fa, genome, ks, strand=True, src_bed=None, contexted_bed=None, mode='bothfix', single_side_pad_len=350)
    
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
            # 0.984420120716095	3.0	chr10_101017466_101017851_+_h38_mk_AluSq2_1_335_1_bothfix_0_0_NA::3_t_ttttttttttt
            # id line: h38_mk_AluSq2_2_329_2_bothfix_0_0_NA::chr11:112229347-112229726(-)::378_A_ATTTTTTTTTTT
            mutated_rslt[line_lst[2].split('::')[1]].append((line_lst[2].split('::')[-1], float(line_lst[0]))) # {id:(mutation, score)}
    # generate mutated score dict
    gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, os.path.join(work_dir, 'tables'))
    # peak detection
    if has_peaks:
        k_peak_dict=peak_detect_store(work_dir, ks)
        draw_plot(work_dir=work_dir, ks=ks, plot_mode=plot_mode, strand=True, has_peaks=has_peaks, k_peak_dict=k_peak_dict)
    else:
        draw_plot(work_dir=work_dir, ks=ks, plot_mode=plot_mode, strand=True, has_peaks=has_peaks)

