import re
import os
import pybedtools
import pickle

def gen_mutated_score_dict(work_dir, baseline_rslt, mutated_rslt, save_table_dir=None, bed=None):
    save_path = os.path.join(work_dir, 'muta.pickle')
    # if os.path.exists(save_path):
        # print(save_path, 'already exists!')
        # return
    # else:
    # print(save_path, 'doesnt exist... saving...')
    all_mutated_score_dict = {}
    for _id in mutated_rslt.keys():
        # _id -> chr1:123-456(-)
        id_lst = re.split('[:()]', _id)
        id_lst = [id_lst[0]] + id_lst[1].split('-') + [id_lst[2]]
        ori_alu_id_str = f'{id_lst[0]}:{int(id_lst[1]) + 350}-{int(id_lst[2]) - 350}({id_lst[3]})'
        if save_table_dir:
            os.makedirs(save_table_dir, exist_ok=True)
            table_path = os.path.join(save_table_dir, ori_alu_id_str + '.tsv')
            save_table_fh = open(table_path, 'w')
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
            if 325 <= int(pos) < int(id_lst[2]) - int(id_lst[1]) - 325:
                # new_pos here are those pos from 325 to 700 (max) actually
                # the contexts are 25bp now, truncate from above original pos, in order to show only 25 bp context instead of 350 bp.
                new_pos = int(pos) - 325
                char_result_dict[alt][0].append(new_pos)
                char_result_dict[alt][1].append(change)
        if save_table_dir:
            save_table_fh.close()
        all_mutated_score_dict[tuple(id_lst)] = char_result_dict
    # else:
    #     for _id in mutated_rslt.keys():
    #         # _id: random name
    #         if save_table_dir:
    #             os.makedirs(save_table_dir, exist_ok=True)
    #             table_path = os.path.join(save_table_dir, _id + '.tsv')
    #             save_table_fh = open(table_path, 'w')
    #             save_table_fh.write('AluID\tOffset\tPOS\tREF\tALT\tMutatedScore\tBaselineScore\tChange\n')
    #         char_result_dict = {}
    #         for mutation, score in mutated_rslt[_id]:
    #             pos, ref, alt = mutation.split('_')
    #             change = score - baseline_rslt[_id]
    #             if save_table_dir:
    #                 # pos here are from 0 to 1050 (max), the contexts are 350bp for each
    #                 save_table_fh.write(f'{_id}\t{pos}\t{int(id_lst[1]) + int(pos)}\t{ref}\t{alt}\t{score}\t{baseline_rslt[_id]}\t{change}\n')
    #             if alt not in char_result_dict:
    #                 char_result_dict[alt] = [[], []]
    #             if 325 <= int(pos) < int(id_lst[2]) - int(id_lst[1]) - 325:
    #                 # new_pos here are those pos from 325 to 700 (max) actually
    #                 # the contexts are 25bp now, truncate from above original pos, in order to show only 25 bp context instead of 350 bp.
    #                 new_pos = int(pos) - 325
    #                 char_result_dict[alt][0].append(new_pos)
    #                 char_result_dict[alt][1].append(change)
    #         if save_table_dir:
    #             save_table_fh.close()
    #         all_mutated_score_dict[_id] = char_result_dict
    pickle.dump(all_mutated_score_dict, open(save_path, 'wb'))

def read_fa(fa_file, genome, strand, bed_file=None):
    '''
    read bed file, output dict [key: each seq's id in bed, value: the seq's fa seq]
    
    fa example: 
    >h38_mk_AluJb_2_both_19_20_3::chr11:3361721-3362121(+)
    tacatgctcttaggagattaccagtga......
    '''
    if bed_file:
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
    # else:
    #     with open(fa_file, 'r') as fa_fh:
    #         while(id_line := fa_fh.readline().rstrip()):
    #             seq_line = fa_fh.readline().rstrip()
    #             seq_fa_dict[id_line] = (seq_line, id_line)
    return seq_fa_dict
