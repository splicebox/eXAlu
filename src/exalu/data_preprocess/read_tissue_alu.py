import os
import sys
import pybedtools
import pandas
import csv


# alu_tissue_dir = '../data/Fixed/Amygdala/'
data_dir = os.getcwd() + '/data/'
# hg38_file = data_dir + 'shared/hg38/hg38c.fa'
SEQ_LEN_TH_UP = 350
SEQ_LEN_TH_LOW = 100

def split(src_bed_file, exon_bed_file, alu_bed_file):
    '''
    BE AWARE: the src_bed_file's bed format has bugs, it now looks,
    chr22	50622719	50624718	SRR1317771	+	0	chr22	50623919	50624078	h38_mk_AluJ	0	-	159
    '''
    src_bed_fh = open(src_bed_file, 'r')
    exon_bed_fh = open(exon_bed_file, 'w')
    alu_bed_fh = open(alu_bed_file, 'w')
    src_bed_reader = csv.reader(src_bed_fh, delimiter='\t')
    exon_bed_writer = csv.writer(exon_bed_fh, delimiter='\t')
    alu_bed_writer = csv.writer(alu_bed_fh, delimiter='\t')
    for idx, row in enumerate(src_bed_reader):
        i, j = 7, 11
        l = int(row[i])
        r = int(row[i + 1])
        seq_len = r - l
        if seq_len > SEQ_LEN_TH_UP:
            continue # should skip both exon and alu
        if seq_len < SEQ_LEN_TH_LOW:
            continue
        alu_bed_writer.writerow([row[i - 1], str(l), str(r), \
            f'{row[i + 2]}_{idx + 1}_{seq_len}', row[5], row[j]]) # row[5] is exon event label
        i, j = 1, 4
        l = int(row[i])
        r = int(row[i + 1])
        seq_len = r - l
        exon_bed_writer.writerow([row[i - 1], str(l), str(r), \
            f'{row[i + 2]}_{idx + 1}_{seq_len}', row[5], row[j]]) # row[5] is exon event label
    src_bed_fh.close()
    exon_bed_fh.close()
    alu_bed_fh.close()
    return
    
def read_tissue_alu(tissue, genome, strand, fixed_dir):
    '''
    deprecated; see mode='none' in read_tissue_alu_context
    '''
    src_tissue_dir = fixed_dir + tissue
    dst_tissue_dir = fixed_dir + f'/f0/{tissue}/'
    # init dict of every sample's bed/seq
    file_dict = {}
    sample_counter = 0
    for i in os.listdir(src_tissue_dir):
        if i.endswith('_Alu.r.overlap.filtered.bed') or i.endswith('_Alu.r.overlap.filtered.bed.withGeneNames') or \
            i == 'GENCODE.v36.ALUs.all.overlap.filtered.bed':
            sample_counter += 1
            seq_dir = os.path.join(dst_tissue_dir, 'alu_seq')
            split_bed_dir = os.path.join(dst_tissue_dir, 'split_bed')
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            if not os.path.exists(split_bed_dir):
                os.makedirs(split_bed_dir)
            sample_id = i.split('_')[0]
            exon_seq_file = sample_id + '_exon.fa' 
            exon_bed_file = sample_id + '_exon.bed'
            alu_seq_file = sample_id + '_alu.fa'
            alu_bed_file = sample_id + '_alu.bed'
            file_dict[sample_id] = [os.path.join(src_tissue_dir, i),
                                    os.path.join(seq_dir, exon_seq_file),
                                    os.path.join(split_bed_dir, exon_bed_file),
                                    os.path.join(seq_dir, alu_seq_file),
                                    os.path.join(split_bed_dir, alu_bed_file)]
    # split original bed to exon/alu bed file
    print(f'{tissue}: {sample_counter}')
    for sample_id, [src_bed_file, _, exon_bed_file, _, alu_bed_file] in file_dict.items():
        split(src_bed_file, exon_bed_file, alu_bed_file)
    # process to get seq file
    for sample_id, [_, exon_seq_file, exon_bed_file, alu_seq_file, alu_bed_file] in file_dict.items():
        exon_bed = pybedtools.BedTool(exon_bed_file)
        alu_bed = pybedtools.BedTool(alu_bed_file)
        ref_fa = pybedtools.example_filename(genome)
        exon_bed.sequence(fi=ref_fa, fo=exon_seq_file, s=strand)
        alu_bed.sequence(fi=ref_fa, fo=alu_seq_file, s=strand)

def add_context(src_bed_file, dst_bed_file, mode='none', single_side_pad_len=0, len_check=True):
    '''
    mode: left - only left context; right - only right context; both - left and right context
          none - no context
    '''
    print('data_preprocess/read_tissue_alu/add_context')
    too_long_counter = 0
    too_short_counter = 0
    all_counter = 0
    src_bed_fh = open(src_bed_file, 'r')
    dst_bed_fh = open(dst_bed_file, 'w')
    src_bed_reader = csv.reader(src_bed_fh, delimiter='\t')
    dst_bed_writer = csv.writer(dst_bed_fh, delimiter='\t')
    for idx, row in enumerate(src_bed_reader):
        i = 1
        j = 5
        writer = dst_bed_writer
        l = int(row[i])
        r = int(row[i + 1])
        seq_len = r - l
        l_pad, r_pad = 0, 0
        if len_check and (seq_len > SEQ_LEN_TH_UP):
            too_long_counter += 1
            continue
        if len_check and (seq_len < SEQ_LEN_TH_LOW):
            too_short_counter += 1
            continue
        all_counter += 1
        if mode == 'none': # classic mode for EAD
            pass
        elif mode == 'bothfix':
            l -= single_side_pad_len
            r += single_side_pad_len
        elif mode == 'leftfix':
            l -= single_side_pad_len
        elif mode == 'rightfix':
            r += single_side_pad_len
        elif mode == 'both': # classic mode for DP
            l_pad = int((SEQ_LEN_TH_UP - seq_len) / 2)
            r_pad = SEQ_LEN_TH_UP - seq_len - l_pad
            l = l - l_pad - single_side_pad_len
            r = r + r_pad + single_side_pad_len
        elif mode == 'left':
            l_pad = int(SEQ_LEN_TH_UP - seq_len)
            l = l - l_pad - single_side_pad_len
        elif mode == 'right':
            r_pad = int(SEQ_LEN_TH_UP - seq_len)
            r = r + r_pad + single_side_pad_len
        else:
            print(mode)
            raise TypeError('wrong flank mode')
        if l < 0 or r < 0:
            raise ValueError('split_context index out of control')
        writer.writerow([row[i - 1], str(l), str(r), \
                        row[3], row[4], row[j]])
    src_bed_fh.close()
    dst_bed_fh.close()
    return all_counter, too_long_counter, too_short_counter


def split_context(src_bed_file, exon_bed_file, alu_bed_file, mode='none', single_side_pad_len=0):
    '''
    read alu/exon sequences for each tissue
    mode: left - only left context; right - only right context; both - left and right context
          none - no context

    BE AWARE: the src_bed_file's bed format has bugs, it now looks,
    chr22	50622719	50624718	SRR1317771	+	0	chr22	50623919	50624078	h38_mk_AluJ	0	-	159
    '''
    too_long_counter, too_short_counter = 0, 0
    all_counter = 0
    src_bed_fh = open(src_bed_file, 'r')
    exon_bed_fh = open(exon_bed_file, 'w')
    alu_bed_fh = open(alu_bed_file, 'w')
    src_bed_reader = csv.reader(src_bed_fh, delimiter='\t')
    exon_bed_writer = csv.writer(exon_bed_fh, delimiter='\t')
    alu_bed_writer = csv.writer(alu_bed_fh, delimiter='\t')
    for idx, row in enumerate(src_bed_reader):
        for i, j, writer in [(1, 4, exon_bed_writer), (7, 11, alu_bed_writer)]:
            l = int(row[i])
            r = int(row[i + 1])
            seq_len = r - l
            l_pad, r_pad = 0, 0
            if i == 7:
                # only alu will do len check
                if seq_len > SEQ_LEN_TH_UP:
                    too_long_counter += 1
                    continue
                if seq_len < SEQ_LEN_TH_LOW:
                    too_short_counter += 1
                    continue
                all_counter += 1
            if mode == 'none': # classic mode for EAD
                pass
            elif mode == 'bothfix':
                l -= single_side_pad_len
                r += single_side_pad_len
            elif mode == 'leftfix':
                l -= single_side_pad_len
            elif mode == 'rightfix':
                r += single_side_pad_len
            elif mode == 'both': # classic mode for DP
                l_pad = int((SEQ_LEN_TH_UP - seq_len) / 2)
                r_pad = SEQ_LEN_TH_UP - seq_len - l_pad
                l = l - l_pad - single_side_pad_len
                r = r + r_pad + single_side_pad_len
            elif mode == 'left':
                l_pad = int(SEQ_LEN_TH_UP - seq_len)
                l = l - l_pad - single_side_pad_len
            elif mode == 'right':
                r_pad = int(SEQ_LEN_TH_UP - seq_len)
                r = r + r_pad + single_side_pad_len
            else:
                print(mode)
                raise TypeError('wrong flank mode')
            if l < 0 or r < 0:
                raise ValueError('split_context index out of control')
            writer.writerow([row[i - 1], str(l), str(r), \
                row[i + 2], row[5], row[j]]) # row[4] is exon event label
    src_bed_fh.close()
    exon_bed_fh.close()
    alu_bed_fh.close()
    return all_counter, too_long_counter, too_short_counter

    
def read_tissue_alu_context(genome, strand, mode='none', single_side_pad_len=0, tissue='', fixed_dir=None, work_dir=None):
    '''
    read alu/exon sequences for each tissue
    mode: left - only left context; right - only right context; both - left and right context
            none - no context
    '''
    src_tissue_dir = fixed_dir + tissue
    dst_tissue_dir = os.path.join(work_dir, 'after_read_Fixed', tissue)
    # init dict of every sample's bed/seq
    file_dict = {}
    sample_counter = 0
    all_counter, too_long_counter, too_short_counter = 0, 0, 0
    for i in os.listdir(src_tissue_dir):
        if i.endswith('_Alu.r.overlap.filtered.bed') or i.endswith('_Alu.r.overlap.filtered.bed.withGeneNames') or \
        i in ['GENCODE.v36.ALUs.3.overlap.filtered.bed', 'GENCODE.v36.ALUs.all.overlap.filtered.bed', 'ncbi_refseq_all.ALUs.overlap.filtered.bed',\
        'ncbi_refseq_curated.ALUs.overlap.filtered.bed', 'UCSC_refseq_refGene.ALUs.overlap.filtered.bed', 'rheMac_ENSEMBL_Alu.overlap.filtered.bed']:
            sample_counter += 1
            seq_dir = os.path.join(dst_tissue_dir, 'alu_seq')
            split_bed_dir = os.path.join(dst_tissue_dir, 'split_bed')
            if not os.path.exists(seq_dir):
                os.makedirs(seq_dir)
            if not os.path.exists(split_bed_dir):
                os.makedirs(split_bed_dir)
            sample_id = i.split('_')[0]
            exon_seq_file = sample_id + '_exon.fa' 
            exon_bed_file = sample_id + '_exon.bed'
            alu_seq_file = sample_id + '_alu.fa'
            alu_bed_file = sample_id + '_alu.bed'
            file_dict[sample_id] = [os.path.join(src_tissue_dir, i),
                                    os.path.join(seq_dir, exon_seq_file),
                                    os.path.join(split_bed_dir, exon_bed_file),
                                    os.path.join(seq_dir, alu_seq_file),
                                    os.path.join(split_bed_dir, alu_bed_file)]
    # split original bed to exon/alu bed file
    for sample_id, [src_bed_file, _, exon_bed_file, _, alu_bed_file] in file_dict.items():
        tissue_all_counter, tissue_too_long_counter, tissue_too_short_counter = split_context(src_bed_file, exon_bed_file, alu_bed_file, mode, single_side_pad_len)
        all_counter += tissue_all_counter
        too_long_counter += tissue_too_long_counter
        too_short_counter += tissue_too_short_counter
    print(f'{tissue}\t#sample: {sample_counter}\tproper: {all_counter}\ttoo_long (>={SEQ_LEN_TH_UP}): {too_long_counter}\ttoo_short (<={SEQ_LEN_TH_LOW}): {too_short_counter}')
    # process to get seq file
    for sample_id, [_, exon_seq_file, exon_bed_file, alu_seq_file, alu_bed_file] in file_dict.items():
        exon_bed = pybedtools.BedTool(exon_bed_file)
        alu_bed = pybedtools.BedTool(alu_bed_file)
        ref_fa = pybedtools.example_filename(genome)
        exon_bed.sequence(fi=ref_fa, fo=exon_seq_file, name=True, s=strand)
        alu_bed.sequence(fi=ref_fa, fo=alu_seq_file, name=True, s=strand)

def read_tissue_aluonly_context(genome, strand, mode='none', single_side_pad_len=0, tissue='', fixed_dir=None):
    '''
    TODO: not finished
    read alu/exon sequences for each tissue
    mode: left - only left context; right - only right context; both - left and right context
            none - no context
    '''
    src_tissue_dir = fixed_dir + tissue
    dst_tissue_dir = fixed_dir + f'padding_{mode}_{single_side_pad_len}/{tissue}/'
    # src_tissue_dir = data_dir + 'Fixed/{}/'.format(tissue)
    # dst_tissue_dir = data_dir + 'Fixed/{}/'.format(f'padding_{mode}_{single_side_pad_len}')
    # init dict of every sample's bed/seq
    file_dict = {}
    sample_counter = 0
    all_counter, too_long_counter = 0, 0
    for i in os.listdir(src_tissue_dir):
       if i.endswith('filtered.bed'):
            sample_counter += 1
            
            sample_id = i.split('_')[0]
            exon_seq_file = sample_id + '_exon.fa' 
            exon_bed_file = sample_id + '_exon.bed'
            alu_seq_file = sample_id + '_alu.fa'
            alu_bed_file = sample_id + '_alu.bed'
            file_dict[sample_id] = [os.path.join(src_tissue_dir, i),
                                    os.path.join(seq_dir, exon_seq_file),
                                    os.path.join(split_bed_dir, exon_bed_file),
                                    os.path.join(seq_dir, alu_seq_file),
                                    os.path.join(split_bed_dir, alu_bed_file)]
    # split original bed to exon/alu bed file
    for sample_id, [src_bed_file, _, exon_bed_file, _, alu_bed_file] in file_dict.items():
        tissue_all_counter, tissue_too_long_counter = split_context(src_bed_file, exon_bed_file, alu_bed_file, mode, single_side_pad_len)
        all_counter += tissue_all_counter
        too_long_counter += tissue_too_long_counter
    print(f'{tissue} #sample: {sample_counter} all(<=350): {all_counter} too_long: {too_long_counter}')
    # process to get seq file
    for sample_id, [_, exon_seq_file, exon_bed_file, alu_seq_file, alu_bed_file] in file_dict.items():
        exon_bed = pybedtools.BedTool(exon_bed_file)
        alu_bed = pybedtools.BedTool(alu_bed_file)
        ref_fa = pybedtools.example_filename(genome)
        exon_bed.sequence(fi=ref_fa, fo=exon_seq_file, name=True, s=strand)
        alu_bed.sequence(fi=ref_fa, fo=alu_seq_file, name=True, s=strand)

if __name__ == '__main__':
    import time
    s = time.time()
    read_tissue_alu('')
    print('time: {}'.format(time.time() - s))