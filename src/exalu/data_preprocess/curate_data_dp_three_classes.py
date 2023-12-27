import os
import pybedtools
import csv
import read_tissue_alu

data_dir = os.getcwd() + '/data'

human_alu_bed_file = data_dir + '/Annotation/Alu_repeats_hg38.bed'
human_alu_filtered_bed_file = data_dir + \
    '/Gencode/Alu_repeats_hg38_filtered.bed'
hg38_file = data_dir + '/shared/hg38/hg38c.fa'


'''
775 GENCODE.v36.ALUs.overlap.filtered.bed
The third file is the results of bedIntersect between the human GENCODE v36 exons and the Alu annotation + filtered to retain only internal exons (col 6 has a 3 in it) + the overlap is >=15 bp + the ALu and the exon/gene are on opposite strands.
'''
# gencode_bed_file = data_dir + '/Gencode/GENCODE.v36.ALUs.overlap.filtered.bed'
'''
25662 GENCODE.v36.ALUs.all.overlap.filtered.bed
The second file is the result of bedIntersect between the human GENCODE v36 exons (all exons) and the Alu annotations + filtered to retain only lines in which the Alu and the exon/gene are on opposite strands.  All overlaps >=1bp (i.e., ANY overlap) are included.
'''
gencode_bed_file = data_dir + '/Gencode/GENCODE.v36.ALUs.all.overlap.filtered.bed'
'''
45983 GENCODE.v36.ALUs.all.overlap.15bp.bed
The first file contains the result of bedIntersect between the human GENCODE v36 exons (_all_ exons, regardless of their position within the transcript) and the Alu annotations + filtered to retain only overlaps of 15 bp or longer.'''
# gencode_bed_file = data_dir + '/Gencode/GENCODE.v36.ALUs.all.overlap.15bp.bed'

gencode_dir = data_dir + '/Gencode/'

def curate_data_dp_three_calsses_context(strand, mode='none', single_side_pad_len=0):
    
    read_tissue_alu.split_context(gencode_bed_file,
                gencode_dir + 'gencode_exon.bed',
                gencode_dir + 'gencode_alu.bed',
                mode='none', single_side_pad_len=0)
    # read_file
    ref_fa = pybedtools.example_filename(hg38_file)
    human_alu_bed = pybedtools.BedTool(human_alu_filtered_bed_file)
    alu_bed = pybedtools.BedTool(gencode_dir + 'gencode_alu.bed')
    neg_alu_bed = human_alu_bed.subtract(alu_bed, A=True)
    alu_bed.sequence(fi=ref_fa, fo=gencode_dir +
                     'gencode_alu.fa', name=True, s=strand)
    neg_alu_bed.sequence(fi=ref_fa, fo=gencode_dir +
                         'neg_alu.fa', name=True, s=strand)

def curate_data_dp_three_calsses(strand):
    read_tissue_alu.split(gencode_bed_file,
                          gencode_dir + 'gencode_exon.bed',
                          gencode_dir + 'gencode_alu.bed')
    # read_file
    ref_fa = pybedtools.example_filename(hg38_file)
    with open(human_alu_bed_file, 'r') as src_fh,\
            open(human_alu_filtered_bed_file, 'w') as dst_fh:
        src_bed_reader = csv.reader(src_fh, delimiter='\t')
        dst_bed_writer = csv.writer(dst_fh, delimiter='\t')
        for row in src_bed_reader:
            if int(row[2]) - int(row[1]) > 350:
                continue
            dst_bed_writer.writerow(row)
    human_alu_bed = pybedtools.BedTool(human_alu_filtered_bed_file)
    alu_bed = pybedtools.BedTool(gencode_dir + 'gencode_alu.bed')
    neg_alu_bed = human_alu_bed.subtract(alu_bed, A=True)
    alu_bed.sequence(fi=ref_fa, fo=gencode_dir +
                     'gencode_alu.fa', name=True, s=strand)
    neg_alu_bed.sequence(fi=ref_fa, fo=gencode_dir +
                         'neg_alu.fa', name=True, s=strand)
    

if __name__ == '__main__':
    # curate_data_dp_three_calsses()
    curate_data_dp_three_calsses_context()
