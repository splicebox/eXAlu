import pandas
# extract bed file from MOAT_v3.txt file

SEQ_LEN_TH = 350

# TODO HL
df = pandas.read_csv(open('data/Fixed/MOAT/MOAT_v3.txt', 'r'), sep='\t', dtype=str)
df = df.dropna()
df = df[['#exon_chrom', 'exon_from', 'exon_to', 'exon_strand', 'exon_genes',
         'alu_chrom', 'alu_from', 'alu_to', 'alu_name', 'alu_strand']]
df = df.drop_duplicates(subset=['alu_chrom', 'alu_from', 'alu_to', 'alu_strand'])
df['exon_type'] = '0'
df['alu_qual'] = '0'
df.to_csv(open('data/Fixed/MOAT/MOAT_v3_Alu.r.overlap.filtered.bed', 'w'), sep='\t',
        columns=['#exon_chrom', 'exon_from', 'exon_to', 'exon_genes', 'exon_strand', 'exon_type',
                 'alu_chrom', 'alu_from', 'alu_to', 'alu_name', 'alu_qual', 'alu_strand'],
        header=None,
        index=None)
print(df)