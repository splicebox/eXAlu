from exalu.run_model import run_ead
from exalu.data_preprocess.curate_data import add_context
import pybedtools
import sys
import argparse
import os


def get_arguments():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        help='select bed or fasta input mode', dest='input_mode')

    parser_bed = subparsers.add_parser('bed', help='infer with bed file')
    parser_bed.add_argument('-b', metavar='ALU_BED_FILE',
                            type=str, required=True, help='the input Alu bed file')
    parser_bed.add_argument('-r', metavar='REF_GENOME_FILE', type=str,
                            required=True, help='the reference genome file')
    parser_bed.add_argument('-m', metavar='MODEL_WEIGHTS_FILE',
                            type=str, required=True, help='the trained model weights file')
    parser_bed.add_argument('-o', metavar='OUTPUT_DIR', type=str, required=True,
                            help='the directory contains temp files and final output file')

    parser_fa = subparsers.add_parser('fasta', help='infer with fasta file')
    parser_fa.add_argument('-f', metavar='ALU_FASTA_FILE',
                           type=str, required=True, help='the input Alu fa file')
    parser_fa.add_argument('-m', metavar='MODEL_WEIGHTS_FILE',
                           type=str, required=True, help='the trained model weights file')
    parser_fa.add_argument('-o', metavar='OUTPUT_DIR', type=str, default='./out',
                           help='the directory contains temp files and final output file, default ./out')

    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    return parser.parse_args()


def infer(model_weights_file, output_dir, alu_bed_file=None, ref_genome_file=None, alu_fa_file=None):
    os.makedirs(output_dir, exist_ok=True)
    if alu_bed_file:
        contexted_bed_file = os.path.join(output_dir, 'temp_contexted_alu.bed')
        fa_file = os.path.join(output_dir, 'temp_contexted_alu.fa')
        ref_genome = pybedtools.example_filename(ref_genome_file)
        add_context(alu_bed_file, contexted_bed_file, 'bothfix', 350)
        alu_bed = pybedtools.BedTool(contexted_bed_file)
        alu_bed.sequence(fi=ref_genome, fo=fa_file, name=True, s=True)
    if alu_fa_file:
        fa_file = alu_fa_file
    run_ead(strand=True,
            model_name='cnet',
            context_mode='bothfix',
            single_side_pad_len=350,
            run_mode=4,
            work_dir=output_dir,
            infer_set='simpleinfer',
            infer_file=fa_file,
            prd_file='results.txt',
            model_wts_name=model_weights_file)


if __name__ == "__main__":
    args = get_arguments()
    if args.input_mode == 'bed':
        infer(model_weights_file=args.m, output_dir=args.o,
              alu_bed_file=args.b, ref_genome_file=args.r)
    if args.input_mode == 'fasta':
        infer(model_weights_file=args.m, output_dir=args.o, alu_fa_file=args.f)
