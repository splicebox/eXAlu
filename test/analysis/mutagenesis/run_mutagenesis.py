import argparse
import sys
from mutagenesis_substitution import gen_saturation_mutagenesis_graphs_substitution
from mutagenesis_deletion import gen_saturation_mutagenesis_graphs_deletion

def get_arguments():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        help='select bed or fasta input mode', dest='input_mode')

    # Creating subparsers for 'bed' and 'fasta' commands
    parser_bed = subparsers.add_parser('bed', help='infer with bed file')
    parser_bed.add_argument('-t', metavar='TYPE', choices=['deletion', 'substitution'], required=True,
                            help='the type of mutagenesis plot, please choose from substitution or deletion, -k is required when the type is deletion')
    parser_bed.add_argument('-b', metavar='ALU_BED_FILE', type=str, required=True, 
                            help='the input Alu bed file')
    parser_bed.add_argument('-r', metavar='REF_GENOME_FILE', type=str, required=True, 
                            help='the reference genome file')
    parser_bed.add_argument('-m', metavar='MODEL_WEIGHTS_FILE', type=str, required=True, 
                            help='the trained model weights file')
    parser_bed.add_argument('-o', metavar='OUTPUT_DIR', type=str, required=True, 
                            help='the directory contains temp files and final output file')
    parser_bed.add_argument('--yaxis', metavar='Y_AXIS_MODE', type=str, default='fixed', 
                            help='limits of y-axis is fixed to +/-0.3 or adaptive. The default is fixed mode')
    
    # Add the conditional -k argument
    parser_bed.add_argument('-k', metavar='K_BP_DELETION', type=str,
                            help='the number of deleting bases for deletion mutagenesis plot, k should be ')

    parser_fa = subparsers.add_parser('fasta', help='infer with fasta file')
    parser_fa.add_argument('-t', metavar='TYPE', choices=['deletion', 'substitution'], required=True,
                            help='the type of mutagenesis plot, please choose from substitution or deletion, -k is required when the type is deletion')
    parser_fa.add_argument('-f', metavar='ALU_FASTA_FILE', type=str, required=True, 
                            help='the input Alu fa file')
    parser_fa.add_argument('-m', metavar='MODEL_WEIGHTS_FILE', type=str, required=True, 
                            help='the trained model weights file')
    parser_fa.add_argument('-o', metavar='OUTPUT_DIR', type=str, default='./out',
                            help='the directory contains temp files and final output file, default ./out')
    parser_fa.add_argument('--yaxis', metavar='Y_AXIS_MODE', type=str, default='fixed', 
                            help='limits of y-axis is fixed to +/-0.3 or adaptive. The default is fixed mode')
    
    # Check if no arguments were provided
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Check if type is 'deletion' and -k is not provided
    if args.input_mode in ['bed', 'fasta'] and args.type == 'deletion' and not getattr(args, 'k', None):
        parser.error("'-k' is required when type is 'deletion'")

    return args


if __name__ == "__main__":
    args = get_arguments()
    if args.input_mode == 'bed':
        if args.type == 'subsititution':
            gen_saturation_mutagenesis_graphs_substitution(work_dir=args.o, genome=args.r, model_wts_name=args.m, alu_bed_file=args.b, plot_mode=args.yaxis)
        elif args.type == 'deletion':
            gen_saturation_mutagenesis_graphs_deletion()

    if args.input_mode == 'fasta':
        if args.type == 'subsititution':
            gen_saturation_mutagenesis_graphs_substitution(work_dir=args.o, model_wts_name=args.m, alu_fa_file=args.f, plot_mode=args.yaxis)
        elif args.type == 'deletion':
            gen_saturation_mutagenesis_graphs_deletion()
