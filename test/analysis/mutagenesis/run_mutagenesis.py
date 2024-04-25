import argparse
import sys
from mutagenesis_substitution import gen_saturation_mutagenesis_graphs_substitution
from mutagenesis_deletion import gen_saturation_mutagenesis_graphs_deletion
import warnings

def parse_comma_separated_integers(value):
    try:
        # Split the string by comma and convert each part to an integer
        parts = list(map(int, value.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError("Each value in K_BP_DELETION_LIST should be an integer.")

    # Check if all elements are within the desired range
    if not all(1 <= part <= 30 for part in parts):
        raise argparse.ArgumentTypeError("Each integer in K_BP_DELETION_LIST should be in the range [1, 30].")

    # Check the number of elements in the list
    if not 1 <= len(parts) <= 4:
        raise argparse.ArgumentTypeError("K_BP_DELETION_LIST must contain between 1 and 4 integers.")
    
    # Issue a warning for parts larger than 15
    for part in parts:
        if part > 15:
            warnings.warn("We didn't test K larger than 15", UserWarning)
            break

    return parts

def get_arguments():
    parser = argparse.ArgumentParser()


    subparsers = parser.add_subparsers(
        help='select bed or fasta input mode', dest='input_mode')

    # Creating subparsers for 'bed' and 'fasta' commands
    parser_bed = subparsers.add_parser('bed', help='infer with bed file')
    parser_bed.add_argument('-t', metavar='TYPE', choices=['deletion', 'substitution'], required=True,
                            help='the type of mutagenesis plot, please choose from substitution or deletion, -k is required when the type is deletion')
    parser_bed.add_argument('-k', metavar='K_BP_DELETION_LIST', type=parse_comma_separated_integers,
                            help='a comma-separated list of integers (1 to 4 elements, each between 1 and 30) specifying the number of deleting bases for deletion mutagenesis plot')
    parser_bed.add_argument('-p', action='store_true',
                            help='draw peaks on plot and output peaks file')
    parser_bed.add_argument('-b', metavar='ALU_BED_FILE', type=str, required=True, 
                            help='the input Alu bed file')
    parser_bed.add_argument('-r', metavar='REF_GENOME_FILE', type=str, required=True, 
                            help='the reference genome file')
    parser_bed.add_argument('-m', metavar='MODEL_WEIGHTS_FILE', type=str, required=True, 
                            help='the trained model weights file')
    parser_bed.add_argument('-o', metavar='OUTPUT_DIR', type=str, required=True, 
                            help='the directory contains temp files and final output file')
    parser_bed.add_argument('--yaxis', metavar='Y_AXIS_MODE', type=str, default='fixed', 
                            help='limits of y-axis is fixed to +/-0.3 or adaptive, the default is fixed mode')
    parser_bed.add_argument('--no-alu-boundaries', action='store_true',
                            help='Do not draw Alu boundaries (grey vertical dashed lines) on plot')

    parser_fa = subparsers.add_parser('fasta', help='infer with fasta file')
    parser_fa.add_argument('-t', metavar='TYPE', choices=['deletion', 'substitution'], required=True,
                            help='the type of mutagenesis plot, please choose from substitution or deletion, -k is required when the type is deletion')
    parser_fa.add_argument('-k', metavar='K_BP_DELETION_LIST', type=parse_comma_separated_integers,
                            help='a comma-separated list of integers (1 to 4 elements, each between 1 and 30) specifying the number of deleting bases for deletion mutagenesis plot')
    parser_fa.add_argument('-p', action='store_true',
                            help='draw peaks on plot and output peaks file')
    parser_fa.add_argument('-f', metavar='ALU_FASTA_FILE', type=str, required=True, 
                            help='the input Alu fa file')
    parser_fa.add_argument('-m', metavar='MODEL_WEIGHTS_FILE', type=str, required=True, 
                            help='the trained model weights file')
    parser_fa.add_argument('-o', metavar='OUTPUT_DIR', type=str, default='./out',
                            help='the directory contains temp files and final output file, default ./out')
    parser_fa.add_argument('--yaxis', metavar='Y_AXIS_MODE', type=str, default='fixed', 
                            help='limits of y-axis is fixed to +/-0.3 or adaptive, the default is fixed mode')
    parser_fa.add_argument('--no-alu-boundaries', action='store_true',
                            help='Do not draw Alu boundaries (grey vertical dashed lines) on plot')
    
    # Check if no arguments were provided
    if len(sys.argv) < 2:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()

    # Check if type is 'deletion' and -k is not provided
    if args.input_mode in ['bed', 'fasta'] and args.t == 'deletion' and not getattr(args, 'k', None):
        parser.error("'-k' is required when type is 'deletion'")

    return args


if __name__ == "__main__":
    args = get_arguments()
    if args.input_mode == 'bed':
        if args.t == 'substitution':
            gen_saturation_mutagenesis_graphs_substitution(work_dir=args.o, genome=args.r, model_wts_name=args.m, alu_bed_file=args.b, plot_mode=args.yaxis, has_peaks=args.p, no_alu_boundaries=args.no_alu_boundaries)
        elif args.t == 'deletion':
            gen_saturation_mutagenesis_graphs_deletion(work_dir=args.o, model_wts_name=args.m, ks=args.k, genome=args.r, alu_bed_file=args.b, plot_mode=args.yaxis, has_peaks=args.p, no_alu_boundaries=args.no_alu_boundaries)

    if args.input_mode == 'fasta':
        if args.t == 'substitution':
            gen_saturation_mutagenesis_graphs_substitution(work_dir=args.o, model_wts_name=args.m, alu_fa_file=args.f, plot_mode=args.yaxis, has_peaks=args.p, no_alu_boundaries=args.no_alu_boundaries)
        elif args.t == 'deletion':
            gen_saturation_mutagenesis_graphs_deletion(work_dir=args.o, model_wts_name=args.m, ks=args.k, alu_fa_file=args.f, plot_mode=args.yaxis, has_peaks=args.p, no_alu_boundaries=args.no_alu_boundaries)
