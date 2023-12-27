from .nucleotides import *
import numpy as np
import torch

# R: purine, = A or G
# Y: pyrimidine, = C or T
# M is A or C
base_dict = {
            'A': [1., 0., 0., 0.], 'a': [1., 0., 0., 0.],
            'C': [0., 1., 0., 0.], 'c': [0., 1., 0., 0.],
            'G': [0., 0., 1., 0.], 'g': [0., 0., 1., 0.],
            'T': [0., 0., 0., 1.], 't': [0., 0., 0., 1.],
            'U': [0., 0., 0., 1.], 'u': [0., 0., 0., 1.],
            'X': [0.25, 0.25, 0.25, 0.25], 'x': [0.25, 0.25, 0.25, 0.25], 
            'N': [0., 0., 0., 0.], 'n': [0., 0., 0., 0.], 
            'R': [0.5, 0., 0.5, 0.], 'r': [0.5, 0., 0.5, 0.], 
            'Y': [0., 0.5, 0., 0.5], 'y': [0., 0.5, 0., 0.5], 
            'M': [0.5, 0.5, 0, 0], 'm': [0.5, 0.5, 0, 0],
            '.': [1., 0., 0.], '(': [0., 1., 0.], ')': [0., 0., 1.], ',': [0., 0., 0.]}

# base_dict = {
#             'A': [1., 0., 0., 0., 0.], 'a': [1., 0., 0., 0., 0.],
#             'C': [0., 1., 0., 0., 0.], 'c': [0., 1., 0., 0., 0.],
#             'G': [0., 0., 1., 0., 0.], 'g': [0., 0., 1., 0., 0.],
#             'T': [0., 0., 0., 1., 0.], 't': [0., 0., 0., 1., 0.],
#             # 'N': [0.25, 0.25, 0.25, 0.25], 'n': [0.25, 0.25, 0.25, 0.25], 
#             'N': [0., 0., 0., 0., 1.], 'n': [0., 0., 0., 0., 1.], 
#             'R': [0.5, 0., 0.5, 0.], 'r': [0.5, 0., 0.5, 0.], 
#             'Y': [0., 0.5, 0., 0.5], 'y': [0., 0.5, 0., 0.5], 
#             'M': [0.5, 0.5, 0, 0], 'm': [0.5, 0.5, 0, 0]}

def seq_encoding_onehot_padN(seq, max_len):
    seq_len = len(seq)
    if len(seq) != max_len:
        tail_pad_len = int((max_len - seq_len) / 2)
        head_pad_len = max_len - seq_len - tail_pad_len
        # new_pep_seq = pep_seq[0:4] + 'X' * head_pad_len + pep_seq[4:-4] + 'X' * tail_pad_len + pep_seq[-4:]
        if seq[0] in '.()':
            new_seq = ',' * head_pad_len + seq + ',' * tail_pad_len
        else:
            new_seq = 'N' * head_pad_len + seq + 'N' * tail_pad_len
        matrix = [base_dict[base] for base in new_seq]
    else:
        matrix = [base_dict[base] for base in seq]
    # return matrix
    # return np.array(matrix).T
    return torch.tensor(matrix).T

def seq_encoding_onehot(seq):
    matrix = [base_dict[base] for base in seq]
    return torch.tensor(matrix).T 

if __name__ == "__main__":
    pass