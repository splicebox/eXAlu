"""
Functions for encoding fixed length sequences of amino acids into various
vector representations, such as one-hot and BLOSUM62.
"""

from __future__ import (
    print_function,
    division,
    absolute_import,
)
import collections
from copy import copy

import pandas
from six import StringIO


COMMON_NUCLEOTIDES = collections.OrderedDict(sorted({
    "A": "Adenine",
    "G": "Guanine",
    "C": "Cytosine",
    "T": "Thymine"
    # "a": "Adenine",
    # "g": "Guanine",
    # "c": "Cytosine",
    # "t": "Thymine"
}.items()))
COMMON_NUCLEOTIDES_WITH_UNKNOWN = copy(COMMON_NUCLEOTIDES)
COMMON_NUCLEOTIDES_WITH_UNKNOWN["N"] = "Unknown"

NUCLEOTIDES_INDEX = dict(
    (letter, i) for (i, letter) in enumerate(COMMON_NUCLEOTIDES_WITH_UNKNOWN))

NUCLEOTIDES = list(COMMON_NUCLEOTIDES_WITH_UNKNOWN.keys())


ENCODING_DATA_FRAMES = {
    "one-hot": pandas.DataFrame([
        [1 if i == j else 0 for i in range(len(NUCLEOTIDES))]
        for j in range(len(NUCLEOTIDES))
    ], index=NUCLEOTIDES, columns=NUCLEOTIDES)
}


def available_vector_encodings():
    """
    Return list of supported amino acid vector encodings.
    Returns
    -------
    list of string
    """
    return list(ENCODING_DATA_FRAMES)


def vector_encoding_length(name):
    """
    Return the length of the given vector encoding.
    Parameters
    ----------
    name : string
    Returns
    -------
    int
    """
    return ENCODING_DATA_FRAMES[name].shape[1]


def index_encoding(sequences, letter_to_index_dict):
    """
    Encode a sequence of same-length strings to a matrix of integers of the
    same shape. The map from characters to integers is given by
    `letter_to_index_dict`.
    Given a sequence of `n` strings all of length `k`, return a `k * n` array where
    the (`i`, `j`)th element is `letter_to_index_dict[sequence[i][j]]`.
    Parameters
    ----------
    sequences : list of length n of strings of length k
    letter_to_index_dict : dict : string -> int
    Returns
    -------
    numpy.array of integers with shape (`k`, `n`)
    """
    df = pandas.DataFrame(iter(s) for s in sequences)
    result = df.replace(letter_to_index_dict)
    return result.values


def fixed_vectors_encoding(index_encoded_sequences, letter_to_vector_df):
    """
    Given a `n` x `k` matrix of integers such as that returned by `index_encoding()` and
    a dataframe mapping each index to an arbitrary vector, return a `n * k * m`
    array where the (`i`, `j`)'th element is `letter_to_vector_df.iloc[sequence[i][j]]`.
    The dataframe index and columns names are ignored here; the indexing is done
    entirely by integer position in the dataframe.
    Parameters
    ----------
    index_encoded_sequences : `n` x `k` array of integers
    letter_to_vector_df : pandas.DataFrame of shape (`alphabet size`, `m`)
    Returns
    -------
    numpy.array of integers with shape (`n`, `k`, `m`)
    """
    (num_sequences, sequence_length) = index_encoded_sequences.shape
    target_shape = (
        num_sequences, sequence_length, letter_to_vector_df.shape[0])
    result = letter_to_vector_df.iloc[
        index_encoded_sequences.reshape((-1,))  # reshape() avoids copy
    ].values.reshape(target_shape)
    return result

# print(ENCODING_DATA_FRAMES["one-hot"][list('AGCTN')])