import sys
import numpy as np

import doitlib


def mprint(string):
    print(string, end="")


# Prints the network for the certifier on one line: its layers in order, separated by commas.
#   dense layer: [[w,...,w],...,[w,...,w]]  its CSV matrix, one row per input (as in CAV 2025)
#   conv layer:  conv(C_in,N,C,K,S,P)[[w,...,w],...,[w,...,w]]  c(C,K,S,P) on an input of
#                C_in channels of N x N. One row per output channel o, listing the kernel
#                weights K[o][c][a][b] by input channel c, then kernel row a, then column b.
# The certifier lists a conv layer's inputs and outputs channel by channel, as the banded
# doubly block Toeplitz matrix does (Delattre et al. 2024, Sec. V-B).
if len(sys.argv) != 3 and len(sys.argv) != 5:
    print(
        f"Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir [dataset input_size]\n")
    sys.exit(1)

# "[128,128]" or "c(16,2,1).d(10)"; doitlib parses both
INTERNAL_LAYER_SIZES = sys.argv[1]

csv_loc = sys.argv[2]+"/"

# a CNN needs dataset and input_size: they give the input shape of its first conv layer
if len(sys.argv) == 5:
    dataset = sys.argv[3]
    input_size = int(sys.argv[4])
    channels = doitlib.datasets[dataset]["channels"]
    num_classes = doitlib.datasets[dataset]["num_classes"]
elif INTERNAL_LAYER_SIZES.strip().startswith("["):
    input_size = channels = num_classes = None
else:
    print(
        f"A CNN also needs dataset and input_size. Usage: {sys.argv[0]} INTERNAL_LAYER_SIZES model_weights_csv_dir dataset input_size\n")
    sys.exit(1)

layers = doitlib.parse_architecture(
    INTERNAL_LAYER_SIZES, num_classes, input_size)

weights = []  # (header, matrix) for each layer
size = input_size
for i, layer in enumerate(layers):
    mat = np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=",")
    header = ""
    if layer[0] == "conv":
        _, C, K, S, P = layer
        # the CSV holds Keras' kernel, of shape (K, K, C_in, C); reorder it to K[o][c][a][b]
        mat = mat.reshape(K, K, channels, C).transpose(
            3, 2, 0, 1).reshape(C, channels*K*K)
        header = f"conv({channels},{size},{C},{K},{S},{P})"
        size, channels = doitlib.conv_output_size(size, K, S, P), C
    elif i > 0 and layers[i-1][0] == "conv":
        # Keras' Flatten lists the last conv layer's outputs position by position, but
        # the certifier lists them channel by channel, so reorder this layer's rows
        D = mat.shape[1]
        mat = mat.reshape(size, size, channels, D).transpose(
            2, 0, 1, 3).reshape(channels*size*size, D)
    weights.append((header, mat.tolist()))

m = 0

weightslen = len(weights)
w = 0
for header, mat in weights:
    # print(f"Matrix {m} has dimensions: {len(mat)} x {len(mat[0])}")
    m = m + 1

    mprint(header)
    mprint("[")
    matlen = len(mat)
    r = 0
    for row in mat:
        mprint("[")
        count = 0
        n = len(row)
        for num in row:
            mprint(f"{num:.150f}")
            count = count + 1
            if count < n:
                mprint(",")
        mprint("]")
        r = r + 1
        if r < matlen:
            mprint(",")
    mprint("]")
    w = w + 1
    if w < weightslen:
        mprint(",")
