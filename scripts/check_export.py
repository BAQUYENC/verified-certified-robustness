"""Checks, on the test set, that the network file that
make_certifier_format.py wrote computes the outputs of the Keras model that
the certified output vectors come from.

The certifier's guarantee is about the network in the file, as parsing.dfy
reads it, whereas the output vectors that it certifies come from the Keras
model that zero_bias_saved_model.py builds from the weight CSVs. This script
reads the file as ParseCNN does, checks its layers against
INTERNAL_LAYER_SIZES, and evaluates it in double precision as the Dafny
definitions say: Conv2DZeroPad (convolution.dfy), which ToeplitzIsConv
proves equal to the layer's matrix, for a convolutional layer, the
transposed matrix for a dense one, and ReLU after every layer except the
last. A CNN's input lists the image channel by channel, as Vec does: this
only reorders Keras' pixels, so l2 distances are unchanged. It then compares
the outputs on every test point with the Keras model's, rebuilt as
zero_bias_saved_model.py builds it, and with the output file that the
certifier read, which keeps 5 decimal places. It is a numerical check, not a
proof.

Usage: python3 check_export.py dataset INTERNAL_LAYER_SIZES
           model_weights_csv_dir neural_net_file model_outputs_file
           input_size [--rtol R] [--atol A]
Exits with status 0 if every check passes and 1 otherwise."""
import argparse
import os
import re
import sys

import numpy as np

import doitlib

# IsReal and IsInt (parsing.dfy) allow only the ASCII digits, so [0-9], not \d
REAL = re.compile(r"-?[0-9]+\.[0-9]+")
INT = re.compile(r"-?[0-9]+")


def read_network(text):
    """The layers in a network file, read as ParseCNN (parsing.dfy) reads
    them: ("conv", cIn, n, cOut, k, stride, pad, K) with K[o][c][a][b], and
    ("dense", W) with W the transposed matrix, one row per output. Raises
    ValueError exactly where ParseCNN returns None. Like ParseCNN, it skips
    the character after each row and each matrix, whatever it is:
    make_certifier_format.py writes a comma there, or nothing at the end."""
    layers = []
    i = 0
    while i < len(text):
        header = None
        if text.startswith("conv(", i):
            j = text.find(")", i + 5)
            if j < 0:
                raise ValueError(f"no ')' after 'conv(' at character {i}")
            fields = text[i + 5:j].split(",")
            # ParseNats: integers (IsInt) that are not negative, so -0 is 0
            if not all(INT.fullmatch(f) and int(f) >= 0 for f in fields) \
                    or len(fields) != 6:
                raise ValueError(f"bad header conv({text[i + 5:j]})")
            header = [int(f) for f in fields]
            i = j + 1
        if not text.startswith("[[", i):
            raise ValueError(f"no matrix at character {i}")
        end = text.find("]]", i + 1)
        if end < 0:
            raise ValueError(f"no ']]' after the matrix at character {i}")
        # the rows "[r,...,r]" between i and end, one at a time
        rows = []
        start = i + 2
        while True:
            stop = text.find("]", start)
            reals = text[start:stop].split(",")
            bad = [r for r in reals if not REAL.fullmatch(r)]
            if bad:
                raise ValueError(f"'{bad[0][:40]}' in the row at character "
                                 f"{start - 1} is not a real (IsReal)")
            rows.append(np.array(reals, dtype=np.float64))
            if stop + 2 > end:
                break
            # ParseMatrix skips the character after ']', then needs '['
            if text[stop + 2] != "[":
                raise ValueError(f"no '[' at character {stop + 2}")
            start = stop + 3
        if len({len(row) for row in rows}) != 1:
            raise ValueError(f"rows of different lengths at character {i}")
        m = np.array(rows)
        if header is None:
            layers.append(("dense", m.T))
        else:
            cIn, n, cOut, k, stride, pad = header
            if m.shape[1] != cIn * k * k:
                raise ValueError(
                    f"conv({','.join(map(str, header))}) has rows of "
                    f"{m.shape[1]} weights, not {cIn * k * k}")
            if min(cIn, n, cOut, k, stride) < 1 or k > n + 2 * pad \
                    or m.shape[0] != cOut:
                raise ValueError(
                    f"conv({','.join(map(str, header))}) is not ValidConv")
            layers.append(("conv", cIn, n, cOut, k, stride, pad,
                           m.reshape(cOut, cIn, k, k)))
        i = end + 3  # ParseMatrix skips the character after "]]"
    # IsCNN (convolution.dfy), which ParseCNN checks last
    if not layers:
        raise ValueError("no layers (IsCNN)")
    if layers[-1][0] != "dense":
        raise ValueError("the last layer is not dense (IsCNN)")
    for i in range(len(layers) - 1):
        outputs, inputs = rows_cols(layers[i])[0], rows_cols(layers[i + 1])[1]
        if outputs != inputs:
            raise ValueError(f"layer {i + 1} has {outputs} outputs, but layer "
                             f"{i + 2} has {inputs} inputs (IsCNN)")
    return layers


def out_size(layer):
    """OutSize (convolution.dfy): the output's height and width."""
    _, cIn, n, cOut, k, stride, pad, _ = layer
    return (n + 2 * pad - k) // stride + 1


def rows_cols(layer):
    """LayerRows and LayerCols (convolution.dfy): the numbers of outputs and
    inputs of the layer's matrix."""
    if layer[0] == "dense":
        return layer[1].shape
    _, cIn, n, cOut = layer[:4]
    return cOut * out_size(layer) ** 2, cIn * n * n


def describe(layer):
    if layer[0] == "dense":
        rows, cols = layer[1].shape
        return f"dense: {cols} inputs, {rows} outputs"
    m = out_size(layer)
    return (f"conv({','.join(map(str, layer[1:7]))}): "
            f"{layer[1]} x {layer[2]} x {layer[2]} inputs, "
            f"{layer[3]} x {m} x {m} outputs")


def check_layers(layers, arch, channels, input_size):
    """Problems with the layers, which read_network has checked form a CNN
    (IsCNN): they are not the layers that INTERNAL_LAYER_SIZES gives on
    channels x input_size x input_size images."""
    problems = []
    if len(layers) != len(arch):
        problems.append(f"{len(layers)} layers, but INTERNAL_LAYER_SIZES "
                        f"has {len(arch)}")
    shape = (channels, input_size)  # the next layer's input image, if any
    for i, (layer, expected) in enumerate(zip(layers, arch)):
        if layer[0] != expected[0]:
            problems.append(f"layer {i + 1} is {layer[0]}, not {expected[0]}")
        elif layer[0] == "conv":
            if (layer[1], layer[2]) != shape:
                problems.append(
                    f"layer {i + 1} has {layer[1]} x {layer[2]} x {layer[2]} "
                    f"inputs, not {shape[0]} x {shape[1]} x {shape[1]}")
            if layer[3:7] != expected[1:]:
                problems.append(
                    f"layer {i + 1} has (C, K, S, P) = {layer[3:7]}, not "
                    f"{expected[1:]}")
            shape = (layer[3], out_size(layer))
        else:
            if i == 0 and layer[1].shape[1] != channels * input_size ** 2:
                problems.append(f"layer 1 has {layer[1].shape[1]} inputs, "
                                f"not {channels * input_size ** 2}")
            if layer[1].shape[0] != expected[1]:
                problems.append(f"layer {i + 1} has {layer[1].shape[0]} "
                                f"outputs, not {expected[1]}")
            shape = None
    return problems


def conv2d_zero_pad(layer, images):
    """Conv2DZeroPad (convolution.dfy) on a batch of images (B, cIn, n, n):
    output (o, y, x) is the sum over c, a and b of K[o][c][a][b] times the
    zero-padded input at (c, y * stride + a, x * stride + b)."""
    _, cIn, n, cOut, k, stride, pad, K = layer
    m = out_size(layer)
    padded = np.pad(images, ((0, 0), (0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((len(images), cOut, m, m))
    span = stride * (m - 1) + 1
    for a in range(k):
        for b in range(k):
            window = padded[:, :, a:a + span:stride, b:b + span:stride]
            out += np.tensordot(window, K[:, :, a, b],
                                axes=([1], [1])).transpose(0, 3, 1, 2)
    return out


def network_outputs(layers, images, batch_size=200):
    """NN (neural_networks.dfy) of the network for Keras images (N, n, n, C),
    in double precision. A dense first layer reads the image in the order of
    Keras' Flatten (row, column, channel), as CAV 2025's files do, and a
    convolutional one channel by channel, as Vec does."""
    outputs = []
    for start in range(0, len(images), batch_size):
        x = np.asarray(images[start:start + batch_size], dtype=np.float64)
        if layers[0][0] == "conv":
            x = x.transpose(0, 3, 1, 2)
        v = x.reshape(len(x), -1)
        for i, layer in enumerate(layers):
            if layer[0] == "conv":
                v = conv2d_zero_pad(
                    layer, v.reshape(len(v), layer[1], layer[2], layer[2]))
                v = v.reshape(len(v), -1)
            else:
                v = v @ layer[1].T
            if i < len(layers) - 1:
                v = np.maximum(v, 0.0)
        outputs.append(v)
    return np.concatenate(outputs)


def read_outputs(path):
    """The output vectors in a file that zero_bias_saved_model.py wrote, one
    per line, ignoring the epsilon that the doit script appends."""
    with open(path) as f:
        lines = [line.split()[0] for line in f if line.strip()]
    return np.array([line.split(",") for line in lines], dtype=np.float64)


def compare(label, ours, theirs, rtol, atol):
    """Prints how far the network file's outputs (ours) are from theirs, and
    returns whether every output is within the tolerance."""
    if ours.shape != theirs.shape:
        print(f"{label}: {theirs.shape[0]} x {theirs.shape[1]} outputs, but "
              f"the network file gives {ours.shape[0]} x {ours.shape[1]}")
        return False
    diff = np.abs(ours - theirs)
    close = np.isclose(ours, theirs, rtol=rtol, atol=atol).all(axis=1)
    same = ours.argmax(axis=1) == theirs.argmax(axis=1)
    print(f"{label}:")
    print(f"  largest |difference|: {diff.max():.3g} "
          f"(largest |output|: {np.abs(theirs).max():.3g})")
    print(f"  test points with every output within the tolerance: "
          f"{close.sum()} of {len(close)}")
    print(f"  test points with the same predicted class: "
          f"{same.sum()} of {len(same)}")
    if not close.all():
        worst = np.argsort(-diff.max(axis=1))[:5]
        print("  largest differences at test points "
              + ", ".join(str(i) for i in worst))
    return bool(close.all())


def keras_outputs(dataset, arch, csv_dir, input_size):
    """The test images and the Keras model's outputs on them, with the model
    built from the weight CSVs as zero_bias_saved_model.py builds it."""
    from tensorflow.keras.layers import Input, Flatten, Dense
    from tensorflow.keras.models import Model
    inputs, outputs = doitlib.build_model(
        Input, Flatten, Dense, input_size=input_size, dataset=dataset,
        internal_layer_sizes=arch)
    model = Model(inputs, outputs)
    doitlib.load_and_set_weights(os.path.join(csv_dir, ""), arch, model)
    x_test, _ = doitlib.load_test_data(dataset=dataset, input_size=input_size)
    return x_test, model.predict(x_test, verbose=0)


def main(argv):
    parser = argparse.ArgumentParser(
        description="Check that make_certifier_format.py's network file "
        "computes the Keras model's outputs.")
    parser.add_argument("dataset")
    parser.add_argument("INTERNAL_LAYER_SIZES")
    parser.add_argument("model_weights_csv_dir")
    parser.add_argument("neural_net_file")
    parser.add_argument("model_outputs_file")
    parser.add_argument("input_size", type=int)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-4)
    args = parser.parse_args(argv)

    info = doitlib.datasets[args.dataset]
    arch = doitlib.parse_architecture(
        args.INTERNAL_LAYER_SIZES, info["num_classes"], args.input_size)

    print(f"Network file: {args.neural_net_file}")
    # main.dfy's ReadFromFile makes each byte one character: Latin-1, with no
    # newline translation
    with open(args.neural_net_file, encoding="latin-1", newline="") as f:
        try:
            layers = read_network(f.read())
        except ValueError as e:
            print(f"FAIL: the certifier would not read it: {e}")
            return 1
    for i, layer in enumerate(layers):
        print(f"  layer {i + 1}: {describe(layer)}")
    problems = check_layers(layers, arch, info["channels"], args.input_size)
    for problem in problems:
        print(f"  problem: {problem}")
    if problems:
        print("FAIL: the layers are not those of INTERNAL_LAYER_SIZES")
        return 1
    print(f"  the layers are those of {args.INTERNAL_LAYER_SIZES} on "
          f"{info['channels']} x {args.input_size} x {args.input_size} "
          "images")

    x_test, keras = keras_outputs(args.dataset, args.INTERNAL_LAYER_SIZES,
                                  args.model_weights_csv_dir, args.input_size)
    ours = network_outputs(layers, x_test)
    print(f"Test points: {len(x_test)}; tolerance: |difference| <= "
          f"{args.atol:g} + {args.rtol:g} x |output|")
    ok = compare("Keras model (float32) and network file (float64)",
                 ours, np.asarray(keras, dtype=np.float64),
                 args.rtol, args.atol)
    ok = compare(f"Output file that the certifier read "
                 f"({args.model_outputs_file}) and network file",
                 ours, read_outputs(args.model_outputs_file),
                 args.rtol, args.atol) and ok
    if ok:
        print("PASS: the network file computes the Keras model's outputs, "
              "within the tolerance, on every test point")
        return 0
    print("FAIL: the network file does not compute the Keras model's outputs")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
