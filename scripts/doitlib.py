import ast
import re

import utils
import numpy as np
import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor


def my_load_mnist_data(keras_loader):
    (x_train, y_train), (x_test, y_test) = keras_loader()
    # Add channel dimension for grayscale images
    x_test = x_test[..., tf.newaxis]
    return (x_train, y_train), (x_test, y_test)


datasets = {
    "mnist": {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 10,
        "keras_load_data_method": lambda: my_load_mnist_data(tf.keras.datasets.mnist.load_data)
    },
    "fashion_mnist": {
        "channels": 1,
        "default_input_size": 28,
        "num_classes": 10,
        "keras_load_data_method": lambda: my_load_mnist_data(tf.keras.datasets.fashion_mnist.load_data)
    },
    "cifar10": {
        "channels": 3,
        "default_input_size": 32,
        "num_classes": 10,
        "keras_load_data_method": tf.keras.datasets.cifar10.load_data
    }
}


# ============================================================================
# Model architectures
#
# CAV 2025 (dense networks): a list of hidden-layer sizes, such as [128,128]
# or the string "[128,128]". The output layer d(10) is added automatically.
#
# CNN extension: a string in the c(C,K,S) / d(D) notation of Leino et al.
# (2021, App. B), such as "c(16,2,1).c(2,2,1).d(100).d(10)", the
# GloRo-inspired MNIST 2C2F model in README.md:
#   c(C,K,S)    convolution with C output channels, a K x K kernel, stride S
#               and maximal non-trivial zero padding: K-1 zeros on each side,
#               so the input is padded to (n+2K-2) x (n+2K-2) (Prach & Lampert
#               2022, App. A.1; Leino et al. use SAME padding instead)
#   c(C,K,S,P)  the same, with P zeros of padding on each side instead
#   d(D)        dense layer with D outputs
# All convolutional layers come before the dense layers, and the last layer
# must be d(<number of classes>). As in CAV 2025, no layer has a bias and
# every layer except the last is followed by ReLU.
# ============================================================================

CONV_LAYER = re.compile(r"c\((\d+),(\d+),(\d+)(?:,(\d+))?\)")
DENSE_LAYER = re.compile(r"d\((\d+)\)")


def conv_output_size(n, k, stride, pad):
    """Height (= width) of a convolution's output for an n x n input."""
    return (n + 2 * pad - k) // stride + 1


def parse_architecture(arch, num_classes, input_size):
    """The model's layers, output layer included, as a list of
    ("conv", C, K, S, P) and ("dense", D) tuples."""
    if isinstance(arch, str) and arch.strip().startswith("["):
        arch = ast.literal_eval(arch.strip())
    if not isinstance(arch, str):
        # CAV 2025: hidden dense layers, then the output layer
        return [("dense", int(d)) for d in arch] + [("dense", num_classes)]

    layers = []
    n = input_size
    for token in arch.replace(" ", "").split("."):
        conv = CONV_LAYER.fullmatch(token)
        dense = DENSE_LAYER.fullmatch(token)
        if conv:
            C, K, S = (int(conv.group(g)) for g in (1, 2, 3))
            P = K - 1 if conv.group(4) is None else int(conv.group(4))
            if layers and layers[-1][0] == "dense":
                raise ValueError(
                    f"{arch}: convolutional layers must come before dense layers")
            if min(C, K, S) < 1 or n + 2 * P < K:
                raise ValueError(
                    f"{arch}: {token} does not fit a {n}x{n} input")
            layers.append(("conv", C, K, S, P))
            n = conv_output_size(n, K, S, P)
        elif dense and int(dense.group(1)) > 0:
            layers.append(("dense", int(dense.group(1))))
        else:
            raise ValueError(f"{arch}: cannot read layer '{token}'")
    if layers[-1] != ("dense", num_classes):
        raise ValueError(f"{arch}: the last layer must be d({num_classes})")
    return layers


def build_model(Input, Flatten, Dense, input_size=28, dataset='mnist', internal_layer_sizes=[]):
    """set input_size to something smaller if the model is downsampled.
    internal_layer_sizes: a CAV 2025 list of hidden-layer sizes, or a CNN
    architecture string (see "Model architectures" above)"""
    if dataset not in datasets.keys():
        raise ValueError(
            f"Unsupported dataset. Choose one of: {dataset.keys()}")

    channels = datasets[dataset]["channels"]
    layers = parse_architecture(
        internal_layer_sizes, datasets[dataset]["num_classes"], input_size)

    inputs = Input((input_size, input_size, channels))
    z = inputs
    flattened = False
    for i, layer in enumerate(layers):
        if layer[0] == "conv":
            _, C, K, S, P = layer
            # Keras' Conv2D only offers 'valid' (no zeros) or 'same' padding, so
            # ZeroPadding2D adds the P zeros and the convolution adds none.
            if P > 0:
                z = tf.keras.layers.ZeroPadding2D(padding=P)(z)
            z = tf.keras.layers.Conv2D(C, K, strides=S, padding='valid', use_bias=False,
                                       activation='relu')(z)
        else:
            if not flattened:
                z = Flatten()(z)
                flattened = True
            is_output = i == len(layers) - 1
            z = Dense(layer[1], use_bias=False,
                      activation=None if is_output else 'relu')(z)
    return (inputs, z)


def load_and_set_weights(csv_loc, internal_layer_sizes, model):
    """model should already be built. This will compile it too.
    layer_<i>_weights.csv holds the kernel of the model's i-th layer that has
    weights (dense or convolutional). A conv kernel, of Keras shape
    (K, K, C_in, C_out), is stored row-major as a (K*K*C_in) x C_out matrix.
    internal_layer_sizes is no longer needed; it is kept so that existing
    calls still work."""
    weighted_layers = [layer for layer in model.layers if layer.get_weights()]
    for i, layer in enumerate(weighted_layers):
        kernel = np.loadtxt(csv_loc+f"layer_{i}_weights.csv", delimiter=",")
        layer.set_weights([kernel.reshape(layer.get_weights()[0].shape)])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(
                      from_logits=True),
                  metrics=['accuracy'])


def load_gloro_data(batch_size=256, augmentation='none', input_size=28, dataset='mnist'):
    """set input_size to resize the dataset. Returns a pair (train, test)"""
    train, test, metadata = utils.get_data(dataset, batch_size, augmentation)

    if dataset not in datasets.keys():
        raise ValueError(
            f"Unsupported dataset. Choose one of: {dataset.keys()}")

    default_input_size = datasets[dataset]["default_input_size"]

    if input_size != default_input_size:
        def resize(image, label):
            image = tf.image.resize(image, [input_size, input_size])
            return image, label
        train = train.map(resize)
        test = test.map(resize)

    return (train, test)


def load_test_data(dataset='mnist', input_size=None):
    """Load and preprocess test data for the specified dataset ('mnist' or 'cifar10').                                                              
    Set input_size to resize the test dataset. Returns a pair (x_test, y_test).                                                                     
    """
    # Turn off SSL certificate checking :(
    import ssl
    ssl._create_default_https_context = ssl._create_unverified_context

    import tensorflow as tf
    import numpy as np

    if dataset not in datasets.keys():
        raise ValueError(
            f"Unsupported dataset. Choose one of: {dataset.keys()}")

    load_data = datasets[dataset]["keras_load_data_method"]
    (x_train, y_train), (x_test, y_test) = load_data()

    num_classes = datasets[dataset]["num_classes"]
    default_size = datasets[dataset]["default_input_size"]

    # Normalize pixel values to [0, 1]
    x_test = x_test.astype('float32') / 255.0

    # Set default input_size if not provided
    if input_size is None:
        input_size = default_size

    # Resize the test dataset if input_size differs from the default size
    if input_size != default_size:
        resized_tensor = tf.image.resize(x_test, [input_size, input_size])
        if tf.executing_eagerly():
            x_test = resized_tensor.numpy()
        else:
            # Convert the tensor to NumPy using a session
            with tf.compat.v1.Session() as sess:
                x_test = sess.run(resized_tensor)

    # Convert labels to one-hot encoded format
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=num_classes)

    return (x_test, y_test)
