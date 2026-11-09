# A Formally Verified Robustness Certifier for Neural Networks

This is the robustness certifier of CAV 2025 (Tobler, Syeda and Murray),
written and verified in Dafny, extended to convolutional neural networks (CNNs)
with zero-padded convolutional layers and set up to run on Apple silicon
(arm64) Macs. It still reproduces the CAV 2025 results for dense networks.

- [A Formally Verified Robustness Certifier for Neural Networks](#a-formally-verified-robustness-certifier-for-neural-networks)
  - [Setting up](#setting-up)
    - [Docker on x86-64](#docker-on-x86-64)
    - [Apple silicon, natively](#apple-silicon-natively)
    - [Apple silicon, in Docker](#apple-silicon-in-docker)
  - [Verifying and building the certifier](#verifying-and-building-the-certifier)
  - [Running the certifier](#running-the-certifier)
    - [Input format](#input-format)
  - [Reproducing the CAV 2025 results](#reproducing-the-cav-2025-results)
    - [Table 1](#table-1)
    - [Figure 8](#figure-8)
  - [The CNN extension](#the-cnn-extension)
  - [CNN experiments](#cnn-experiments)
    - [Architectures](#architectures)
    - [Results](#results)

## Setting up

Verifying and building the certifier needs Dafny 4.9.0, Z3 4.13.4 and a .NET
SDK. The scripts need Python with TensorFlow 2.13 and gloro. The CAV 2025
Docker container provides all of these on x86-64 hosts, but does not work on
ARM hosts: on Apple silicon, set up natively or build an arm64 container. The
experiments ran on machines with 16 GB of memory.

### Docker on x86-64

```
gunzip cav2025-artifact.tar.gz
docker image load < cav2025-artifact.tar
docker run -it cav2025-artifact:latest bash
source cav2025-artifact/bin/activate
```

The container's `/workspace` holds `robustness-verifier/` (this repository),
`cav2025-artifact/` (the Python virtual environment), `cav2025-models/` (the
paper's three models and their result files) and `gloro/`.

### Apple silicon, natively

Install a .NET SDK, Dafny 4.9.0, Z3 4.13.4, `ts` (in moreutils), `jq` and
Python 3.11. Homebrew's `dafny` and `z3` are newer than `verify_all.sh`
accepts.

```
brew install --cask dotnet-sdk
brew install moreutils jq python@3.11
dotnet tool install --global dafny --version 4.9.0 --allow-roll-forward
mkdir -p ~/tools && cd ~/tools
curl -LO https://github.com/Z3Prover/z3/releases/download/z3-4.13.4/z3-4.13.4-arm64-osx-13.7.1.zip
unzip z3-4.13.4-arm64-osx-13.7.1.zip && xattr -dr com.apple.quarantine z3-4.13.4-arm64-osx-13.7.1
echo 'export PATH="$HOME/.dotnet/tools:$HOME/tools/z3-4.13.4-arm64-osx-13.7.1/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
dafny --version && z3 --version
```

Then, in this repository's directory, create the virtual environment `.venv`:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install tensorflow==2.13.0 numpy==1.24.3 tensorflow-metadata==1.17.0 tensorflow_datasets matplotlib scikit-learn pandas mpmath adversarial-robustness-toolbox==1.19.1
git clone https://github.com/klasleino/gloro /tmp/gloro
(cd /tmp/gloro && git checkout a218dcdaaa41951411b0d520581e96e7401967d7 && pip install --no-deps .)
rm -rf /tmp/gloro
pip check
```

The pinned versions matter:

- `tensorflow==2.13.0` installs `tensorflow-macos`. pip does not count that as
  the `tensorflow` that gloro requires, so install gloro with `--no-deps`, or
  pip adds a newer TensorFlow with Keras 3.
- `numpy==1.24.3` is the newest numpy that TensorFlow 2.13 accepts.
- `tensorflow-metadata==1.17.0` needs protobuf 4.25, the newest that
  TensorFlow 2.13 accepts. Newer releases need protobuf 6 and break
  `tensorflow_datasets`.
- `adversarial-robustness-toolbox==1.19.1` is the release that the CAV 2025
  artifact used. ART 1.20.0 needs Keras 3 and breaks `run_pgd_attack.py`.

Finally, unzip the models into this repository's parent directory
(`unzip cav2025-models.zip -d ..`), and use `.venv` wherever the instructions
below say `cav2025-artifact`.

### Apple silicon, in Docker

In this repository's directory, which holds `cav2025-models.zip`, build and
enter an arm64 container. It has the CAV 2025 container's layout, with .NET 10,
Dafny 4.9.0 and Z3 4.13.4.

```
docker build -t cav2025-artifact:arm64 .
docker run -it cav2025-artifact:arm64 bash
source cav2025-artifact/bin/activate
```

For larger models, raise Docker Desktop's memory limit (Settings > Resources).

## Verifying and building the certifier

In `robustness-verifier/` (this repository's directory, outside the container),
run:

```
./verify_all.sh
```

Every file should verify with 0 errors. This takes about a minute (58 s on an
M1 MacBook Air).

| File                                               | Verified |
| -------------------------------------------------- | -------- |
| `basic_arithmetic.dfy`                             | 52       |
| `convolution.dfy`                                  | 68       |
| `l2_extra.dfy`                                     | 4        |
| `linear_algebra.dfy`                               | 96       |
| `main.dfy`                                         | 2        |
| `neural_networks.dfy`                              | 10       |
| `parsing.dfy`                                      | 37       |
| `robustness_certification.dfy`                     | 44       |
| `operator_norms.dfy` (with `--isolate-assertions`) | 2,300    |

The verified property is soundness: the certifier says that an output is robust
only if it is. `main.dfy` asserts it after each call to `CertifyMargin`:

```
assert robust ==> forall v: Vector |
  IsInput(v, neuralNet) && NN(neuralNet, v) == outputVector ::
  Robust(v, outputVector, errorMargin, neuralNet);
```

`Robust` is defined in `robustness_certification.dfy` (CAV 2025, Section 4.4),
and `IsInput` and `NN` in `neural_networks.dfy` (Sections 4.1 and 4.2, where
`NN` is called ApplyNN). For a CNN, `neuralNet` is the ghost network
`CNNToNeuralNetwork(cnn)`, so
the guarantee covers CNNs unchanged (see [The CNN extension](#the-cnn-extension)).

To build the certifier binary `main`, run:

```
dafny build --unicode-char:false --target:cs main.dfy IO/FileIO.cs
```

The first build downloads NuGet packages. The generated C# project targets
.NET 6 and rolls forward to a newer installed .NET.

## Running the certifier

```
Usage: main <neural_network_input.txt> <GRAM_ITERATIONS> [<CONV_ITERATIONS>]
```

- `GRAM_ITERATIONS` is the number of Gram iterations for each dense layer. More
  iterations give tighter bounds but take longer. 12 suffice for the paper's
  models.
- `CONV_ITERATIONS` is the number of iterations of Algorithm 3 (norm2_toep) of
  Delattre et al. (2024) for each convolutional layer. The default is the
  smaller of `GRAM_ITERATIONS` and 3. Each iteration grows the filter from
  k x k to (2k-1) x (2k-1), so keep this number small.

The certifier first computes the margin Lipschitz bounds. It then reads lines
from stdin, each with an output vector and a perturbation bound, such as
`3.0,3.0,4.0 0.49`, and certifies each output. Its JSON output also gives an
upper bound on each layer's spectral norm (`spectral_norm_bounds`).

### Input format

The network file lists the layers in order, separated by commas, and contains
only the characters `0123456789.,-[]` and convolutional layers' headers.

- A dense layer is a matrix `[[...],...,[...]]` with one row per input, as
  Keras stores its weights. Each number is an optional `-`, digits, `.` and
  digits.
- A convolutional layer is a header `conv(C_in,N,C,K,S,P)` followed by a matrix
  with C rows, one per output channel, each with C_in·K·K weights, listed by
  input channel, then kernel row, then kernel column. The layer applies
  `c(C,K,S,P)` (see [CNN experiments](#cnn-experiments)) to an input of C_in
  channels of N x N. Inputs and outputs are listed channel by channel, then row
  by row: pixel (i, j) of channel c is entry c·N·N + i·N + j.

Convolutional layers come first, the last layer is dense, and each layer's
output size is the next layer's input size.

## Reproducing the CAV 2025 results

Run these commands in `robustness-verifier/scripts/`, in the virtual
environment. The models of the paper's Table 1 are in `cav2025-models/`:
`2025-01-25_09:27:46-mnist/`, `2025-01-30_10:58:01-fashion_mnist/` and
`2025-01-28_20:39:32-cifar10/`. They already hold the paper's result files,
which the evaluation script will not overwrite, so delete the result JSON
first. For MNIST:

```
rm "../../cav2025-models/2025-01-25_09:27:46-mnist/results_epsilon_0.45_[128,128,128,128,128,128,128,128]_500_eval_0.3_gram_11.json"
```

### Table 1

```
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-25_09:27:46-mnist"/ mnist 0.45 "[128,128,128,128,128,128,128,128]" 0.3 ../main 11 500 32
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-30_10:58:01-fashion_mnist"/ fashion_mnist 0.26 "[256,128,128,128,128,128,128,128,128,128,128,128]" 0.25 ../main 12 500 64
./doit_verified_robust_gloro.sh "../../cav2025-models/2025-01-28_20:39:32-cifar10"/ cifar10 0.1551 "[512,256,128,128,128,128,128,128]" 0.141 ../main 12 800 256
```

These take about 20 to 40 minutes, 20 to 50 minutes and 20 to 40 hours. The
arguments are the model's directory, the dataset, the training ε, the
architecture ("Hidden Neurons" in Table 1), the evaluation ε, the certifier,
the number of Gram iterations, and the training epochs and batch size. Answer
`y` when the script asks whether to proceed. The MNIST run ends like this:

```
Proportion robust: 0.9574
Proportion correct: 0.984
Proportion robust and correct: 0.954
...
Certifier started at:         2025-04-09 08:11:11
Certifier produced bounds at: 2025-04-09 08:27:10
```

"Proportion robust" is the verified robustness (95.74%), and "Proportion robust
and correct" is the VRA (95.40%). The time between the two timestamps is the
time taken to compute the bounds: here 15:59, against 17:02 in Table 1. The run
also prints gloro's unverified robustness and VRA (95.76% and 95.41%), from
which Table 1's differences (-0.02 and -0.01) come.

### Figure 8

Figure 8 plots the MNIST model's results against the number of Gram iterations.

- Verified robustness and running time: run the MNIST command above with each
  number of Gram iterations from 1 to 12. `scripts/results/` already holds the
  values.
- Unverified robustness (95.76%): gloro's statistics in
  `cav2025-models/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500/gloro_model_stats.json`.
- Measured robustness (97.46%): attack the model, which takes 30 to 80
  minutes. In the CAV 2025 container, first run
  `pip install mpmath adversarial-robustness-toolbox==1.19.1`.

  ```
  python3 run_pgd_attack.py mnist "[128,128,128,128,128,128,128,128]" "../../cav2025-models/2025-01-25_09:27:46-mnist/model_weights_epsilon_0.45_[128,128,128,128,128,128,128,128]_500"/ 0.3 500 28
  ```

  It prints "Robustness on PGD adversarial samples: 97.46%", or a close value,
  since the attacks are random. It also prints "Robust accuracy on PGD
  adversarial samples", an upper bound on the VRA.

## The CNN extension

The certifier also accepts CNNs: zero-padded convolutional layers followed by
dense layers, all without biases, and each except the last followed by ReLU.
The scripts build each convolutional layer as `ZeroPadding2D(P)` followed by
`Conv2D(C, K, strides=S, padding='valid', use_bias=False)`, with `Flatten`
before the first dense layer.

What is verified:

- `convolution.dfy` models each convolutional layer as a ghost banded doubly
  block Toeplitz matrix (`ToeplitzMatrix`), and `CNNToNeuralNetwork` turns a
  CNN into a CAV 2025 `NeuralNetwork` with one matrix per layer. The matrices
  are never built at run time.
- `ToeplitzIsConv` (`convolution.dfy`) proves that this matrix computes
  `Conv2DZeroPad`, which is defined as Keras computes `ZeroPadding2D(P)`
  followed by `Conv2D(C, K, strides=S, padding='valid', use_bias=False)`. That
  this definition matches Keras is not proved.
- `Norm2Toep` (`operator_norms.dfy`) bounds a convolutional layer's spectral
  norm by Algorithm 3 (norm2_toep) of Delattre et al. (2024), run on the
  layer's filter. Each Gram iterate is rescaled and rounded to 16 decimal
  places, and the rounding errors are added back. Dense layers use the CAV
  2025 Gram iteration.

What is assumed: the CAV 2025 axioms (`SpecNorm`, `Sqrt`, `Assumption1` to
`Assumption4` and `CauchySchwartz`) and four more, at the end of
`operator_norms.dfy`, each a published or textbook fact:

- `ConvGramIdentity`: with stride 1 and maximal non-trivial zero padding,
  TᵀT, where T is the layer's matrix, is the matrix of the convolution with
  the Gram filter (Prach & Lampert 2022, Eq. 17; Delattre et al. 2024, Th. 4).
- `SpecNormTriangle`: the triangle inequality for the spectral norm.
- `ConvCrop`: any other zero padding or stride cannot increase the spectral
  norm, because each row of the layer's matrix is zero or a different row of
  that matrix (Prach & Lampert 2022, App. A.1; Thompson 1972).
- `ConvSchur`: by Schur's test, the spectral norm is at most the larger of the
  largest absolute row sum and column sum, which `SchurBound` bounds from the
  filter's weights.

`scripts/check_conv_axioms.py` checks the three axioms about convolutions, and
the bound that `Norm2Toep` computes, numerically on 2,592 random layers.

## CNN experiments

### Architectures

The experiments use three CNNs inspired by the MNIST 2C2F, MNIST 4C3F and
CIFAR-10 6C2F models of Leino et al. (2021, App. B), in their notation:
`c(C,K,S)` is a convolutional layer with C output channels, a K x K kernel,
stride S and K-1 zeros of padding on each side (`c(C,K,S,P)` has P zeros), and
`d(D)` is a dense layer with D outputs.

| Model | Dataset   | ε (training / evaluation) | Architecture                                                              | Inputs to first dense layer | Weights (original)    |
| ----- | --------- | ------------------------- | ------------------------------------------------------------------------- | --------------------------- | --------------------- |
| 2C2F  | `mnist`   | 0.45 / 0.3                | `c(16,2,1).c(2,2,1).d(100).d(10)`                                         | 2·30·30 = 1,800             | 181,192 (166,248)     |
| 4C3F  | `mnist`   | 1.75 / 1.58               | `c(32,2,1).c(32,2,1).c(64,2,1).c(2,2,1).d(512).d(512).d(10)`              | 2·32·32 = 2,048             | 1,328,768 (1,973,536) |
| 6C2F  | `cifar10` | 0.1551 / 0.141            | `c(32,2,1).c(32,2,1).c(32,2,1).c(64,2,1).c(64,2,1).c(2,2,1).d(512).d(10)` | 2·38·38 = 2,888             | 1,517,440 (2,249,568) |

Each keeps the original's numbers of layers, dense-layer sizes and channel
widths, but every convolutional layer has a 2 x 2 kernel, stride 1 and one zero
of padding on each side, and the last one has 2 channels. The originals, with
SAME padding, are `c(16,4,2).c(32,4,2).d(100).d(10)`,
`c(32,3,1).c(32,4,2).c(64,3,1).c(64,4,2).d(512).d(512).d(10)` and
`c(32,3,1).c(32,3,1).c(32,4,2).c(64,3,1).c(64,3,1).c(64,4,2).d(512).d(10)`.

The 2 channels keep the first dense layer small: the cost of its Gram
iteration grows with its number of inputs.

Training follows Leino et al.'s ReLU models (2021, Table B.2), as CAV 2025
did. Each model is trained with a larger ε than it is evaluated with, for 500,
300 and 800 epochs, with batches of 256, Adam, a learning rate decaying from
10⁻³ to 10⁻⁶, cross-entropy loss, a fixed ε and no data augmentation. Leino et
al. also let 4C3F's learning rate decay only to 5 × 10⁻⁶, and trained 6C2F with
data augmentation, warm-up, a logarithmic ε schedule and orthogonal
initialisation.

[`scripts/README.md`](scripts/README.md#cnn-experiments) gives the commands for
all the experiments.

### Results

These results use 6 Gram iterations for the dense layers and the default 3
iterations of Algorithm 3, on an M1 MacBook Air with 16 GB of memory.
Percentages are of the 10,000 test points. GloRo's figures are its unverified
statistics from training, and the measured ones are the upper bounds from the
attacks of `run_pgd_attack.py`. The times are those taken to compute the
bounds.

| Model                             | Accuracy | Robustness: verified / GloRo / measured | VRA: verified / GloRo / measured | Time: verified / GloRo |
| --------------------------------- | -------- | --------------------------------------- | -------------------------------- | ---------------------- |
| 2C2F                              | 97.59%   | 91.98% / 93.98% / 96.88%                | 91.56% / 93.43% / 95.75%         | 6,466 s / 0.1 s        |
| 4C3F                              | 90.44%   | 10.08% / 44.33% / 60.69%                | 10.08% / 44.25% / 60.12%         | 11,321 s / 1.5 s       |
| 4C3F, 4 iterations of Algorithm 3 | 90.44%   | 27.16% / 44.33% / 60.69%                | 27.15% / 44.25% / 60.12%         | 13,096 s / 1.5 s       |
| 6C2F                              | 53.52%   | 5.86% / 46.62% / 66.33%                 | 5.49% / 33.18% / 42.12%          | 28,872 s / 2.0 s       |
| CAV 2025 MNIST (dense)            | 98.40%   | 94.18% / 95.76% / –                     | 93.94% / 95.41% / –              | 637 s / 2.2 s          |
| CAV 2025 CIFAR-10 (dense)\*       | 57.74%   | 39.13% / 46.42% / –                     | 31.66% / 36.17% / –              | – / 1.0 s              |

\* The verified figures come from CAV 2025's result file for 6 Gram iterations
in `cav2025-models.zip`: we did not certify this model again. With 12 Gram
iterations, CAV 2025 reports 46.12% and 35.95%, with the bounds computed in
19:02:32 (Table 1).

With 11 Gram iterations, the certifier reproduces the CAV 2025 MNIST model's
published verified robustness and VRA, 95.74% and 95.40%, exactly, in 1,104 s.

The attacks found adversarial examples for 312, 3,931 and 3,367 test points of
2C2F, 4C3F and 6C2F, but none for an output that the certifier, with 6 Gram
iterations, or GloRo had certified robust.
