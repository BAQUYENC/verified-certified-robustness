# Evaluation scripts

Run the scripts from this directory, in the virtual environment (see the
top-level [`README.md`](../README.md#setting-up)).

| Script                                            | Purpose                                                                    |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `doit_verified_robust_gloro.sh`                   | Train or load a model, certify it, and report accuracy, robustness and VRA |
| `cnn_experiments.sh`                              | Run the CNN experiments, one step at a time                                |
| `run_pgd_attack.py`                               | Measure robustness and robust accuracy by adversarial attacks              |
| `get_gloro_robustness_results.py`                 | Recompute gloro's unverified statistics and time its bound computation     |
| `gram_sweep_csv.py`, `plot_robustness_vs_gram.py` | Plot robustness or VRA against the number of Gram iterations               |
| `check_conv_axioms.py`                            | Check the convolution axioms numerically                                   |

## The evaluation script

```
./doit_verified_robust_gloro.sh "train_and_eval" dataset training_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]
./doit_verified_robust_gloro.sh basedir dataset training_epsilon INTERNAL_LAYER_SIZES eval_epsilon robustness_certifier_binary GRAM_ITERATIONS epochs batch_size [model_input_size]
```

The first form trains a new model with gloro (`train_gloro.py`), in a new
directory named after the date and the dataset. The second evaluates the model
that an earlier run created in `basedir`. The script then:

1. builds a zero-bias copy of the model and computes its outputs on the test
   set (`zero_bias_saved_model.py`);
2. writes the weights in the certifier's input format
   (`make_certifier_format.py`);
3. certifies the outputs with the certifier;
4. prints the proportions robust, correct, and robust and correct (the VRA)
   (`test_verified_certified_robust_accuracy.py`), gloro's unverified
   statistics, and the certifier's timestamps. The time taken to compute the
   bounds is the difference between "Certifier started at" and "Certifier
   produced bounds at".

The arguments:

- `dataset`: `mnist`, `fashion_mnist` or `cifar10`.
- `training_epsilon` and `eval_epsilon`: the perturbation bounds for training
  and for certification.
- `INTERNAL_LAYER_SIZES`: either the hidden-layer sizes of a dense network,
  such as `"[128,128]"` (`[]` for none; the output layer is added), or a CNN
  in the notation of Leino et al. (2021, App. B), such as
  `"c(16,2,1).c(2,2,1).d(100).d(10)"`. `c(C,K,S)` is a convolutional layer with
  C output channels, a K x K kernel, stride S and K-1 zeros of padding on each
  side (`c(C,K,S,P)` has P zeros), and `d(D)` is a dense layer with D outputs.
  Convolutional layers come first, and the last layer is `d(10)`. No layer has
  a bias, and every layer except the last is followed by ReLU.
- `robustness_certifier_binary`: usually `../main`.
- `GRAM_ITERATIONS`: the number of Gram iterations for the dense layers.
  Convolutional layers get the certifier's default `CONV_ITERATIONS`, the
  smaller of `GRAM_ITERATIONS` and 3. To choose it, run the certifier directly,
  e.g. `cat outputs_eval.txt | ../main neural_net.txt 6 2 > results.json`.
- `epochs` and `batch_size`: for training.
- `model_input_size` (optional): resizes the images to `model_input_size` x
  `model_input_size`. The default is 28 for MNIST and Fashion MNIST, and 32 for
  CIFAR-10.

For example, this trains and certifies the MNIST 2C2F model, and then
certifies it again with `n` Gram iterations:

```
./doit_verified_robust_gloro.sh train_and_eval mnist 0.45 "c(16,2,1).c(2,2,1).d(100).d(10)" 0.3 ../main 6 500 256
./doit_verified_robust_gloro.sh dir mnist 0.45 "c(16,2,1).c(2,2,1).d(100).d(10)" 0.3 ../main n 500 256
```

For a CNN:

- The model is `ZeroPadding2D(P)` and
  `Conv2D(C, K, strides=S, padding='valid', use_bias=False, activation='relu')`
  for each convolutional layer, then `Flatten` and the dense layers. gloro
  warns that it uses 1.0 as the Lipschitz constant of `ZeroPadding2D`. That is
  correct: padding only adds zeros.
- `train_gloro.py` saves a convolutional layer's kernel, of Keras shape
  `(K, K, C_in, C)`, in `layer_<i>_weights.csv` as a `(K·K·C_in) x C` matrix,
  and a dense layer's weights as its inputs x outputs matrix.
- `make_certifier_format.py INTERNAL_LAYER_SIZES model_weights_csv_dir dataset input_size`
  writes each convolutional layer as `conv(C_in,N,C,K,S,P)` and its kernel. It
  also reorders the first dense layer's rows from the order of Keras' `Flatten`
  (row, column, channel) to the certifier's (channel, row, column).
- Keep the first dense layer's number of inputs, C·M·M for the last
  convolutional layer's C channels of M x M, small. The Gram iteration's matrix
  has one row and one column per input.

## CNN experiments

`./cnn_experiments.sh STEP MODEL [N...]` runs one step of the experiments for
one model, and keeps the model's logs, CSVs, plots and numbers in
`results/cnn/MODEL/`. Run it without arguments for its usage.

Run one step at a time: the certifier's and gloro's bound computation times
are results, and a job running alongside slows them down. If `certify` stops,
run it again: it skips the numbers of Gram iterations that finished. If `train`
stops while training, delete its new run directory, and `model_weights_csv/`
if it exists, before training again.

These are all the experiments, in order. First, CAV 2025's MNIST model,
exported again with today's `make_certifier_format.py`:

```
./cnn_experiments.sh setup-cav cav-mnist
./cnn_experiments.sh certify cav-mnist 6 11
./cnn_experiments.sh gloro cav-mnist
./cnn_experiments.sh summary cav-mnist 6
./cnn_experiments.sh summary cav-mnist 11
```

The three CNNs. `train` certifies each one with 6 Gram iterations, and
`certify` adds the other numbers of Gram iterations:

```
./cnn_experiments.sh train 2c2f
./cnn_experiments.sh gloro 2c2f
./cnn_experiments.sh certify 2c2f 1 2 3 4 5 7 8 9 10
./cnn_experiments.sh pgd 2c2f
./cnn_experiments.sh plot 2c2f
./cnn_experiments.sh summary 2c2f
```

```
./cnn_experiments.sh train 4c3f
./cnn_experiments.sh gloro 4c3f
./cnn_experiments.sh certify 4c3f 1 2 3 4 8
./cnn_experiments.sh pgd 4c3f
./cnn_experiments.sh plot 4c3f
./cnn_experiments.sh summary 4c3f
```

```
./cnn_experiments.sh train 6c2f
./cnn_experiments.sh gloro 6c2f
./cnn_experiments.sh certify 6c2f 1 2 3 4 8
./cnn_experiments.sh pgd 6c2f
./cnn_experiments.sh plot 6c2f
./cnn_experiments.sh summary 6c2f
```

`train` runs the evaluation script:

```
./doit_verified_robust_gloro.sh train_and_eval mnist 0.45 "c(16,2,1).c(2,2,1).d(100).d(10)" 0.3 ../main 6 500 256
./doit_verified_robust_gloro.sh train_and_eval mnist 1.75 "c(32,2,1).c(32,2,1).c(64,2,1).c(2,2,1).d(512).d(512).d(10)" 1.58 ../main 6 300 256
./doit_verified_robust_gloro.sh train_and_eval cifar10 0.1551 "c(32,2,1).c(32,2,1).c(32,2,1).c(64,2,1).c(64,2,1).c(2,2,1).d(512).d(10)" 0.141 ../main 6 800 256
```

4C3F with 4 iterations of Algorithm 3 instead of 3 shows what the default
costs. `cnn_experiments.sh` has no step for this, so run the certifier
directly, and then compute the proportions. This takes about 3.7 hours. On a
Mac, put `caffeinate -i` before `../main` to keep it awake.

```
D=$(cat results/cnn/4c3f/run_dir.txt)
A="c(32,2,1).c(32,2,1).c(64,2,1).c(2,2,1).d(512).d(512).d(10)"
cat "$D/all_mnist_outputs_epsilon_1.75_${A}_300_eval_1.58.txt" | ../main "$D/neural_net_mnist_epsilon_1.75_${A}_300.txt" 6 4 | tee results/cnn/4c3f/conv4-gram-6.json | ts "%Y-%m-%d %H:%M:%S" > results/cnn/4c3f/conv4-gram-6.log
python3 test_verified_certified_robust_accuracy.py mnist "$A" results/cnn/4c3f/conv4-gram-6.json "$D/model_weights_epsilon_1.75_${A}_300" 28
```

The bound computation time is the difference between the timestamps of the
log's first line, `[`, and its line with `"lipschitz_bounds"`.

Last, CAV 2025's CIFAR-10 model. Certifying it again takes many hours: the CAV
2025 run with 6 Gram iterations took 16.4 hours. The results therefore use the
result file of that run, kept in `cav2025-models.zip`. `setup-cav` and `gloro`
time gloro's bound computation (about 1 s). `setup-cav` deletes the result
files from its copy of the model, so unzip the model again to compute the
proportions from the result file:

```
./cnn_experiments.sh setup-cav cav-cifar10
./cnn_experiments.sh gloro cav-cifar10
unzip -q ../cav2025-models.zip 'cav2025-models/2025-01-28_20:39:32-cifar10/*' -d /tmp/cav
D='/tmp/cav/cav2025-models/2025-01-28_20:39:32-cifar10'
A='[512,256,128,128,128,128,128,128]'
python3 test_verified_certified_robust_accuracy.py cifar10 "$A" "$D/results_epsilon_0.1551_${A}_800_eval_0.141_gram_6.json" "$D/model_weights_epsilon_0.1551_${A}_800" 32
```

The last command ends with:

```
Proportion robust: 0.3913
Proportion correct: 0.5774
Proportion robust and correct: 0.3166
```

To certify the model again instead, run `./cnn_experiments.sh certify cav-cifar10 6`
and then `./cnn_experiments.sh summary cav-cifar10 6`.

## Measured robustness

```
python3 run_pgd_attack.py dataset INTERNAL_LAYER_SIZES model_weights_csv_dir eval_epsilon 500 model_input_size
```

`model_weights_csv_dir` is the directory in which the evaluation script saved
the weights. Quote it, because its name contains `[` or `(`. The script needs
mpmath and ART (in the CAV 2025 container, run
`pip install mpmath adversarial-robustness-toolbox==1.19.1` first). It runs
FGSM, the Momentum Iterative Method and PGD, each targeted and untargeted,
with 500 iterations for the iterative attacks. It ends with lines like these:

```
Model accuracy: 72.8
Robustness on PGD adversarial samples: 53.28%
Norms of non-false-positive vectors that cause classification changes: min: 0.00599679290863438; max: 0.289841752514313
False positives in PGD attack: 0
Number of PGD attacks succeeding against certified robust outputs: 0
```

"Robustness on PGD adversarial samples" is the measured robustness. The next
line, "Robust accuracy on PGD adversarial samples", is the measured robust
accuracy. These bound the verified robustness and VRA of any sound certifier
from above. False positives are adversarial examples that lie beyond ε when
their distance is computed exactly, so they do not count. The model accuracy
should match the evaluation script's "Proportion correct".

## gloro's bound computation time

```
python3 get_gloro_robustness_results.py mnist "c(16,2,1).c(2,2,1).d(100).d(10)" "dir/model_weights_epsilon_0.45_c(16,2,1).c(2,2,1).d(100).d(10)_500" 0.3 28 100 > 2c2f-gloro-rerun.json
jq -r '.[].debug_msg // empty' 2c2f-gloro-rerun.json
```

The last line it prints, `gloro Lipschitz bound computation time (seconds): …`,
is gloro's time in the results table. The last argument, 100, is the maximum
number of power-iteration steps.

## Plots

`cnn_experiments.sh plot` plots robustness and VRA against the number of Gram
iterations. The plot script needs pandas (in the CAV 2025 container, run
`pip install pandas numpy==1.24.3` first). By hand, save each evaluation run's
output (add
`2>&1 | tee 2c2f-gram-n.log` to the command) and the measured robustness as a
proportion in a text file (`0.972` for 97.20%, as the plot script multiplies
its inputs by 100), and then run:

```
python3 gram_sweep_csv.py robust 2c2f-gram-*.log > 2c2f-sweep.csv
jq '.[0].robustness' "dir/model_weights_epsilon_0.45_c(16,2,1).c(2,2,1).d(100).d(10)_500/gloro_model_results.json" > 2c2f-gloro.txt
python3 plot_robustness_vs_gram.py 2c2f-sweep.csv 2c2f-measured.txt 2c2f-gloro.txt 2c2f.pdf
```

For the VRA, use `gram_sweep_csv.py vra`, `jq '.[0].vra'` and the measured
robust accuracy, and add `vra` at the end of the plot command.

The data for CAV 2025's Figure 8 are in `results/`:

```
python3 plot_robustness_vs_gram.py "results/robustness_vs_gram_input_28_[128,128,128,128,128,128,128,128].txt" "results/measured_robustness_input_28_[128,128,128,128,128,128,128,128].txt" "results/gloro_robustness_28_[128,128,128,128,128,128,128,128].txt" performance_28.pdf
```

`results/` also holds the data for two smaller MNIST models, with 7 x 7 and
14 x 14 inputs. To redo them, train with 1 Gram iteration, and certify the
model in the directory that this creates (`dir`) with 2 to 10:

```
./doit_verified_robust_gloro.sh train_and_eval mnist 0.3 "[16]" 0.3 ../main 1 500 256 7
./doit_verified_robust_gloro.sh dir mnist 0.3 "[16]" 0.3 ../main n 500 256 7
```

For 14 x 14, use `"[64]"` and `14`. Record each run's proportion robust and
time in seconds in `results/robustness_vs_gram_input_7_[16].csv` (or
`results/robustness_vs_gram_input_14_[64].csv`), and the measured robustness
from `run_pgd_attack.py` in `results/measured_robustness_input_7_[16].txt`.
Then save gloro's robustness and plot:

```
jq '.[0].robustness' "dir/model_weights_epsilon_0.3_[16]_500/gloro_model_results.json" > results/gloro_robustness_input_7_\[16\].txt
python3 plot_robustness_vs_gram.py results/robustness_vs_gram_input_7_\[16\].csv results/measured_robustness_input_7_\[16\].txt results/gloro_robustness_input_7_\[16\].txt performance_7.pdf
```

## Checking the convolution axioms

`check_conv_axioms.py` checks numerically the three axioms that the CNN
extension adds about convolutions (`ConvGramIdentity`, `ConvCrop` and
`ConvSchur`, in `operator_norms.dfy`), and the bound that `Norm2Toep` computes.
It builds the matrices of 2,592 random layers (Gaussian filters with 1 to 3
input and output channels and kernels of size 1 to 3, on inputs of size 2 to 5,
with strides 1 to 3 and 0 to K zeros of padding), and compares them with their
spectral norms, computed by SVD in double precision. It needs only numpy and
takes about a minute:

```
python3 check_conv_axioms.py
```

Each deviation should be at the level of floating-point rounding. On an M1
MacBook Air (the last digits depend on the platform), it prints:

```
2592 layers (random filters, strides 1-3, paddings 0..k):
  ConvGramIdentity: max |T^T T - SameConv(GramStep)| = 1.42e-14
  ConvCrop:  max sigma(T) - sigma(T_full)  = 7.11e-15  (must be <= 0)
  ConvSchur: max sigma(T) - SchurBound(K)   = 0.00e+00  (must be <= 0)
  Norm2Toep, N = 1..3: max sigma(T) - bound = 4.44e-16  (must be <= 0)
```

Such tests cannot prove the axioms, but they would catch a misstatement, such
as an off-by-one index in the Gram filter or the padding. The fourth axiom,
`SpecNormTriangle`, is the triangle inequality for the spectral norm.
