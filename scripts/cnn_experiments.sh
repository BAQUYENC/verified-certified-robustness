#!/usr/bin/env bash
# Runs the CNN experiments with doit_verified_robust_gloro.sh and the other
# scripts in this directory. Each model's logs, CSVs, plots and numbers go in
# results/cnn/MODEL/. The doit script's run directories (weights, network file,
# model outputs, certifier results) stay in this directory, as before.
#
# Usage: ./cnn_experiments.sh STEP MODEL [N...]
#   train MODEL         train MODEL and certify it with its default Gram iterations
#   certify MODEL N...  certify MODEL again with N Gram iterations, for each N
#                       (an N that finished is skipped, so a sweep can be resumed)
#   pgd MODEL           measured robustness and robust accuracy by PGD attacks
#                       (needs mpmath and ART)
#   gloro MODEL         gloro's unverified statistics and Lipschitz bound time
#   plot MODEL          robustness and VRA against Gram iterations (needs pandas)
#   summary MODEL [N]   the numbers for the research question B table, for N
#                       Gram iterations (default: the MODEL's own)
#   setup-cav MODEL     copy a CAV 2025 model out of ../cav2025-models.zip and
#                       export it again with today's make_certifier_format.py
# MODEL: smoke, 2c2f, 4c3f, 6c2f, gloro2c2f, cav-mnist or cav-cifar10.
# Set CERTIFIER to use a certifier other than ../main.

set -eo pipefail
cd "$(dirname "$0")"

usage() {
    sed -n '/^# Usage/,/^# Set CERTIFIER/s/^# \{0,1\}//p' "$0"
    exit 1
}

[ $# -ge 2 ] || usage
STEP=$1
MODEL=$2
shift 2

CERTIFIER=${CERTIFIER:-../main}
# keep the Mac awake during long runs
if command -v caffeinate > /dev/null 2>&1; then CAFF="caffeinate -i"; else CAFF=""; fi

# TRAIN_EPS, EVAL_EPS, EPOCHS and BATCH follow Leino et al.'s ReLU GloRo nets
# (2021, Table B.2), as CAV 2025 did: a model is trained with a larger epsilon
# than it is evaluated with. Keep them once a model is trained, because its run
# directory's file names hold them. PGD_ITERS is the attacks' MAX_ITER (500 in
# CAV 2025).
CAV_NAME=""
case "$MODEL" in
    smoke) # a few minutes: checks the whole pipeline before the long runs
        DATASET=mnist TRAIN_EPS=0.3 EVAL_EPS=0.3 EPOCHS=2 BATCH=256 INPUT=14 GRAM=2 PGD_ITERS=10
        ARCH="c(4,2,1).c(2,2,1).d(16).d(10)" ;;
    2c2f)
        DATASET=mnist TRAIN_EPS=0.45 EVAL_EPS=0.3 EPOCHS=500 BATCH=256 INPUT=28 GRAM=6 PGD_ITERS=500
        ARCH="c(16,2,1).c(2,2,1).d(100).d(10)" ;;
    4c3f)
        DATASET=mnist TRAIN_EPS=1.75 EVAL_EPS=1.58 EPOCHS=300 BATCH=256 INPUT=28 GRAM=6 PGD_ITERS=500
        ARCH="c(32,2,1).c(32,2,1).c(64,2,1).c(2,2,1).d(512).d(512).d(10)" ;;
    6c2f)
        DATASET=cifar10 TRAIN_EPS=0.1551 EVAL_EPS=0.141 EPOCHS=800 BATCH=256 INPUT=32 GRAM=6 PGD_ITERS=500
        ARCH="c(32,2,1).c(32,2,1).c(32,2,1).c(64,2,1).c(64,2,1).c(2,2,1).d(512).d(10)" ;;
    gloro2c2f) # optional: Leino et al.'s own 2C2F (stride 2, 1 zero of padding)
        DATASET=mnist TRAIN_EPS=0.45 EVAL_EPS=0.3 EPOCHS=500 BATCH=256 INPUT=28 GRAM=6 PGD_ITERS=500
        ARCH="c(16,4,2,1).c(32,4,2,1).d(100).d(10)" ;;
    cav-mnist) # the CAV 2025 dense models, as in the top-level README.md
        DATASET=mnist TRAIN_EPS=0.45 EVAL_EPS=0.3 EPOCHS=500 BATCH=32 INPUT=28 GRAM=11 PGD_ITERS=500
        ARCH="[128,128,128,128,128,128,128,128]" CAV_NAME="2025-01-25_09:27:46-mnist" ;;
    cav-cifar10)
        DATASET=cifar10 TRAIN_EPS=0.1551 EVAL_EPS=0.141 EPOCHS=800 BATCH=256 INPUT=32 GRAM=12 PGD_ITERS=500
        ARCH="[512,256,128,128,128,128,128,128]" CAV_NAME="2025-01-28_20:39:32-cifar10" ;;
    *)
        usage ;;
esac

R="results/cnn/$MODEL"
mkdir -p "$R"
RUN_DIR=""
if [ -f "$R/run_dir.txt" ]; then RUN_DIR=$(cat "$R/run_dir.txt"); fi

weights_dir() { echo "$RUN_DIR/model_weights_epsilon_${TRAIN_EPS}_${ARCH}_${EPOCHS}"; }
results_json() { echo "$RUN_DIR/results_epsilon_${TRAIN_EPS}_${ARCH}_${EPOCHS}_eval_${EVAL_EPS}_gram_$1.json"; }

need_run_dir() {
    if [ -z "$RUN_DIR" ]; then
        echo "No run directory for $MODEL yet: run '$0 train $MODEL' (or setup-cav) first."
        exit 1
    fi
}

# gloro_stat accuracy|robustness|vra: gloro's (unverified) statistic from training
gloro_stat() {
    if [ -f "$(weights_dir)/gloro_model_results.json" ]; then
        jq ".[0].$1" "$(weights_dir)/gloro_model_results.json"
    else # the CAV 2025 models
        jq ".$1" "$(weights_dir)/gloro_model_stats.json"
    fi
}

# doit FIRST_ARGUMENT GRAM_ITERATIONS LOG: the doit script, answering "yes" to its question
doit() {
    echo yes | $CAFF ./doit_verified_robust_gloro.sh "$1" "$DATASET" "$TRAIN_EPS" "$ARCH" "$EVAL_EPS" \
        "$CERTIFIER" "$2" "$EPOCHS" "$BATCH" "$INPUT" 2>&1 | tee "$3"
}

# save_measured WHAT FILE: the percentage on pgd.log's "WHAT on PGD adversarial
# samples" line, saved in FILE as the proportion that plot_robustness_vs_gram.py reads
save_measured() {
    pct=$(sed -n "s/^$1 on PGD adversarial samples: \([0-9.]*\)%\$/\1/p" "$R/pgd.log" | tail -1)
    if [ -z "$pct" ]; then echo "No \"$1 on PGD adversarial samples\" line in $R/pgd.log"; exit 1; fi
    awk -v p="$pct" 'BEGIN { printf "%.4f\n", p / 100 }' > "$2"
    echo "Saved $pct% as $(cat "$2") in $2"
}

case "$STEP" in
    train)
        if [ -n "$CAV_NAME" ]; then echo "$MODEL is trained already: use setup-cav."; exit 1; fi
        if [ -n "$RUN_DIR" ]; then
            echo "$MODEL is trained already, in $RUN_DIR (delete $R/run_dir.txt to train it again)."
            exit 1
        fi
        log="$R/gram-$GRAM.log"
        status=0
        doit train_and_eval "$GRAM" "$log" || status=$?
        RUN_DIR=$(sed -n 's/^Artifacts and results will live in: \(.*\)\/$/\1/p' "$log" | head -1)
        if [ -n "$RUN_DIR" ] && [ -d "$(weights_dir)" ]; then echo "$RUN_DIR" > "$R/run_dir.txt"; fi
        if [ "$status" -ne 0 ]; then echo "The doit script failed: see $log"; exit "$status"; fi
        ;;
    certify)
        need_run_dir
        [ $# -ge 1 ] || usage
        for n in "$@"; do
            json=$(results_json "$n")
            log="$R/gram-$n.log"
            if [ -f "$json" ]; then
                if jq empty "$json" > /dev/null 2>&1 && grep -q '^All done\.' "$log" 2>/dev/null; then
                    echo "$MODEL with $n Gram iterations is done already: $log"
                    continue
                fi
                echo "Moving the results of an unfinished run aside: $json.unfinished"
                mv "$json" "$json.unfinished"
            fi
            doit "$RUN_DIR" "$n" "$log"
        done
        ;;
    pgd)
        need_run_dir
        if [ -e "$R/pgd-disagree" ]; then echo "Delete $R/pgd-disagree (from an earlier run) first."; exit 1; fi
        # with the certifier's and gloro's answers, it also counts attacks that beat a certified answer
        if [ -f "$(weights_dir)/gloro_model_results.json" ]; then
            set -- "$(results_json "$GRAM")" "$(weights_dir)/gloro_model_results.json" "$R/pgd-disagree"
        else
            set --
        fi
        $CAFF python3 run_pgd_attack.py "$DATASET" "$ARCH" "$(weights_dir)" "$EVAL_EPS" "$PGD_ITERS" "$INPUT" "$@" \
            2>&1 | tee "$R/pgd.log"
        # the upper bounds on the proportion certified robust and on the VRA
        save_measured "Robustness" "$R/measured-robustness.txt"
        save_measured "Robust accuracy" "$R/measured-robust-accuracy.txt"
        ;;
    gloro)
        need_run_dir
        python3 get_gloro_robustness_results.py "$DATASET" "$ARCH" "$(weights_dir)" "$EVAL_EPS" "$INPUT" 100 \
            > "$R/gloro-rerun.json"
        jq -r '.[].debug_msg // empty' "$R/gloro-rerun.json"
        ;;
    plot)
        need_run_dir
        if [ ! -f "$R/measured-robustness.txt" ] || [ ! -f "$R/measured-robust-accuracy.txt" ]; then
            echo "Run '$0 pgd $MODEL' first: the plots show the measured robustness and robust accuracy."
            exit 1
        fi
        python3 gram_sweep_csv.py robust "$R"/gram-*.log > "$R/sweep-robust.csv"
        python3 gram_sweep_csv.py vra "$R"/gram-*.log > "$R/sweep-vra.csv"
        gloro_stat robustness > "$R/gloro-robustness.txt"
        gloro_stat vra > "$R/gloro-vra.txt"
        python3 plot_robustness_vs_gram.py "$R/sweep-robust.csv" "$R/measured-robustness.txt" "$R/gloro-robustness.txt" "$R/robustness.pdf"
        python3 plot_robustness_vs_gram.py "$R/sweep-vra.csv" "$R/measured-robust-accuracy.txt" "$R/gloro-vra.txt" "$R/vra.pdf" vra
        echo "Saved $R/robustness.pdf and $R/vra.pdf"
        ;;
    summary)
        need_run_dir
        n=${1:-$GRAM}
        log="$R/gram-$n.log"
        {
            echo "$MODEL: $ARCH on $DATASET, epsilon $EVAL_EPS"
            echo "Verified certifier, $n Gram iterations ($log):"
            grep -E '^Proportion (robust|correct|robust and correct):' "$log" | sed 's/^/  /' || true
            python3 gram_sweep_csv.py robust "$log" | awk -F, 'NR == 2 { print "  Seconds to compute the bounds: " $3 }'
            echo "Unverified gloro certifier (statistics from training):"
            for s in accuracy robustness vra; do echo "  $s: $(gloro_stat $s)"; done
            if [ -f "$R/gloro-rerun.json" ]; then
                jq -r '.[].debug_msg // empty' "$R/gloro-rerun.json" | grep 'computation time' | sed 's/^/  /' || true
            fi
            if [ -f "$R/measured-robustness.txt" ]; then echo "Measured robustness (PGD): $(cat "$R/measured-robustness.txt")"; fi
            if [ -f "$R/measured-robust-accuracy.txt" ]; then
                echo "Measured robust accuracy (PGD): $(cat "$R/measured-robust-accuracy.txt")"
            fi
        } | tee "$R/summary-gram-$n.txt"
        ;;
    setup-cav)
        if [ -z "$CAV_NAME" ]; then echo "setup-cav is for cav-mnist and cav-cifar10."; exit 1; fi
        RUN_DIR="cav2025-rerun/$CAV_NAME"
        if [ -e "$RUN_DIR" ]; then echo "$RUN_DIR exists already."; exit 1; fi
        rm -rf cav2025-rerun.tmp
        unzip -q ../cav2025-models.zip "cav2025-models/$CAV_NAME/*" -d cav2025-rerun.tmp
        mkdir -p cav2025-rerun
        mv "cav2025-rerun.tmp/cav2025-models/$CAV_NAME" "$RUN_DIR"
        rm -rf cav2025-rerun.tmp
        # this copy is for new results; cav2025-models.zip keeps the CAV 2025 ones
        rm -f "$RUN_DIR"/results_*
        # CAV 2025 exported weights to 5 decimal places; today's exporter prints them exactly
        python3 make_certifier_format.py "$ARCH" "$(weights_dir)" "$DATASET" "$INPUT" \
            > "$RUN_DIR/neural_net_mnist_epsilon_${TRAIN_EPS}_${ARCH}_${EPOCHS}.txt"
        echo "$RUN_DIR" > "$R/run_dir.txt"
        echo "Ready: $RUN_DIR"
        echo "(The doit script will say gloro_model_results.json does not exist: CAV 2025 models keep"
        echo "gloro's statistics in gloro_model_stats.json, which summary reads.)"
        ;;
    *)
        usage ;;
esac