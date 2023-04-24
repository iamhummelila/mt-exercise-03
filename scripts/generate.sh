#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools
samples=$base/samples

mkdir -p $samples

num_threads=4
device=""

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/poetry \
        --words 100 \
        --checkpoint $models/model_0.pt \
        --outf $samples/sample_0
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/poetry \
        --words 100 \
        --checkpoint $models/model_3.pt \
        --outf $samples/sample_3
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/poetry \
        --words 100 \
        --checkpoint $models/model_5.pt \
        --outf $samples/sample_5
)

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python generate.py \
        --data $data/poetry \
        --words 100 \
        --checkpoint $models/model_7.pt \
        --outf $samples/sample_7
)

