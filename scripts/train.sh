#! /bin/bash

scripts=$(dirname "$0")
base=$(realpath $scripts/..)

models=$base/models
data=$base/data
tools=$base/tools

mkdir -p $models

num_threads=4
device=""

SECONDS=0

(cd $tools/pytorch-examples/word_language_model &&
    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python main.py --data $data/poetry \
        --epochs 40 \
        --log-interval 100 \
        --emsize 200 --nhid 200 --dropout 0.3 --tied \
        --save $models/model.pt
        # TODO: change the dropout once the validation stuff works
        # choose values between 200 and 300 for embed and hidden,
        # and something else for the dropout
        # including no dropout for once! not sure if we gotta do 0 then
)

echo "time taken:"
echo "$SECONDS seconds"
