#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

data=$base/data

mkdir -p $data

tools=$base/tools

# link default training data for easier access

mkdir -p $data/wikitext-2

for corpus in train valid test; do
    absolute_path=$(realpath $tools/pytorch-examples/word_language_model/data/wikitext-2/$corpus.txt)
    ln -snf $absolute_path $data/wikitext-2/$corpus.txt
done

# download a different interesting data set!

mkdir -p $data/poetry

mkdir -p $data/poetry/raw

# wget https://www.gutenberg.org/files/52521/52521-0.txt
wget https://www.gutenberg.org/files/20158/20158-0.txt # I want poetry! this includes a little codeswitching and a good amount of numbers tho
mv 20158-0.txt $data/poetry/raw/byron.txt

# preprocess slightly

cat $data/poetry/raw/byron.txt | python $base/scripts/preprocess_raw.py > $data/poetry/raw/byron.cleaned.txt

# tokenize, fix vocabulary upper bound

cat $data/poetry/raw/byron.cleaned.txt | python $base/scripts/preprocess.py --vocab-size 50000 --tokenize --lang "en" --sent-tokenize > \
    $data/poetry/raw/byron.preprocessed.txt

# split into train, valid and test

head -n 440 $data/poetry/raw/byron.preprocessed.txt | tail -n 400 > $data/poetry/valid.txt
head -n 840 $data/poetry/raw/byron.preprocessed.txt | tail -n 400 > $data/poetry/test.txt
tail -n 3075 $data/poetry/raw/byron.preprocessed.txt | head -n 2955 > $data/poetry/train.txt
