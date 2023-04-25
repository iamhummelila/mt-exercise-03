#! /bin/bash

scripts=$(dirname "$0")
base=$scripts/..

tools=$base/tools
mkdir -p $tools

echo "Make sure this script is executed AFTER you have activated a virtualenv"

pip install numpy torch sacremoses nltk pandas seaborn

# install Moses scripts for preprocessing

git clone https://github.com/bricksdont/moses-scripts $tools/moses-scripts

# install pytorch examples

git clone https://github.com/pytorch/examples $tools/pytorch-examples
