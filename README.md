# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts, and Moritz Steiner, for preparing his repo for us to adapt (https://github.com/moritz-steiner/mt-exercise-03)

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps if you wish to repeat what we did

It is possible to clone this repository in the desired place:

    git clone https://github.com/iamhummelila/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

# Changes to the original Exercise repo

The following changes were made to the original code:

./scripts/download.sh
The data was adapted, from the original Grimm stories to our desired Lord Byron poetry. This includes how we called our folders.

./scripts/generate.sh
The data folder (name) was adapted as well

./scripts/preprocess_raw.py
One small change: The underscores were replaced with empty strings for the preprocessing. This concerns the sample - the old sample still used the preprocessing without underscores.

samples/old_sample
samples/sample
Here, we added a sample. At first, there was a sample when we had not yet removed the underscores (which were primarily used for formatting reasons). This, we called the old_sample. The new sample contains generated text based on a language model that has never seen any underscores, since we removed all of them in a preprocessing step.

There was not a lot to change to get a functioning language model. A big thank you to both Moritz and Emma for their amazing code, and their making available of the code.