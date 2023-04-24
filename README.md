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

## TODO: On aspects of dropout and perplexities

If you change your pytorch main.py file as we did in our copy, (note: you will have to use your local file by clicking through the path # TODO write path), then you can create two files, pplvalidation.txt and ppltraining.txt in which you can save the perplexities in relation to epoch and dropout that you use, for both the training and the validation step separately. You may also use plotting_ppls.py (where you might need to modify the path or put the files into the correct folder) to create a pandas series out of those two tables, which you can later use to analyse or draw lineplots.

It is currently not possible to save a perplexities model if you interrupt via keyboard during training. This may be improved in the future.

Additionally to the previous repo, we are also using pandas. This will be installed with the install_packages.sh script.

Alternatively, to use the module 'pandas', type the following into your command line:

`pip install pandas`

The current train.sh script contains multiple models that only use varying dropouts. You may remove or add any number you wish. Additionally, the generate.sh file contains multiple models as well, for which it will produce a separate sample. Those samples can be considered if the wish to compare samples is present.

### IMPORTANT!
When forking this repo, you will have to put up additional effort for the dropout and the perplexity steps. You will have to copy paste our code in main.py into tools/pytorch-examples/word_language_model/main.py 

Else, you will not get the dropout files.

## Bugs 

Unfortunately, the training perplexity table is currently not operating correctly. 