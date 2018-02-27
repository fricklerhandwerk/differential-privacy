# Experiments with Differentially Private Algorithms

This is the accompanying source code to my Bachelor's thesis at TUHH "Computational Verification of Differentially Private Algorithms" (2018).

## Installation

The package and its dependencies can be installed in a virtual environment like this:

```shell
virtualenv -p python3 env
source env/bin/activate
pip3 install -e .
```

## Usage

The GUI programs can be called with either of

```shell
diffpriv-single
diffpriv-reportnoisymax
diffpriv-svt
```

`algorithms.py` implements basic differentially private algorithms in terms of random distributions.

`naive.py` contains the same algorithms in their original forms, which produce single random values. They can be used for Monte-Carlo trials.

`accuracy.py` provides different accuracy estimates for the Sparse Vector Technique in a library. If executed as a script, it plots example graphs.

`plot_*` produce some example plots to illustrate the principles behind the algorithms or statistics about experimental data.

`gui_*` contain the source code for the interactive experiments.

`data_*` retrieve and pre-process test data to be usable in the computation-intensive `experiments.py`. The unpacked data requires about 590 MB. Use in this order: `data_get.sh, data_histogram*, data_flatten`. `data_count*` is there to verify the item counts.

`experiments.py` must be used interactively with

```shell
python3 -i experiments.py
```

where one can choose which part to run. Preparing the data and computing probabilities may take multiple days on mid-range hardware. The intermediate files take up around 5.5 GB.
