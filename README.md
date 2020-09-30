# How to compare adversarial robustness of classifiers from a global perspective
Venue: **ICLR 2021**

Authors: **Will be added after end of reviewing period.**

Institution: **Will be added after end of reviewing period.**


## What is in this repository?
+ The code to calculate robustness curves for a chosen model and dataset
+ The code to reproduce all experiments (with figures) from the paper, including the following:
  + experiments on inter and intra class distances for different datasets and norms
  + experiments on robustness curves for different models and datasets
  
## Main idea of the paper
<p align="center"><img src="images/readme_gif.gif" width="500"></p>
Adversarial robustness of trained models has attracted considerable attention over recent years.
Adversarial attacks undermine the reliability of and trust in machine learning models, but the construction of more robust models hinges on a rigorous understanding of how one should characterize adversarial robustness as a property of a given model.
Point-wise measures for specific threat models are currently the most popular tool for comparing the robustness of classifiers and are used in most recent publications on adversarial robustness.
In this work, we use recently proposed robustness curves to show that point-wise measures of robustness continuously fail to capture important global properties that are essential to reliably compare the robustness of different classifiers.
We introduce new ways in which robustness curves can be used to systematically uncover these properties and provide concrete recommendations for researchers and practitioners when assessing and comparing the robustness of trained models.
Furthermore, we characterize scale as an inherent property of data sets, and we analyze it for certain common data sets, demonstrating that robustness thresholds must be chosen accordingly.

## How to generate robustness curves
The python script `generate_robustness_curves.py` contains methods to calculate robustness curves. You can either directly execute the script or import the methods from the file. If you directly execute the script, you can define parameters via arguments. Example of usage (estimated runtime: 4 Minutes):

`python generate_robustness_curves.py --dataset=mnist --n_points=10 --model_path='provable_robustness_max_linear_regions/models/mmr+at/2019-02-17 01:54:16 dataset=mnist nn_type=cnn_lenet_small p_norm=inf lmbd=0.5 gamma_rb=0.2 gamma_db=0.2 ae_frac=0.5 epoch=100.mat' --nn_type='cnn' --norms 2 1 inf --save=True --plot=True`

This calculates and plots the robustness curves for a model trained by Croce et al. for 10 datapoints of the MNIST test set for the l_2, l_1 and l_\infty norms.

The datasets are available in the folder `provable_robustness_max_linear_regions/datasets`. You can choose between the following: `mnist`, `fmnist`, `gts` and `cifar10`. The models are available in the folder `provable_robustness_max_linear_regions/models`. You can execute `python generate_robustness_curves.py --help` to get more information about the different arguments of the script.

## Installation

We manage python dependencies with anaconda. You can find information on how to install anaconda at: https://docs.anaconda.com/anaconda/install/. After installing, create the environment with executing `conda env create` in the root directory of the repository. This automatically finds and uses the file `environment.yml`, which creates an environment called `robustness` with
everything needed to run our python files and notebooks. Activate the environment with `conda activate robustness`.

We use tensorflow-gpu 2.1 to calculate adversarial examples. To correctly set up tensorflow for your GPU, follow the instructions from: https://www.tensorflow.org/install/gpu.

We use the julia package [MIPVerify](https://github.com/vtjeng/MIPVerify.jl) with [Gurobi](https://www.gurobi.com/documentation/quickstart.html) to calculate exact minimal adversarial examples in the notebook `experiments/rob_curves_true_vs_approximative.ipynb`. To install julia, follow the instructions from: https://julialang.org/downloads/. To install gurobi, follow the instructions from  https://www.gurobi.com/documentation/quickstart.html (free academic licenses available). You need to install the following julia packages: MIPVerify, Gurobi, JuMP, Images, Printf, MAT, CSV and NPZ. More information on MIPVerify can be found here: https://vtjeng.github.io/MIPVerify.jl/latest/#Installation-1.

## Contact
**Will be added after end of reviewing period.**
## Citation
**Will be added after end of reviewing period.**