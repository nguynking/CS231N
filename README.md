<h1 align="center">CS231n: Deep Learning for Computer Vision</h1>
<p align="center"><i>Stanford - Spring 2023</i></p>

## About

### Overview

These are my solutions for the **CS231N** course assignments offered by _Stanford University_ (Spring 2023). Written questions are explained in detail, the code is brief and commented.

### Main sources (official)
* [**Course page**](http://cs231n.stanford.edu/)
* [**Course Notes and Assignments**](https://cs231n.github.io/)
* [**Lecture videos (2016)**](https://youtube.com/playlist?list=PLkt2uSq6rBVctENoVBg1TpCC7OQi31AlC&si=zzTJkoGBTSrT_L1U) or [**2017**](https://youtube.com/playlist?list=PLC1qU-LWwrF64f4QKQT-Vg5Wr4qEE1Zxk&si=CkZbxLfFRDfvPJUA) version.

## Requirements
For **conda** users, the instructions on how to set-up the environment are given in the handouts. For `pip` users, I've gathered all the requirements in one [file](requirements.txt). Please set up the virtual environment and install the dependencies (for _linux_ users):

```shell
$ python -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

You can install everything with **conda** too (see [this](https://stackoverflow.com/questions/51042589/conda-version-pip-install-r-requirements-txt-target-lib)). For code that requires **Azure** _Virtual Machines_, I was able to run everything successfully on **Google Colab** with a free account.

> Note: Python 3.8 or newer should be used

## Solutions

### Structure

For every assignment, i.e., for directories `assigment1` through `assignment5`, there is coding and written parts. The `solutions.pdf` files are generated from latex directories where the provided templates were filled while completing the questions in `handout.pdf` files and the code.

### Assignments

* [**A1**](assignment1): Image Classification, kNN, SVM, Softmax, Fully Connected Neural Network
* [**A2**](assignment2): Fully Connected and Convolutional Nets, Batch Normalization, Dropout, Pytorch & Network Visualization
* [**A3**](assignment3): Network Visualization, Image Captioning with RNNs and Transformers, Generative Adversarial Networks, Self-Supervised Contrastive Learning
