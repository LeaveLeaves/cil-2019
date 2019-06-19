Road segmentation
=================

Team Member: Jingyuan Ma, Ye Zhi, Yongqi Wang

This repository contain the tools and models for the the course project
of `Computational Intelligence
Lab <http://da.inf.ethz.ch/teaching/2019/CIL/project.php>`__ (Spring
2019): Road Segmentaion.

Prerequisites
-------------

In our setting, the models are being run inside a `Docker
container <https://hub.docker.com/r/ufoym/deepo/>`__ (using the default
tag: ``latest``)

To use the docker container:

.. code:: shell

    docker pull foym/deepo:latest
    # change the volume mount before
    docker run -it -v ./cil-2019:/home/cil-2019 foym/deepo bash
    docker ps -a 
    # CONTAINER ID        IMAGE               COMMAND             CREATED             STATUS              PORTS               NAMES
    # ae2f6abc24d5        ufoym/deepo         "bash"              2 hours ago         Up About an hour    6006/tcp            eager_ganguly
    docker ps -a -q
    # ae2f6abc24d5
    # entering the bash with container ID
    docker exec -it ae2f6abc24d5 bash
    # follow instructions below

-  PyTorch 1.0
-  ``pip3 install torch torchvision``
-  Easydict
-  ``pip3 install easydict``
-  tqdm
-  ``pip3 install tqdm``

Usage
-----

A sample workflow:

.. code:: shell

    git clone https://github.com/wyq977/cil-2019.git
    cd cil-2019
    cd model/final
    # train with CUDA device 0
    python train.py -d 0
    # eval using the default last epoh
    python eval.py -d 0 -p ./val_pred
    # generate predicted groundtruth
    python pred.py -d 0 -p ./pred
    # generate submission.csv
    python ../../cil-road-segmentation-2019/mask_to_submission.py --name submission -p ./pred/

Model dir:

.. code:: shell

    ├── config.py
    ├── dataloader.py
    ├── eval.py
    ├── network.py
    ├── pred.py
    └── train.py

Prepare data
~~~~~~~~~~~~

With a tab-separated files specifying the path of images and
groundtruth, ``train.txt``, ``val.txt``, ``test.txt``.

``train.txt`` or ``val.txt``:

.. code:: shell

    path-of-the-image   path-of-the-groundtruth

Noted that the ``test.txt``:

.. code:: shell

    path-of-the-image   path-of-the-image

A handy script using the package
`glob <https://docs.python.org/3/library/glob.html>`__ can be found
inside the dataset directory

Training
~~~~~~~~

Currently, distributed training from ``torch.distributed.launch`` is not
supported.

To specify which CUDA device used for training, one can parse the index
to ``train.py``

A simple use case using the first CUDA device:

.. code:: shell

    python train.py -d 0

Training can be restored from saved checkpoints

.. code:: shell

    python train.py -d 0 -e log/snapshot/epoch-last.pth

Predictive groudtruth labels
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similar to training

.. code:: shell

    python pred.py -d 0 -p ../../cil-road-segmentation-2019/pred/ -e log/snapshot/epoch-last.pth

Evalaute
~~~~~~~~

.. code:: shell

    python pred.py -d 0 -p ../../cil-road-segmentation-2019/val_pred/ -e log/snapshot/epoch-last.pth

Create submission.csv
~~~~~~~~~~~~~~~~~~~~~

.. code:: shell

    cd ../../cil-road-segmentation-2019/
    python mask_to_submission.py --name submission -p pred/

Structure
---------

.. code:: shell

    ├── README.rst
    ├── cil-road-segmentation-2019 # datasets and submission script
    ├── docs
    ├── utils # helper function and utils for model
    ├── log
    └── model

Under ``model`` directory, one can train, predict groundtruth (test
images) and evaluate a model, details usage see the usage section above.

Different helpers functions used in constructing models, training,
evaluation and IO operations regarding to ``pytorch`` could be found
under ``utils`` folder. Functions or modules adapted from
`TorchSeg <https://github.com/ycszen/TorchSeg/tree/master/model>`__ is
clearly marked and referenced in the files.

Logistics
---------

Links:
~~~~~~

1. `Projects
   description <http://da.inf.ethz.ch/teaching/2019/CIL/project.php>`__
2. `Road
   seg <https://inclass.kaggle.com/c/cil-road-segmentation-2019>`__
3. `Road seg kaggle sign
   in <https://www.kaggle.com/t/c83d1c6de17c433ca64b3a9174205c44>`__
4. `Link for
   dataset.zip <https://storage.googleapis.com/public-wyq/cil-2019/cil-road-segmentation-2019.zip>`__
5. `Course <http://da.inf.ethz.ch/teaching/2019/CIL/project.php>`__
6. `How to write
   paper <http://da.inf.ethz.ch/teaching/2019/CIL/material/howto-paper.pdf>`__

.. code:: bash

    wget https://storage.googleapis.com/public-wyq/cil-2019/cil-road-segmentation-2019.zip

Grading
~~~~~~~

**Project Grading** The project grade is composed of a competitive (30%)
and a non-competitive (70%) part.

**Competitive grade (30%):** The ranks in the Kaggle competition system
will be converted on a linear scale to a grade between 4 and 6.

**Non-competitive grade:** The following criteria are graded based on an
evaluation by the teaching assistants: quality of paper (30%),
creativity of solution (20%), quality of implementation (20%). Each
project is graded by two independent reviewers. The grades of each
reviewer are de-biased such that the aveage grade across all projects
that the reviewer graded is comparable for each reviewer.

report grading guidlines Your paper will be graded by two independent
reviewers according to the following three criteria:

1. Quality of paper (30%)
-------------------------

6.0: Good enough for submission to an international conference. 5.5:
Background, method, and experiment are clear. May have minor issues in
one or two sections. Language is good. Scores and baselines are well
documented. 5.0: Explanation of work is clear, and the reader is able to
identify the novelty of the work. Minor issues in one or two sections.
Minor problems with language. Has all the recommended sections in the
howto-paper. 4.5: Able to identify contribution. Major problems in
presentation of results and or ideas and or reproducibility/baselines.
4.0: Hard to identify contribution, but still there. One or two good
sections should get students a pass. 3.5: Unable to see novelty. No
comparison with any baselines.

2. Creativity of solution (20%)

3. 0: Elegant proposal, either making a useful assumption, studying a
   particular class of data, or using a novel mathematical fact. 5.5: A
   non-obvious combination of ideas presented in the course or published
   in a paper (Depending on the difficulty of that idea). 5.0: A novel
   idea or combination not explicitly presented in the course. 4.5: An
   idea mentioned in a published paper with small extensions / changes,
   but not so trivial to implement. <=4.0: A trivial idea taken from a
   published paper.

4. Quality of implementation (20%)

5. 0: Idea is executed well. The experiments done make sense in order to
   answer the proposed research questions. There are no obvious
   experiments not done that could greatly increase clarity. The
   submitted code and other supplementary material is understandable,
   commented, complete, clean and there is a README file that explains
   it and describes how to reproduce your results.

Subtractions from this grade will be made if: - the submitted code is
unclear, does not run or experiments cannot be reproduced or there is no
description of it - experiments done are useless to gain understanding
or of unclear nature or obviously useful experiments have been left
undone - comparison to baselines are not done

Computational resources
~~~~~~~~~~~~~~~~~~~~~~~

1. https://scicomp.ethz.ch/wiki/Leonhard
2. https://scicomp.ethz.ch/wiki/CUDA\_10\_on\_Leonhard#Available\_frameworks
3. https://scicomp.ethz.ch/wiki/Using\_the\_batch\_system#GPU

Project submission
~~~~~~~~~~~~~~~~~~

1. Submit the final report:
   https://cmt3.research.microsoft.com/ETHZCIL2019
2. Signed form here:
   http://da.inf.ethz.ch/teaching/2019/CIL/material/Declaration-Originality.pdf
3. Kaggle: https://inclass.kaggle.com/c/cil-road-segmentation-2019

