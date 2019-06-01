# cil-2019

## __DO NOT COMMIT OR PUSH TO DEVELOP/MASTER__

## Master branch

`master` branch contains the final version

Do not commit or re-branch until submission

## Develop branch


Make pll request from `feature/wyq` to `develop` branch after testing.

Code should be reviewed before merging into `develop` branch.

Code in develop branch should pass the submission requirements.

## Submit paper

1. Submit the [final report](https://cmt3.research.microsoft.com/ETHZCIL2019) here.
2. Signed form here: http://da.inf.ethz.ch/teaching/2019/CIL/material/Declaration-Originality.pdf
3. Kaggle sign in: https://www.kaggle.com/t/3e9d21e9198d44b6994d7ffe5b838f96

### Project Grading

The project grade is composed of a competitive (30%) and a non-competitive (70%) part.

Competitive grade (30%): The ranks in the Kaggle competition system will be converted on a linear scale to a grade between 4 and 6.

Non-competitive grade: The following criteria are graded based on an evaluation by the teaching assistants: quality of paper (30%), creativity of solution (20%), quality of implementation (20%). Each project is graded by two independent reviewers. The grades of each reviewer are de-biased such that the aveage grade across all projects that the reviewer graded is comparable for each reviewer.

## Data

1. `train_pos.txt` and `train_neg.txt`:

    a small set of training tweets for each of the two classes. (Dataset available in the zip file, see link below)

2. `train_pos_full.txt` and `train_neg_full.txt`:
   
   a complete set of training tweets for each of the two classes, about 1M tweets per class. (Dataset available in the zip file, see link below)

3. `test_data.txt`:

   the test set, that is the tweets for which you have to predict the sentiment label.

4. `sampleSubmission.csv`:

   a sample submission file in the correct format, note that each test tweet is numbered. (submission of predictions: -1 = negative prediction, 1 = positive prediction)

Note that all tweets have been tokenized already, so that the words and punctuation are properly separated by a whitespace.

### How to download data
```bash
wget https://storage.googleapis.com/public-wyq/cil-2019/data.zip
```

More details info. on the dataset can be found in the `README.md` under `./data`
