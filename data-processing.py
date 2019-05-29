# cil-2019

## __DO NOT COMMIT OR PUSH TO DEVELOP/MASTER__

## Master branch

`master` branch contains the final version

Do not commit or re-branch until submission

## Develop branch


Make pll request from `feature/wyq` to `develop` branch after testing.

Code should be reviewed before merging into `develop` branch.

Code in develop branch should pass the submission requirements.

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