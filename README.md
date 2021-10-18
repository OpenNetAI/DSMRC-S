# DSMRC-S
This is the implementation of [Distant Supervision based Machine Reading Comprehension for Extractive
               Summarization in Customer Service, SIGIR 2021].

DSMRC-S is a Extractive summarization method based on Machine Reading Comprehension and Distant Supervision.

The overall architecture of DSMRC-S is:
![overall.png](https://i.loli.net/2021/10/18/sqYgroMp9Fh4IAE.png)

## Requirements
```
python == 3.6
numpy == 1.18.1
pytorch == 1.4.0
```

## Running 
In DSMRC-S, the text summarization task is transformed into a two-stage machine reading comprehension (MRC) task.

The first stage: predicts the probability of token appearing in the answer.

```
python big_data_train_file.py
```

The second stage: extracts the answer based on the probabilities and the exctration strategy.

```
python big_data_predict_file.py
```
The results will be saved in predict_result folder.
