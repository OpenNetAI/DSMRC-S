# DSMRC-S
This is the implementation of [Distant Supervision based Machine Reading Comprehension for Extractive
               Summarization in Customer Service, SIGIR 2021].

DSMRC-S is a Extractive summarization method based on Machine Reading Comprehension and Distant Supervision.

The overall architecture of DSMRC-S is:

![image](https://github.com/marmorb/DSMRC-S/blob/main/picture/overall.png)

## Requirements
```
python == 3.6
numpy == 1.18.1
pytorch == 1.4.0
```

##Running 
In DSMRC-S, the text summarization task is transformed into a two-stage machine reading comprehension (MRC) task.

The first stage: predicts the probability of token appearing in the answer.

```
python train_file.py
```

The second stage: extracts the answer based on the probabilities and the exctration strategy.

```
python predict_file.py
```
The results will be saved in predict_result folder.
