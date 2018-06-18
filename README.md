# Named Entity Recognition - RDD Corpus
This document explains the details for the reproduction of the results obtained with the RDD corpus in the task of detecting rare diseases and disabilities[2].

## Directory Structure
This repository is divided as follows:

    .
    ├── src 
    │   ├── files
    │   │   ├── data.txt (Text file with the corpus)
    │   ├── pkl_reduc (Intermediate directory with the preprocessed information.)
    │   │   ├── *.pk 
    │   └── BIOF1Validation_5tags.py (Different Python functions for the evaluation)
    │   ├── case_wordNER.py (Python code for creating and training the model)
    │   ├── preprocess.py (Script for corpus preprocessing)
    │   ├── utils.py (tool box library)
    ├── trained-model (or build)
    │   ├── case-embedding-model.h5
    │   ├── prediction-case.txt
    │   ├── test-case.txt
    ├── Readme.md
    ├── requirements.txt

## Corpus RDD (BIO format)
To carry out this experiment we have used the relationships file provided in the RDD corpus. These files include annotations about disabilities and rare diseases in the BIO-Format (Begin-In-Out) that appear in the different sentences. An example can be found below
```
0	Furthermore	O	O
1	such	O	O
2	disruption	O	O
3	in	O	O
4	white	O	O
5	matter	O	O
6	organization	O	O
7	appears	O	O
8	to	O	O
9	be	O	O
10	a	O	O
11	feature	O	O
12	specific	O	O
13	to	O	O
14	Aicardi	BU	O
15	syndrome	IU	O
16	and	O	O
17	not	O	O
18	shared	O	O
19	by	O	O
20	other	O	O
21	neurodevelopmental	BI	O
22	disorders	II	O
23	with	O	O
24	similar	O	O
25	anatomic	O	O
26	manifestations	O	O
27 . O O
```
This file is located at: src/files/data.txt




## Instructions for running the experiment

For this experiment Dependency-Based Word Embeddings has been used [1]
```bash
curl http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2 --output deps.words.bz2
bzip2 -d deps.words.bz2
rm -rf levy_word_emb && mkdir levy_word_emb
mv deps.words levy_word_emb
```
To run the experiment
```bash
python preprocess.py
python case_wordNER.py
```
The model shown here has the following configuration:
```python
experiment_configuration = {
    "lstm_param": 100,
    "dropout": 0.5,
    "decay": 1e-8,
    "lr": 0.03,
    "number_of_epochs": 150,
    "minibatch_size": 128,
    "n_folds": 10,
    "numHiddenUnits": 100
}
```
```
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_1 (InputLayer)            (None, 300)          0                                            
__________________________________________________________________________________________________
input_2 (InputLayer)            (None, 300)          0                                            
__________________________________________________________________________________________________
embedding_1 (Embedding)         (None, 300, 300)     895500      input_1[0][0]                    
__________________________________________________________________________________________________
embedding_2 (Embedding)         (None, 300, 8)       64          input_2[0][0]                    
__________________________________________________________________________________________________
concatenate_1 (Concatenate)     (None, 300, 308)     0           embedding_1[0][0]                
                                                                 embedding_2[0][0]                
__________________________________________________________________________________________________
dense_1 (Dense)                 (None, 300, 100)     30900       concatenate_1[0][0]              
__________________________________________________________________________________________________
bidirectional_1 (Bidirectional) (None, 300, 200)     160800      dense_1[0][0]                    
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 300, 100)     20100       bidirectional_1[0][0]            
__________________________________________________________________________________________________
dropout_1 (Dropout)             (None, 300, 100)     0           dense_2[0][0]                    
__________________________________________________________________________________________________
time_distributed_1 (TimeDistrib (None, 300, 6)       606         dropout_1[0][0]                  
==================================================================================================
Total params: 1,107,970
Trainable params: 212,470
Non-trainable params: 895,500
__________________________________________________________________________________________________
```

### Making predictions
Below is a fragment of code for predicting the label set for a sentence.
```python
from keras.preprocessing import sequence
from keras.models import load_model
import gzip
import cPickle as pkl
from utils import getCasing
import numpy as np

# Instructions for loading the pre-trained model
model = load_model('trained-model/case-embedding-model.h5')

# Loading dictionaries
word2Idx, label2Idx, case2Idx = pkl.load(gzip.open('pkl_reduc/utils.pkl.gz', 'rb'))
idx2Label = {label2Idx[y]:y for y in label2Idx }

sentence = "Furthermore such disruption in white matter organization appears to be a feature specific to Aicardi syndrome and not shared by other neurodevelopmental disorders with similar anatomic manifestations ."


# Translation of terms based on dictionaries - 300 is the maximum sentence length allowed by the experiment
trad_tokens = [word2Idx[x] if x in word2Idx else word2Idx["UNKNOWN_TOKEN"] for x in sentence.split(" ")]
trad_tokens = sequence.pad_sequences([trad_tokens],maxlen=300,padding='post',value=word2Idx["PADDING_TOKEN"])

trad_casing = [getCasing(x,case2Idx) for x in sentence.split(" ")]
trad_casing = sequence.pad_sequences([trad_casing],maxlen=300,padding='post',value=word2Idx["PADDING_TOKEN"])

zip(sentence.split(" "),[idx2Label[x] for x in np.argmax(model.predict([trad_tokens,trad_casing], verbose=0), axis=2)[0]][:len(sentence.split(" "))])
```

```
Out[37]: 
[('Furthermore', 'BI'),
 ('such', 'II'),
 ('disruption', 'II'),
 ('in', 'O'),
 ('white', 'O'),
 ('matter', 'O'),
 ('organization', 'O'),
 ('appears', 'O'),
 ('to', 'O'),
 ('be', 'O'),
 ('a', 'O'),
 ('feature', 'O'),
 ('specific', 'O'),
 ('to', 'O'),
 ('Aicardi', 'BU'),
 ('syndrome', 'IU'),
 ('and', 'O'),
 ('not', 'O'),
 ('shared', 'O'),
 ('by', 'O'),
 ('other', 'O'),
 ('neurodevelopmental', 'BI'),
 ('disorders', 'II'),
 ('with', 'O'),
 ('similar', 'O'),
 ('anatomic', 'O'),
 ('manifestations', 'O'),
 ('.', 'O')]
```

## Results

We have evaluated our model using a 10-fold cross validation. In the following table you can see that we have considered two different forms of evaluation. On the one hand, in the first evaluation the "B-" and "I-" labels are only considered correct if they are in the correct sequence. Moreover, the second evaluation takes into account the concordance of the different labels (excluding O) separately. We provide overall results and separate results for both types of entities.

|                 	|           	| Evaluation 1 	|           	|           	| Evaluation 2 	|           	|
|-----------------	|-----------	|--------------	|-----------	|-----------	|--------------	|-----------	|
|                 	| Precision 	| Recall       	| F-measure 	| Precision 	| Recall       	| F-measure 	|
| LSTM-W+C(RD+DI) 	| 76.75     	| 68.44        	| 72.33     	| 79.64     	| 75.79        	| 77.65     	|
| LSTM-W+C(RD)    	| 63.24     	| 61.90        	| 62.52     	| 69.69     	| 70.03        	| 69.81     	|
| LSTM-W+C(DI)    	| 76.31     	| 74.96        	| 75.58     	| 80.85     	| 81.44        	| 81.11     	|


[1] - Levy, O., & Goldberg, Y. (2014). Neural word embedding as implicit matrix factorization. In Advances in neural information processing systems (pp. 2177-2185).

[2] - Hermenegildo Fabregat Marcos, Lourdes Araujo, Juan Martinez-Romo (2018). Deep neural models for extracting entities and relationships in the new RDD corpus relating disabilities and rare diseases (In revision)
