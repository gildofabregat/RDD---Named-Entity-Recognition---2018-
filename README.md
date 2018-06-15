# Named Entity Recognition - 2018 - RDD Corpus
Named-entity recognition (NER) (also known as entity identification, entity chunking and entity extraction) is a subtask of information extraction that seeks to locate and classify named entities in text into pre-defined categories such as the names of persons, organizations, locations, expressions of times, quantities, monetary values, percentages, etc. This work explains the details for the reproduction of the results obtained with the RDD corpus in the task of detecting rare diseases and disabilities.

## Corpus RDD (BIO format)
We have made use of the relationship files provided in the RDD corpus to carry out this experiment. These files include annotations about disabilities and rare diseases in the BIO-Format (Begin-In-Out) that appear in the different sentences. An example can be found below
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

## Instructions for run the experiment

For this experiment Dependency-Based Word Embeddings has been used [1]
```bash
curl http://u.cs.biu.ac.il/~yogo/data/syntemb/deps.words.bz2 --output deps.words.bz2
bzip2 -d deps.words.bz2
rm -r levy_word_emb && mkdir levy_word_emb
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

## Instructions for loading the pre-trained model
```python
from keras.models import load_model
model = load_model('trained-model/case_embedding_model.h5')
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
