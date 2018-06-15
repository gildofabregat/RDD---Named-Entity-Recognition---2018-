# Named Entity Recognition - 2018 - RDD Corpus
Use of deep learning for the recognition of named entities in the RDD corpus.

Instructions for loading the model
```python
from keras.models import load_model
model = load_model('case_embedding_model.h5')
```

## Results

|                 	|           	| Evaluation 1 	|           	|           	| Evaluation 2 	|           	|
|-----------------	|-----------	|--------------	|-----------	|-----------	|--------------	|-----------	|
|                 	| Precision 	| Recall       	| F-measure 	| Precision 	| Recall       	| F-measure 	|
| LSTM-W+C(RD+DI) 	| 76.75     	| 68.44        	| 72.33     	| 79.64     	| 75.79        	| 77.65     	|
| LSTM-W+C(RD)    	| 63.24     	| 61.90        	| 62.52     	| 69.69     	| 70.03        	| 69.81     	|
| LSTM-W+C(DI)    	| 76.31     	| 74.96        	| 75.58     	| 80.85     	| 81.44        	| 81.11     	|



