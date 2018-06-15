"""
The file preprocesses the files/train.txt and files/test.txt files.

I requires the dependency based embeddings by Levy et al.. Download them from his website and change 
the embeddingsPath variable in the script to point to the unzipped deps.words file.
"""

import numpy as np
import cPickle as pkl
import theano
import sys
import gzip
from utils import createMatrices, readFile, getCasing


embeddingsPath = 'levy_word_emb/deps.words'
max_sentence_length = 300 if len(sys.argv) > 1 and sys.argv[1] == "full" else 40
folder = 'files/'
files = [folder+'data.txt']

# At which column position is the token and the tag, starting at 1. Position 0 is for the word position in the sentence
tokenPosition = 1
tagPosition = 2

# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #
#      Start of the preprocessing
# ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::: #

outputFilePath = 'pkl_reduc/data.pkl.gz'
embeddingsPklPath = 'pkl_reduc/embeddings.pkl.gz'
trainSentences = readFile(files[0], tokenPosition, tagPosition)

# Mapping of the labels to integers
labelSet = set()
words = {}
for dataset in [trainSentences]:
    for sentence in dataset:
        for token, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True
# PADDING label is added
labelSet.add('X')

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)


# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower': 1,
            'allUpper': 2, 'initialUpper': 3,
            'other': 4, 'mainly_numeric': 5,
            'contains_digit': 6, 'PADDING_TOKEN': 7}
caseEmbeddings = np.identity(len(case2Idx), dtype=theano.config.floatX)
        
# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

for line in gzip.open(embeddingsPath) if embeddingsPath.endswith('.gz') else open(embeddingsPath):
    split = line.strip().split(" ")
    word = split[0]
    if len(word2Idx) == 0:   # Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1)   # Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
      
embeddings = {'wordEmbeddings': np.array(wordEmbeddings), 'word2Idx': word2Idx,
              'caseEmbeddings': caseEmbeddings, 'case2Idx': case2Idx,
              'label2Idx': label2Idx}

f = gzip.open(embeddingsPklPath, 'wb')
pkl.dump(embeddings, f, -1)
f.close()

f = gzip.open(outputFilePath, 'wb')
pkl.dump(createMatrices(trainSentences, word2Idx,  label2Idx, case2Idx, max_sentence_length), f, -1)
f.close()

print("Data stored in pkl folder")
