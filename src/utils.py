import numpy as np
from functools import reduce

def summary(ap,cvprec,cvrec,cvf1):
    print(ap+"PREC FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvprec), np.std(cvprec)))
    print(ap+"REC FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvrec), np.std(cvrec)))
    print(ap+"F1 FINAL : %.2f%% (+/- %.2f%%)" % (np.mean(cvf1), np.std(cvf1)))


def createMatrices(sentences, word2Idx, label2Idx, case2Idx,max_sentence_length):
    unknownIdx = word2Idx['UNKNOWN_TOKEN']
    paddingIdx = word2Idx['PADDING_TOKEN']    
    
    xMatrix, caseMatrix, yVector = [], [], []
    
    wordCount = 0
    unknownWordCount = 0
    print("number of sentences = ", len(sentences))
    for sentence in sentences:
        wordIndices = []    
        caseIndices = []
        labelIndices = []
        for targetWordIdx in range(len(sentence)):
            if targetWordIdx < max_sentence_length:
                # Get the context of the target word and map these words to the index in the embeddings matrix
                word = sentence[targetWordIdx][0]
                wordCount += 1
                if word in word2Idx:
                    wordIdx = word2Idx[word]
                elif word.lower() in word2Idx:
                    wordIdx = word2Idx[word.lower()]
                else:
                    wordIdx = unknownIdx
                    unknownWordCount += 1
                #Get the label and map to int
                labelIdx = label2Idx[sentence[targetWordIdx][1]]
                labelIndices.append(labelIdx)
                wordIndices.append(wordIdx)
                caseIndices.append(getCasing(word, case2Idx))

        # PADDING
        for targetWordIdx in range(len(sentence),max_sentence_length):
            #labelIdx = len(label2Idx)  # 'PADDING_TOKEN' de los labels
            labelIndices.append(label2Idx['X'])
            wordIndices.append(paddingIdx)
            caseIndices.append(case2Idx['PADDING_TOKEN'])
        xMatrix.append(wordIndices)

        caseMatrix.append(caseIndices)
        yVector.append(labelIndices)

    xMatrix2 = reduce(lambda x, y: x+y, xMatrix)
    xMatrix3 = np.reshape(xMatrix2, (-1, max_sentence_length))
    caseMatrix2 = reduce(lambda x, y: x+y,caseMatrix)
    caseMatrix3 = np.reshape(caseMatrix2, (-1, max_sentence_length))
    yVector2 = reduce(lambda x, y: x+y,yVector)
    yVector3 = np.reshape(yVector2, (-1, max_sentence_length))
    return (np.asarray(xMatrix3), np.asarray(caseMatrix3), np.asarray(yVector3))

def readFile(filepath, tokenPosition, tagPosition):
    sentences = []
    sentence = []
    for line in open(filepath):
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            if len(sentence) > 0:
                sentences.append(sentence)
                sentence = []
            continue
        splits = line.split('\t')
        sentence.append([splits[tokenPosition], splits[tagPosition]])
    
    if len(sentence) > 0:
        sentences.append(sentence)

    print(filepath, len(sentences), "sentences")
    return sentences


def getCasing(word, caseLookup):   
    casing = 'other'    
    numDigits = sum([int(char.isdigit()) for char in word])
    digitFraction = numDigits / float(len(word))
    if word.isdigit(): #Is a digit
        casing = 'numeric'
    elif digitFraction > 0.5:
        casing = 'mainly_numeric'
    elif word.islower(): #All lower case
        casing = 'allLower'
    elif word.isupper(): #All upper case
        casing = 'allUpper'
    elif word[0].isupper(): #is a title, initial char upper, then all lower
        casing = 'initialUpper'
    elif numDigits > 0:
        casing = 'contains_digit'   
    return caseLookup[casing]
