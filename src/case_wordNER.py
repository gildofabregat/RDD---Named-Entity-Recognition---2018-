"""
Code was written & tested with:
- Python 2.7
- Theano 0.8.1
- Keras 1.1.1

"""
import numpy as np

import os
import gzip
import cPickle as pkl
import sys

import keras
from keras.optimizers import Adagrad
from keras.utils import np_utils
from keras.layers import LSTM, Bidirectional, TimeDistributed, Input
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Dropout
from keras.models import Model

# cross validation
from sklearn.model_selection import StratifiedKFold
import BIOF1Validation_5tags
from utils import summary

def getPrecision(pred_test, yTest, targetLabel):
    # Precision for non-vague
    targetLabelCount, correctTargetLabelCount = 0, 0
    for idx in range(len(pred_test)):
        targetLabelCount += int(pred_test[idx] == targetLabel)
        correctTargetLabelCount += int(pred_test[idx] == targetLabel) * int(pred_test[idx] == yTest[idx])
    return 0 if correctTargetLabelCount == 0 else float(correctTargetLabelCount) / targetLabelCount


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

train_tokens, train_case, train_y = pkl.load(gzip.open('pkl_reduc/data.pkl.gz', 'rb'))

embeddings = pkl.load(gzip.open('pkl_reduc/embeddings.pkl.gz', 'rb'))
label2Idx = embeddings['label2Idx']
wordEmbeddings = embeddings['wordEmbeddings']
caseEmbeddings = embeddings['caseEmbeddings']
# Inverse label mapping
idx2Label = {v: k for k, v in label2Idx.items()}

n_in = train_tokens.shape[1]
n_out = len(label2Idx)
max_sentence_length = 300 if len(sys.argv) > 1 and sys.argv[1] == "full" else 40


#####################################
#
# Create the  Network
#
#####################################

# Create the train and predict_labels function


def create_model():
    # API MODEL
    words_input = Input(shape=(train_tokens.shape[1],))
    words = Embedding(input_dim=wordEmbeddings.shape[0], output_dim=wordEmbeddings.shape[1], input_length=n_in,
                      weights=[wordEmbeddings], trainable=False)(words_input)
    casing_input = Input(shape=(train_case.shape[1],))
    casing = Embedding(input_dim=caseEmbeddings.shape[0], output_dim=caseEmbeddings.shape[1], input_length=n_in,
                       weights=[caseEmbeddings], trainable=True)(casing_input)
    conc = keras.layers.concatenate([words, casing])
    words = Dense(output_dim=experiment_configuration["numHiddenUnits"], activation='tanh')(conc)
    conc = Bidirectional(LSTM(experiment_configuration["lstm_param"], return_sequences=True))(words)
    conc = Dense(output_dim=experiment_configuration["numHiddenUnits"], activation='tanh')(conc)
    conc = Dropout(experiment_configuration["dropout"])(conc)
    conc = TimeDistributed(Dense(output_dim=n_out, activation='softmax'))(conc)
    model_r = Model(inputs=[words_input, casing_input], outputs=conc)
    adg = Adagrad(lr=experiment_configuration["lr"], decay=experiment_configuration["decay"])
    model_r.compile(optimizer=adg, loss='categorical_crossentropy', metrics=['accuracy'])
    return model_r
##################################
#
# Training of the Network
#
##################################


max_prec, max_rec, max_acc, max_f1 = 0, 0, 0, 0

skf = StratifiedKFold(n_splits=experiment_configuration["n_folds"])

cvf1, cvrec, cvprec = [], [], []
cvf1I, cvrecI, cvprecI = [], [], []
cvf1U, cvrecU, cvprecU = [], [], []

cvf1_tag, cvrec_tag, cvprec_tag = [], [], []
cvf1I_tag, cvrecI_tag, cvprecI_tag = [], [], []
cvf1U_tag, cvrecU_tag, cvprecU_tag = [], [], []

fold = 0
fpred = open('predictions-case.txt', 'w')
fpred2 = open("data-to-test-case.txt", "w")
for train_index, test_index in skf.split(train_tokens, train_y[:, 0]):
    print("TRAIN:", train_index, "TEST:", test_index)
    y_train, y_test = train_y[train_index], train_y[test_index]  # etiquetas de salida
    train_y_cat2 = np.reshape(np_utils.to_categorical(y_train, n_out), (-1, max_sentence_length, n_out))
    model = create_model()
    model.fit([train_tokens[train_index], train_case[train_index]], train_y_cat2,
              nb_epoch=experiment_configuration["number_of_epochs"],
              batch_size=experiment_configuration["minibatch_size"], verbose=True, shuffle=True)
    if os.path.exists("./case-embedding-model.h5"):
        os.remove("./case-embedding-model.h5")
    model.save("./case-embedding-model.h5")

    pred_test = np.argmax(model.predict([train_tokens[test_index], train_case[test_index]], verbose=0), axis=2)
    for idxsen in range(len(pred_test)):
        i = 0
        while (i < max_sentence_length) and (idx2Label[pred_test[idxsen][i]] != 'X'):
            fpred.write("%s " % idx2Label[pred_test[idxsen][i]])
            fpred2.write("%s " % idx2Label[y_test[idxsen][i]])
            i += 1
        fpred.write("\n")
        fpred2.write("\n")
    ((_, pre_test, rec_test, f1_test), (_, pre_testI, rec_testI, f1_testI), (_, pre_testU, rec_testU, f1_testU)) = BIOF1Validation_5tags.compute_f1(pred_test, y_test, idx2Label,False)
    print("prec %.2f%%, rec %.2f%%, f1 %.2f%%" % (pre_test, rec_test, f1_test))

    cvprec.append(pre_test * 100)
    cvrec.append(rec_test * 100)
    cvf1.append(f1_test * 100)

    cvprecI.append(pre_testI * 100)
    cvrecI.append(rec_testI * 100)
    cvf1I.append(f1_testI * 100)

    cvprecU.append(pre_testU * 100)
    cvrecU.append(rec_testU * 100)
    cvf1U.append(f1_testU * 100)

    ((_, pre_test, rec_test, f1_test), (_, pre_testI, rec_testI, f1_testI), (_, pre_testU, rec_testU, f1_testU)) = BIOF1Validation_5tags.compute_f1(pred_test, y_test, idx2Label,True)
    print("prec %.2f%%, rec %.2f%%, f1 %.2f%%" % (pre_test, rec_test, f1_test))

    cvprec_tag.append(pre_test * 100)
    cvrec_tag.append(rec_test * 100)
    cvf1_tag.append(f1_test * 100)

    cvprecI_tag.append(pre_testI * 100)
    cvrecI_tag.append(rec_testI * 100)
    cvf1I_tag.append(f1_testI * 100)

    cvprecU_tag.append(pre_testU * 100)
    cvrecU_tag.append(rec_testU * 100)
    cvf1U_tag.append(f1_testU * 100)

    fold = fold + 1
    print("PREC fold %d: %.2f%% " % (fold, pre_test))
    print("REC fold %d: %.2f%% " % (fold, rec_test))
    print("F1 fold %d: %.2f%% " % (fold, f1_test))
fpred.close()
fpred2.close()

print("NORMAL")

summary("", cvprec, cvrec, cvf1)
summary("Rare Disease: ", cvprecU, cvrecU, cvf1U)
summary("Disability: ", cvprecI, cvrecI, cvf1I)

print("TAG")
summary("", cvprec_tag, cvrec_tag, cvf1_tag)
summary("Rare Disease: ", cvprecU_tag, cvrecU_tag, cvf1U_tag)
summary("Disability: ", cvprecI_tag, cvrecI_tag, cvf1I_tag)
