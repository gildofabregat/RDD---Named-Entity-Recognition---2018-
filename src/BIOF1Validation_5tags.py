"""
Computes the F1 score on BIO tagged data

Some of these functions have been adapted from the code on the repository:
https://github.com/UKPLab/deeplearning4nlp-tutorial

"""


def compute_precision(guessed, correct):
    d = {"U": {"aci": 0, "error": 0}, "I": {"aci": 0, "error": 0}}
    correctCount = 0
    count = 0
    idx = 0

    if (len(guessed) != len(correct)):
        print("ERROR!!!!!!!!!!!!!!: compute_precision: differnt length")
        a = input()
   
    while idx < len(guessed) and idx < len(correct) and correct[idx] != 'X' and guessed[idx] != 'X':
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1
            d[guessed[idx][1]]["error"] += 1
            correctlyFound = guessed[idx] == correct[idx]
            idx += 1
            if (idx < len(correct)):
                if guessed[idx][0] == 'I':
                    while idx < len(guessed) and guessed[idx][0] == 'I':  
                        if guessed[idx] != correct[idx]:
                            correctlyFound = False
                        idx = idx + 1
            if correctlyFound:
                correctCount += 1
                d[guessed[idx-1][1]]["aci"] += 1
        else:
            idx += 1

    precision = 0 if count == 0 else float(correctCount) / count
    precisionI = 0 if d["I"]["error"] == 0 else float(d["I"]["aci"]) / d["I"]["error"]
    precisionU = 0 if d["U"]["error"] == 0 else float(d["U"]["aci"]) / d["U"]["error"]

    return precision, precisionI, precisionU


def compute_precision_tag(guessed, correct):
    d = {"U": {"aci": 0, "error": 0}, "I": {"aci": 0, "error": 0}}
    correctCount = 0
    count = 0
    idx = 0

    if len(guessed) != len(correct):
        print("ERROR!!!!!!!!!!!!!!: compute_precision: distintaslongitudes")
        a = input()

    # guessed and correct change order to compute precision and recall
    while idx < len(guessed) and idx < len(correct) and correct[idx] != 'X' and guessed[idx] != 'X':
        if len(guessed[idx]) == 2:
            count += 1
            d[guessed[idx][1]]["error"] += 1
            # print "etiqueta de 2 caracteres"
            if (correct[idx][0] == guessed[idx][0]) and (correct[idx][1] == guessed[idx][1]):
                correctCount = correctCount + 1
                d[guessed[idx][1]]["aci"] += 1
        idx += 1

    precision = 0 if count == 0 else float(correctCount) / count
    precisionI = 0 if d["I"]["error"] == 0 else float(d["I"]["aci"]) / d["I"]["error"]
    precisionU = 0 if d["U"]["error"] == 0 else float(d["U"]["aci"]) / d["U"]["error"]

    return precision, precisionI, precisionU


def compute_f1(predictions, dataset_y, idx2Label,tag):
    fun = compute_precision if not tag else compute_precision_tag

    # convert indices to labels
    label_y = [[idx2Label[element] for element in dataset_y[i]] for i in range(len(dataset_y))]
    pred_labels = [[idx2Label[element] for element in predictions[i]] for i in range(len(predictions))]

    prec, precI, precU = [], [], []
    rec, recI, recU = [], [], []
    f1, f1I, f1U = [], [], []

    for i in range(len(predictions)):
        # check the labels sequences
        p = fun(pred_labels[i], label_y[i])
        prec.append(p[0])
        precI.append(p[1])
        precU.append(p[2])
        r = fun(label_y[i], pred_labels[i])
        rec.append(r[0])
        recI.append(r[1])
        recU.append(r[2])
        f1.append(2.0 * p[0] * r[0] / (p[0] + r[0]) if (r[0] + p[0]) > 0 else 0)
        f1I.append(2.0 * p[1] * r[1] / (p[1] + r[1]) if (r[1] + p[1]) > 0 else 0)
        f1U.append(2.0 * p[2] * r[2] / (p[2] + r[2]) if (r[2] + p[2]) > 0 else 0)
    precr = sum(prec) / len(predictions)
    precrI = sum(precI) / len(predictions)
    precrU = sum(precU) / len(predictions)

    recr = sum(rec) / len(predictions)
    recrI = sum(recI) / len(predictions)
    recrU = sum(recU) / len(predictions)

    f1r = 0 if (recr+precr) == 0 else 2.0 * precr * recr / (precr + recr)
    f1Ir = 0 if (recrI+precrI) == 0 else 2.0 * precrI * recrI / (precrI + recrI)
    f1Ur = 0 if (recrU+precrU) == 0 else 2.0 * precrU * recrU / (precrU + recrU)

    return ("G",precr, recr, f1r),("I",precrI, recrI, f1Ir),("U",precrU, recrU, f1Ur)
