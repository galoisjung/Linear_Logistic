import numpy as np
import json
import os.path
import re
from os import listdir
from struct import pack, unpack
from nltk.tokenize import sent_tokenize
from konlpy.tag import Komoran
from konlpy.tag import Komoran
import mail_extraction
import naver_extraction
import Dao_email
from math import log
import hashlib
import matplotlib.pyplot as plt


def hashing(input):
    result = int.from_bytes(hashlib.sha256(input.encode()).digest()[:4], 'little')
    return result


class split:
    def split(self, doc):
        return doc.split()


class split_sound:
    def split(self, doc):
        doc = re.sub("\s", "", doc)
        result = []
        result.extend(doc)
        return result


class bigram:
    def split(self, doc):
        result = []
        tokens = doc.split()
        for i in range(len(tokens) - 1):
            result.append(' '.join(tokens[i:i + 2]))  # [i,i+2) 형태라는 것에 주의!
        return result


class bigram_sound:
    def split(self, doc):
        result = []
        tokens = split_sound().split(doc)
        for i in range(len(tokens) - 1):
            result.append(' '.join(tokens[i:i + 2]))
        return result


class morphs:
    def __init__(self):
        self.ma = Komoran()
        if os.path.isfile("morphs.json"):
            with open("morphs.json", "r") as fp:
                self.storedic = json.load(fp)
        else:
            self.storedic = dict()

    def split(self, doc):
        doc = re.sub("(?:\s)+", " ", doc)
        emoji_pattern = re.compile("["
                                   u"\U0001F600-\U0001F64F"  # emoticons
                                   u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                   u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                   u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                   u"\U0001F914"
                                   u"\U0001F9B8"
                                   "]+", flags=re.UNICODE)
        doc = emoji_pattern.sub("", doc)
        try:
            if str(hashing(doc)) in self.storedic:
                result = self.storedic[str(hashing(doc))]
            else:
                with open("morphs.json", "w") as f:
                    result = self.ma.morphs(doc)
                    self.storedic[hashing(doc)] = result
                    json.dump(self.storedic, f)
        except UnicodeDecodeError:
            print("error")
            result = []
        return result


class noun:
    def __init__(self):
        self.ma = Komoran()
        if os.path.isfile("noun.json"):
            with open("noun.json", "r") as fp:
                self.storedic = json.load(fp)
        else:
            self.storedic = dict()

    def split(self, doc):
        hash_doc = hashing(doc)
        result = []
        for sentence in sent_tokenize(doc):
            sentence = sentence.strip()
            print(sentence)
            if len(sentence) > 1:
                try:
                    if hash_doc in self.storedic:
                        result = self.storedic[hash_doc]
                    else:
                        with open("noun.json", "w") as f:
                            result = self.ma.nouns(sentence)
                            self.storedic[hash_doc] = result
                            json.dump(self.storedic, f)
                except UnicodeDecodeError:
                    print("error")
                    result = []
        return result


def separate_target(doc, method, v):
    data = []
    target = []
    for i in doc:
        data.append(i[0])
        if i[1] == "True":
            target.append(1)
        else:
            target.append(0)
    tdm = making_dtm(data, method, v)
    t = np.array(target)
    return tdm, t


def making_BOV(data, method):
    v = list()
    for i in data:
        v.extend(method.split(i[0]))
    return list(set(v))


def making_dtm(doc, method, v):
    result = []
    for d in doc:
        row = [1] + ([0] * len(v))
        dov = method.split(d)
        for i, j in enumerate(v):
            if j in dov:
                row[i + 1] = 1
        result.append(row)
    return np.array(result)


def linear_regression(binX, Y, cnt, h):
    linearFn = lambda X, W: X.dot(W)
    linearDerivFn = lambda X, Y, W: 2 * X.T.dot(Y - X.dot(W))

    lossFn = lambda X, Y, W: np.linalg.norm(Y - linearFn(X, W))
    theta = np.random.rand(binX.shape[-1])
    i = cnt

    history = list()

    for j in range(i):
        print(j)
        v = linearDerivFn(binX, Y, theta)
        u = v / np.linalg.norm(v)
        theta = theta + h * u
        if j % 10 == 0:
            history.append(lossFn(binX, Y, theta))

    return theta, history


def logistic_regression(binX, Y, cnt, h):
    logisticFn = lambda X, W: 1 / (1 + np.exp(-X.dot(W)))
    logisticDerivFn = lambda X, Y, W: X.T.dot(Y - logisticFn(X, W))

    theta = np.random.rand(binX.shape[-1])
    i = cnt
    history = list()

    logisticLossFn = lambda X, Y, W: Y[Y == 1].dot(np.log(1e-10 + logisticFn(X[Y == 1], W))) \
                                     + (1 - Y[Y == 0]).dot(np.log(1e-10 + 1 - logisticFn(X[Y == 0], W)))

    for j in range(i):
        print(j)
        v = logisticDerivFn(binX, Y, theta)
        u = v / np.linalg.norm(v)
        theta = theta + h * u
        if j % 10 == 0:
            history.append((-logisticLossFn(binX, Y, theta)))

    return theta, history


def scoreing(Test_X, theta, actual, nb=0):
    true_count = 0
    false_positive = 0
    false_negative = 0
    count = 0

    if nb == 0:
        logisticFn = lambda X, W: 1 / (1 + np.exp(-X.dot(W)))
        predict = logisticFn(Test_X, theta) > 0.5
    else:
        linearFn = lambda X, W: X.dot(W)
        predict = linearFn(Test_X, theta) > 0.5

    for i in range(len(predict)):
        if predict[i] == actual[i]:
            true_count += 1
        elif predict[i] == True and actual[i] == False:
            false_positive += 1
        elif predict[i] == False and actual[i] == True:
            false_negative += 1

        count += 1
    precision = true_count / count
    accuracy = true_count / (true_count + false_positive)
    recall = true_count / (true_count + false_negative)
    f1 = 2 / ((1 / precision) + (1 / recall))

    print("precision:" + str(precision))
    print("accuracy:" + str(accuracy))
    print("Recall:" + str(recall))
    print("F1-score" + str(f1))

    return (precision, accuracy, recall, f1)
