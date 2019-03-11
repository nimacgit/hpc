#!/usr/bin/env python
# coding: utf-8
# import locale
#
# locale.setlocale(locale.LC_ALL, 'Persian')
# get_ipython().run_line_magic('config', 'IPCompleter.greedy=True')

import numpy as np
import pandas as pd
# import math
# from scipy import stats
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import joblib
import time

from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# import os
# os.getcwd()
# df = pd.read_csv("C:\\Users/nimac/RProjects/alpha-sun/src/resources/divar_posts_dataset.csv")
print("start at " + str(time.ctime()))
print("reading csv")
df = pd.read_csv("/home/n.behrang78.student.sharif/nimac/divar_posts_dataset.csv")
print("read done")

dfcat1 = df.loc[df.cat2 != df.cat2]
dfcat1["cat"] = dfcat1["cat1"]
dfcat2 = df.loc[(df.cat3 != df.cat3) & (df.cat2 == df.cat2)]
dfcat2["cat"] = (dfcat2["cat1"] + ":" + dfcat2["cat2"])
dfcat3 = df.loc[(df.cat3 == df.cat3)]
dfcat3["cat"] = (dfcat3["cat1"] + ":" + dfcat3["cat2"] + ":" + dfcat3["cat3"])
data = dfcat1.append(dfcat2.append(dfcat3))
df = data

print("cating done")

descs = df.desc
titles = df.title
ntitles = []
ndescs = []
print(titles.shape)
print(descs.shape)
descs = df.desc
treecat = df.cat

text_clf = Pipeline([
    ('vect', CountVectorizer(max_features=1500, min_df=5, max_df=0.7)),
    # ('vect', CountVectorizer(max_features=1500, min_df=5, max_df=0.7)),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
    alpha=1e-5, random_state=42,
    max_iter=100, tol=None)),
    # ('clf', RandomForestClassifier(n_estimators=1000, max_depth=5, random_state=0, n_jobs=-1)),
    # ('clf', MultinomialNB()),
    # ('clf', DecisionTreeClassifier(max_depth=5)),
    # ('clf', SVC(gamma='scale', decision_function_shape='ovo')),
])
print("build pipeline done")
text_clf.fit(descs, treecat)
print("fit done")

predicted = text_clf.predict(descs)
# joblib.Parallel(n_jobs=4)(text_clf.predict(descs))
print("predict done")
np.mean(predicted == treecat)
print("mean done")



import tempfile
import itertools as IT
import os

def uniquify(path, sep = ''):
    def name_sequence():
        count = IT.count()
        yield ''
        while True:
            yield '{s}{n:d}'.format(s = sep, n = next(count))
    orig = tempfile._name_sequence
    with tempfile._once_lock:
        tempfile._name_sequence = name_sequence()
        path = os.path.normpath(path)
        dirname, basename = os.path.split(path)
        filename, ext = os.path.splitext(basename)
        fd, filename = tempfile.mkstemp(dir = dirname, prefix = filename, suffix = ext)
        tempfile._name_sequence = orig
    return filename


joblib.dump(text_clf, uniquify("trained_clf"), compress=9)
print("save done")
print("finished at " + str(time.ctime()))