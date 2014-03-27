from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectPercentile, SelectKBest, f_classif

import networkx as nx

import numpy as np
from numpy import zeros
from numpy import transpose
from numpy import array

import sys
import nltk
import re
import collections
import difflib

from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize, sent_tokenize

from operator import itemgetter

from multiprocessing import Pool

import urllib2

from HTMLParser import HTMLParser
import time
import random

pattern = r'''(?x)              # set flag to allow verbose regexps
    https?\://([^ ,;:()`'"])+   # URLs
    | [<>]?[:;=8][\-o\*\']?[\)\]\(\[dDpP/\:\}\{@\|\\]   # emoticons
    | [\)\]\(\[dDpP/\:\}\{@\|\\][\-o\*\']?[:;=8][<>]?   # emoticons, reverse orientation
    | ([A-Z]\.)+                # abbreviations, e.g. "U.S.A."
    | \w+(-\w+)*                # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?          # currency and percentages
    | \.\.\.                    # ellipsis
    | @+[\w_]+                  # user names
    | \#+[\w_]+[\w\'_\-]*[\w_]+ # hashtags
    | [.,;"'?():-_`]            # these are separate tokens
    '''



def load_dataset(filename):
	'''
	Read the input file. 
	For each entity and filler pair, 
	'''
	lines = [line.strip() for line in open(filename,'r')]
	samples = []
	tmp = {}
	for l in lines:
		if (l != ""):
			samples.append(l)
	return samples

def load_states(filename):
    lines = [line.strip() for line in open(filename,'r')]
    states = {}
    for l in lines:
        ls = l.split()
        number = ls[4].replace(',','')
        states[ls[2]] = np.around(int(number)/50000)
    return states

def print_data(f, dataset):
    ra = random.sample(range(len(train)), len(train))
    for v in ra:
        f.write(train[v]+"\n")

samples = []
states = {}
dataset = {}
if len(sys.argv) > 2:
    samples = load_dataset(sys.argv[1])
    states = load_states(sys.argv[2])

count = FreqDist();
for s in samples:
    words = nltk.regexp_tokenize(s, pattern)
    count.inc(words[1])
    if words[0] in dataset:
        dataset[words[1]].append(s)
    else:
        dataset[words[1]] = []
        dataset[words[1]].append(s)

barrier = float("inf")
for key in count.keys():
    rate = count[key]/float(states[key])
    if barrier > rate:
        barrier = rate

print 'barrier: ',barrier

train = []
dev = []
test = []

s = len(count.keys())
rand = random.sample(range(s), s)
print len(rand), rand
for j, key in enumerate(rand):
    key = (count.keys())[key]
    size = int(np.around(states[key] * barrier))
    print key
    print len(dataset[key])
    ra = random.sample(range(len(dataset[key])), size)
    for i,v in enumerate(ra):
        if j < s*2/3:
            train.append(dataset[key][v])
        else:
            test.append(dataset[key][v])

f1 = open('sample1.train','w')
f2 = open('sample1.dev','w')
f3 = open('sample1.test','w')
print "Writing Training Data..."
print_data(f1, train)
print "Writing Development Data..."
print_data(f2, dev)
print "Writing Testing Data..."
print_data(f3, test)
