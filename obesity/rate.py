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
    | [\xF0-\xF7][\x80-\xBF][\x80-\xBF][\x80-\xBF]|[\xC2-\xE3][\x80-\xBF][\x80-\xBF]|[\xC2-\xE3][\xA9-\xAE]
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

def load_rates(filename):
    lines = [line.strip() for line in open(filename,'r')]
    rates = {}
    for l in lines:
        ls = l.split()
        rates[ls[0]] = (ls[1],ls[2])
    return rates

def print_data(f, dataset):
    ra = random.sample(range(len(dataset)), len(dataset))
    for v in ra:
        f.write(dataset[v]+"\n")

def print_data(f, dataset):
    ra = random.sample(range(len(dataset)), len(dataset))
    for v in ra:
        f.write(dataset[v]+"\n")

def load_states(filename):
    lines = [line.strip() for line in open(filename,'r')]
    states = {}
    for l in lines:
        ls = l.split()
        number = ls[4].replace(',','')
        states[ls[2]] = np.around(int(number)/50000)
    return states

def load_label(filename):
    lines = [line.strip('\r') for line in open(filename,'r')]
    rates = {}
    for l in lines:
        ls = l.split()
        if float(ls[1]) >= 30:
            rates[ls[0]] = 0
        elif float(ls[1]) < 25:
            rates[ls[0]] = 2
        else:
            rates[ls[0]] = 1
    return rates

def state_rate(data, pattern, states):
    fat = FreqDist()
    fit = FreqDist()
    ok = FreqDist()
    states = FreqDist()
    for d in data:
        words = nltk.regexp_tokenize(d, pattern)
        if words[0] == 'fit':
            fit.inc(words[1])
        elif words[0] == 'fat':
            fat.inc(words[1])
        else:
            ok.inc(words[1])
        states.inc(words[1])
    f = 0.0
    t = 0.0
    for s in states.keys():
        r = fat[s]/float(states[s])
        f += fat[s]
        t += states[s]
        print s ,fat[s], fit[s], ok[s], r
    print "total", f/float(t)
    print len(states.keys())

samples = []
rates = {}
dataset = {}
data = []
ss = {}

if len(sys.argv) > 2:
    samples = load_dataset(sys.argv[1])
    rates = load_rates(sys.argv[2])
    ss = load_states(sys.argv[3])
    sl = load_label(sys.argv[4])

count = FreqDist();
for s in samples:
    words = nltk.regexp_tokenize(s, pattern)
    rate = 0.0
    c = 0
    for w in words:
        if w in rates:
            rate += float(rates[w][0])
            c += 1
    if c != 0:
        r = rate/float(c)
        if np.isnan(r):
            string = "ok\t"+s
        elif r >= 14:
            string = "fit\t"+s
        elif r > 10:
            string = "ok\t"+s
        elif r <= 10:
            string = "fat\t"+s
        else:
            print r
        count.inc(words[0])
        if words[0] in dataset:
            dataset[words[0]].append(string)
        else:
            dataset[words[0]] = []
            dataset[words[0]].append(string)

        data.append(string)

#f = open('rate_sample','w')
#print "Writing Training Data..."
#print_data(f, data)


barrier = float("inf")
for key in count.keys():
    rate = count[key]/float(ss[key])
    if barrier > rate:
        barrier = rate

print 'barrier: ',barrier

train = []
dev = []
test = {}

s = len(count.keys())
while True:
    print "HERE"
    train = []
    dev = []
    test = {}
    rand = random.sample(range(s), s)
    has0 = False
    has1 = False
    has2 = False
    has10 = False
    has11 = False
    has12 = False
    for j, key in enumerate(rand):
        key = (count.keys())[key]
        size = int(np.around(ss[key] * barrier))
        ra = random.sample(range(len(dataset[key])), size)
        k = 0
        if j < s*.7:
            print "TRAIN..."
            ra = random.sample(range(len(dataset[key])), size)
            te = []
            for i,v in enumerate(ra):
                train.append(dataset[key][v])
                if sl[key] == 0:
                    has0 = True
                elif sl[key] == 1:
                    has1 = True
                elif sl[key] == 2:
                    has2 = True
        else:
            print "TEST..."
            while k < 10:
                ra = random.sample(range(len(dataset[key])), size)
                te = []
                for i,v in enumerate(ra):
                    te.append(dataset[key][v])
                    if sl[key] == 0:
                        has10 = True
                    elif sl[key] == 1:
                        has11 = True
                    elif sl[key] == 2:
                        has12 = True
                if k in test:
                    test[k].extend(te)
                else:
                    test[k] = []
                    test[k].extend(te)                    
                k += 1
                print "Loop"+str(k+2)+"..."
    if has0 and has1 and has2 and has10 and has11 and has12:
        print "here"
        break
    else:
        train = []
        test = []
        dev = {}
        print "Looping..."

f1 = open('sample.train','w')
f2 = open('sample.dev','w')
print "Writing Training Data..."
state_rate(train, pattern, ss)
print_data(f1, train)
#print "Writing Development Data..."
#state_rate(dev, pattern, ss)
#print_data(f2, dev)
print "Writing Testing Data..."
#state_rate(test, pattern, ss)
for i in range(10):
    te = test[i]
    state_rate(te, pattern, ss)
    f3 = open('sample'+str(i)+'.test','w')
    print_data(f3, te)

