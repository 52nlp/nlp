#!/usr/bin/python

######################################################################
#   Yunhao Xu
#
#   Based on the tweets about foods from each states, rate the obesity of
#   a state in 3 levels ("fit", "ok", and "fat").
#
######################################################################

# Supervised classification with scikit-learn 
# See: http://scikit-learn.org/stable/tutorial/basic/tutorial.html
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
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
    | [\xF0-\xF7][\x80-\xBF][\x80-\xBF][\x80-\xBF]|[\xC2-\xE3][\x80-\xBF][\x80-\xBF]|[\xC2-\xE3][\xA9-\xAE]
    | ([A-Z]\.)+                # abbreviations, e.g. "U.S.A."
    | \w+(-\w+)*                # words with optional internal hyphens
    | \$?\d+(\.\d+)?%?          # currency and percentages
    | \.\.\.                    # ellipsis
    | @+[\w_]+                  # user names
    | \#+[\w_]+[\w\'_\-]*[\w_]+ # hashtags
    | [.,;"'?():-_`]            # these are separate tokens
    '''

def pmi():
    pass

def load_dis(filename):
    '''
    Read the input emoji file.  
    match the emoji code and its distribution
    '''
    lines = [line.strip() for line in open(filename,'r')]
    dis = {}
    for l in lines:
        ls = l.split()
        dis[ls[0]] = ls[1]
    return dis

def load_dataset(filename):
    '''
    Read the input file.  
    '''
    lines = [line.strip() for line in open(filename,'r')]
    samples = []
    tmp = {}
    for l in lines:
        if (l != ""):
            samples.append(l)
    return samples

def load_test_dataset(filename):
    '''
    Read the input file for testing. 
    '''
    lines = [line.strip() for line in open(filename,'r')]
    samples = []
    tmp = {}
    for l in lines:
        if (l != ""):
            samples.append(l)
    return samples

def load_rates(filename):
    '''
    Read the input file.  
    match the health rate for each terms
    '''
    lines = [line.strip() for line in open(filename,'r')]
    rates = {}
    for l in lines:
        ls = l.split()
        rates[ls[0]] = (ls[1],ls[2])
    return rates

def load_emoji(filename):
    '''
    Read the input file.  
    match the happy rate for each emoji
    '''
    lines = [line.strip() for line in open(filename,'r')]
    rates = {}
    for l in lines:
        ls = l.split()
        rates[ls[0]] = ls[2]
    return rates

def load_label(filename):
    '''
    Read the input file.  
    match the obesity rate for each states
    '''
    lines = [line.strip('\r') for line in open(filename,'r')]
    rates = {}
    for l in lines:
        ls = l.split()
        if float(ls[1]) >= 30:
            rates[ls[0]] = 0
        elif float(ls[1]) < 25:
            rates[ls[0]] = 2
#            rates[ls[0]] = 1
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

def construct_unigram_count(words, emoji_rate, negInclude=False, emoInclude=False):
    '''
    Construct the unigram features for normal terms
    '''
    feature = {}
    has_neg = False
    emo_sum = 0
    emo_count = 0
    emo_rate = 1
    if emoInclude:
        for w in words:
            if w in emoji_rate:
                emo_sum += int(emoji_rate[w])
                emo_count += 1
        if emo_count != 0:
            emo_rate = emo_sum/emo_count
    for w in words:
        w = w.lower()
        if re.match('[^\w\s]', w):
            has_neg = False
        if re.match('^(|n[o\']t)$', w):
            has_neg = True
        if len(w) > 2:
            s = w.lower()
            if re.match("^#", w):
                s = w[1:]
            if re.match("^@", w):
                s = w[1:]
            if has_neg and negInclude:
                s = 'NOT_'+s
            if emoInclude:
                s = str(emo_rate)+"_"+s
            if s in feature:
                feature[s] = feature[s] + 1
            else:
                feature[s] = 1
    return feature

def construct_unigram_rate(words, term_rates, emoji_rate, negInclude=False, emoInclude=False):
    '''
    Construct the unigram features for alimental terms
    '''
    feature = {}
    emo_sum = 0
    emo_count = 0
    emo_rate = 1
    if emoInclude:
        for w in words:
            if w in emoji_rate:
                emo_sum += int(emoji_rate[w])
                emo_count += 1
        if emo_count != 0:
            emo_rate = emo_sum/emo_count

    for w in words:
        w = w.lower()
        if len(w) > 2:
            s = w
            if re.match("^#", w):
                s = w[1:]
            if re.match("^@", w):
                s = w[1:]
            string = s
            if emoInclude:
                string = str(emo_rate)+"_"+s
            if s in term_rates:
                string = string + "_"+term_rates[s][0]
                feature[string] = term_rates[s]
    return feature

def construct_bigram_count(words, emoji_rate, bigramInclude=False, negInclude=False, emoInclude=False):
    '''
    Construct the bigram features for normal terms
    '''
    feature = {}
    has_neg = False
    emo_sum = 0
    emo_count = 0
    emo_rate = 1
    if bigramInclude:
        if emoInclude:
            for w in words:
                if w in emoji_rate:
                    emo_sum += int(emoji_rate[w])
                    emo_count += 1
            if emo_count != 0:
                emo_rate = emo_sum/emo_count
        for i, w in enumerate(words):
            w1 = w.lower()
            w2 = w1
            if i+1 < len(words):
                w2 = words[i+1].lower()
            if re.match('[^\w\s]', w1):
                has_neg = False
            if re.match('[^\w\s]', w2):
                has_neg = False
            if re.match('^(|n[o\']t)$', w1):
                has_neg = True
            if re.match('^(|n[o\']t)$', w2):
                has_neg = True
            if len(w1) > 2 and len(w2) > 2:
                s1 = w1
                s2 = w2
                if re.match("^#", w1):
                    s1 = w1[1:]
                if re.match("^#", w2):
                    s2 = w2[1:]
                if re.match("^@", w1):
                    s1 = w1[1:]
                if re.match("^@", w2):
                    s2 = w2[1:]
                s = s1.lower() +"_"+ s2.lower()
                if has_neg and negInclude:
                    s = 'NOT_'+s
                if emoInclude:
                    s = str(emo_rate)+"_"+s
                if s in feature:
                    feature[s] = feature[s] + 1
                else:
                    feature[s] = 1
    return feature

def construct_bigram_rate(words, term_rates, emoji_rate, bigramInclude=False, negInclude=False, emoInclude=False):
    '''
    Construct the bigram features for alimental terms
    '''
    feature = {}
    if bigramInclude:
        emo_sum = 0
        emo_count = 0
        emo_rate = 1
        if emoInclude:
            for w in words:
                if w in emoji_rate:
                    emo_sum += int(emoji_rate[w])
                    emo_count += 1
            if emo_count != 0:
                emo_rate = emo_sum/emo_count

        for i, w in enumerate(words):
            w1 = w.lower()
            w2 = w1
            if i+1 < len(words):
                w2 = words[i+1].lower()
            if len(w1) > 2 and len(w2) > 2:
                s1 = w1
                s2 = w2
                if re.match("^#", w1):
                    s1 = w1[1:]
                if re.match("^#", w2):
                    s2 = w2[1:]
                if re.match("^@", w1):
                    s1 = w1[1:]
                if re.match("^@", w2):
                    s2 = w2[1:]
                s = s1.lower()+'_'+s2.lower()
                string = s
                if emoInclude:
                    string = str(emo_rate)+"_"+s
                if s in term_rates:
                    string = string + "_"+term_rates[s][0]
                    feature[string] = term_rates[s]
    return feature

def construct_features(samples, term_rates, pattern, emoji_rate, bigramInclude=False, negInclude=False, emoInclude=False):
    '''
    Scan through the input training data, extract all possible 
    features for unigram, bigram, normal terms and alimental terms. 
    It will output a list of all features along with the count of the 
    occurrences of each feature (this would be used for filtering)
    '''
    features_count = {}
    features_rate = {}
    bi_features_count = {}
    bi_features_rate = {}
#    unigram = FreqDist()
#    uni_rate = FreqDist()
    for s in samples:
        # default information
        feature = []
        words = nltk.regexp_tokenize(s, pattern)
        label = 1
        if words[0] == "fat":
            label = 0
        elif words[0] == "fit":
            label = 2
        state = words[1]
        #unigram word count
        uni_count = construct_unigram_count(words[2:], emoji_rate, negInclude, emoInclude)
        #unigram word count + rate
        uni_rate = construct_unigram_rate(words[2:], term_rates, emoji_rate, negInclude, emoInclude)
        #bigrams word count
        bi_count = construct_bigram_count(words[2:], emoji_rate, bigramInclude, negInclude, emoInclude)
        #bigrams word count + rate
        bi_rate = construct_bigram_rate(words[2:], term_rates, emoji_rate, bigramInclude, negInclude, emoInclude)

        if state in features_count:
            features_count[state].append(uni_count)
            features_rate[state].append(uni_rate)
            bi_features_count[state].append(bi_count)
            bi_features_rate[state].append(bi_rate)
        else:
            features_count[state] = []
            features_count[state].append(uni_count)
            features_rate[state] = []
            features_rate[state].append(uni_rate)
            bi_features_count[state] = []
            bi_features_count[state].append(bi_count)
            bi_features_rate[state] = []
            bi_features_rate[state].append(bi_rate)
    return features_count, features_rate, bi_features_count, bi_features_rate

def filter_features(features_count, features_rate, bi_features_count, bi_features_rate, state_label, negInclude=True, emoInclude=True):
    '''
    Filter all features based on their count
    Return a list of features
    '''
    count_threshold = 100
    fat_count = FreqDist()
    fit_count = FreqDist()
    ok_count = FreqDist()
    fat_count1 = FreqDist()
    fit_count1 = FreqDist()
    ok_count1 = FreqDist()
    fat_count2 = FreqDist()
    fit_count2 = FreqDist()
    ok_count2 = FreqDist()
    fat_count0 = FreqDist()
    fit_count0 = FreqDist()
    ok_count0 = FreqDist()

    fat_rate = FreqDist()
    fit_rate = FreqDist()
    ok_rate = FreqDist()
    fat_rate0 = FreqDist()
    fit_rate0 = FreqDist()
    ok_rate0 = FreqDist()
    fat_rate1 = FreqDist()
    fit_rate1 = FreqDist()
    ok_rate1 = FreqDist()
    fat_rate2 = FreqDist()
    fit_rate2 = FreqDist()
    ok_rate2 = FreqDist()

    bi_fat_count = FreqDist()
    bi_fit_count = FreqDist()
    bi_ok_count = FreqDist()
    bi_fat_count1 = FreqDist()
    bi_fit_count1 = FreqDist()
    bi_ok_count1 = FreqDist()
    bi_fat_count2 = FreqDist()
    bi_fit_count2 = FreqDist()
    bi_ok_count2 = FreqDist()
    bi_fat_count0 = FreqDist()
    bi_fit_count0 = FreqDist()
    bi_ok_count0 = FreqDist()

    bi_fat_rate = FreqDist()
    bi_fit_rate = FreqDist()
    bi_ok_rate = FreqDist()
    bi_fat_rate0 = FreqDist()
    bi_fit_rate0 = FreqDist()
    bi_ok_rate0 = FreqDist()
    bi_fat_rate1 = FreqDist()
    bi_fit_rate1 = FreqDist()
    bi_ok_rate1 = FreqDist()
    bi_fat_rate2 = FreqDist()
    bi_fit_rate2 = FreqDist()
    bi_ok_rate2 = FreqDist()

    states = {}
    for s in features_count:
        count = FreqDist()
        for feature in features_count[s]:
            for f in feature:
                count.inc(f, count=feature[f])
                if state_label[s] == 0:
                    fat_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        fat_count0.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        ok_count0.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        fit_count0.inc(f,count=feature[f])
                elif state_label[s] == 1:
                    ok_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        fat_count1.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        ok_count1.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        fit_count1.inc(f,count=feature[f])
                elif state_label[s] == 2:
                    fit_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        fat_count2.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        ok_count2.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        fit_count2.inc(f,count=feature[f])
        states[s] = count
    for s in features_rate:
        count = FreqDist()
        for feature in features_rate[s]:
            for f in feature:
                if state_label[s] == 0 and feature[f][1] != float('NaN'):
                    fat_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate0.inc(f)
                    elif re.match('^1_',f):
                        ok_rate0.inc(f)
                    elif re.match('^2_',f):
                        fit_rate0.inc(f)
                elif state_label[s] == 1 and feature[f][1] != float('NaN'):
                    ok_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate1.inc(f)
                    elif re.match('^1_',f):
                        ok_rate1.inc(f)
                    elif re.match('^2_',f):
                        fit_rate1.inc(f)
                elif state_label[s] == 2 and feature[f][1] != float('NaN'):
                    fit_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate2.inc(f)
                    elif re.match('^1_',f):
                        ok_rate2.inc(f)
                    elif re.match('^2_',f):
                        fit_rate2.inc(f)
    bi_states = {}
    for s in bi_features_count:
        bi_count = FreqDist()
        for feature in bi_features_count[s]:
            for f in feature:
                bi_count.inc(f, count=feature[f])
                if state_label[s] == 0:
                    bi_fat_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        bi_fat_count0.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        bi_ok_count0.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        bi_fit_count0.inc(f,count=feature[f])
                elif state_label[s] == 1:
                    bi_ok_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        bi_fat_count1.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        bi_ok_count1.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        bi_fit_count1.inc(f,count=feature[f])
                elif state_label[s] == 2:
                    bi_fit_count.inc(f, count=feature[f])
                    if re.match('^0_',f):
                        bi_fat_count2.inc(f,count=feature[f])
                    elif re.match('^1_',f):
                        bi_ok_count2.inc(f,count=feature[f])
                    elif re.match('^2_',f):
                        bi_fit_count2.inc(f,count=feature[f])
        bi_states[s] = bi_count
    for s in bi_features_rate:
        count = FreqDist()
        for feature in bi_features_rate[s]:
            for f in feature:
                if state_label[s] == 0 and feature[f][1] != float('NaN'):
                    fat_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate0.inc(f)
                    elif re.match('^1_',f):
                        ok_rate0.inc(f)
                    elif re.match('^2_',f):
                        fit_rate0.inc(f)
                elif state_label[s] == 1 and feature[f][1] != float('NaN'):
                    ok_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate1.inc(f)
                    elif re.match('^1_',f):
                        ok_rate1.inc(f)
                    elif re.match('^2_',f):
                        fit_rate1.inc(f)
                elif state_label[s] == 2 and feature[f][1] != float('NaN'):
                    fit_rate.inc(f)
                    if re.match('^0_',f):
                        fat_rate2.inc(f)
                    elif re.match('^1_',f):
                        ok_rate2.inc(f)
                    elif re.match('^2_',f):
                        fit_rate2.inc(f)

    feature = {}
    if negInclude or emoInclude:
        for i,k in enumerate(fat_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        #####
        for i,k in enumerate(bi_fat_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_count0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_count1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_count2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_rate0.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_rate1.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_rate2.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
    else:
        for i,k in enumerate(fat_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fat_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(ok_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(fit_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        #####
        for i,k in enumerate(bi_fat_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_count.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fat_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_ok_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break
        for i,k in enumerate(bi_fit_rate.keys()):
            if i < count_threshold*3:
                feature[k] = 1
            else:
                break

    filtered = []
    for f in feature:
        filtered.append(f)
    return filtered

def construct_dataset(features_count, features_rate, bi_features_count, bi_features_rate, state_label, filtered_features):
    '''
    Construct datums and labels matries according to the given 
    features.
    '''
    table = []
    label = []
    total = 0
    for s in features_count:
        for f in features_count:
            total += 1
    for s in state_label:
        if s in features_count:
            y = []
            for f in filtered_features:
                v = 0
                t = 0
                for feature in features_count[s]:
                    t += 1
                    if f in feature:
                        v += feature[f]
                for feature in features_rate[s]:
                    t += 1
                    if f in feature:
                        v += 1
                for feature in bi_features_count[s]:
                    t += 1
                    if f in feature:
                        v += feature[f]
                for feature in bi_features_rate[s]:
                    t += 1
                    if f in feature:
                        v += 1
#                vs = v*total/t
                vs = v
                y.append(vs)
            table.append(y)
            label.append(state_label[s])
    return table, label

def score(side_by_side):
    '''
    Output the score
    '''
    total = 0
    correct = 0
    total_per_class = {}
    predicted_per_class = {}
    correct_per_class = {}
    counts_per_label = {}
    for i in range(side_by_side.shape[0]):
        gold = side_by_side[i,0]
        pred = side_by_side[i,1]
        total += 1
        if(gold == pred):
            correct += 1
            correct_per_class[gold] = correct_per_class.get(gold, 0) + 1
        total_per_class[gold] = total_per_class.get(gold, 0) + 1
        predicted_per_class[pred] = predicted_per_class.get(pred, 0) + 1

    acc = float(correct) / float(total)
    print "Accuracy:", acc, "correct:", correct, "total:", total
    for l in total_per_class.keys():
        my_correct = correct_per_class.get(l, 0)
        my_pred = predicted_per_class.get(l, 0)
        my_total = total_per_class.get(l, 0)
        p = float(my_correct) / float(my_pred)
        r = float(my_correct) / float(my_total)
        f1 = 0
        if p != 0 and r != 0:
            f1 = 2*p*r / (p + r)
        print "Label", l, " => Precision:", p, "Recall:", r, "F1:", f1


#c = SVC()
c = MultinomialNB()
#c = DecisionTreeClassifier()
samples = []
tests_samples = []
term_rates = {}
emoji_rate = {}
state_label = {}
if len(sys.argv) > 5:
    samples = load_dataset(sys.argv[1])
    tests_samples = load_dataset(sys.argv[2])
    term_rates = load_rates(sys.argv[3])
    emoji_rate = load_emoji(sys.argv[4])
    state_label = load_label(sys.argv[5])
else:
    sys.exit("Usage")

negInclude=False
emoInclude=False
bigramInclude=True
features_count, features_rate, bi_features_count, bi_features_rate = construct_features(samples, term_rates, pattern, emoji_rate, bigramInclude, negInclude, emoInclude)

filtered_features = filter_features(features_count, features_rate, bi_features_count, bi_features_rate, state_label, negInclude, emoInclude)
datrum, label = construct_dataset(features_count, features_rate, bi_features_count, bi_features_rate, state_label, filtered_features)
c.fit(datrum, label)

print "Testing..."
test_features_count, test_features_rate, test_bi_features_count, test_bi_features_rate = construct_features(tests_samples, term_rates, pattern, emoji_rate, bigramInclude, negInclude, emoInclude)
test_datrum, test_label = construct_dataset(test_features_count, test_features_rate, test_bi_features_count, test_bi_features_rate, state_label, filtered_features)
s = c.predict(test_datrum)

side_by_side = transpose(array([test_label, s]))
score(side_by_side)

