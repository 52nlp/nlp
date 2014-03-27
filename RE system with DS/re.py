#!/usr/bin/python

######################################################################
#   Yunhao Xu
#   
#   Inmplement an Relation Extraction (RE) system using data automatically
#   generated using Distant Supervision (DS).
#
######################################################################

# Supervised classification with scikit-learn 
# See: http://scikit-learn.org/stable/tutorial/basic/tutorial.html

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

K = 20
threshold = 20
eva_data = []


def resample(samples):
	'''
	Randomize the given samples
	Output a collection of sample that has the same
	size of the input but the order is randomized, and
	the duplicate sample is allowed in this collection
	'''
	rand = np.random.randint(len(samples), size=len(samples))
	sa = []
	ra = {}
	for r in rand:
		sa.append(samples[r])
	return sa

# pointwise mutual information (npmi)
def pmi(features):
	'''
	Compute the PMI value for all features
	'''
	dic = FreqDist()
	dic_pos = FreqDist()
	pos = 0.0
	N = 0.0
	for i,feature in enumerate(features):
		N = N + 1
		for f in feature:
			if f[-1] == 1:
				pos = pos + 1
				for t in f[:-3]:
					dic_pos.inc(t)
					dic.inc(t)
			else:
				for t in f[:-3]:
					dic.inc(t)
	N = N + len(dic.keys())
	pos = pos + len(dic.keys())
	pmi_pos = {}
	for t in dic.keys():
		pmi_pos[t]=np.log(float((dic_pos[t]+1)*N)/float((dic[t]+1)*pos))
	pmi_pos = dict(sorted(pmi_pos.items(), key=itemgetter(1)))
	return pmi_pos

def check_name(words, name, index):
	'''
	Check the given name occurs in the sentence of 
	the given position
	'''
	i = index

	possible_names = []
	string_name = ""
	string = ""
	for n in name:
		string_name += n 
		string += words[i]['word'] if i in words else ""
		possible_names.append(string)
		i+=1
	if i in words:
		string += words[i]['word']
		possible_names.append(string)

	m = 0
	lens = 0
	for l, pn in enumerate(possible_names):
		diff = difflib.SequenceMatcher(a=pn.lower(), b=string_name.lower()).ratio()
		if diff > m:
			m = diff
			lens = l + 1
	return m > 0.9, lens

def convert_to_string(words, index, len):
	'''
	convert list of words to a string
	'''
	s = ""
	for i in np.arange(len):
		s += " " + words[index+i]['word']
	return s

def is_not_bet(e_i, e_l, s_i, s_l):
	'''
	Check if two range of indexes are not
	overlap
	'''
	if e_i > s_i+s_l:
		return True
	elif s_i > e_i + e_l:
		return True
	else:
		return False

def search_for_entity(e_name, tokens):
	'''
	Find the starting index and its length of 
	the entity in a sentence.
	'''
	indexes = []
	lens = []
	pre = 0
	pre_l = 0
	for i in tokens:
		r, l = check_name(tokens, e_name, i)
		if r and is_not_bet(pre, pre_l, i, l):
			indexes.append(i)
			lens.append(l)
			pre = i
			pre_l = l
	return indexes, lens

def search_for_filler(s_ne, tokens):
	'''
	Given the NE label find all fillers that
	matches that label in the sentence
	'''
	indexes = []
	lens = []
	is_inside = False
	l = 0
	for i in tokens:
		if re.match(s_ne, tokens[i]['ne']) and not is_inside:
			indexes.append(i)
			l = 1
			is_inside = True
		elif re.match(s_ne, tokens[i]['ne']) and is_inside:
			l += 1
		elif not re.match(s_ne, tokens[i]['ne']) and is_inside:
			lens.append(l)
			l = 0
			is_inside = False
	if is_inside or l > 0:
		lens.append(l)
	return indexes, lens

def construct_lexical(token, entity_index, entity_len, slot_index, slot_len, sem):
	'''
	Construct the lexical features
	'''
	feature = []
	words_between = ""
	entity_ne = sem[0]
	filler_ne = sem[1]
	is_flaged = sem[-1]
	words_between += entity_ne
	f_u = []
	for i in range(np.minimum(entity_index+entity_len, slot_index+slot_len), np.maximum(entity_index, slot_index)):
		# the between words
		b = entity_ne
		b += ' ' + token[i]['word']
		b += ' ' + filler_ne + ' ' + is_flaged
		feature.append(b)
		if re.match("^VB",token[i]['pos'],re.IGNORECASE):
			f_u.append(b)
		words_between += ' ' + token[i]['word']
	words_between += ' ' + filler_ne + ' ' + is_flaged
	feature.append(words_between)
	return feature, f_u

def find_index(dependencies, index, l):
	'''
	Given a range of index, find all indexes
	that occurs in the dependency path
	'''
	hs = []
	ms = []
	ids = []
	for d in dependencies:
		h = int(d['h'])
		m = int(d['m'])
		if h >= index and h <= index+l and h not in hs:
			hs.append(h)
			if h not in ids:
				ids.append(h)
		if m >= index and m <= index+l and m not in ms:
			ms.append(m)
			if m not in ids:
				ids.append(m)
	return ids, hs, ms

def construct_graph(dep):
	'''
	Given dependencies, contruct a graph
	'''
	G = nx.Graph()
	for d in dep:
		h = int(d['h'])
		m = int(d['m'])
		G.add_edge(h, m)
	return G

def get_dep_path(token, depend, path):
	'''
	Given a dependency path, then for 
	each edge in the path create a feature.
	'''
	ss = []
	vbs = []
	for i, e in enumerate(path[:-1]):
		for d in depend:
			h = int(d['h'])
			m = int(d['m'])
			if m == e and h == path[i+1]:
				s = token[path[i+1]]['word'] +' '+ d['rule']
				ss.append(s)
				if re.match("^VB",token[path[i+1]]['pos'], re.IGNORECASE):
					vbs.append(s)
			if h == e and m == path[i+1]:
				s = d['rule']+' '+token[path[i+1]]['word']
				ss.append(s)
				if re.match("^VB",token[path[i+1]]['pos'], re.IGNORECASE):
					vbs.append(s)
	return ss, vbs

def extract_dependency_path(token, depend, entity_index, entity_len, slot_index, slot_len):
	'''
	Extract the shortest path between a given entity and filler pair
	'''
	e_f = 0
	s_f = 0
	ds = []
	ms = []
	paths = []
	e_ids = find_index(depend, entity_index, entity_len)
	s_ids = find_index(depend, slot_index, slot_len)
	G = construct_graph(depend)
	for e in e_ids:
		for s in s_ids:
			for ei in e:
				for si in s:
					if nx.has_path(G, source=ei, target=si):
						paths.append(nx.shortest_path(G, source=ei, target=si))
	shortest = []
	minimum = sys.maxint
	for p in paths:
		if minimum > len(p):
			minimum = len(p)
			shortest = p
	dep_path, vbs = get_dep_path(token, depend, shortest)
	return dep_path, vbs

def construct_syntax(token, depend, entity_index, entity_len, slot_index, slot_len, sem):
	'''
	Construct the syntax features
	'''
	feature = []
	syn_between = ""
	entity_ne = sem[0]
	filler_ne = sem[1]
	is_flaged = sem[-1]
	syn_between += entity_ne
	f_u = []
	dep_path, vbs = extract_dependency_path(token, depend, entity_index, entity_len, slot_index, slot_len)
	for d in dep_path:
		b = entity_ne
		b += ' ' + d
		b += ' ' + filler_ne + ' ' + is_flaged
		feature.append(b)
		if d in vbs:
			f_u.append(b)
		syn_between += ' ' + d
		# the between words
	syn_between += ' ' + filler_ne + ' ' + is_flaged
	feature.append(syn_between)
	return feature, f_u

def construct_semantics(token, entity_index, slot_index):
	'''
	Construct the semantic features
	'''
	feature = []
	# first two features contains the info 
	feature.append(token[entity_index]['ne'])
	feature.append(token[slot_index]['ne'])
	# second two features have the order 
	# last feature is the flag: 1 - slot comes before entity
	if entity_index > slot_index:
		feature.append(token[slot_index]['ne'])
		feature.append(token[entity_index]['ne'])
		feature.append('unflaged')
	else:
		feature.append(token[entity_index]['ne'])
		feature.append(token[slot_index]['ne'])
		feature.append('flaged')
	return feature

def construct_possible_features_test(samples, E_ne, S_ne, has_Window=False):
	'''
	This part is used to compute the possible NE labels for the 
	filler

	Scan through the input training data, extract all semantics 
	features. It will output a list of all features along with 
	the count of the occurrences of each feature.
	'''
	global K
	window = K
	possible_features = {}
	feature_dist_sem = FreqDist()
	feature_dist_lex = FreqDist()
	feature_dist_lex_u = FreqDist()
	feature_dist_lex_n = FreqDist()
	feature_dist_syn = FreqDist()
	feature_dist_syn_u = FreqDist()
	feature_dist_syn_n = FreqDist()
	features  = []
	for s in samples:
		for k, token in enumerate(s['tokens']):
			features_of_sample = []
			e_indexes, e_lens = search_for_entity(s['entity'], token)
			s_indexes, s_lens = search_for_entity(s['slot'], token)
			for i, e_i in enumerate(e_indexes):
				for j, s_i in enumerate(s_indexes):
					if (np.abs(e_i-s_i) <= window or not has_Window) and is_not_bet(e_i, e_lens[i], s_i, s_lens[j]):
						semantics_features = construct_semantics(token, e_i, s_i)

						feature_dist_sem.inc(semantics_features[-1])
						features_of_sample.append(semantics_features)
		features.append(features_of_sample)
	return features, feature_dist_lex, feature_dist_lex_u, feature_dist_syn, feature_dist_syn_u, feature_dist_sem, feature_dist_lex_n, feature_dist_syn_n

def construct_possible_features(samples, E_ne, S_ne, has_Window=False):
	'''
	Scan through the input training data, extract all possible 
	features for lexical, syntax and semantic. It will output
	a list of all features along with the count of the occurrences
	of each feature (this would be used for filtering)
	'''
	global K
	window = K
	possible_features = {}
	feature_dist_sem = FreqDist()
	feature_dist_lex = FreqDist()
	feature_dist_lex_u = FreqDist()
	feature_dist_lex_n = FreqDist()
	feature_dist_syn = FreqDist()
	feature_dist_syn_u = FreqDist()
	feature_dist_syn_n = FreqDist()
	features  = []
	for s in samples:
		for k, token in enumerate(s['tokens']):
			features_of_sample = []
			e_indexes, e_lens = search_for_entity(s['entity'], token)
			s_indexes, s_lens = search_for_filler(S_ne, token)
			for i, e_i in enumerate(e_indexes):
				for j, s_i in enumerate(s_indexes):
					if (np.abs(e_i-s_i) <= window or not has_Window) and is_not_bet(e_i, e_lens[i], s_i, s_lens[j]):
						semantics_features = construct_semantics(token, e_i, s_i)
						lexical_features, lexical_features_u = construct_lexical(token, e_i, e_lens[i], s_i, s_lens[j], semantics_features)
						syntax_features, syntax_features_u = construct_syntax(token, s['depend'][k], e_i, e_lens[i], s_i, s_lens[j], semantics_features)
						r, l = check_name(token, s['slot'], s_i)
						if r:
							label = [1] 
						else:
							label = [0]
						e_str = convert_to_string(token, e_i, e_lens[i])
						s_str = convert_to_string(token, s_i, s_lens[j])
						es = [e_str]
						ss = [s_str]
						feature = lexical_features + syntax_features + es + ss + label				
						feature_dist_lex_n.inc(lexical_features[0])
						if len(lexical_features) >= 2:
							feature_dist_lex_n.inc(lexical_features[-2])
						else:
							feature_dist_lex_n.inc(lexical_features[-1])
						for f in lexical_features_u:
							feature_dist_lex_n.inc(f)

						feature_dist_syn_n.inc(syntax_features[0])
						if len(syntax_features) >=2:
							feature_dist_syn_n.inc(syntax_features[-2])
						else:
							feature_dist_syn_n.inc(syntax_features[-1])					
						for f in syntax_features_u:
							feature_dist_syn_n.inc(f)

						for f in semantics_features:
							feature_dist_sem.inc(f)
						for f in lexical_features[:-1]:
							feature_dist_lex.inc(f)
						for f in syntax_features[:-1]:
							feature_dist_syn.inc(f)
						feature_dist_lex_u.inc(lexical_features[-1])
						feature_dist_syn_u.inc(syntax_features[-1])

						features_of_sample.append(feature)
		features.append(features_of_sample)
	return features, feature_dist_lex, feature_dist_lex_u, feature_dist_syn, feature_dist_syn_u, feature_dist_sem, feature_dist_lex_n, feature_dist_syn_n

def process_tokens(line):
	'''
	Input is the line of the tokens.
	split the input string, and make a dictionary
	for the given string, where key is the word index,
	and value is the dictionary of the word info, which
	contains the word, its pos, and its ner
	'''
	tokens = (re.split('\s+',line))[1:]
	words = {}
	i = 1
	for t in tokens:
		tags = re.split('\/',t[::-1])
		ne = (tags[0])[::-1]
		pos = (tags[1])[::-1]
		size = len(ne)+1+len(pos)+1
		word = t[:-size]
		w = {}
		w['word'] = word
		w['pos'] = pos
		w['ne'] = ne
		words[i] = w
		i += 1
	return words


def process_dependencies(line):
	'''
	Input is the line of the dependency path.
	Split the input string, and make a list of 
	the dependencies. Each element in the list is
	a dictionary that contains the header, modifier,
	and the rule.
	'''
	dp = []
	for t in re.findall("\d+\s+\d+\s+\w+",line):
		ts = re.split('\s+',t)
		d = ts[0]
		r = ts[1]
		rule = ts[2]
		w = {}
		w['h'] = d
		w['m'] = r
		w['rule'] = rule
		dp.append(w)
	return dp

def load_train(filename, dataset):
	'''
	Read the input file. 
	For each entity and filler pair, store
	all information into a dictionary
	'''
	lines = [line.strip() for line in open(filename,'r')]
	samples = []
	tmp = {}
	for l in lines:
		if (l != ""):
			entity = []
			slot = []
			tokens = {}
			dp = []
			words = re.split('\s+',l)
			if words[0] == "ENTITY:":
				if tmp != {}:
					samples.append(tmp)
				tmp = {}
				tmp['tokens'] = []
				tmp['depend'] = []
				entity = words[1:]
				tmp['entity'] = entity
			elif words[0] == "SLOT:":
				slot = words[1:]
				tmp['slot'] = slot
			elif words[0] == "TOKENS:":
				tokens = process_tokens(l)
				tmp['tokens'].append(tokens)
			elif words[0] == "DEPENDENCIES:":
				dp = process_dependencies(l)
				tmp['depend'].append(dp)
	return samples

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
			words = re.split('\t', l)
			string = words[1] + " " + words[2]
			samples.append(string)
	return samples

# TODO
# work out threshold for each
# PMI threshold
def filter_features(features, lex_feature, lex_feature_u, syn_feature, syn_feature_u, sem_feature, lex_feature_n, syn_feature_n):
	'''
	Filter all features based on their count and pmi
	Return a list of features
	'''
	global threshold
	feature = []
	pmi_pos = pmi(features)
	lex_threshold = 0.0001 * lex_feature.N()
	lex_u_threshold = 0.001 * lex_feature_u.N()
	lex_n_threshold = 0.001 * lex_feature_n.N()
	syn_threshold = 0.0001 * syn_feature.N()
	syn_u_threshold = 0.001 * syn_feature_u.N()
	syn_n_threshold = 0.001 * syn_feature_n.N()
	for i, lex in enumerate(lex_feature.keys()):
		if lex_feature[lex] >= threshold and lex_threshold > i:
			features.append(lex)
	for i, lex in enumerate(lex_feature_u.keys()):
		if lex_u_threshold > i and pmi_pos.has_key(lex) and pmi_pos[lex] > -0.5:
			feature.append(lex)
	for i, lex in enumerate(lex_feature_n.keys()):
		if lex_n_threshold > i and pmi_pos.has_key(lex) and pmi_pos[lex] > -0.5:
			feature.append(lex)
	for i, syn in enumerate(syn_feature.keys()):
		if syn_feature[lex] >= threshold and syn_threshold > i:
			features.append(syn)
	for i, syn in enumerate(syn_feature_u.keys()):
		if syn_u_threshold > i and pmi_pos.has_key(syn) and pmi_pos[syn] > -0.5:
			feature.append(syn)
	for i, syn in enumerate(syn_feature_n.keys()):
		if syn_n_threshold > i and pmi_pos.has_key(syn) and pmi_pos[syn] > -0.5:
			feature.append(syn)

	return feature

def construct_dataset(samples, features, filtered_features, additional_keys=[]):
	'''
	Construct datums and labels matries according to the given 
	features.
	'''
	global eva_data
	datums = []
	labels = []
	for i, sample in enumerate(samples):
		for feature in features[i]:
			X = []
			for f in filtered_features:
				if f in feature:
					X.append(1)
				else:
					X.append(0)
			datums.append(X)
			lab = feature[-1]
			s = feature[-3] + ' ' + feature[-2]
			if s in additional_keys:
				labels.append(1)
			else:
				labels.append(feature[-1])
	for i, d in enumerate(datums):
		if labels[i] == 1:
			eva_data = d
			break
	return datums, labels

def construct_test_dataset(samples, features, filtered_features, additional_keys=[]):
	'''
	Construct datums and labels matries according to the given 
	features.
	'''
	datums = {}
	labels = {}
	for i, sample in enumerate(samples):
		data = []
		label = []
		for feature in features[i]:
			X = []
			for f in filtered_features:
				if f in feature:
					X.append(1)
				else:
					X.append(0)
			data.append(X)
			s = feature[-3] + ' ' + feature[-2]
			contain = False
			for string in additional_keys:
				diff = difflib.SequenceMatcher(a=s.lower(), b=string.lower()).ratio()
				if diff > 0.9:
					contain = True
					break

			if contain:
				label.append(1)
			else:
				label.append(feature[-1])

		datums[i] = data
		labels[i] = label
	return datums, labels

def test_predict(c, datums, labels, features, dictionary, additional_keys=[]):
	'''
	Predict the datums
	'''
	r = []	
	r_total = []
	l = []
	l_total = []
	keys = {}
	for i in datums.keys():
		if len(datums[i]) != 0:
			dic = {}
			tru = {}
			lab = []
			for j, f in enumerate(features[i]):
				s = f[-3] + ' ' + f[-2]
				if dic.has_key(s):
					dic[s].append(j)
				else:
					dic[s] = [j]
					tru[s] = 1 if f[-1] == 1 else 0
					contain = False
					for string in additional_keys:
						diff = difflib.SequenceMatcher(a=s.lower(), b=string.lower()).ratio()
						if diff > 0.9:
							contain = True
							break

					if contain:
						tru[s] = 1
					lab.append(tru[s])

			ps = []
			for d in dic.keys():
				la = []
				for k in np.arange(len(datums[i])):
					if k in dic[d]:
						la.append(1)
					else:
						la.append(0)
				p = c.score(array(datums[i]), array(la))
				ps.append(p)
				if keys.has_key(d):
					keys[d].append(p)
				if tru[d] != 1:
					keys[d] = [p]
				l.append(tru[d])

			m = 0
			index = 0
			for ind, p in enumerate(ps):
				if p > m:
					m = p
					index = ind


			for ind, p in enumerate(ps):
				if ind == index:
					r.append(1)
				else:
					r.append(0)

			p = c.predict(array(datums[i]))
			r_total.extend(p)
			l_total.extend(labels[i])
		else:
			r.append(0)
			l.append(1)
			r_total.append(0)
			l_total.append(1)
	return r, l, r_total, l_total, keys

def noise_or(keys):
	'''
	Using the noise or algorithm, compute 
	the score of each keys
	'''
	key = {}
	for k in keys.keys():
		m = 1.0
		for p in keys[k]:
			m = m * (1-p)
		key[k] = 1 - m
	result = {}
	for k in key.keys():
		if key[k] > 0.8:
			result[k] = key[k]
	return result

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
	l = 1
	my_correct = correct_per_class.get(l, 0)
	my_pred = predicted_per_class.get(l, 0)
	my_total = total_per_class.get(l, 0)
	p = float(my_correct) / float(my_pred)
	r = float(my_correct) / float(my_total)
	f1 = 0
	if p != 0 and r != 0:
	    f1 = 2*p*r / (p + r)
	print "Label", l, " => Precision:", p, "Recall:", r, "F1:", f1

#files = ["per_spouse", "per_employee_of", "org_top_membersSLASHemployees", "org_top_membersSLASHemployees"]
files = ["org_country_of_headquarters"]
file_nes = {'org_country_of_headquarters': {'s': 'COUNTRY', 'e': 'ORGANIZATION'},
'per_spouse': {'s': 'PERSON', 'e': 'PERSON'},
'per_employee_of': {'s': 'PERSON', 'e': 'ORGANIZATION'},
'org_top_membersSLASHemployees': {'s': 'ORGANIZATION', 'e': 'PERSON'},
'sample': {'s': 'COUNTRY', 'e': 'ORGANIZATION'}}
files_argv = []
if len(sys.argv) > 1:
	for a in sys.argv[1:]:
		files_argv.append(a)
	files = files_argv

for f in files:
#	c = DecisionTreeClassifier()
	c = SVC()
	train_data = {}
	test_data = {}
#	window = False
	window = True
	print "Load the files..."
	samples = load_train(f+".train",train_data)
	test_samples = load_train(f+".test",test_data)
	optional_samples = load_train(f+".train.optional",test_data)
	dictionary = load_dataset(f)
	print "DONE.\nConstruct all possible features..."

	resample = resample(samples)
	features, lex, lex_u, syn, syn_u, sem, lex_n, syn_n = construct_possible_features(samples, file_nes[f]['e'], file_nes[f]['s'], window)

	print "DONE.\nFilter features..."
	filtered_features = filter_features(features, lex, lex_u, syn, syn_u, sem, lex_n, syn_n)
	print "DONE.\nTraining..."
	train_datums, train_labels = construct_dataset(samples, features, filtered_features)
	c.fit(train_datums, train_labels)
	print "DONE.\nEvaluating..."
	t_features, t_lex, t_lex_u, t_syn, t_syn_u, t_sem, t_lex_n, t_syn_n = construct_possible_features(test_samples, file_nes[f]['e'], file_nes[f]['s'], window)

	test_datums, test_labels = construct_test_dataset(test_samples, t_features, filtered_features)

	p, l, pt, lt, keys = test_predict(c, test_datums, test_labels, t_features, dictionary)
	side_by_side = transpose(array([l, p]))
	score(side_by_side)

	side_by_side = transpose(array([lt, pt]))
	score(side_by_side)

	print "DONE.\nProcess on optional dataset..."
	o_features, o_lex, o_lex_u, o_syn, o_syn_u, o_sem, o_lex_n, o_syn_n = construct_possible_features_test(optional_samples, file_nes[f]['e'], file_nes[f]['s'], window)
	summ = 0
	for sem in o_sem.keys():
		if sem != 'O' and sem != file_nes[f]['s']:
			summ += o_sem[sem]
	ne_label = file_nes[f]['s']
	for sem in o_sem.keys():
		if sem != 'O' and o_sem[sem] > summ/o_sem.N():
			ne_label += "|" + sem

	features, lex, lex_u, syn, syn_u, sem, lex_n, syn_n = construct_possible_features(samples, file_nes[f]['e'], ne_label, window)
	t_features, t_lex, t_lex_u, t_syn, t_syn_u, t_sem, t_lex_n, t_syn_n = construct_possible_features(test_samples, file_nes[f]['e'], ne_label, window)

	test_datums, test_labels = construct_test_dataset(test_samples, t_features, filtered_features)
	o_p, o_l, o_pt, o_lt, keys = test_predict(c, test_datums, test_labels, t_features, dictionary)
	key = noise_or(keys)
	print "DONE.\nRe-training..."
	train_datums, train_labels = construct_dataset(samples, features, filtered_features, key)
	c.fit(train_datums, train_labels)
	test_datums, test_labels = construct_test_dataset(test_samples, t_features, filtered_features, key)
	print "DONE.\nRe-evaluating..."
	p, l, pt, lt, keys = test_predict(c, test_datums, test_labels, t_features, dictionary, key)
	side_by_side = transpose(array([l, p]))
	score(side_by_side)
	side_by_side = transpose(array([lt, pt]))
	score(side_by_side)
