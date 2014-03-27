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
from nltk.tag.stanford import NERTagger

from operator import itemgetter

from multiprocessing import Pool

import urllib2

from HTMLParser import HTMLParser
import time
import warnings

warnings.simplefilter("ignore", RuntimeWarning) 

code = {}
urls = []
stop = 0.5

class CodeParser(HTMLParser):
	is_inside_code = False
	is_inside_li = False
	is_inside_p = False
	new_current = False
	current = ""
	last = "1"
	current_code = 0
	value = {
	"  ": 0,
	"F ": 1,
	"F+": 2,
	"E-": 3,
	"E ": 4,
	"E+": 5,
	"D-": 6,
	"D ": 7,
	"D+": 8,
	"C-": 9,
	"C ": 10,
	"C+": 11,
	"B-": 12,
	"B ": 13,
	"B+": 14,
	"A-": 15,
	"A ": 16,
	}
	def handle_starttag(self, tag, attrs):
		if tag == "code":
			self.is_inside_code = True
		if tag == "li":
			self.is_inside_li = True

	def handle_endtag(self, tag):
		global code
		if tag == "code":
			self.is_inside_code = False
		if tag == "li":
			if self.new_current and self.current_code != 0:
				code[self.current].append(self.current_code)
				self.current_code = 0
				self.new_current = False
			self.is_inside_li = False

	def handle_data(self, data):
		global code
		if self.is_inside_li and data == ") ":
			self.is_inside_p = False
		if self.is_inside_code:
			self.current_code = self.value[data]
		if self.is_inside_p:
			self.current = data
			if data not in code:
				code[data] = []
				self.new_current = True
		if self.is_inside_li and data == "(":
			self.is_inside_p = True
		self.last = data

class PageParser(HTMLParser):
	isInside = False
	def handle_starttag(self, tag, attrs):
		global urls
		if tag == "div":
			for attr in attrs:
				if attr == ('class', 'pagination'):
					self.isInside = True
		if self.isInside:
			for attr in attrs:
				if attr[0] == "href":
					urls.append('http://caloriecount.about.com'+attr[1])

	def handle_endtag(self, tag):
		if tag == "div":
			self.isInside = False


def get_codes(keyword):
	global code
	global urls
	global stop
	code = {}
	urls = []
	response = urllib2.urlopen('http://caloriecount.about.com/cc/search.php?searchpro='+keyword+'&generic=&s_order=points&manu=')
	html = response.read()
	time.sleep(stop)

	pp = PageParser()
	pp.feed(html)

	cp = CodeParser()
	cp.feed(html)
	for url in urls[1:-1]:
		h = ""
		r = urllib2.urlopen(url)
		h = r.read()
		time.sleep(stop)
		cp = CodeParser()
		cp.feed(h)

	a1 = []
	a2 = []
	for k in code.keys():
		if re.match(keyword, k, re.IGNORECASE):
			a2.extend(code[k])
		a1.extend(code[k])
	mid1 = np.around(np.average(a1))
	mid2 = np.around(np.average(a2))
	return mid1, mid2 
	
stop_words = ["a", "an", "and", "are", "as", "at", "be", "but", "by",
"for", "if", "in", "into", "is", "it",
"no", "not", "of", "on", "or", "such",
"that", "the", "their", "then", "there", "these",
"they", "this", "to", "was", "will", "with"]
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


samples = []
filename = ""
if len(sys.argv) > 2:
	samples = load_dataset(sys.argv[1])
	filename = sys.argv[2]

f = open(filename,'w')

for s in samples:
	if len(s) > 2:
		print s
		w = s
		if re.match("^#", s):
			w = s[1:]
		if re.match("^@", s):
			w = s[1:]
		try:
			v1, v2 = get_codes(w)		
			string = w + "\t" + str(v1) + "\t" + str(v2)
			f.write(string+"\n")
		except Exception:
			pass

