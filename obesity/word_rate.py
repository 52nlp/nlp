import sys

filename = sys.argv[1]

lines = [line.strip() for line in open(filename,'r')]

words = {}

for l in lines:
	ls = l.split()
	n1 = ls[1]
	n2 = ls[2]
	w = ls[0]
	#if not (n1 == "nan" and n2 == "nan"):
	if not (n2 == "nan"):
		words[w] = (n1, n2)
for w in sorted(words.iterkeys()):
	print w+"\t"+words[w][0]+"\t"+words[w][1]
