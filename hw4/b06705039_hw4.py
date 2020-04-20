from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
import string

import os
# import csv
# import nltk
# import json
import math 
import numpy as np
from numpy.linalg import norm
from numpy import dot

UseDocNum = 100

###################  write file   ########################

def writefile(classK, name):
	with open(name+'.txt', 'w') as f:
		for i in range(0, len(classK)):
			classK[i] = list(classK[i])
			classK[i].sort()
			for j in range(0, len(classK[i])):
				f.write("%s\n"%(classK[i][j]))
			f.write("\n")
	f.close()


###################  (end) write file   ########################
###################  classify function   ########################


def classifyK(K, A):
	kClass = []

	group = set()
	pairGroupNum = []

	for i in range(0, UseDocNum-1):																	# see every pair we order
		x, y = A[i]	
		for j in range(0, len(kClass)):
			if(x in kClass[j] or y in kClass[j]):
				pairGroupNum.append(j)

		if(i < UseDocNum-K):	
			if(len(pairGroupNum)==2):
				kClass[pairGroupNum[0]] = kClass[pairGroupNum[0]].union(kClass[pairGroupNum[1]])
				del kClass[pairGroupNum[1]]
			elif(len(pairGroupNum)==1):
				kClass[pairGroupNum[0]].add(x)
				kClass[pairGroupNum[0]].add(y)
			elif(len(pairGroupNum)==0):
				group.add(x)
				group.add(y)
				kClass.append(group)
				group = set()
			
		else:
			if(len(pairGroupNum)==1):
				if(x in kClass[pairGroupNum[0]]):
					group.add(y)
				else:
					group.add(x)
				kClass.append(group)
				group = set()
			elif(len(pairGroupNum)==0):
				kClass.append({x})
				kClass.append({y})
		pairGroupNum = []
	return kClass

###################  (end) classify function   ########################

###########  class heap   ################

class MaxHeap:
    def __init__(self, collection=None):
        self._heap = []

        if collection is not None:
            for el in collection:
                self.push(el)

    def push(self, value,index):
        temp = [value, index]
        self._heap.append(temp)
        _sift_up(self._heap, len(self) - 1)

    def pop(self):
        _swap(self._heap, len(self) - 1, 0)
        el = self._heap.pop()
        _sift_down(self._heap, 0)
        return el

    def getMax(self):
    	return self._heap[0]

    def __len__(self):
        return len(self._heap)

    def print(self, idx=1, indent=0):
        print("\t" * indent, f"{self._heap[idx - 1]}")
        left, right = 2 * idx, 2 * idx + 1
        if left <= len(self):
            self.print(left, indent=indent + 1)
        if right <= len(self):
            self.print(right, indent=indent + 1)

    def delete(self, doc):
         for i in range(0,len(self._heap)):
         	if(self._heap[i][1]==doc):
         		_swap(self._heap, len(self) - 1, i)
         		self._heap.pop()
         		if(i==len(self)):
         			return
         		idx = _sift_up(self._heap,i)
         		_sift_down(self._heap, idx)
         		return
         return


def _swap(L, i, j):
    L[i], L[j] = L[j], L[i]


def _sift_up(heap, idx):
    parent_idx = (idx - 1) // 2
    # If we've hit the root node, there's nothing left to do
    if parent_idx < 0:
        return idx#my

    # If the current node is larger than the parent node, swap them
    if heap[idx][0] > heap[parent_idx][0]:
        _swap(heap, idx, parent_idx)
        _sift_up(heap, parent_idx)

    return idx#my

def _sift_down(heap, idx):
    child_idx = 2 * idx + 1
    # If we've hit the end of the heap, there's nothing left to do
    if child_idx >= len(heap):
        return

    # If the node has a both children, swap with the larger one
    if child_idx + 1 < len(heap) and heap[child_idx][0] < heap[child_idx + 1][0]:
        child_idx += 1

    # If the child node is smaller than the current node, swap them
    if heap[child_idx][0] > heap[idx][0]:
        _swap(heap, child_idx, idx)
        _sift_down(heap, child_idx)


# def heap_sort(collection):
#     heap = MaxHeap(collection)
#     sorted_arr = []
#     while len(heap) > 0:
#         sorted_arr.append(heap.pop())
#     return sorted_arr

#############  (end) class heap   ################



########### toTerm funciton ################
def toTerm(filename):
	f = open(filename, 'r', encoding='windows-1252')
	#token
	orque =[]
	tokenize = RegexpTokenizer(r'\w+')
	orque = tokenize.tokenize(f.read())


	punc = ['\'','/',':']#\,/_:           
	punF = ['-','_']#-_
	orque = [w for w in orque if w not in string.punctuation and w not in punc and len(w)>2
				and not any(c in w for c in punc) and w.isdigit() == False]
	

	#lowercase
	orque = [orque.lower() for orque in orque]
	#stemming
	ps = PorterStemmer()
	pss = []
	for w in orque:
		#if(len(ps.stem(w))>3):
		#	pss.append(ps.stem(w))
		#else:
			pss.append(w)
	#stopword
	stop_words = set(stopwords.words('english'))
	query = []
	for r in pss:
		if r[0] =='_':
			r = r[1:]
		if (not r in stop_words):
			if r != "":
				query.append(r)
	return query

############ (END) toTerm function ##############

#we have 1095 documents
############ dictionary ##############
dict = [] 				# dictionary for all training data [ dict ]
idf = []					# correspond with dict, idf
tf = []					# term in all doc frequency
doc_token = [] 		# store [ doc with token(with frequency) ]
doc_token_a = {}   # for temp one doc, token with frequency
doc_all = []			# doc with normorlize tf-idf
vec_tfidf = []			# temp vec for a doc




path = 'IRTM/'
fileNames = range(1,UseDocNum+1) 
for name in fileNames:																		#read every document
	name = str(name)
	if(name != '.DS_Store'):
		w = toTerm('./'+path+name+'.txt')
		for token in w:
			if(token not in dict):  															# create dictionary
				dict.append(token);
			if(doc_token_a.get(token)):													# for this doc, store token and it's frequency
				doc_token_a[token]=doc_token_a[token]+1
			else:
				doc_token_a.setdefault(token, 1)
		doc_token.append(doc_token_a)									
		doc_token_a = {}



for term in range(0,len(dict)):															# create dict -> idf
	idf.append(0)
	tf.append(0)
	for docN in range(0,UseDocNum):												# go through every doc for this term
		if(doc_token[docN].get(dict[term])):
			idf[term] = idf[term] + 1														# count term appear in how many doc
			tf[term] = tf[term] + doc_token[docN][dict[term]]						# count term total frequency
	idf[term] = math.log(UseDocNum/idf[term]) 								# calculate idf



for docN in range(0,UseDocNum):													# for every doc, calculate normalize tfidf
	for i in range(0,len(idf)):																# along the dictionary word to form the matrix
		vec_tfidf.append(0);																	# create assistant vector for all_doc
		if(dict[i] in doc_token[docN].keys()):
			vec_tfidf[i] = doc_token[docN][dict[i]]/tf[i]*idf[i]				# calculate tf-idf
	vec_tfidf= vec_tfidf / norm(vec_tfidf)
	doc_all.append(vec_tfidf)																# normalize tf-idf vector for all doc
	vec_tfidf = []		



################# (end) dictionary ###################


#################    set similarity matrix & heap   ###################
sim = []                        				    # store every similarity for every doc
sim_a = []                      				# every similarity for a doc    
doc_heap = []								# store every heap together
cosineV = 0
I = []												# if this cluster still alive

for docN in range(0, UseDocNum):		
	I.append(1)																				# every document is alive now
	AHeap = MaxHeap();																	# a doc heap	
	for i in range(0, UseDocNum):
		cosineV = 0
		cosineV = dot(doc_all[docN],doc_all[i])
		sim_a.append(cosineV)
		if(i != docN):
			AHeap.push(cosineV,i)
	sim.append(sim_a);
	sim_a = []
	doc_heap.append(AHeap)


#################  (end) set similarity matrix & heap   ###################


#################     HAC   ###################

A = []										# store all merge pairs
A_a = []									# merge pairs
MaxIndex = 0
MaxSim = 0
curIndex = 0
curSim = 0
doc = 0

for i in range(0, UseDocNum-1):
	MaxSim, MaxIndex = 0, 0
	for docN in range(0, UseDocNum):														# see which one has max similarity
		if(I[docN]==1):
			curSim, curIndex = doc_heap[docN].getMax();
			if(curSim>MaxSim):
				MaxSim, MaxIndex = curSim, curIndex
				doc = docN	
	A_a.append(doc)																				# store pair into A
	A_a.append(MaxIndex)
	A.append(A_a)	
	A_a = []
	
	

	I[MaxIndex] = 0;																					# set the index been merged
	#doc_heap[MaxIndex] = []	# keep it for print the similarity					# delete heap of a index that will be merge
	doc_heap[doc] = MaxHeap();																# update doc heap
	for docN in range(0, UseDocNum):
		if(I[docN]==1 and docN!=doc):					
			cosineV = 0	

			cosineV = dot(doc_all[doc], doc_all[docN])									# different method to count cosine similarity 
			tempcos = dot(doc_all[docN],doc_all[MaxIndex])
			if(tempcos<cosineV):																	# -- use complete link
				cosineV = tempcos	
			# if(tempcos>cosineV):																	# -- use single link
			# 	cosineV = tempcos	


			doc_all[doc][docN] = cosineV						
			doc_all[docN][doc] = cosineV
			doc_heap[doc].push(cosineV,docN)												# finish update heap	

			doc_heap[docN].delete(MaxIndex)												# update every other doc
			doc_heap[docN].delete(doc)
			doc_heap[docN].push(cosineV,doc)	

			# print("\n", docN)
			# doc_heap[docN].print()																		# trytry


#################    (end) HAC   ###################


#################  classify document  ###################


k8 = classifyK(8, A)

k13 = classifyK(13, A)
k20 = classifyK(20, A)



writefile(k8, 'k8')
writefile(k13, 'k13')
writefile(k20, 'k20')


# print(A)
# k10 = classifyK(10, A)																			# trytry
# # writefile(k10, 'k10')
# print(	k10)																							# trytry
	



#################  (end) classify document  ###################



