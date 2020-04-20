import os
import csv
import nltk
import json
import math 
import string
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords

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

path = 'IRTM/'
fileNames = os.listdir(path) 
classList = {1: ["11","19","29","113","115","169","278","301","316","317","321","324","325","338","341"],
			 2: ["1","2","3","4","5","6","7","8","9","10","12","13","14","15","16"],
			 3: ["813","817","818","819","820","821","822","824","825","826","828","829","830","832","833"],
			 4: ["635","680","683","702","704","705","706","708","709","719","720","722","723","724","726"],
			 5: ["646","751","781","794","798","799","801","812","815","823","831","839","840","841","842"],
			 6: ["995","998","999","1003","1005","1006","1007","1009","1011","1012","1013","1014","1015","1016","1019"],
			 7: ["700","730","731","732","733","735","740","744","752","754","755","756","757","759","760"],
			 8: ["262","296","304","308","337","397","401","443","445","450","466","480","513","533","534"],
			 9: ["130","131","132","133","134","135","136","137","138","139","140","141","142","143","145"],
			 10:["31","44","70","83","86","92","100","102","305","309","315","320","326","327","328"],
			 11:["240","241","243","244","245","248","250","254","255","256","258","260","275","279","295"],
			 12:["535","542","571","573","574","575","576","578","581","582","583","584","585","586","588"],
			 13:["485","520","523","526","527","529","530","531","532","536","537","538","539","540","541"]}



dict = [] 		# dictionary for all training data [dict x 1]
N = 13*15		# count training document
Nc = []			# count the number of document/in a class, [1 x class]
prior = []		# ratio of Nc/N, for every class, [1 x class]
text = []		# every term appear in each class, and frequency, [depends x class]
text_a = {}		# assistant text
T = []			# every term in document, count how many time appear in every class, [dict x class] ==>> condprob
T_a = []		# assistant T, [dict x 1]
TwT = []
TwT_a = {}
ntc = []		# count token for every class
termN = 0
DinC_a = []
DinCaa = {}		# store eveny term for every doc

################### training #######################

for classN in range(0,13):
	Nc.append(15)#maybe can be deleted
	prior.append(1/13)
	for docN in classList[classN+1]:
		term = toTerm(path+docN+'.txt')
		for t in term:
			termN=termN+1
			if(t in dict):
				dict = dict
			else:
				dict.append(t)
			if(text_a.get(t)):
				text_a[t]=text_a[t]+1
			else:
				text_a.setdefault(t,1)
			DinCaa.setdefault(t)
		DinC_a.append(DinCaa)
		DinCaa = {}


	text.append(text_a)
	text_a = {}

for classN in range(0,13):
	ntc.append(len(dict)+sum(text[classN].values()))
	for t in dict:
		if(text[classN].get(t)):
			T_a.append(text[classN][t])
		else:
			T_a.append(0)
	T.append(T_a)
	T_a = []
T = np.asarray(T)


################# (end) training #####################

################# feature selection #####################

AllN = []
dict_AClass = []
termbox = []		#n11, n10, n01, n00
for classN in range(0,13):
	for term in dict:
		for i in range(0,4):
			termbox.append(0)
		for doc in range(0,195):
			if(doc//15==classN and term in DinC_a[doc]):
				termbox[0] = termbox[0]+1
			elif(doc//15==classN and term not in DinC_a[doc]):
				termbox[1] = termbox[1]+1
			elif(doc//15!=classN and term in DinC_a[doc]):
				termbox[2] = termbox[2]+1
			elif(doc//15!=classN and term not in DinC_a[doc]):
				termbox[3] = termbox[3]+1
		dict_AClass.append(termbox)
		termbox = []
	AllN.append(dict_AClass)
	dict_AClass = []




fs = []		#feature selection [dict x class]
fs_a = []	#help feature selection [dict x 1]
five = {}
for classN in range(0,13):
	for term in range(0,len(dict)):
		n1110 = AllN[classN][term][0]+AllN[classN][term][1]
		n1101 = AllN[classN][term][0]+AllN[classN][term][2]
		n11 = AllN[classN][term][0]
		n10 = AllN[classN][term][1]
		n01 = AllN[classN][term][2]
		n00 = AllN[classN][term][3]
		numr = math.pow(n1101/N,n11)*math.pow(1-(n1101/N),n10)*math.pow(n1101/N,n01)*math.pow(1-(n1101/N),n00)
		deno = math.pow(n11/n1110,n11)*math.pow(1-(n11/n1110),n10)*math.pow(n01/(n01+n00),n01)*math.pow(1-(n01/(n01+n00)),n00)
		if(deno==0 or numr==0):
			p = 0
		else:
			p = -2*math.log(numr/deno) 
		if(len(five)>=500 and p>five[min(five)]):
			fs_a.append(p)
			inindex = min(five)
			fs_a[inindex] = 0
			five.pop(inindex)
			five.setdefault(term,p)
		elif(len(five)>=500):
			fs_a.append(0)
		elif(len(five)<500):
			fs_a.append(p)
			five.setdefault(term,p)
	fs.append(fs_a)
	fs_a = []
	five = {}
fs = np.asarray(fs)

# for c in range(0,13):
# 	count = 0
# 	for term in range(0,len(dict)):
# 		if(fs[c][term]>0):
# 			count=count+1
# 	print(count)

################# (end) feature selection #####################


####################### testing #######################

training_data = ["11","19","29","113","115","169","278","301","316","317","321","324","325","338","341",
			 "1","2","3","4","5","6","7","8","9","10","12","13","14","15","16",
			 "813","817","818","819","820","821","822","824","825","826","828","829","830","832","833",
			 "635","680","683","702","704","705","706","708","709","719","720","722","723","724","726",
			 "646","751","781","794","798","799","801","812","815","823","831","839","840","841","842",
			 "995","998","999","1003","1005","1006","1007","1009","1011","1012","1013","1014","1015","1016","1019",
			 "700","730","731","732","733","735","740","744","752","754","755","756","757","759","760",
			 "262","296","304","308","337","397","401","443","445","450","466","480","513","533","534",
			 "130","131","132","133","134","135","136","137","138","139","140","141","142","143","145",
			 "31","44","70","83","86","92","100","102","305","309","315","320","326","327","328",
			 "240","241","243","244","245","248","250","254","255","256","258","260","275","279","295",
			 "535","542","571","573","574","575","576","578","581","582","583","584","585","586","588",
			 "485","520","523","526","527","529","530","531","532","536","537","538","539","540","541"]

score = []
result = {}
w = {}
path = 'IRTM/'
fileNames = range(1,1096) 
for name in fileNames:#read every document
	name = str(name)
	if(name in training_data and name != '.DS_Store'):
		bestClass = 0
		w = toTerm('./'+path+name+'.txt')
		#print(name)
		for classN in range(0,13):
			score.append(0)
			score[classN] = math.log(prior[classN])
			for t in dict:
				if(t in w):
					index = dict.index(t)
					if(fs[classN][index]!=0):
						score[classN] = score[classN] + math.log((T[classN][index]+1)/ntc[classN])
					else:
						score[classN] = score[classN] + math.log((1)/ntc[classN])
			if score[classN]>score[bestClass] and score[classN]!=0:
				bestClass = classN
				#print(bestClass,score[bestClass])
		result.setdefault(name,bestClass+1)
		score = []
		w = {}


with open('rresult.csv', 'w') as f:
	f.write("%s,%s\n"%('Id','Value'))
	for key in result.keys():
		f.write("%s,%s\n"%(key,result[key]))
f.close()
# ####################### (end) testing #######################










