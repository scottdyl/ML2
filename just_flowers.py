# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 21:27:20 2022

@author: drsco
"""



import itertools
from collections import Counter
from itertools import permutations
import matplotlib.pylab as plt
from itertools import combinations
import numpy as np


###  Flowers
flower_orders=['W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
               'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
               'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B',
               'W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R/B','W/R','W/R','W/R',
               'W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R','W/R',
               'W/R','W/R','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y','R/V/Y',
               'R/V/Y','R/V/Y','R/V/Y','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V','W/R/V',
               'W/R/V','W/R/V','W/R/V','W/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V',
               'W/N/R/V','W/N/R/V','W/N/R/V','W/N/R/V','W/R/B/Y','W/R/B/Y','W/R/B/Y','W/R/B/Y',
               'W/R/B/Y','W/R/B/Y','B/Y','B/Y','B/Y','B/Y','B/Y','R/B/Y','R/B/Y','R/B/Y','R/B/Y',
               'R/B/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/N/R/B/V/Y','W/G',
               'W/G','W/G','W/G','R/Y','R/Y','R/Y','R/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','N/R/V/Y','W/R/B/V',
               'W/R/B/V','W/R/B/V','W/R/B/V','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','W/N/R/V/Y','N/R/Y','N/R/Y',
               'N/R/Y','W/V/O','W/V/O','W/V/O','W/N/R/Y','W/N/R/Y','W/N/R/Y','R/B/V/Y','R/B/V/Y','R/B/V/Y','W/R/V/Y',
               'W/R/V/Y','W/R/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/R/B/V/Y','W/N/R/B/Y','W/N/R/B/Y','W/N/R/B/Y','R/G','R/G',
               'B/V/Y','B/V/Y','N/B/Y','N/B/Y','W/B/Y','W/B/Y','W/N/B','W/N/B','W/N/R','W/N/R','W/N/B/Y','W/N/B/Y','W/B/V/Y',
               'W/B/V/Y','W/N/R/B/V/Y/G/M','W/N/R/B/V/Y/G/M','B/R','N/R','V/Y','V','N/R/V','N/V/Y','R/B/O','W/B/V','W/V/Y','W/N/R/B',
               'W/N/R/O','W/N/R/G','W/N/V/Y','W/N/Y/M','N/R/B/Y','N/B/V/Y','R/V/Y/O','W/B/V/M','W/B/V/O','N/R/B/Y/M',
               'N/R/V/O/M','W/N/R/Y/G','N/R/B/V/Y','W/R/B/V/Y/P','W/N/R/B/Y/G','W/N/R/B/V/O/M','W/N/R/B/V/Y/M',
               'W/N/B/V/Y/G/M','W/N/B/V/V/Y/P']


#define future lists
doublelists = []
testlist = []



#Counter objects below
colcount = Counter()
flowerCounter = Counter()
paircounter = Counter()
tripletCounter = Counter()

#1. Build a counter object and use the counter and confirm they have the same values.

for f in flower_orders:
     flowerCounter[f] += 1

print(flowerCounter)

#2. Count how many objects have color W in them.
#convert list of strings to a list of lists and break the obj up by /
for f in flower_orders:
    mylist = list(f.split("/")) 
    doublelists.append(mylist)
    
wordCount = 0
for f in doublelists:
    wordCount += f.count('W')
print(wordCount)

#3. Make histogram of colors

interList = (itertools.chain.from_iterable(doublelists))
interList = list(interList)

#loop to iterate
for col in interList:
    colcount[col] += 1

colcount
colcountdict = dict(colcount)

plt.hist(colcountdict.values(),bins=10)
plt.show()

#4. Rank the pairs of colors in each order regardless of how many colors are in an order.

doubles = []
for color in flower_orders:
    for letter in permutations(color.split("/"),2):
        doubles.append(letter)
print(doubles)
        


#5. Rank the triplets of colors in each order regardless of how many colors are in an order.
tripples  = []
for color in flower_orders:
    for letter in permutations(color.split("/"),3):
        tripples.append(letter)
print(tripples)

#6. Make dictionary color for keys and values are what other colors it is ordered with.
#I assume this to mean make a dictionary where the key is the first value and the other colors follow
# colors are key and the other values it is ordered with are in the dict
#I may be wrong but given the next question I assume you mean pairs here

colorDict = {}
#split the list of strings into a list of touples
#res = [tuple(map(str, sub.split(","))) for sub in flower_orders]
#print(res)
#print(Convert(res,colorDict))
#this isn't 

def Convert(tup, di):
    for a, b in tup:
        di.setdefault(a, []).append(b)
    return di
      

print (Convert(doubles, colorDict))

#doubles count dict
#this is now the counter of every double
#this will help us with the graph
dubCounter = Counter()
for f in doubles:
     dubCounter[f] += 1
#convert to dict
dictdubcounter = dict(dubCounter)
print(dictdubcounter)

# 7. Make a graph showing the probability of having an edge between two colors based on how often they co-occur.
#probability = how likely an event is to occur

