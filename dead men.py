# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 00:22:08 2022

@author: drsco
"""
import collections
import itertools
from collections import Counter
from itertools import permutations
import matplotlib.pylab as plt
from itertools import combinations
import numpy as np

### Dead men tell tales
dead_men_tell_tales = ['Four score and seven years ago our fathers brought forth on this',
'continent a new nation, conceived in liberty and dedicated to the',
'proposition that all men are created equal. Now we are engaged in',
'a great civil war, testing whether that nation or any nation so',
'conceived and so dedicated can long endure. We are met on a great',
'battlefield of that war. We have come to dedicate a portion of',
'that field as a final resting-place for those who here gave their',
'lives that that nation might live. It is altogether fitting and',
'proper that we should do this. But in a larger sense, we cannot',
'dedicate, we cannot consecrate, we cannot hallow this ground.',
'The brave men, living and dead who struggled here have consecrated',
'it far above our poor power to add or detract. The world will',
'little note nor long remember what we say here, but it can never',
'forget what they did here. It is for us the living rather to be',
'dedicated here to the unfinished work which they who fought here',
'have thus far so nobly advanced. It is rather for us to be here',
'dedicated to the great task remaining before us--that from these',
'honored dead we take increased devotion to that cause for which',
'they gave the last full measure of devotion--that we here highly',
'resolve that these dead shall not have died in vain, that this',
'nation under God shall have a new birth of freedom, and that',
'government of the people, by the people, for the people shall',
'not perish from the earth.']

#start dictionary
probdictdead = {}
probtransdict = {}


#1. Join everything

joinstr  = " " 
joinstr = joinstr.join(dead_men_tell_tales)

#2. Remove spaces

noSpace = joinstr.replace(" ", "")


#3. Occurrence probabilities for letters
#total probability
letterprob = len(noSpace)

letter_counts = collections.Counter(noSpace)

#letter_counts

#make dict
letter_count_dict = dict(letter_counts)


#run for loop to 
for k, l in letter_count_dict.items():
    
    percent = l / letterprob * 100
    
    dictUpdate2= {k:percent}
    
    probdictdead.update(dictUpdate2)


#4. Tell me transition probabilities for every letter pairs

#Count all
transProbs = Counter(zip(noSpace[:-1], noSpace[1:]))
transProbs=dict(transProbs)
sumprob = sum(transProbs.values())


for m, n in transProbs.items():

    probPercent = n / sumprob * 100
   
    dictUpdate3 = {m:probPercent}
    
    probtransdict.update(dictUpdate3)
#probtransdict gives us our final result for the matrix
print(probtransdict)


#6. plot graph of transition probabilities from letter to letter
#create the plot matrix
keys = np.array(probtransdict.keys())
vals = np.array(probtransdict.values())
print(keys)
print(vals)

unq_keys, key_idx = np.unique(keys, return_inverse=True)
#key_idx = key_idx.reshape(-1, 2)
print(unq_keys)
print(key_idx)

#7. Flatten a nested list

#List of lists
testList= [["red","green","blue"], ["orange","blue","red"], ["purple"], ["red","blue"]]
merged = list(itertools.chain(*testList))
print(merged)

