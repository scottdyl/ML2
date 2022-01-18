# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 13:02:43 2022

@author: drsco
"""

### Imports
import collections
import itertools
from collections import Counter
from itertools import permutations
import matplotlib.pylab as plt
from itertools import combinations

### Lists

# pratice list
testList = ["red","green", "blue"]

#sppend()
testList.append("yellow")
print(testList)
#this adds and item to the end of a list

#extend()
colorList = ["purple","blue"]
testList.extend(colorList)
print(testList)
#adds two lists together


#index()
print(testList.index("green"))
#gives the number position of the item in the list starting from 0 to inf

#index(value, integer)
print(testList.index("blue",1))
#this searches throuogh the list and retuns the index of the list element starting at the 
#index number given

#insert(position)
testList.insert(5,"white")
print(testList)
#insterts the element at that index position

#remove()
testList.remove("blue")
#removes all of that element in the list

#pop()
testList.pop()
print(testList)
#removes the last item in the list

#count()
print(testList.count('yellow'))
#counts the occurance of that element in the list

#reverse()
testList.reverse()
print(testList)
#flips the order of the list

#sort()
testList.sort()
print(testList)
#sorts the list

#[1]+[1]
#This function joins two lists to give us a lists within a list
newList = [testList]+[colorList]
print(newList)

#[2]*2
#This function duplicates a list
twocolorList = [colorList]*2
print(twocolorList)

listex = [1,2][1:]
#This function allows a users to retrieve a list from an index
print(listex)
#for this case it will return 2 since 1 has an index of 0
#lists index is always count-1 since the first index is 0

listloop = [x for x in [2,3]]
#This function will loop through x for everything in the in statement
#for this case 
print(listloop)


iflist = [x for x in [1,2]if x==1]
#This line loops through the list between index 1 and 2 while looking for the value 1
print(iflist)

forlist = [y*2 for x in [[1,2],[3,4]] for y in x]
#This line is a loop that multiples whatever value is y by 2 inside a list of lists and then outputs a list 
print(forlist)
#in this case it counts by evens

#A=[1]
#This function assigns the list to the variable A and in this case that is a value of 1
A=[1]
print(A)


#### Tuple

testTuple = ("red","green", "blue")

#index()
print(testTuple.index("red"))
#This function returns the index of a specified value

#build a dictionary from tuples
#This function turns a list of tuples into a dictionary of tuples
tupleList = dict([('red', 1), ('green', 2), ('blue', 3)])

#unpack tuples
for n, v in tupleList.items():
    print(n,v)
#This function unpacks the tuples

#### Dicts

#a_dict= {'I hate':'you', 'You should':’leave’}
a_dict= {'I hate':'you', 'You should':'leave'}
print(a_dict)
#this creates a dictionary with a key and a value
#here we have I hate is the key to you

#keys()
print("keys")
print(a_dict.keys())
#This function prints the keys to the dictionary 

#items()
print(a_dict.items())
#This function prints the items within the dictionary

#hasvalues()
print("hasvalues has been sunset")
#print(a_dict.hasvalues())
#This function has been sunset


#'never' in a dict
print('never' in a_dict)
# this function checks to see if the key "never" is in the dict object
#returns true or false

#del a_dict['me']
#del a_dict['me']
#This function should remove "me" from the dict as a key
#in this case me isn't a key so it will error out

#a_dict.clear()
a_dict.clear()
print(a_dict)
#This function clears the dictionary
#prints a blank dict object

### my turn using dir()

print(a_dict.get("I hate"))
#this returns the key value from the dict

print(a_dict.keys())
#this will print out all the keys

print(a_dict.setdefault("You should"))
#this will set the default key

print(a_dict.pop("You should"))
#this will pop off the term specified or the last term if not given one


### sets: 11

#Define a set
testSet = {"red","green", "blue"}

#add()
testSet.add("orange")
#this adds something to the se
print(testSet)

#clear()
testSet.clear()
#poof! this removes everything from the set
print(testSet)

#make a new set since we cleared it
testSet = {"red","green", "blue"}

#copy()
testSet2=testSet.copy()
#this returns a copy of the set

testSet.add("orange")
#adding a new element to test set for the difference

#difference()
testSet.difference(testSet2)
#returns a set that is the difference between two sets

#discard()
testSet.discard("orange")
#this removes an element from the set

#intersection()
print(testSet.intersection(testSet2))
#this will look at the intersection of the two sets

#issubset()
print(testSet2.issubset(testSet))
#returns true of all in testSet is in testSet2

#pop()
testSet.pop()
#pops off the last element in the set
print(testSet)

#remove()
testSet.remove("green")
#removes a certain value from the set
print(testSet)

#union()
testSet.union(testSet2)
#joins two sets

#update()
testSet.update(testSet2)
print(testSet)
#updates the set to include the values in testSet2


####Strings
testString="Colors"
testString2="taste the rainbow"

#capitalize()
print(testString2.capitalize())
#makes the strong proper

#casefold()
print(testString.casefold())
#makes the string all lower case

#center()
print(testString2.center(7))
#Justifies the string x characrters

#count()
print(testString2.count("a"))
#counts the number of charactors

#encode()
print("encode")
print(testString.encode())
#method returns an encoded version of the given string.

#find()
print(testString2.find("r"))
#Returns first occurance of letter

#partition()
print(testString2.partition("is"))
#this splits the string by the value and returns the before and after as tuples

#replace()
print("replace")
replaceTest = testString2.replace('rainbow', 'candy')
print(replaceTest)
#this replaces the matching string with another value

#split()
print(testString2.split(' '))
#splits the text by the specified value

#title()
print(testString2.title())
#this makes it like a title. Cap after each space

#zfill()
print(testString2.zfill(3))
#method returns a copy of the string with '3' characters padded to the left.

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

### 10 business questions
#1) What is the most popular?
#2) Could there be seasonality?
#3) Are alike colors simular in sales?
#4) Is it the combinations or the colors itself?
#5) Will a sale improve our top peformer?
#6) Could type of flower be more important than color?
#7) What is the shape of our data?
#8) What are our botom peformers?
#9) Can you work overtime? We need more people.
#10) Can we account for seasonality to predcit order sizes?
#11) Why is this important?

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


