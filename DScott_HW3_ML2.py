import numpy as np
#help with 0s like
#https://numpy.org/doc/stable/reference/generated/numpy.zeros_like.html
import scipy.stats as ss
from sklearn.cluster import KMeans

# Decision Making With Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch. You have to pick a restaurant that’s best for everyone.  
# Then you should decided if you should split into two groups so everyone is happier.  

# Despite the simplicity of the process you will need to make decisions regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  
# This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.  

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object. 
# people and restrant infastruction was provided by Andy Heroy
# some scores were changed for fun :)
people = {'Jane': {'willingness to travel': 1,
				  'desire for new experience':5,
				  'cost':1,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 1,
				  },
		  'Bob': {'willingness to travel': 5,
				  'desire for new experience':2,
				  'cost':0,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 1,
				  },
		  'Mary': {'willingness to travel': 4 ,
				  'desire for new experience': 4,
				  'cost': 5,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 3,
				  },
		  'Mike': {'willingness to travel': 5,
				  'desire for new experience': 1,
				  'cost': 4,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 3,
				  },
		  'Alice': {'willingness to travel': 4,
				  'desire for new experience': 5,
				  'cost': 1,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 2,
				  },
		  'Skip': {'willingness to travel': 2,
				  'desire for new experience': 2,
				  'cost': 5,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 2,
				  },
		  'Kira': {'willingness to travel': 1,
				  'desire for new experience': 2,
				  'cost': 5,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 1,
				  },
		  'Moe': {'willingness to travel': 1,
				  'desire for new experience': 3,
				  'cost': 3,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 5,
				  },
		  'Sara': {'willingness to travel': 2,
				  'desire for new experience': 5,
				  'cost': 1,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 1,
				  },
		  'Tom': {'willingness to travel': 5,
				  'desire for new experience': 1,	
				  'cost': 1,
				  #'indian food':1,
				  #'Mexican food':1,
				  #'hipster points':3,
				  'vegetarian': 1,
				  }                  
		  }         

# Transform the user data into a matrix(M_people). Keep track of column and row ids.   
people_names = list(people)
people_cols =list(people[people_names[1]])

M_people = np.zeros((len(people_names),len(people_cols)))
for i, p in enumerate(people):
	M_people[i,] = np.array(list(people[p].values()))

print('M_people: \n',M_people)
# Next you collected data from an internet website. You got the following information.

restaurants  = {'flacos':{'distance' : 2,
						'novelty' : 3,
						'cost': 4,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 5
						},
			  'Joes':{'distance' : 5,
						'novelty' : 1,
						'cost': 5,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 3
					  },
			  'Poke':{'distance' : 4,
						'novelty' : 2,
						'cost': 4,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 4
					  },                      
			  'Sush-shi':{'distance' : 4,
						'novelty' : 3,
						'cost': 4,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 4
					  },
			  'Chick Fillet':{'distance' : 3,
						'novelty' : 2,
						'cost': 5,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 5
					  },
			  'Mackie Des':{'distance' : 2,
						'novelty' : 3,
						'cost': 4,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 3
					  },
			  'Michaels':{'distance' : 2,
						'novelty' : 1,
						'cost': 1,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 5
					  },
			  'Amaze':{'distance' : 3,
						'novelty' : 5,
						'cost': 2,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 4
					  },
			  'Kappa':{'distance' : 5,
						'novelty' : 1,
						'cost': 2,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 3
					  },
			  'Mu':{'distance' : 3,
						'novelty' : 1,
						'cost': 5,
						#'average rating': 5,
						#'cuisine': 5,
						'vegetarian': 3
					  }                      
}
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
rest_names = list(restaurants)
rest_cols =list(restaurants[rest_names[1]])
M_restaurants = np.zeros((len(rest_names),len(rest_cols)))
for i, r in enumerate(restaurants):
	M_restaurants[i,] = np.array(list(restaurants[r].values()))

print('M_restaurants: \n',M_restaurants)

#get rank from the ss package comes partly from this scipy.org page
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rankdata.html
def getRank(data):
	return(ss.rankdata(len(data)-ss.rankdata(data)+2,method='min'))

# The most important idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.
print('Linear Combination: ','It is math! Or in other words it is an expression made of a set of terms (x) multiplied by a constant (beta)')

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 
# I bet Skip knows where some fire tacos are
p = people_names.index('Skip')
skip=np.dot(M_people[p,],M_restaurants.T)

print('Skip:', skip)
print('Each entry vector represents: ','the weights of Skip\'s prefered restraunt and which ones he would like best')

# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 
M_usr_x_rest=np.dot(M_people,M_restaurants.T)

print('M_usr_x_rest: \n',M_usr_x_rest)
print('Each a_ij matrix represents: ','each peron\'s weight for each restraunt preference. \n for example 1:1 is Jane and her value for flacos')

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?
rest_score  = M_usr_x_rest.sum(axis=1) 
rank=0
for i in reversed(np.argsort(rest_score)):
    rank=rank+1
    print(rank,rest_names[i], ":", rest_score[i])  
rest_rank = getRank(rest_score)
best_r_id = np.flatnonzero(rest_rank ==1)[0]
best_r_score = rest_score[best_r_id]
best_r_name = rest_names[best_r_id]
 
print("The winner is! ",best_r_name," : ", best_r_score)
print('Each entery represents: ', 'the total weighted scores for all restraunts of all the people giving us the best restraunt for the group')

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   
# Do the same as above to generate the optimal restaurant choice.  
M_usr_x_rest_rank = np.zeros_like(M_usr_x_rest)
for r in range(M_usr_x_rest.shape[1]):
    M_usr_x_rest_rank[:,r] = getRank(M_usr_x_rest[:,r])
rest_score2   = np.sum(M_usr_x_rest_rank,axis=1)
c=0
for i in (np.argsort(rest_score2)):
    c+=1
    print(c,". Restaurant",rest_names[i], " tot Score:", rest_score2[i])
rest_rank2 = ss.rankdata(rest_score2,method='min')
best_r_id2 = np.flatnonzero(rest_rank2 ==1)[0]
best_r_score2 = rest_score2[best_r_id2]
best_r_name2 = rest_names[best_r_id2]

print("The winner is! ",best_r_name2," : ", best_r_score2)
#print('Each entery represents: ', '')

# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?
print('I believe that the first methoid simply takes the best of all options where the second accounts for those who really really \
	want certain aspects. Making the second methoid less soseptial to outliars. In the real would world this would be equal to that one asshole who \
		really wants a certain place to eat but does not like the system of choosing')

# How should you preprocess your data to remove this problem. 
print("I think logging the data will do just fine. This will prevent skewness")
log_r_score  = [np.log(M_usr_x_rest[x,:]).sum() for x in range(M_usr_x_rest.shape[0])]
c=0
print('log: \r')
for i in reversed(np.argsort(log_r_score)):
    c+=1
    print(c,rest_names[i], ":", round(log_r_score[i],3)) 

# Find  user profiles that are problematic, explain why?
# bias people are hard to find in faked data
_=rest_names.index(best_r_name)
print(best_r_name, " ranked",M_usr_x_rest_rank[_,:]," final rank ",rest_rank2[_])
print('poke ranked so high on each person list. I do not think this is correct therefore problematic' )

# Think of two metrics to compute the disatistifaction with the group.  
# metric 1 selected resraunt vs what they wanted
rest_best_max_score = M_usr_x_rest[best_r_id,:].max()
people_dissatisfaction =  abs(M_usr_x_rest[best_r_id,:] - rest_best_max_score)
i=0
for x in reversed(np.argsort(people_dissatisfaction) ) :
    i=i+1
    print(i,". ",people_names[x],":( ",people_dissatisfaction[x])
	# for example kira is not happy with this choice
# metric 2 could be to calculate the overall dissatisfaction with the picked restruant
rest_mean_score = M_usr_x_rest.mean(axis=1)
people_no_like  =  np.mean(abs(M_usr_x_rest.T - rest_mean_score),axis=0)
i=0
for x in reversed(np.argsort(people_no_like) ) :
    i=i+1
    print(i,".",rest_names[x],"overall :( ",round(people_no_like[x],2))

# Should you split in two groups today? 
# just based on the last question I will say yes but lets see using math!
# if the two clusters match then they should not split up
# we want to use the log to avoid skewness
X = M_usr_x_rest.T
k=2 #for kmeans 
km = KMeans(n_clusters=k, random_state=42,n_init=10,max_iter=500).fit(X)

def bestRest(data):
    rest_score_  = np.sum((data),axis=1)
    print(rest_score_)
       
    rest_rank_  = getRank(rest_score_)
    rest_best_id_  = np.flatnonzero(rest_rank_ ==1)[0]
    rest_best_score_  = log_r_score[rest_best_id_]
    rest_best_name_  = rest_names[rest_best_id_]

    rest_mean_score_ = (data[rest_best_id_,:]).mean()
    
    diss  =  np.mean(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    diss_sd  =  np.std(abs(data[rest_best_id_].T - rest_mean_score_),axis=0)
    
    print("Gruop 1 ",rest_best_name_,":", round(rest_best_score_,3))
    return(rest_best_id_)
    
for i in range(k):
    print("CLUSTER ",i)
    data=np.dot(M_restaurants,M_people[km.labels_ == i].T)   
    
    print("people:", data.shape[1]," ->",np.where(km.labels_==i))
    _=bestRest(data)
print('looks like according to knn that they should stay as a group and go to sush-shi')
# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?
print("this just means that budget isn't a factor too much so we should adjust everyones cost to max")
boss_people = M_people.copy()
boss_people[:,people_cols.index('cost')]=5
M_usr_x_rest2=np.dot(boss_people,M_restaurants.T)

rest2_score  = [M_usr_x_rest2[:,x].sum() for x in range(M_usr_x_rest2.shape[1])]

for i in np.argsort(rest2_score):
    print("Restaurant",rest_names[i], " tot Score:", rest2_score[i])  

rank_r_2 = getRank(rest2_score)
best_r_id2 = np.flatnonzero(rank_r_2 ==1)[0]
best_r_score_rank2 = rest2_score[best_r_id2]
best_r_name2 = rest_names[best_r_id2]

print("if the boss is buying this is whhere we go",best_r_name2,":", best_r_score_rank2)


# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 




