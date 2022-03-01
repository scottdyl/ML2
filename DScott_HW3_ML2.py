import numpy as np
# Decision Making With Matrices

# This is a pretty simple assignment.  You will do something you do everyday, but today it will be with matrix manipulations. 

# The problem is: you and your work friends are trying to decide where to go for lunch. You have to pick a restaurant that’s best for everyone.  
# Then you should decided if you should split into two groups so everyone is happier.  

# Despite the simplicity of the process you will need to make decisions regarding how to process the data.
  
# This process was thoroughly investigated in the operation research community.  
# This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.  

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.  
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

print(M_people)
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

print(M_restaurants)




# The most important idea in this project is the idea of a linear combination.  
# Informally describe what a linear combination is  and how it will relate to our restaurant matrix.
print('Linear Combination: ','')

# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent? 


# Next, compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent? 

# Sum all columns in M_usr_x_rest to get the optimal restaurant for all users.  What do the entries represent?

# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal restaurant choice.  

# Why is there a difference between the two?  What problem arrives?  What does it represent in the real world?

# How should you preprocess your data to remove this problem. 

# Find  user profiles that are problematic, explain why?

# Think of two metrics to compute the disatistifaction with the group.  

# Should you split in two groups today? 

# Ok. Now you just found out the boss is paying for the meal. How should you adjust? Now what is the best restaurant?

# Tomorrow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix? 




