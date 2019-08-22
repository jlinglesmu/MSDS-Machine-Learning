# Decision making with Matrices

# This is a pretty simple assingment.  You will do something you do everyday, but today it will be with matrix manipulations.
# The problem is: you and your work firends are trying to decide where to go for lunch. You have to pick a resturant thats best for everyone.  
# Then you should decide if you should split into two groups so eveyone is happier.
# Dispite the simplictiy of the process you will need to make decisions regarding how to process the data.
# This process was thoughly investigated in the operation research community.  
# This approach can prove helpful on any number of decision making problems that are currently not leveraging machine learning.

#--------------- Assignment begins here ------------------

# You asked your 10 work friends to answer a survey. They gave you back the following dictionary object.

#names  = ['Jason', 'Amanda', 'Dylan', 'Bradyn','Marlene', 'Gerry', 'Maryanne','Al','Sara','Cameron']
#cats = ['Close Distance', 'Desire for New Experience', 'Cost', 'American', 'Asian', 'Italian', 'Indian', 'Mexican', 'Hipster Points', 'Vegetarian']
import numpy as np

people = {'Jason': {'willingness to travel':.1 ,
                  'desire for new experience':.3,
                  'cost':.1,
                  'mexican food':.5 ,
                  },
        'Amanda': {'willingness to travel':.3 ,
                  'desire for new experience':.4,
                  'cost':.1,
                  'mexican food': .2,
                  },
        'Dylan': {'willingness to travel':.1 ,
                  'desire for new experience':.2,
                  'cost':.1,
                  'mexican food': .6,
                  },
        'Bradyn': {'willingness to travel': .05,
                  'desire for new experience':.1,
                  'cost':.05,
                  'mexican food': .8,
                  },
        'Marlene': {'willingness to travel': .2,
                  'desire for new experience':.2,
                  'cost':.25,
                  'mexican food': .35,
                  },
        'Gerry': {'willingness to travel': .4,
                  'desire for new experience':.4,
                  'cost':.15,
                  'mexican food': .05,
                  },
        'Al': {'willingness to travel': .1,
                  'desire for new experience':.05,
                  'cost':.8,
                  'mexican food': .05,
                  },
        'Maryanne': {'willingness to travel':.45 ,
                  'desire for new experience':.45,
                  'cost':.05,
                  'mexican food':.05,
                  },
        'Sara': {'willingness to travel':.05,
                  'desire for new experience':.25,
                  'cost':.3,
                  'mexican food':.4,
                  },
        'Cameron': {'willingness to travel':.3,
                  'desire for new experience':.3,
                  'cost':.3,
                  'mexican food':.1,
                  },

          }
#people = {'Jane': {'willingness to travel':
#                  'desire for new experience':
#                  'cost':
#                  'indian food':
#                  'mexican food':
#                  'hipster points':
#                  'vegetarian'           }         }

# Question 1
# Transform the user data into a matrix(M_people). Keep track of column and row ids.
# Added: normalize the points for each user -- make their preferences add to 1 in the actual weights matrix you use for analysis.

# convert each person's values to a list

peopleKeys, peopleValues = [], []
lastKey = 0
for k1, v1 in people.items():
    row = []
    
    for k2, v2 in v1.items():
        peopleKeys.append(k1+'_'+k2)
        if k1 == lastKey:
            row.append(v2)      
            lastKey = k1
            
        else:
            peopleValues.append(row)
            row.append(v2)   
            lastKey = k1
            
M_people = np.array(peopleValues)


# Next you collected data from an internet website. You got the following information.
# Added: make these scores /10, on a scale of 0-10, where 10 is good. So, 10/10 for distance means very close. 
#resturants  = {'flacos':{'distance' :
#                        'novelty' :
#                        'cost':
#                        'average rating':
#                        'cuisine':
#                        'vegitarians'
#                        }    }
#1 is bad, 10 is great
restaurants = {'Thirsty Lion': {'willingness to travel':9,
                  'new experience':2,
                  'cost':7,
                  'mexican food':1 ,
                  },
        'Lava Grill': {'willingness to travel':10,
                  'new experience':4,
                  'cost':2,
                  'mexican food': 1,
                  },
        'Lazy Dog': {'willingness to travel':9 ,
                  'new experience':7,
                  'cost':8,
                  'mexican food': 4,
                  },
        'Big Fish': {'willingness to travel': 4,
                  'new experience':8,
                  'cost':5,
                  'mexican food': 2,
                  },
        'Anamias': {'willingness to travel': 3,
                  'new experience':5,
                  'cost':7,
                  'mexican food': 10,
                  },
        'Fuzzys': {'willingness to travel': 9,
                  'new experience':4,
                  'cost':9,
                  'mexican food': 10,
                  },
        'Mi Cocina': {'willingness to travel':5,
                  'new experience':6,
                  'cost':6,
                  'mexican food': 9,
                  },
        'Seasons 52': {'willingness to travel':2,
                  'new experience':10,
                  'cost':4,
                  'mexican food':1,
                  },
        'Winewood': {'willingness to travel':8,
                  'new experience':9,
                  'cost':3,
                  'mexican food':3,
                  },
        'Cafe Italia': {'willingness to travel':7,
                  'new experience':8,
                  'cost':7,
                  'mexican food':2,
                  },

          }
restaurantsKeys, restaurantsValues = [], []

for k1, v1 in restaurants.items():
    for k2, v2 in v1.items():
        restaurantsKeys.append(k1+'_'+k2)
        restaurantsValues.append(v2)
            

#-------- Data processing ends --------
#-------- Start with 2 numpy matrices if you're not excited to do data processing atm ------ 
# Question 2
# Transform the restaurant data into a matrix(M_resturants) use the same column index.
#restaurantsMatrix = np.reshape(restaurantsValues, (2,4))

restaurantsValues

# Question 3
# The most imporant idea in this project is the idea of a linear combination.
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.
print('In mathematical terms, a linear combination is where an expression is built from a set of terms (2 or more) by multiplying them.  However, it is something that we do in our lives.  One simple example is multiple people deciding which restaurant or movie to go to.  We assign weights based on how much each person cares about different factors in their preference, and we then are able to select the best option based on the end result.')

    
# Question 4
# Choose a person and compute(using a linear combination) the top restaurant for them.  What does each entry in the resulting vector represent.
#'Jason': {'willingness to travel':.1 ,
#                  'desire for new experience':.3,
#                  'cost':.1,
#                  'mexican food':.5 ,
#                  },
#Jason's Scores
#Thirsty Lion = 2.7
.1*9 + .3*2 + .1*7 + .5*1
#Lava Grill = 2.9
.1*10 + .3*4 + .1*2 + .5*1
#Lazy Dog = 5.8
.1*9 + .3*7 + .1*8 + .5*4
#Big Fish = 4.3
.1*4 + .3*8 + .1*5 + .5*2
#Anamias = 7.5
.1*3 + .3*5 + .1*7 + .5*10
#Fuzzys = 8
.1*9 + .3*4 + .1*9 + .5*10
#Mi COcina = 7.4
.1*5 + .3*6 + .1*6 + .5*9
#Seasons 52 = 4.1
.1*2 + .3*10 + .1*4 + .5*1
#Winewood = 5.3
.1*8 + .3*9 + .1*3 + .5*3
#Cafe Italia = 4.8
.1*7 + .3*8 + .1*7 + .5*2
print('Each entry in the resulting vector represents the score for each restaurant for the individual that we are reviewing.  The highest ranked restaurant for Jason is Fuzzys with a score of 8 out of 10')
# Question 5
# Next compute a new matrix (M_usr_x_rest  i.e. an user by restaurant) from all people.  What does the a_ij matrix represent?
restaurantsMatrix.shape, peopleMatrix.shape

#M_usr_x_rest_rank_temp = []
#for person in M_usr_x_rest:
#   M_usr_x_rest_rank_temp.append(rankdata(person,method = 'ordinal'))
#M_usr_x_rest_rank = np.array(M_usr_x_rest_rank_temp)
# Question 6
# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?

#------------- CLASS ENDS -----------

#-------- Discuss with class mates ---------

# Question 7
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal resturant choice.
#Jack scores for 
#                    scores        ranking 
#Tacos             74            1    
#tapas              50             3   
#bar                  70             2

# Question 8
# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?

# Question 9
# How should you preprocess your data to remove this problem.

#------------ Clustering stuff ------------

# Question 10
# Find  user profiles that are problematic, explain why?

# Question 11
# Think of two metrics to compute the disatistifaction with the group.

# Question 12
# Should you split in two groups today?

#---- Did you understand what's going on? ---------

# Question 13
# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?

# Question 14
# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
