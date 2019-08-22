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
from scipy.stats import rankdata
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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

M_restaurants= np.reshape(restaurantsValues, (10,4))

# Question 3
# The most imporant idea in this project is the idea of a linear combination.
# Informally describe what a linear combination is  and how it will relate to our resturant matrix.
print('Question 3: In mathematical terms, a linear combination is where an expression is built from a set of terms (2 or more) by multiplying them.  However, it is something that we do in our lives.  One simple example is multiple people deciding which restaurant or movie to go to.  We assign weights based on how much each person cares about different factors in their preference, and we then are able to select the best option based on the end result.')

    
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
#Verify that the matrix for the people and restaurants are the same size
#M_restaurants.shape, M_people.shape

M_people_swap = np.swapaxes(M_people, 0, 1)
M_usr_x_rest = np.matmul(M_restaurants, M_people_swap)
print('Question 5This matrix(M_usr_x_rest) consists of the scores for each restaurant per row with the calculated score of each individual based on their preferences from the M_People matrix.')

# Question 6
# Sum all columns in M_usr_x_rest to get optimal restaurant for all users.  What do the entry’s represent?
tot_rest_scores = np.sum(M_usr_x_rest, 1)
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html
rank_rests = rankdata(tot_rest_scores, method='ordinal')
#Step to reverse order to rank by the highest ranking restraurant to 10 being the lowest rank and 1 being the highest rank
rank_rests = rankdata([-1 * i for i in tot_rest_scores]).astype(int)
zipped_rest = zip(restaurants.keys(), tot_rest_scores, rank_rests)
sorted_rest_ranks = sorted(zipped_rest, key=lambda x: x[1])
#Sort by rankings
sorted_rest_ranks = sorted_rest_ranks[::-1]
#restaurants.keys()
print('Question 6: The entries in this matrix represent the total score for each restaurant (sum of each persons score per restaurant).  The highest rated total score is Fuzzys with 79.85')

#Use of tabulate to print the table of restaurants, score, and ranking
print(tabulate(sorted_rest_ranks, headers=['Restaurant', 'Score', 'Ranking']))

#------------- CLASS ENDS -----------

#-------- Discuss with class mates ---------
# Question 7
# Now convert each row in the M_usr_x_rest into a ranking for each user and call it M_usr_x_rest_rank.   Do the same as above to generate the optimal 
# resturant choice.
#Jack scores for 
#                    scores        ranking 
#Tacos             74            1    
#tapas              50             3   
#bar                  70             2

tot_people_scores = np.sum(M_usr_x_rest, axis = 0)
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html
rank_people = rankdata(tot_people_scores, method='ordinal')
#Step to reverse order to rank by the highest ranking restraurant to 10 being the lowest rank and 1 being the highest rank
rank_people= rankdata([-1 * i for i in tot_people_scores]).astype(int)
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html
zipped_people = zip(restaurants.keys(), tot_people_scores, rank_people)
M_usr_x_rest_rank = sorted(zipped_people, key=lambda x: x[1])
M_usr_x_rest_rank = M_usr_x_rest_rank[::-1]
#Sort by rankings
print('Question 7: Seasons 52 is the top ranked restaurant using this methodology.')
print(tabulate(M_usr_x_rest_rank, headers=['Restaurant', 'Score', 'Ranking']))

# Question 8
# Why is there a difference between the two?  What problem arrives?  What does represent in the real world?
print('Question 8: With the second approach based on each users scores, it potentially skews the results for some people.  For example, if a person is solely interested in the cost and a restaurant has a very favorable rating based on cost, then that restaurant is likely to be favored with that calcuation.  For example, Lazy Dog was the third ranked restaurant in the initial evaluation, but fell to ninth in the second one as it is skewed by Bradyns preferences that are almost solely based on Mexican food.  Because Lazy dog has a lower score for Mexican food, the overall ranking is lowered using the second approach.')
# Question 9
# How should you preprocess your data to remove this problem.
#http://wiki.gis.com/wiki/index.php/Weighted_Linear_Combination
print('Question 9: A better option would be to normalize the scores so that they would not be so heavily weighted by people who have such dramatic preferences.')

#------------ Clustering stuff ------------
# Question 10
# Find  user profiles that are problematic, explain why?
#https://stackoverflow.com/questions/32723798/how-do-i-add-a-title-to-seaborn-heatmap
#https://seaborn.pydata.org/generated/seaborn.heatmap.html
plot_dims = (12,10)
xticklabels = people.keys()
yticklabels = restaurants.keys()
fig, ax = plt.subplots(figsize=plot_dims)
sns.heatmap(ax=ax, data=M_usr_x_rest, annot=True, xticklabels=xticklabels, yticklabels=yticklabels)
ax.set_title('Individual Preference Score Calculated by Linear Combination by Restaurant')
plt.show()
print('Question 10: The heatmap better illustrates the issue pointed out previously with individual preferences that are skewed for one category.  The darker colored blocks highlight areas where restaurant choice is skewed by these individual preferences which is wehre the score is below 3.  The most influential individual is Bradyn with scores of 1.8 for Thirsty Lion & Lava Grill being the most impactful.  He also had 5 out of 10 scores below 5 while he had 3 scores about 8 which shows the influence that he had in skewing both directions.')


# Question 11
# Think of two metrics to compute the disatistifaction with the group.
print('There are a couple of comparisons that can be done such as evaluating an individual preference with the ranking OR compare the individual linear combination scores to the group linear combination scores.  Clustering could also be used to see if an individual preference stands out compared to the rest of the group.  A heatmap could also give a visual representation of how personal vs. group preferences.')

# Question 12
# Should you split in two groups today?
#https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
#In order to better visualize a cluster, first need to reduce to 2 dimensions; choice of using PCA in this case
pca = PCA(n_components=2)
pcaPeopleMatrix = pca.fit_transform(M_people)  

#Review components and explained variance for top 2 components
print(pca.components_)
print(pca.explained_variance_)

# https://scikit-learn.org/stable/modules/clustering.html 
#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html#sphx-glr-auto-examples-cluster-plot-kmeans-digits-py
#3 clusters appears to return the best results
kmeans = KMeans(n_clusters=3)
kmeans.fit(pcaPeopleMatrix)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
fig.subplots_adjust(left=0.0625, right=0.95, wspace=0.1)

#https://matplotlib.org/users/colors.html
colors = ["g.","r.","c."]

#people.keys()
labelList = ['Jason', 'Amanda', 'Dylan', 'Bradyn', 'Marlene', 'Gerry', 'Al', 'Maryanne', 'Sara', 'Cameron']
ax.set_title('K-Means Cluster of People', size=25)
for i in range(len(pcaPeopleMatrix)):
   print ("coordinate:" , pcaPeopleMatrix[i], "label:", labels[i])
   ax.plot(pcaPeopleMatrix[i][0],pcaPeopleMatrix[i][1],colors[labels[i]],markersize=10)
   #https://matplotlib.org/users/annotations_intro.html
   #https://matplotlib.org/users/text_intro.html
   ax.annotate(labelList[i], (pcaPeopleMatrix[i][0],pcaPeopleMatrix[i][1]), size=15)
ax.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)
plt.show()
#cluster 0 is green on the bottom left, cluster 1 is red on the right side, cluster 2 is cyan (blue) in the upper left corner
print('It appears that there are at least 3 distinct groups of people.  Bradyn is close to being an outlier in the (or possibly a separate) cluster on the bottom right while Al is distinctly in his own cluster and should probably decide which group that he wants to go with based on the option that is closest to his preference')


#---- Did you understand what's going on? ---------
# Question 13
# Ok. Now you just found out the boss is paying for the meal. How should you adjust. Now what is best restaurant?
bossRestMatrix = M_restaurants[:,[0,2, 3]]
M_bossRestaurants= np.reshape(bossRestMatrix, (10,3))
bossPeopleMatrix = M_people[:,[0,2, 3]]
M_boss_people_swap = np.swapaxes(bossPeopleMatrix, 0, 1)
M_usr_x_rest_boss = np.matmul(M_bossRestaurants, M_boss_people_swap)
#M_boss_people_swap - verify size of the matrix
#M_bossRestaurants - verify size of the matrix

tot_rest_boss_scores = np.sum(M_usr_x_rest_boss, 1)
#https://docs.scipy.org/doc/scipy-0.16.0/reference/generated/scipy.stats.rankdata.html
boss_rank_rests = rankdata(tot_rest_boss_scores, method='ordinal')
#Step to reverse order to rank by the highest ranking restraurant to 10 being the lowest rank and 1 being the highest rank
boss_rank_rests = rankdata([-1 * i for i in tot_rest_boss_scores]).astype(int)
zipped_boss_rest = zip(restaurants.keys(), tot_rest_boss_scores, boss_rank_rests)
sorted_rest_boss_ranks = sorted(zipped_boss_rest, key=lambda x: x[1])
#Sort by rankings
sorted_rest_boss_ranks = sorted_rest_boss_ranks[::-1]
sorted_rest_boss_ranks
#restaurants.keys()
print('Question 13: If the boss is paying, then cost would probably not be a factor for anybody, BUT it still may be a preference for the boss if they are expensing the meal and are within a certain budget.  If the cost component is removed from the restaurant and people matrix, the top choice would be Fuzzys')
print(tabulate(sorted_rest_boss_ranks, headers=['Restaurant', 'Score', 'Ranking']))
# Question 14
# Tommorow you visit another team. You have the same restaurants and they told you their optimal ordering for restaurants.  Can you find their weight matrix?
print('Question 14: With only the ordering for restaurants, it would not be possible to get the individual matrix for people.  It may be possible to generalize the range of scores for the group, but it would not be possible to isolate the individual scores for the group.')