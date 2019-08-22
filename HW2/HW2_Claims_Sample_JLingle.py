# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 17:31:47 2019
Jason Lingle
HW2 - Claims Sample Analysis with structured arrays
"""

import functools
import io
import numpy as np
import sys
import numpy.lib.recfunctions as rfn
import numpy_groupies as npg
import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
from sklearn.metrics import accuracy_score
from itertools import product
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from matplotlib.backends.backend_pdf import PdfPages#import resource
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier

#import time
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.model_selection import train_test_split
#from sklearn.model_selection import KFold

#resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
sys.setrecursionlimit(10**6)

#https://stackoverflow.com/questions/5061582/setting-stacksize-in-a-python-script/16248113#16248113

#NumPy Cheatsheet - https://s3.amazonaws.com/assets.datacamp.com/blog_assets/Numpy_Python_Cheat_Sheet.pdf


#to fix a bug in np.genfromtxt when Python Version (sys.version_info) is 3 or greater. 
# https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl


genfromtxt_old = np.genfromtxt
@functools.wraps(genfromtxt_old)
def genfromtxt_py3_fixed(f, encoding="utf-8", *args, **kwargs):
  if isinstance(f, io.TextIOBase):
    if hasattr(f, "buffer") and hasattr(f.buffer, "raw") and \
    isinstance(f.buffer.raw, io.FileIO):
      # Best case: get underlying FileIO stream (binary!) and use that
      fb = f.buffer.raw
      # Reset cursor on the underlying object to match that on wrapper
      fb.seek(f.tell())
      result = genfromtxt_old(fb, *args, **kwargs)
      # Reset cursor on wrapper to match that of the underlying object
      f.seek(fb.tell())
    else:
      # Not very good but works: Put entire contents into BytesIO object,
      # otherwise same ideas as above
      old_cursor_pos = f.tell()
      fb = io.BytesIO(bytes(f.read(), encoding=encoding))
      result = genfromtxt_old(fb, *args, **kwargs)
      f.seek(old_cursor_pos + fb.tell())
  else:
    result = genfromtxt_old(f, *args, **kwargs)
  return result

if sys.version_info >= (3,):
  np.genfromtxt = genfromtxt_py3_fixed

## HW notes:
'''    
A medical claim is denoted by a claim number ('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number ('Claim.Line.Number').

1. J-codes are procedure codes that start with the letter 'J'.

     A. Find the number of claim lines that have J-codes.
     51,029
     
     B. How much was paid for J-codes to providers for 'in network' claims?
     #Provider.Payment.Amount = 2,417,221

     C. What are the top five J-codes based on the payment to providers?
    J1745	  434,232
    J0180	  299,777
    J9310	  168,631
    J3490	  90,199
    J1644	  81,909



2. For the following exercises, determine the number of providers that were paid for at least one J-code. Use the J-code claims for these providers to complete the following exercises.
    A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each 
    provider versus the number of paid claims.
    B. What insights can you suggest from the graph?
    C. Based on the graph, is the behavior of any of the providers concerning? Explain.

3. Consider all claim lines with a J-code.
     A. What percentage of J-code claim lines were unpaid?
     B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
     C. How accurate is your model at predicting unpaid claims?
     D. What data attributes are predominately influencing the rate of non-payment?
'''
#inpstream = io.open('data\claim.sample.csv','r')
#creates array of structured arrays
types = ['S8', 'f8', 'i4', 'i4', 'S14', 'S6', 'S6', 'S6', 'S4', 'S9', 'S7', 'f8',
         'S5', 'S3', 'S3', 'S3', 'S3', 'S3', 'f8', 'f8', 'i4', 'i4', 'i4', 'S3', 
         'S3', 'S3', 'S4', 'S14', 'S14']

CLAIMS = np.genfromtxt('claim.sample.csv', dtype=types, delimiter=',', names=True, 
                       usecols=[0,1,2,3,4,5,
                                6,7,8,9,10,11,
                                12,13,14,15,16,
                                17,18,19,20,21,
                                22,23,24,25,26,
                                27,28])

#String Operations in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.char.html
#Sorting, Searching, and Counting in NumPy - https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.sort.html
# You might see issues here: https://stackoverflow.com/questions/23319266/using-numpy-genfromtxt-gives-typeerror-cant-convert-bytes-object-to-str-impl

test = 'J'
#Encode needed here because we're looking for a string to ensure that we don't get an error
test = test.encode()
#Returns a value of -1 for values that do not have the J--we need to find the rows without a -1
#We only want the non -1 values 2/13/2019: Changed code from !=-1 to ==1 b/c only 0 values should be returned (null values are included in the In Network Claims query, returning incorrect results)
#We can use the index and subset the rows that start with J
JcodesIndexes = np.flatnonzero(np.core.defchararray.find(CLAIMS['ProcedureCode'],test)==1)
#Using those indexes, subset CLAIMS to only Jcodes--this is the value that will provide the count of jcodes with claims
Jcodes = CLAIMS[JcodesIndexes]
#1A. Find the number of claim lines that have J-codes.
print('Question 1A.  The number of claim lines that have a J-Code is:', len(JcodesIndexes))





#1B. How much was paid for J-codes to providers for 'in network' claims?
#https://www.machinelearningplus.com/python/101-numpy-exercises-python/
#Apply Same Logic as 1B but include the in-network filter
in_network = 'I'
#Encode needed here because we're looking for a string to ensure that we don't get an error
in_network = in_network.encode()
#We only want the non -1 values 2/13/2019: Changed code from !=-1 to ==1 b/c only 0 values should be returned
InNetworkIndexes = np.flatnonzero(np.core.defchararray.find(Jcodes['InOutOfNetwork'],in_network)==1)
InNetworkAmount = Jcodes[InNetworkIndexes]['ProviderPaymentAmount'].sum().round(2)
print('Question 1B.  The amount paid for J-Codes to providers for in network claims is:', InNetworkAmount)



#sum_per_jcode_id = npg.aggregate(Jcodes, InNetworkAmount['ProviderPaymentAmount'], func='sum')

#C. What are the top five J-codes based on the payment to providers?
    #Sorted Jcodes, by ProviderPaymentAmount
Sorted_Jcodes = (np.sort(Jcodes, order='ProviderPaymentAmount'))[::-1]
Sorted_Jcodes = np.sort(Jcodes, order='ProviderPaymentAmount')
Sorted_Jcodes = Sorted_Jcodes[::-1]
    #ProviderPayments = Sorted_Jcodes['ProviderPaymentAmount']
    #Jcodes = Sorted_Jcodes['ProcedureCode']
#Join arrays together
    #arrays = [Jcodes, ProviderPayments]

#https://www.numpy.org/devdocs/user/basics.rec.html
    #Jcodes_with_ProviderPayments = rfn.merge_arrays(arrays, flatten = True, usemask = False)

# Print JCode & Provider Payments (displays by line--needs to be grouped)
    #print(Jcodes_with_ProviderPayments[:5])

#http://esantorella.com/2016/06/16/groupby/
#A fast GroupBy class
class Groupby:
    def __init__(self, keys):
        _, self.keys_as_int = np.unique(keys, return_inverse = True)
        self.n_keys = max(self.keys_as_int)
        self.set_indices()
        
    def set_indices(self):
        self.indices = [[] for i in range(self.n_keys+1)]
        for i, k in enumerate(self.keys_as_int):
            self.indices[k].append(i)
        self.indices = [np.array(elt) for elt in self.indices]
        
    def apply(self, function, vector, broadcast):
        if broadcast:
            result = np.zeros(len(vector))
            for idx in self.indices:
                result[idx] = function(vector[idx])
        else:
            result = np.zeros(self.n_keys)
            for k, idx in enumerate(self.indices):
                result[self.keys_as_int[k]] = function(vector[idx])

        return result


#https://stackoverflow.com/questions/42350029/assign-a-number-to-each-unique-value-in-a-list
#https://www.afternerd.com/blog/python-enumerate/

##Create Dictionary with JCode, Index
#jcode_to_index_dict 
jcode_index_dictionary = {ni : indi for indi, ni in enumerate(set(Jcodes['ProcedureCode']))}
#jcode_ids 
Jcodes_ids = [jcode_index_dictionary[ni] for ni in Jcodes['ProcedureCode']]

##Create Dictionary with Index, JCode
index_jcode_dictionary = {indi : ni for indi, ni in enumerate(set(Jcodes['ProcedureCode']))}
#sum_per_jcode_id
Jcode_Payment_Sum = npg.aggregate(Jcodes_ids, Jcodes['ProviderPaymentAmount'], func='sum')

#Groupby function not returning the correct results for jcode sums        
#perform the groupby to get the group sums
#group_sums = Groupby(Jcodes).apply(np.sum, ProviderPayments, broadcast=False)
#unique_keys, indices = np.unique(Jcodes, return_inverse = True)
#zipped = zip(unique_keys, group_sums)  # python 3

#zipped = zip(Jcodes_ids, index_jcode_dictionary.values(), Jcode_Payment_Sum)
zipped = zip(index_jcode_dictionary.values(), Jcode_Payment_Sum)
sorted_group_sums = sorted(zipped, key=lambda x: x[1])
reverse_sorted_group_sums = sorted_group_sums[::-1]
print('Question 1C.  The the top five J-codes based on the payment to providers are:', reverse_sorted_group_sums[:5])
#print(sorted_group_sums)
#https://github.com/ml31415/numpy-groupies
#zipped = zip(unique_keys, group_sums)  # python 3
#sorted_group_sums = sorted(zipped, key=lambda x: x[1])

#2. For the following exercises, determine the number of providers that were paid for at least one J-code. 
#Use the J-code claims for these providers to complete the following exercises.
#Provider.Payment.Amount
#Provider.ID

#unpaid and paid indexes
unpaid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] == 0)
paid_mask = (Sorted_Jcodes['ProviderPaymentAmount'] > 0)

Unpaid_Claims = Sorted_Jcodes[unpaid_mask]
Paid_Claims = Sorted_Jcodes[paid_mask]


#Add fields to structured array: Need to add count columm for number of claims paid & unpaid for plotting
#https://www.numpy.org/devdocs/user/basics.rec.html 
Sorted_Jcodes = rfn.append_fields(Sorted_Jcodes, 'PaidClaims', (Sorted_Jcodes['ProviderPaymentAmount'] > 0)+0)
Sorted_Jcodes = rfn.append_fields(Sorted_Jcodes, 'UnpaidClaims', (Sorted_Jcodes['ProviderPaymentAmount'] == 0)+0)
#Verify the addition of the fields
#Sorted_Jcodes.dtype.names

Paid_ProviderID_dictionary = {ni : indi for indi, ni in enumerate(set(Sorted_Jcodes['ProviderID']))}
Paid_ProviderID_ids = [Paid_ProviderID_dictionary[ni] for ni in Sorted_Jcodes['ProviderID']]
index_provider_dictionary = {indi : ni for indi, ni in enumerate(set(Sorted_Jcodes['ProviderID']))}

#Sum Counts for Paid & Unpaid claims
#Paid_count = npg.aggregate(Paid_ProviderID_ids, Paid_Claims['NumberOfClaims'], func='sum')
#sum_per_provider_paid = npg.aggregate(provider_ids, claims_data_jcodes_paid['isPaidCount'], func='sum')
Tot_Claims_Paid = npg.aggregate(Paid_ProviderID_ids, Sorted_Jcodes['PaidClaims'], func='sum')
Tot_Claims_Unpaid = npg.aggregate(Paid_ProviderID_ids, Sorted_Jcodes['UnpaidClaims'], func='sum')


Paid_zipped = zip(Paid_ProviderID_dictionary.keys(), Tot_Claims_Paid, Tot_Claims_Unpaid, (Tot_Claims_Unpaid + Tot_Claims_Paid))
PAIDAGG = sorted(Paid_zipped, key=lambda x: x[3], reverse=True)

#2A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
# Create the plot object
# Plot the data, set the size (s), color and transparency (alpha)
FIG, AX = plt.subplots()
#AX.scatter(UNPAIDAGG, PAIDAGG)

#ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)
AX.grid(linestyle='-', linewidth='0.75', color='red')

FIG = plt.gcf()
FIG.set_size_inches(15, 10)
plt.rcParams.update({'font.size': 5})

for provider in PAIDAGG:
    #AX.scatter(PAIDAGG[1], PAIDAGG[2], edgecolors='none')
    AX.scatter(provider[1], provider[2], label='Provider #:' + provider[0].decode())
    #AX.annotate(PAIDAGG, provider[0].decode)

#for i, TXT in enumerate(provider):
#    AX.annotate(PAIDAGG, (provider[1], provider[2]))

plt.tick_params(labelsize=15)
plt.xlabel('# of Unpaid claims', fontsize=15)
plt.ylabel('# of Paid claims', fontsize=15)
plt.title('Scatterplot of Unpaid and Paid claims by Provider', fontsize=35)
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
plt.legend(loc=0)
plt.savefig('Paid_Unpaid_Scatterplot.png')# of the points
#B. What insights can you suggest from the graph?
print('Question 2B. There appear to be a distinct separation in providers that pay a higher ratio of claims than those that go unpaid and vice versa.')

#C. Based on the graph, is the behavior of any of the providers concerning? Explain.
print('Question 2C. There are 3 providers at the bottom and close to the middle that have almost the same # of unpaid and paid claims.  There are also 2 other providers that seem to have a relatively high number of unpaid claims to paid claims even though it seems like a much lower ratio FA0001411001 and FA001387002.  Also, there are quite a few providers that have what appears to be a more favorable ratio of payment of claims to unpaid claims.  The top 2 in this group are FA0001387001 and FA0001389001.')

#3. Consider all claim lines with a J-code.
#https://towardsdatascience.com/handling-imbalanced-datasets-in-machine-learning-7a0e84220f28
#create a new column and data type for both structured arrays
#3A. What percentage of J-code claim lines were unpaid?

#InNetworkIndexes = np.flatnonzero(np.core.defchararray.find(Jcodes['InOutOfNetwork'],in_network)==1)
#InNetworkAmount = Jcodes[InNetworkIndexes]['ProviderPaymentAmount'].sum().round(2)
TotPaid= Tot_Claims_Paid.sum(axis=0)
TotUnpaid = Tot_Claims_Unpaid.sum(axis=0)
TotalClaims =  np.add(TotPaid, TotUnpaid)
PercentagePaid = TotUnpaid / TotalClaims
print('Question 3A. ', (PercentagePaid.round(2)*100), ' percent of claims are unpaid.')


#B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.

Unpaid_Jcodes = Sorted_Jcodes[unpaid_mask]
Paid_Jcodes = Sorted_Jcodes[paid_mask]

new_dtype1 = np.dtype(Unpaid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])
new_dtype2 = np.dtype(Paid_Jcodes.dtype.descr + [('IsUnpaid', '<i4')])

#first get the right shape for each.
Unpaid_Jcodes_w_L = np.zeros(Unpaid_Jcodes.shape, dtype=new_dtype1)
Paid_Jcodes_w_L = np.zeros(Paid_Jcodes.shape, dtype=new_dtype2)

Unpaid_Jcodes_w_L['V1'] = Unpaid_Jcodes['V1']
Unpaid_Jcodes_w_L['ClaimNumber'] = Unpaid_Jcodes['ClaimNumber']
Unpaid_Jcodes_w_L['ClaimLineNumber'] = Unpaid_Jcodes['ClaimLineNumber']
Unpaid_Jcodes_w_L['MemberID'] = Unpaid_Jcodes['MemberID']
Unpaid_Jcodes_w_L['ProviderID'] = Unpaid_Jcodes['ProviderID']
Unpaid_Jcodes_w_L['LineOfBusinessID'] = Unpaid_Jcodes['LineOfBusinessID']
Unpaid_Jcodes_w_L['RevenueCode'] = Unpaid_Jcodes['RevenueCode']
Unpaid_Jcodes_w_L['ServiceCode'] = Unpaid_Jcodes['ServiceCode']
Unpaid_Jcodes_w_L['PlaceOfServiceCode'] = Unpaid_Jcodes['PlaceOfServiceCode']
Unpaid_Jcodes_w_L['ProcedureCode'] = Unpaid_Jcodes['ProcedureCode']
Unpaid_Jcodes_w_L['DiagnosisCode'] = Unpaid_Jcodes['DiagnosisCode']
Unpaid_Jcodes_w_L['ClaimChargeAmount'] = Unpaid_Jcodes['ClaimChargeAmount']
Unpaid_Jcodes_w_L['DenialReasonCode'] = Unpaid_Jcodes['DenialReasonCode']
Unpaid_Jcodes_w_L['PriceIndex'] = Unpaid_Jcodes['PriceIndex']
Unpaid_Jcodes_w_L['InOutOfNetwork'] = Unpaid_Jcodes['InOutOfNetwork']
Unpaid_Jcodes_w_L['ReferenceIndex'] = Unpaid_Jcodes['ReferenceIndex']
Unpaid_Jcodes_w_L['PricingIndex'] = Unpaid_Jcodes['PricingIndex']
Unpaid_Jcodes_w_L['CapitationIndex'] = Unpaid_Jcodes['CapitationIndex']
Unpaid_Jcodes_w_L['SubscriberPaymentAmount'] = Unpaid_Jcodes['SubscriberPaymentAmount']
Unpaid_Jcodes_w_L['ProviderPaymentAmount'] = Unpaid_Jcodes['ProviderPaymentAmount']
Unpaid_Jcodes_w_L['GroupIndex'] = Unpaid_Jcodes['GroupIndex']
Unpaid_Jcodes_w_L['SubscriberIndex'] = Unpaid_Jcodes['SubscriberIndex']
Unpaid_Jcodes_w_L['SubgroupIndex'] = Unpaid_Jcodes['SubgroupIndex']
Unpaid_Jcodes_w_L['ClaimType'] = Unpaid_Jcodes['ClaimType']
Unpaid_Jcodes_w_L['ClaimSubscriberType'] = Unpaid_Jcodes['ClaimSubscriberType']
Unpaid_Jcodes_w_L['ClaimPrePrinceIndex'] = Unpaid_Jcodes['ClaimPrePrinceIndex']
Unpaid_Jcodes_w_L['ClaimCurrentStatus'] = Unpaid_Jcodes['ClaimCurrentStatus']
Unpaid_Jcodes_w_L['NetworkID'] = Unpaid_Jcodes['NetworkID']
Unpaid_Jcodes_w_L['AgreementID'] = Unpaid_Jcodes['AgreementID']

#And assign the target label 
Unpaid_Jcodes_w_L['IsUnpaid'] = 1

#copy the data for the paid jcodes
Paid_Jcodes_w_L['V1'] = Paid_Jcodes['V1']
Paid_Jcodes_w_L['ClaimNumber'] = Paid_Jcodes['ClaimNumber']
Paid_Jcodes_w_L['ClaimLineNumber'] = Paid_Jcodes['ClaimLineNumber']
Paid_Jcodes_w_L['MemberID'] = Paid_Jcodes['MemberID']
Paid_Jcodes_w_L['ProviderID'] = Paid_Jcodes['ProviderID']
Paid_Jcodes_w_L['LineOfBusinessID'] = Paid_Jcodes['LineOfBusinessID']
Paid_Jcodes_w_L['RevenueCode'] = Paid_Jcodes['RevenueCode']
Paid_Jcodes_w_L['ServiceCode'] = Paid_Jcodes['ServiceCode']
Paid_Jcodes_w_L['PlaceOfServiceCode'] = Paid_Jcodes['PlaceOfServiceCode']
Paid_Jcodes_w_L['ProcedureCode'] = Paid_Jcodes['ProcedureCode']
Paid_Jcodes_w_L['DiagnosisCode'] = Paid_Jcodes['DiagnosisCode']
Paid_Jcodes_w_L['ClaimChargeAmount'] = Paid_Jcodes['ClaimChargeAmount']
Paid_Jcodes_w_L['DenialReasonCode'] = Paid_Jcodes['DenialReasonCode']
Paid_Jcodes_w_L['PriceIndex'] = Paid_Jcodes['PriceIndex']
Paid_Jcodes_w_L['InOutOfNetwork'] = Paid_Jcodes['InOutOfNetwork']
Paid_Jcodes_w_L['ReferenceIndex'] = Paid_Jcodes['ReferenceIndex']
Paid_Jcodes_w_L['PricingIndex'] = Paid_Jcodes['PricingIndex']
Paid_Jcodes_w_L['CapitationIndex'] = Paid_Jcodes['CapitationIndex']
Paid_Jcodes_w_L['SubscriberPaymentAmount'] = Paid_Jcodes['SubscriberPaymentAmount']
Paid_Jcodes_w_L['ProviderPaymentAmount'] = Paid_Jcodes['ProviderPaymentAmount']
Paid_Jcodes_w_L['GroupIndex'] = Paid_Jcodes['GroupIndex']
Paid_Jcodes_w_L['SubscriberIndex'] = Paid_Jcodes['SubscriberIndex']
Paid_Jcodes_w_L['SubgroupIndex'] = Paid_Jcodes['SubgroupIndex']
Paid_Jcodes_w_L['ClaimType'] = Paid_Jcodes['ClaimType']
Paid_Jcodes_w_L['ClaimSubscriberType'] = Paid_Jcodes['ClaimSubscriberType']
Paid_Jcodes_w_L['ClaimPrePrinceIndex'] = Paid_Jcodes['ClaimPrePrinceIndex']
Paid_Jcodes_w_L['ClaimCurrentStatus'] = Paid_Jcodes['ClaimCurrentStatus']
Paid_Jcodes_w_L['NetworkID'] = Paid_Jcodes['NetworkID']
Paid_Jcodes_w_L['AgreementID'] = Paid_Jcodes['AgreementID']

#And assign the target label 
Paid_Jcodes_w_L['IsUnpaid'] = 0

#now combine the rows together (axis=0)
Jcodes_w_L = np.concatenate((Unpaid_Jcodes_w_L, Paid_Jcodes_w_L), axis=0)

#We need to shuffle the rows before using classifers in sklearn to address the unbalanced sampling
np.random.shuffle(Jcodes_w_L)

label =  'IsUnpaid'

#Removed V1 as it is a unique identifier and would provide little value in prediction; DiagnosisCode also removed b/c it showed no value in the feature importance
#Remove DiagnosisCode or ProviderID?
cat_features = ['ProviderID','LineOfBusinessID','RevenueCode',
                'ServiceCode', 'PlaceOfServiceCode', 'ProcedureCode',
                'DenialReasonCode',
                'PriceIndex', 'InOutOfNetwork', 'ReferenceIndex', 
                'PricingIndex', 'CapitationIndex', 'ClaimSubscriberType',
                'ClaimPrePrinceIndex', 'ClaimCurrentStatus', 'NetworkID',
                'AgreementID', 'ClaimType', ]
#Removed 'MemberID' as it is a unique identifier and would provide little value in prediction
numeric_features = ['ClaimNumber', 'ClaimLineNumber', 
                    'ClaimChargeAmount',
                    'SubscriberPaymentAmount', 'ProviderPaymentAmount',
                    'GroupIndex', 'SubscriberIndex', 'SubgroupIndex']

#separate categorical and numeric features
Mcat = np.array(Jcodes_w_L[cat_features].tolist())
Mnum = np.array(Jcodes_w_L[numeric_features].tolist())

L = np.array(Jcodes_w_L[label].tolist())

#Setup One Hot Encoding
#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#https://medium.com/@contactsunny/label-encoder-vs-one-hot-encoder-in-machine-learning-3fc273365621
#https://towardsdatascience.com/encoding-categorical-features-21a2651a065c

#https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
ohe = OneHotEncoder(sparse=False) #Easier to read
Mcat = ohe.fit_transform(Mcat)
ohe.inverse_transform(Mcat)
ohe_features = ohe.get_feature_names(cat_features).tolist()

M = np.concatenate((Mcat, Mnum), axis=1)

#Concatenate the columns
#M = np.concatenate((Mcat_subset, Mnum_subset), axis=1)

L = Jcodes_w_L[label].astype(int)
n_folds = 5

#EDIT: pack the arrays together into "data"
data = (M,L,n_folds)

#EDIT: A function, "run", to run all our classifiers against our data.
#Changed from K-fold to StratifiedKFold to address issue with unbalanced data (more unpaid than paid claims)
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html
#https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
def run(a_clf, data, clf_hyper={}):
  M, L, n_folds = data #EDIT: unpack the "data" container of arrays
  #kf = KFold(n_splits=n_folds) # JS: Establish the cross validation 
  kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # JS: Establish the cross validation 
  ret = {} # JS: classic explicaiton of results
  
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)): #EDIT: We're interating through train and test indexes by using kf.split
                                                                   #      from M and L.
                                                                   #      We're simply splitting rows into train and test rows
                                                                   #      for our five folds.
    
    clf = a_clf(**clf_hyper) # JS: unpack paramters into clf if they exist   #EDIT: this gives all keyword arguments except 
                                                                             #      for those corresponding to a formal parameter
                                                                             #      in a dictionary.
            
    clf.fit(M[train_index], L[train_index])   #EDIT: First param, M when subset by "train_index", 
                                              #      includes training X's. 
                                              #      Second param, L when subset by "train_index",
                                              #      includes training Y.                             
    
    pred = clf.predict(M[test_index])         #EDIT: Using M -our X's- subset by the test_indexes, 
                                              #      predict the Y's for the test rows.
    
    ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}  
    
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    #https://markhneedham.com/blog/2017/06/16/scikit-learn-random-forests-feature-importance/
    headers = ["name", "score"]
    values = sorted(zip(ohe_features, importances), key=lambda x: x[1] * -1)
    print(tabulate(values[0:5], headers, tablefmt="plain"))
    ###print("Feature ranking:")
#ohe_features
    #for name, importance in zip(iris["feature_names"], rnd_clf.feature_importances_):
       ###for f in range(M[train_index].shape[1]):
    #for f in range(M[train_index].shape[1]):
        ###print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
        #plt.figure()
        #plt.title("Feature importances")
        #plt.bar(range(M[train_index].shape[1]), importances[indices],
        #        color="r", yerr=std[indices], align="center")
        #plt.xticks(range(M[train_index].shape[1]), indices)
        #plt.xticks(range(M[train_index].shape[1]), ohe_features)
        #plt.xlim([-1, M[train_index].shape[1]])
        #plt.xticks(range(M[train_index].shape[1]), indices)
        #plt.xlim([-1, M[train_index].shape[1]])
        #plt.show()

  return ret

def runGridSearch(a_clf, data, clf_hyper={}):
  M, L, n_folds = data # unpack data containter
  kf = StratifiedKFold(n_splits=n_folds, shuffle=True) # Establish the cross validation
  ret = {} # classic explicaiton of results
  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
      clf = a_clf(**clf_hyper)
      clf.GridSearchCV(M[train_index], L[train_index])
      pred = clf.predict(M[test_index])
      ret[ids]= {'clf': clf,                    #EDIT: Create arrays of
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}    
  return ret

def populateClfAccuracyDict(results):
    for key in results:
        k1 = results[key]['clf'] 
        v1 = results[key]['accuracy']
        k1Test = str(k1) #Since we have a number of k-folds for each classifier...
                         #We want to prevent unique k1 values due to different "key" values
                         #when we actually have the same classifer and hyper parameter settings.
                         #So, we convert to a string
                        
        #String formatting            
        k1Test = k1Test.replace('            ',' ') # remove large spaces from string
        k1Test = k1Test.replace('          ',' ')
        
        #Then check if the string value 'k1Test' exists as a key in the dictionary
        if k1Test in clfsAccuracyDict:
            clfsAccuracyDict[k1Test].append(v1) #append the values to create an array (techically a list) of values
        else:
            clfsAccuracyDict[k1Test] = [v1] #create a new key (k1Test) in clfsAccuracyDict with a new value, (v1)            
        
def myHyperSetSearch(clfsList,clfDict):
    #hyperSet = {}
    for clf in clfsList:
    
    #I need to check if values in clfsList are in clfDict
        clfString = str(clf)
        #print("clf: ", clfString)
        
        for k1, v1 in clfDict.items(): # go through the inner dictionary of hyper parameters
            #Nothing to do here, we need to get into the inner nested dictionary.
            if k1 in clfString:
                #allows you to do all the matching key and values
                k2,v2 = zip(*v1.items()) # explain zip (https://docs.python.org/3.3/library/functions.html#zip)
                for values in product(*v2): #for the values in the inner dictionary, get their unique combinations from product()
                    hyperSet = dict(zip(k2, values)) # create a dictionary from their values
                    results = run(clf, data, hyperSet) # pass the clf and dictionary of hyper param combinations to run; get results
                    populateClfAccuracyDict(results) # populate clfsAccuracyDict with results
 
clfsList = [RandomForestClassifier] 
clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3]}}

#clfsList = [RandomForestClassifier, LogisticRegression] 
#clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], "n_jobs": [1,2,3]}, 'LogisticRegression': {"tol": [0.01,0.1, .05], "solver": ['lbfgs', 'sag', 'saga']}}
#clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4], 
#                                      "n_jobs": [1,2,3]}}#,
                                      
           #'LogisticRegression': {"tol": [0.001,0.01,0.1]}}

                   
#Declare empty clfs Accuracy Dict to populate in myHyperSetSearch     
clfsAccuracyDict = {}

#Run myHyperSetSearch
myHyperSetSearch(clfsList,clfDict)    

print(clfsAccuracyDict)

#B. Create a model to predict when a J-code is unpaid. Explain why you choose the modeling approach.
#print('Question 3B.  I used the death to grid search script from HW1.  I initially used the Logistic Regression and Random Forest Classifiers but removed the Logistic regression because the accuracy was consistently lower than the Random Forest models.  Also, I removed the Logistic Regression to make it easier to pull the feature importance for the models.')
#C. How accurate is your model at predicting unpaid claims?
#https://bigdata-madesimple.com/dealing-with-unbalanced-class-svm-random-forest-and-decision-tree-in-python/
#https://www.kdnuggets.com/2016/08/learning-from-imbalanced-classes.html/2
#https://www.kdnuggets.com/2016/04/unbalanced-classes-svm-random-forests-python.html
#print('Question 3C.  Even with the use of StratifiedKFold and shuffling the data, the RandomForestClassifier is returning values of upwards of 99 percent with 2 splits and 1 fold.  This is concerning, and I would typically investigate further options as suggested above to resample the data.')
#D. What data attributes are predominately influencing the rate of non-payment?
#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html

# Plot the feature importances of the forest
#print(ohe_features)
#ohe = OneHotEncoder(sparse=False) #Easier to read
#Mcat = ohe.fit_transform(Mcat)
#ohe.inverse_transform(Mcat)
#ohe_features = ohe.get_feature_names(cat_features).tolist()





#1A. Find the number of claim lines that have J-codes.
print('Question 1A.  The number of claim lines that have a J-Code is:', len(JcodesIndexes))
print('Question 1B.  The amount paid for J-Codes to providers for in network claims is:', InNetworkAmount)
print('Question 1C.  The the top five J-codes based on the payment to providers are:', reverse_sorted_group_sums[:5])
#2A. Create a scatter plot that displays the number of unpaid claims (lines where the ‘Provider.Payment.Amount’ field is equal to zero) for each provider versus the number of paid claims.
# Create the plot object
# Plot the data, set the size (s), color and transparency (alpha)
FIG, AX = plt.subplots()
#AX.scatter(UNPAIDAGG, PAIDAGG)

#ax.scatter(x_data, y_data, s = 10, color = color, alpha = 0.75)
AX.grid(linestyle='-', linewidth='0.75', color='red')

FIG = plt.gcf()
FIG.set_size_inches(15, 10)
plt.rcParams.update({'font.size': 5})

for provider in PAIDAGG:
    #AX.scatter(PAIDAGG[1], PAIDAGG[2], edgecolors='none')
    AX.scatter(provider[1], provider[2], label='Provider #:' + provider[0].decode())
    #AX.annotate(PAIDAGG, provider[0].decode)

#for i, TXT in enumerate(provider):
#    AX.annotate(PAIDAGG, (provider[1], provider[2]))

plt.tick_params(labelsize=15)
plt.xlabel('# of Unpaid claims', fontsize=15)
plt.ylabel('# of Paid claims', fontsize=15)
plt.title('Scatterplot of Unpaid and Paid claims by Provider', fontsize=35)
#https://matplotlib.org/api/_as_gen/matplotlib.pyplot.legend.html
plt.legend(loc=0)
plt.savefig('Paid_Unpaid_Scatterplot.png')# of the points
#B. What insights can you suggest from the graph?
print('Question 2B. There appear to be a distinct separation in providers that pay a higher ratio of claims than those that go unpaid and vice versa.')
print('Question 2C. There are 3 providers at the bottom and close to the middle that have almost the same # of unpaid and paid claims.  There are also 2 other providers that seem to have a relatively high number of unpaid claims to paid claims even though it seems like a much lower ratio FA0001411001 and FA001387002.  Also, there are quite a few providers that have what appears to be a more favorable ratio of payment of claims to unpaid claims.  The top 2 in this group are FA0001387001 and FA0001389001.')
print('Question 3A. ', (PercentagePaid.round(2)*100), ' percent of claims are unpaid.')
print('Question 3B.  I used the death to grid search script from HW1.  I initially used the Logistic Regression and Random Forest Classifiers but removed the Logistic regression because the accuracy was consistently lower than the Random Forest models.  Also, I removed the Logistic Regression to make it easier to pull the feature importance for the models.')
print('Question 3C.  Even with the use of StratifiedKFold and shuffling the data, the RandomForestClassifier is returning values of upwards of 99 percent with 2 splits and 1 fold.  This is concerning, and I would typically investigate further options as suggested above to resample the data.')
print('name                               score')
print('DenialReasonCode_b'"K62"'      0.116045')
print('DenialReasonCode_b'" "'        0.0599372')
print('DenialReasonCode_b'"PDC"'      0.0597075')
print('ClaimType_b'"E"'               0.0345357')
print('AgreementID_b'"A00BLCH0Q001"'  0.0343387')