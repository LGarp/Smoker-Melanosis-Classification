#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
import matplotlib.pyplot as pt
from sklearn.model_selection import cross_val_score, GridSearchCV
from tqdm import tqdm,trange


# In[2]:


data = pd.read_csv("C:\\Users\\mahos\\1. Final Program\\DATA_SMOTE_DOPI.csv")
A = data["Area Segmented"]
B = data["Red"]
C = data["Green"]
D = data["Blue"]
data


# In[3]:


#MENENTUKAN NILAI MAKSIMUM
maksA = max(A)
maksB = max(B)
maksC = max(C)
maksD = max(D)
#MENENTUKAN NILAI MINIMUM
minA = min(A)
minB = min(B)
minC = min(C)
minD = min(D)


# In[4]:


#FUCTION FOR SCALING
def scaling(ket):
  N = len(data)
  new = np.zeros(N)
  for n in range (N):
#SCALING METHOD WITH MIN MAX SCALING
    new[n] = (ket[n]-min(ket))/(max(ket)-min(ket))
  return new


# In[5]:


A = scaling(data["Area Segmented"])
B = scaling(data["Red"])
C = scaling(data["Green"])
D = scaling(data["Blue"])


# In[6]:


data['Area Segmented'] = A
data['Red'] = B
data['Green'] = C
data['Blue'] = D
data


# In[7]:


data.info()


# In[8]:


#PROSES SPLIT DATA FEATURES DAN LABEL
Y = data['CLASS']
X = data.drop(['CLASS'], axis = 1)


# In[9]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
X_train = X_train.values
X_test = X_test.values


# In[10]:


# SOURCE FROM : https://towardsdatascience.com/how-to-build-knn-from-scratch-in-python-5e22b8920bd2
# MENGHITUNG JARAK ANTAR DUA BUAH POINT
def minkowski_distance(a, b, p=0):
    # Store the number of dimensions
    dim = len(a)
    # Set initial distance to 0
    distance = 0
    # Calculate minkowski distance using parameter p
    for n in range (dim):
        distance += abs(a[n] - b[n])**p
    distance = distance**(1/p)
    return distance


# In[33]:


#GENERAL FORMULA FOR KNN
def knn_predict(X_knn,X_input,Y_knn,k,p):
  # Counter to help with label voting
  from collections import Counter
  
  # Make predictions on the test data
  # Need output of 1 prediction per test data point
  y_hat = []

  for input_point in X_input:
    distances = []
    for knn_point in X_knn:
      distance = minkowski_distance(input_point, knn_point, p=p)
      distances.append(distance)
      
    # Store distances in a dataframe
    df_dists = pd.DataFrame(data=distances, columns=['dist'], 
                                index=Y_train.index)
    
    # Sort distances, and only consider the k closest points
    df_nn = df_dists.sort_values(by=['dist'], axis=0)[:k]

    # Create counter object to track the labels of k closest neighbors
    counter = Counter(Y_knn[df_nn.index])

    # Get most common label of all the nearest neighbors
    prediction = counter.most_common()[0][0]
    
    # Append prediction to output list
    y_hat.append(prediction)

  return y_hat

#MENCARI OUTPUT APABILA INPUTNYA TRAINING OR TESTING  
def knn_train_test_predict(X_train, X_test, Y_train, k, p):
    y_hat_test = knn_predict(X_train,X_test,Y_train,k,p)
    y_hat_train = knn_predict(X_train,X_train,Y_train,k,p)   
    return y_hat_train,y_hat_test 

#MENCARI AKURASI TRAINING DAN TESTING
def knn_train_test_accuracy(X_train,X_test,Y_train,Y_test,k,p):
  y_hat_train,y_hat_test = knn_train_test_predict(X_train,X_test,Y_train,k,p)

  # train_acc = accuracy_score(Y_train,y_hat_train)
  # test_acc = accuracy_score(Y_test,y_hat_test)
  trains_acc = classification_report(Y_train,y_hat_train)
  tests_acc = classification_report(Y_test,y_hat_test)
  
  from sklearn.metrics import balanced_accuracy_score
  train_acc = [balanced_accuracy_score(Y_train,y_hat_train)]
  test_acc = [balanced_accuracy_score(Y_test,y_hat_test)]
  # return train_acc,test_acc
  return {"train":(train_acc,trains_acc),"test":(test_acc,tests_acc)}


# Make predictions on test dataset
# knn_train_test_accuracy(X_train, X_test, Y_train,Y_test, k=4, p=2)
kp ={}
for n in trange(1,5):
  kp["k-"+str(n).zfill(2)] = {}
  for i in range(1,5):
    kp["k-"+str(n).zfill(2)]["p-"+str(i).zfill(2)]=knn_train_test_accuracy(X_train, X_test, Y_train,Y_test, k=n, p=i)
print(kp)


# In[34]:


from pprint import pprint
pprint(kp)


# In[13]:


#MASUKAN DATA HASIL Ekstraksi Fitur
Area_test = 105637
Red_test = 58.985143333333326 
Green_test = 32.86272666666664
Blue_test = 36.119633333333354

Area_test = float(Area_test)
Red_test = float(Red_test)
Green_test= float(Green_test)
Blue_test = float(Blue_test)


# In[14]:


Area_ST = (Area_test-minA)/(maksA-minA)
Red_ST = (Red_test-minB)/(maksB-minB)
Green_ST = (Green_test-minC)/(maksC-minC)
Blue_ST = (Blue_test-minD)/(maksD-minD)


# In[21]:


# MENGHITUNG JARAK ANTAR DUA BUAH POINT
def minkowski_distance(a, b, p=3):
    # Store the number of dimensions
    dim = len(a)
    # Set initial distance to 0
    distance = 0
    # Calculate minkowski distance using parameter p
    for n in range (dim):
        distance += abs(a[n] - b[n])**p
    distance = distance**(1/p)
    return distance


# In[22]:


test_pt = [Area_ST,Red_ST,Green_ST,Blue_ST]
distance = []
for i in X.index:
    distance.append(minkowski_distance(test_pt, X.iloc[i]))
    
data_jarak = pd.DataFrame(data=distance, index=X.index, columns=['JARAK DATA UJI TERHADAP DATA SET'])
data_jarak


# In[23]:


# Find the n nearest neighbors
sorted_data_jarak = data_jarak.sort_values(by=['JARAK DATA UJI TERHADAP DATA SET'], axis=0)
sorted_data_jarak


# In[24]:


k = 4
df_nn = sorted_data_jarak[:k]
df_nn


# In[25]:


from collections import Counter
index = df_nn.index
# index
k_label = Y[index]
# k_label
counter = Counter(k_label)
most_common_val = counter.most_common()

nearest_label = most_common_val[0][0]
nearest_count = most_common_val[0][1]
nearest_label
# counter.most_common()[0][0]


# In[26]:


A = knn_predict(X_train,[test_pt],Y_train,k=2,p=2)
A


# In[ ]:




