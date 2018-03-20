
# coding: utf-8

# In[94]:


#purpose finish foward and backward search algorithm and create a high efficient search algorithm
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import neighbors
from sklearn import preprocessing
import sys
print ("this is my AI assginmant")


# In[95]:


def scale_normalized(data):
    max_num = max(data)
    min_num = min(data)
    for i,item in enumerate(data):
        data[i] = (data[i]-min_num)/max_num
    return data


# In[104]:


#dftrain = pd.read_csv("CS205_BIGtestdata__8.txt",delim_whitespace=True,header=None)
dftrain = pd.read_csv("CS205_SMALLtestdata__70.txt",delim_whitespace=True,header=None)
#dftrain = pd.read_csv("CS205_SMALLtestdata__23.txt",delim_whitespace=True,header=None) #read the file and delimeter is whitespace
# header is none is different with header is None
total_features = dftrain.shape[1]#0 for row 1 for columns 
rows = dftrain.shape[0]
print (dftrain.shape)
labels = dftrain[dftrain.columns[0]]# keep the first column as labels
print ("--start normalizing, and it will take some time")
for i in range(1,total_features):#the orocess of normalized
    dftrain[dftrain.columns[i]]=scale_normalized(dftrain[dftrain.columns[i]])
print ("FINISH NORMALIZE")


# In[105]:


def leave_one_out_cross_validation(rows,k,test_data):
    X = np.reshape(test_data, (rows, k))
    res = 0 
    for row in range(0,rows):
        distance_list = []#store each distance
        label_index =-1
        for other_row in range(0,rows):# compare each data with the rest data
            distance = 0.0
            if other_row == row:# do not need to compare with itself
                distance_list.append(1000000.0)
                continue
            for (x,y) in zip (X[row,0:],X[other_row,0:]):# count the distance
                distance += (x-y)**2
            distance_list.append(distance)
        min_distance = min(distance_list)
        same_distance=0
        for i,num in enumerate(distance_list):
            if num == min_distance:
                label_index = i#is is index num is number is distance_list
        if labels[row] == labels[label_index]:#it means the prediction for this label is correct
            res +=1
    print ("The accuracy of data is :", res/100.00)
    
            
                
            
            
            
'''
def leave_one_out_cross_validation(row,k,test_data):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,n_jobs=1)
    #neighbors.KNeighborsClassifier
    # X is testing data
    # Y is label
    #test_data = dftrain[dftrain.columns[1]]# keep the first column as labels
    X = test_data

    X = np.reshape(test_data, (row, k))
    #print (data)
    #data = np.reshape（test_data,(100,10))
    Y = labels
    #clf.fit(test_data,labels)
    clf.fit(X,Y)
    clf.kneighbors(X=None, n_neighbors=None, return_distance= True)
    #clf.predict(X)
    clf.predict_proba(X)
    scores = clf.score(X,Y,sample_weight=None)
    temp = scores
    #print  (scores)
    return (float(scores))
'''


# In[106]:


def call_test_data(temp_items):
    data = []
    sys.stdout.write("using features: ")
    sys.stdout.flush()
    print temp_items,
    
    #print ("using features: "temp_items,)
    for value in (dftrain.values):
        temp = []
        for item in temp_items:
            temp.append(value[item])
        data.append(temp)
    #print (dats
    return list(data)
    


# In[107]:


def forward_search(row,column):
    items=[]
    lawyer_score = []
    for num in range (1,total_features):
        max_score = -1
        test_data = []
        new_element = -1
        for i in range (1,total_features):
            if i in items:
                continue
            else:
                temp_items=[]
                for element in items:# can not assgin temp_items = items, it will affect the items
                    temp_items.append(element)
                temp_items.append(i)
                test_data = call_test_data(temp_items)# based on the element number to get new test_data columns 
                cur_score = leave_one_out_cross_validation(row,num,test_data)# num is num of columns
                sys.stdout.write(" accuracy is ")
                sys.stdout.flush()
                print cur_score
                if cur_score >= max_score:
                    new_element = i
                    max_score = cur_score
        print (max_score,new_element)
        lawyer_score.append(max_score)
        items.append(new_element)
    max_num = 0 # score the maximum scores
    index = -1 # score the best features 
    for i, num in enumerate(lawyer_score):
        if num > max_num:
            print(i)
            max_num = num
            index = i+1 
    print ("max score = " ,max_num)
    print ("feature = " ,items[0:index])
    


# In[108]:


def Backward_Elimination(row,column):
    print ("--start Backward_Elimination")
    items=[]
    remove_list = []
    for i in range(1,total_features):
        items.append(i)
    print (items)
    lawyer_score = []
    for num in range (1,total_features-1):
        max_score = -1
        test_data = []
        new_element = -1
        for i in range (1,total_features):
            if i not in items:
                continue
            else:
                temp_items=[]
                for element in items:# can not assgin temp_items = items, it will affect the items
                    temp_items.append(element)
                temp_items.remove(i)
                test_data = call_test_data(temp_items)# based on the element number to get new test_data columns 
                cur_score = leave_one_out_cross_validation(row,total_features-num-1,test_data)
                sys.stdout.write(" accuracy is ")
                sys.stdout.flush()
                print cur_score
                if cur_score >= max_score:
                    new_element = i
                    max_score = cur_score
        print (max_score,new_element)
        remove_list.append(new_element)
        lawyer_score.append(max_score)
        items.remove(new_element)
    max_num = 0 # score the maximum scores
    index = -1 # score the best features 
    for i, num in enumerate(lawyer_score):
        if num > max_num:
            print(i)
            max_num = num
            index = i+1 
    print ("max score = " ,max_num)
    items=[]
    for i in range(1,total_features):
        items.append(i)
    for item in remove_list[0:index]:
        items.remove(item)
    print ("feature = " ,items)
    


# In[110]:


def main(): 
    print 'this is main function'
    #forward_search(rows,total_features)
    #Backward_Elimination(rows,total_features)
    items = [8,4,1]
    test_data  = call_test_data(items)
    leave_one_out_cross_validation(100,3,test_data)
    #test_data()

if __name__ == "__main__":
    main()

