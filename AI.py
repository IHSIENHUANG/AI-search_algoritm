
# coding: utf-8
#purpose finish foward and backward search algorithm and create a high efficient search algorithm
import pandas as pd
import numpy as np
import sklearn as sk
from sklearn import neighbors
from sklearn import preprocessing
import sys
print ("this is my AI assginmant")

dftrain = pd.read_csv("CS205_SMALLtestdata__23.txt",delim_whitespace=True,header=None) #read the file and delimeter is whitespace
# header is none is different with header is None
#dftrain.columns
#rint (dftrain.info)
total_features = dftrain.shape[1]#0 for row 1 for columns 
#print preprocessing.scale(dftrain[1])
labels = dftrain[dftrain.columns[0]]# keep the first column as labels
items = [6,9,3]#the index column we want to input inside the data
print (len(items))

def leave_one_out_cross_validation(k,test_data):
    clf = neighbors.KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None,n_jobs=1)
    #neighbors.KNeighborsClassifier
    # X is testing data
    # Y is label
    #test_data = dftrain[dftrain.columns[1]]# keep the first column as labels
    X = test_data

    X = np.reshape(test_data, (100, k))
    #print (data)
    #data = np.reshape（test_data,(100,10))
    Y = labels
    #clf.fit(test_data,labels)
    clf.fit(X,Y)
    clf.kneighbors(X=None, n_neighbors=None, return_distance= True)
    #clf.predict(X)
    clf.predict(X)
    scores = clf.score(X,Y,sample_weight=None)
    temp = scores
    #print  (scores)
    return (float(scores))
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
    


# In[179]:


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
                cur_score = leave_one_out_cross_validation(num,test_data)
                sys.stdout.write(" accuracy is ")
                sys.stdout.flush()
                print cur_score
                if cur_score > max_score:
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
                cur_score = leave_one_out_cross_validation(total_features-num-1,test_data)
                sys.stdout.write(" accuracy is ")
                sys.stdout.flush()
                print cur_score
                if cur_score > max_score:
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
def main():
    print 'this is main function'  
   # forward_search(100,11)
    Backward_Elimination(100,11)
    #leave_one_out_cross_validation(3)
if __name__ == "__main__":
    main()

