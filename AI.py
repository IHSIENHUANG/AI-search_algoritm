#purpose finish foward and backward search algorithm and create a high efficient search algorithm
import pandas as pd
import numpy as np
import time
from sklearn import preprocessing
import sys
print ("this is my AI assginmant")
def scale_normalized(data):
    max_num = max(data)
    min_num = min(data)
    for i,item in enumerate(data):
        data[i] = (data[i]-min_num)/max_num
    return data
#dftrain = pd.read_csv("CS205_BIGtestdata__8.txt",delim_whitespace=True,header=None)
#dftrain = pd.read_csv("CS205_SMALLtestdata__70.txt",delim_whitespace=True,header=None)
dftrain = pd.read_csv("CS205_SMALLtestdata__23.txt",delim_whitespace=True,header=None) #read the file and delimeter is whitespace
#dftrain = pd.read_csv("CS205_SMALLtestdata__14.txt",delim_whitespace=True,header=None) #read the file and delimeter is whitespace
# header is none is different with header is None
total_features = dftrain.shape[1]#0 for row 1 for columns 
rows = dftrain.shape[0]
print (dftrain.shape)
labels = dftrain[dftrain.columns[0]]# keep the first column as labels
print ("--start normalizing, and it will take some time")
for i in range(1,total_features):#the orocess of normalized
    dftrain[dftrain.columns[i]]=scale_normalized(dftrain[dftrain.columns[i]])
print ("FINISH NORMALIZE")
def leave_one_out_cross_validation(rows,k,test_data):
    res = 0 
    for i,item in enumerate(test_data):
        min_distance = 100
        distance_list = []
        index = -1
        for j,cmp_item in enumerate(test_data):
            distance =0.0
            if i==j:
                distance_list.append(100.0)
                continue
            for (x,y) in zip(item,cmp_item):
                distance += (x-y)**2
            distance_list.append(distance)
        min_distance = min(distance_list)
        label_index=-1
        for k, num in enumerate(distance_list):
            if num == min_distance:
                label_index = k
                break
        if labels[i] == labels[label_index]:#it means the prediction for this label is correct
            res +=1
    print ("The accuracy of data is :", res/100.00)
    return (res/100.00)
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
    return list(data)
def forward_search(row,column):
    print ("start forward_search")
    items=[]
    lawyer_score = []
    start_time = time.time()
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
                if cur_score > max_score:
                    new_element = i
                    max_score = cur_score
        print (max_score,new_element)
        lawyer_score.append(max_score)
        items.append(new_element)
    max_num = max(lawyer_score) # score the maximum scores
    index = -1 # score the best features 
    for i, num in enumerate(lawyer_score):
        if num == max_num:
            index = i+1 
            break
    print ("max score = " ,max_num)
    print ("feature = " ,items[0:index])
    print ("total time is %s seconds",time.time()-start_time)
def Backward_Elimination(row,column):
    print ("--start Backward_Elimination")
    start_time = time.time()
    print ("start recording time")
    items=[]
    remove_list = []
    for i in range(1,total_features):
        items.append(i)
    print (items)
    lawyer_score = []
    for num in range (1,total_features-1):
        max_score = -1 # store the best score in this lawyer
        min_score = 101
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
                if cur_score > max_score:#delete the less
                    new_element = i
                    max_score = cur_score
        print (max_score,new_element)
        remove_list.append(new_element)
        lawyer_score.append(max_score)
        items.remove(new_element)
    max_num = max(lawyer_score) # score the maximum scores
    index = -1 # score the best features 
    for i, num in enumerate(lawyer_score):
        if num == max_num: # it can eliminate the feature as many as posiible 
            index = i+1 
            break
    print ("max score = " ,max_num)
    items=[]
    for i in range(1,total_features):
        items.append(i)
    for item in remove_list[0:index]:
        items.remove(item)
    print ("feature = " ,items)
    print ("total time is %s seconds",time.time()-start_time)
def original_leave_one_out_cross_validation(rows,k,test_data,max_score):
    # the idea is that if we find the score is lower than before, do not need to finish the calculation
    max_error = 100 - max_score*100
    error_rate = 0 
    res = 0
     
    for i,item in enumerate(test_data):
        min_distance = 100
        distance_list = []
        index = -1
        for j,cmp_item in enumerate(test_data):
            distance =0.0
            if i==j:
                distance_list.append(100.0)
                continue
            if error_rate >= max_error:
                print ("The accuracy of data is :", 0) 
                return 0
            for (x,y) in zip(item,cmp_item):
                distance += (x-y)**2
            if distance < min_distance:
                min_distance = distance
                index = j
        if labels[i] == labels[index]:#it means the prediction for this label is correct
            res +=1
        else:
            error_rate +=1
    print ("The accuracy of data is :", res/100.00)
    return (res/100.00)
def original_search(row,total_features):
    print ("start recording time")
    start_time = time.time()
    print ("--THIS IS MY OWN SEARCH--")
    #basically, it is a forward search, but instead of leave_one_out_cross_validation this algorithm put concept of alpha 
    #beta inside the the leave one out ->  original_leave_one_out_cross_validation
    items=[]
    lawyer_score = []
    index_time = 0.0
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
                cur_score = original_leave_one_out_cross_validation(row,num,test_data,max_score)# num is num of columns
                if cur_score > max_score:
                    new_element = i
                    max_score = cur_score
        print (max_score,new_element)
        lawyer_score.append(max_score)
        items.append(new_element)
    max_num = max(lawyer_score)# score the maximum scores
    index = -1 # score the best features 
    for i, num in enumerate(lawyer_score):
        if num == max_num:
            index = i+1 
            break
    print ("max score = " ,max_num)
    print ("feature = " ,items[0:index])
    print ("total time is %s seconds",time.time()-start_time)
def main(): 
    print ("please input between 1-3 \n 1 for forward search \n 2 for backard_elemination \n 3 for original_search")
    choice = raw_input()
    if int(choice) == 1 :
        forward_search(rows,total_features)
    elif int(choice) == 2:
        Backward_Elimination(rows,total_features)
    elif int(choice) == 3:
        original_search(rows,total_features)
    else:
        print ("please input betweeen 1-3")
if __name__ == "__main__":
    main()
