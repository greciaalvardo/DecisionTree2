#-------------------------------------------------------------------------
# AUTHOR: Grecia Alvarado
# FILENAME: decision_tree_2.py
# SPECIFICATION: Test the accuracy of the decision tree with a max depth of 3
# FOR: CS 4210- Assignment #2
# TIME SPENT: 30 minutes
#-----------------------------------------------------------*/
#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH 
#AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays
#importing some Python libraries
from sklearn import tree
import csv
dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 
'contact_lens_training_3.csv']
for ds in dataSets:
    dbTraining = []
    X = []
    Y = []
    #reading the training data in a csv file
    with open(ds, 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTraining.append (row)
    #transform the original categorical training features to numbers and add to the
    #4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    # so X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    tracker = {}
    num = 1
    for i in dbTraining:
      indices = i
      row = []
      for j in range(0, len(indices)-1):
         if indices[j] in tracker:
            row.append(tracker.get(indices[j]))
            indices[j] = tracker.get(indices[j])
         else:
            tracker[indices[j]] = num
            row.append(num)
            indices[j] = num
            num += 1
      X.append(row)

    #transform the original categorical training classes to numbers and add to the 
    #vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    tracker2 = {}
    num2 = 1
    for j in dbTraining:
      for i in range(len(j)-1, len(j)):
         if j[i] in tracker2:
            Y.append(tracker2.get(j[i]))
         else:
            tracker2[j[i]] = num2
            Y.append(num2)
            num2 += 1
    #loop your training and test tasks 10 times here
    for i in range (10):
       #fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth=3)
       clf = clf.fit(X, Y)

       #read the test data and add this data to dbTest
       dbTest = []
       with open('contact_lens_test.csv', 'r') as csvfile:
         reader = csv.reader(csvfile)
         for i, row in enumerate(reader):
             if i > 0: #skipping the header
                dbTest.append (row)
       rowsClass = 0
       TP = 0
       TN = 0
       FP = 0
       FN = 0
       for data in dbTest:
          row = []
          for j in range(0, len(data)):
            if j == len(data)-1:
               if data[j] in tracker2: 
                  rowsClass = tracker2.get(data[j])
            else:
               if data[j] in tracker:
                  row.append(tracker.get(data[j]))
       #where [0] is used to get an integer as the predicted class label so 
       #that you can compare it with the true label
       # #compare the prediction with the true label (located at data[4]) of the 
        #test instance to start calculating the accuracy.       
          class_predicted = clf.predict([row])[0]
          if class_predicted == 1:
             if rowsClass == 1:
                TP += 1
             elif rowsClass == 2:
                FN += 1
          elif class_predicted == 2:
             if rowsClass == 2:
                TN += 1
             elif rowsClass == 1:
                FP += 1

    #find the lowest accuracy of this model during the 10 runs (training and 
    #test set)
    #print the lowest accuracy of this model during the 10 runs (training and test 
    #set).
    #your output should be something like that: final accuracy when training on 
    #contact_lens_training_1.csv: 0.2
    #add your Python code down below
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    print("Final accuracy when training on", ds, ": ", accuracy)
