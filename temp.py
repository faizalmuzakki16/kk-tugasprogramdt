#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  3 18:16:28 2018

@author: temperantia
"""


from random import seed
from random import randrange
import random

import csv
import math

"""
============================
load dataset + normalization
============================
"""
def loadCsv(filename, isNormalized=0):
    with open(filename, newline='') as f:
        reader = csv.reader(f)
        dataset = list(reader)        
        random.shuffle(dataset)
        dataset = strColumntoFloat(dataset)
        if (isNormalized == 0):
            return dataset
        minmax = datasetMinmax(dataset)
        normalizeDataset(dataset, minmax)
        return dataset
    
 
def strColumntoFloat(dataset):
    newset=[]
    for column in dataset:
        newcolumn = []
        for item in column:
            item = float(item.strip())
            newcolumn.append(item)
        newset.append(newcolumn)
    return newset
            #row[column] = float(row[column].strip())
 
def datasetMinmax(dataset):
    minmax = list()
    for i in range(len(dataset[0])):
        col_values = [row[i] for row in dataset]
        value_min = min(col_values)
        value_max = max(col_values)
        minmax.append([value_min, value_max])
    return minmax
 
def normalizeDataset(dataset, minmax):
    for row in dataset:
        for i in range(len(row)):
            row[i] = (row[i] - minmax[i][0]) / (minmax[i][1] - minmax[i][0])


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split
"""
================
Impurity node
================
"""


def gain_index (groups, classes, parent_entropy):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = parent_entropy
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if(p!=0):
                score -= p * math.log(p,2)
        # weight the group score by its relative size
        gini -= (score) * (size / n_instances)
    return gini

def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def misscalculation_error(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = -999
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            if(score<p):
                score = p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini
"""
Algorithm
"""
# Split a dataset based on an attribute and an attribute value
def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def get_split(dataset, parent):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    if(impurityAlg==0):
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    elif(impurityAlg==1):
        b_score = -999
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = gain_index(groups, class_values, parent)
                if gini > b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    else :
        for index in range(len(dataset[0])-1):
            for row in dataset:
                groups = test_split(index, row[index], dataset)
                gini = misscalculation_error(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups, 'split': b_score}
 
# Create a terminal node value
def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)
 
# Create child splits for a node or make terminal
def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    # check for a no split
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    # check for max depth
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    # process left child
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left, node['split'])
        split(node['left'], max_depth, min_size, depth+1)
    # process right child
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right, node['split'])
        split(node['right'], max_depth, min_size, depth+1)
 
    
def parent_start(train):
    class_col = train[-1]
    unique_class = list(set(class_col))
    p =0.0
    for unique in unique_class:
        a = class_col.count(unique)/ len(class_col)
        p-= a*math.log(a,2)
    return p
    
# Build a decision tree
def build_tree(train, max_depth, min_size):
    root = get_split(train, parent_start(train))
    split(root, max_depth, min_size, 1)
    return root
 
# Make a prediction with a decision tree
def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']
 
# Classification and Regression Tree Algorithm
def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return([predictions,tree])
 
"""
Accuracy
"""

def confusion_matric(actual, predicted):
    """
    [[TP][FP]
     [FN][TN]]
    """
    matrix = [0,0,0,0]
    for a in range (len(predicted)):
        if (actual[a] == predicted[a]):
            if(predicted[a] == 1.0):
                matrix[0]+=1
            else :
                matrix[1]+=1
        else:
            if(predicted[a] == 1.0):
                matrix[2]+=1
            else :
                matrix[3]+=1
    return matrix

def precision(matrix):
    tp= matrix[0]
    alls = matrix[0]+matrix[1]
    return tp/alls * 100

def recall(matrix):
    tp = matrix[0]
    alls = matrix[0]+ matrix[2]
    return tp/alls * 100

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 1
    return correct / float(len(actual)) * 100.0

def MSE(actual, predicted):
    MSE = 0.0
    leng = len(predicted)
    for i in range(leng):
        MSE += pow(actual[i]-predicted[i],2)
    return MSE/leng

class_n = [['Number of times pregnant',
        'Plasma glucose concentration a 2 hours in an oral glucose tolerance test',
        'Diastolic blood pressure (mm Hg)',
        'Triceps skin fold thickness (mm)',
        '2-Hour serum insulin (mu U/ml)',
        'Body mass index (weight in kg/(height in m)^2)',
        'Diabetes pedigree function'
        'Age (years)',
        'Class variable (0 or 1)'
        ],
        ['per capita crime rate by town',
         'proportion of residential land zoned for lots over \
                 25,000 sq.ft.',
         'Charles River dummy variable (= 1 if tract bounds \
                 river; 0 otherwise)',
         'nitric oxides concentration (parts per 10 million)',
         'average number of rooms per dwelling',
         'proportion of owner-occupied units built prior to 1940',
         'weighted distances to five Boston employment centres',
         'index of accessibility to radial highways',
         'full-value property-tax rate per $10,000',
         'pupil-teacher ratio by town',
         '1000(Bk - 0.63)^2 where Bk is the proportion of blacks \
                 by town',
         '% lower status of the population',
         'Median value of owner-occupied homes in $1000\'s'
         ]]
        
itterr=0
def test_predict(node, row,nama):
    global itterr
    for a in range (itterr): 
        print ('|',end='')
    print(' is '+ class_n[nama][node['index']]+" less than :"+str(node['value']))
    
    itterr=itterr+1
    for a in range (itterr): 
        print ('|',end='')
    if row[node['index']] < node['value']:
        print('yes')
        itterr=itterr+1
        if isinstance(node['left'], dict):
            
            return test_predict(node['left'], row,nama)
        else:
            for a in range (itterr): 
                print ('|',end='')
            print("judge : " + str(node['left']))
            return node['left']
    else:
        print('no')
        itterr=itterr+1
        if isinstance(node['right'], dict):
            
            return test_predict(node['right'], row,nama)
        else:
            for a in range (itterr): 
                print ('|',end='')
            print("judge : " + str(node['right']))
            return node['right']
 
    

# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        score = []
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted[0])
        score.append(accuracy)
        matrix = confusion_matric(actual, predicted[0])
        score.append(precision(matrix))
        score.append(recall(matrix))
        scores.append(score)
    returned = {'scores': scores, 'tree':predicted[1]}
    return returned

"""
================
Main
================
"""
impurityAlg=0
impurityName=['gini','entropy', 'miscalculation error']


def main ():
#    select = input("klasifikasi || regresi")
    
    global impurityAlg
#    impurityAlg = input("'gini','entropy', 'miscalculation error' : ")
    # Test CART on Bank Note dataset
    seed(1)
    # load and prepare datay
    filenames = ['pima-indians-diabetes.data','housing.data']
    for select in range (2):        
        if select == 0:
            print('Classification')
        else:
            print('Regression')
        for impurityAlg in range(3):
            global itterr
            itterr = 0
            print(impurityName[impurityAlg])
            if (select ==0):
                filename = filenames[0]
                dataset = loadCsv(filename)
                n_folds = 5
                max_depth = 5
                min_size = 10
                get= evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)
                scores = get['scores']
                tree = get['tree']
                x = 0
                accuracy=[]
                for a in range( len(scores)):
                    accuracy.append(scores[a][0])
                    print('accuracy '+str(x) +': '+ str(scores[a][0]))
                    print('precission'+str(x) +': '+ str(scores[a][1]))
                    print('recall '+str(x) +': '+ str(scores[a][2]))
                    x+=1
            #    print('Scores: %s' % scores)
                print('Mean Accuracy: %.3f%%' % (sum(accuracy)/float(len(accuracy))))
                test = dataset[-1]
                test_predict(tree,test,0)
            
            else:
                filename = filenames[1]
                dataset = loadCsv(filename)
                n_folds = 10
                max_depth = 5
                min_size = 10
                scores = get['scores']
                tree = get['tree']
                x = 0
                accuracy=[]
                for a in range( len(scores)):
                    print('MSE '+str(x) +': '+ str(scores[a]))
                    x+=1
                    
                print('Scores: %s' % scores)
                #print('Mean Accuracy: %.3f%%' % (sum(scores)/float(len(scores))))
                test = dataset[-1]
                test_predict(tree,test,1)
    
main()
