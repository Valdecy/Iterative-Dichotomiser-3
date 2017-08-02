############################################################################

# Created by: Prof. Valdecy Pereira, D.Sc.
# UFF - Universidade Federal Fluminense (Brazil)
# email:  valdecy.pereira@gmail.com
# Course: Data Mining
# Lesson: Decision Trees - ID3

# Citation: 
# PEREIRA, V. (2018). Project: ID3, File: Python-DM-Classification-01-ID3.py, GitHub repository: <https://github.com/Valdecy/ID3>

############################################################################

# Installing Required Libraries
import pandas as pd
import numpy  as np

# Function: dt_id3
def dt_id3(Xdata, ydata):
    
    ################     Part 1 - Preprocessing    #############################
    # Preprocessing - Creating Dataframe
    name = ydata.name
    ydata = pd.DataFrame(ydata.values.reshape((ydata.shape[0], 1)))
    dataset = pd.concat([ydata, Xdata], axis = 1)
    dataset = dataset.applymap(str)
    
    # Preprocessing - Unique Words List
    unique = []
    uniqueWords = []
    for j in range(0, dataset.shape[1]): 
        for i in range(0, dataset.shape[0]):
            token = dataset.iloc[i, j]
            if not token in unique:
                unique.append(token)
        uniqueWords.append(unique)
        unique = []  
    
    # Preprocessing - Label Matrix
    label = np.array(uniqueWords[0])
    label = label.reshape(1, len(uniqueWords[0]))
    
    ################    Part 2 - Initialization    #############################
    # ID3 - Initializing Variables
    i = 0
    branch = [None]*1
    branch[0] = dataset
    gain = np.empty([1, branch[i].shape[1]])
    rule = [None]*1
    rule[0] = "IF "
    root_index = 0
    skip_update = False
    stop = 2
    
    ################     Part 3 - id3 Algorithm    #############################
    # ID3 - Algorithm
    while (i < stop):
        entropy = 0
        denominator_1 = branch[i].shape[0]
        for entp in range(0, label.shape[1]):
            numerator = (branch[i][(branch[i].iloc[:,0] == label[0, entp])].count())[0]
            if numerator > 0:
                entropy = entropy - (numerator/denominator_1)* np.log2((numerator/denominator_1))
        gain.fill(entropy)
        for element in range(1, branch[i].shape[1]):
            if len(branch[i]) == 0:
                skip_update = True
                break
            if len(np.unique(branch[i][0])) == 1 or len(branch[i]) == 1 or gain.sum(axis=1) == 0:
                 rule[i] = rule[i] + " THEN " + name + " = " + branch[i].iloc[0, 0] + "."
                 rule[i] = rule[i].replace(" AND  THEN ", " THEN ")
                 skip_update = True
                 break
            for word in range(0, len(uniqueWords[element])):
                denominator_2 = (branch[i][(branch[i].iloc[:, element] == uniqueWords[element][word])].count())[0]
                for lbl in range(0, label.shape[1]):
                    numerator = (branch[i][(branch[i].iloc[:,0] == label[0, lbl]) & (branch[i].iloc[:, element] == uniqueWords[element][word])].count())[0]
                    if numerator > 0:
                        gain[0, element] = gain[0, element] + (denominator_2/denominator_1)* (numerator/denominator_2)* np.log2((numerator/denominator_2))
    
        if skip_update == False:
            gain[0,0] = -gain[0,0]
            root_index = np.argmax(gain)
            gain[0,0] = -gain[0,0]
            rule[i] = rule[i] + list(branch[i])[root_index]
            
            for word in range(0, len(uniqueWords[root_index])):
                branch.append(branch[i][branch[i].iloc[:, root_index] == uniqueWords[root_index][word]])
                rule.append(rule[i] + " = " + "{" + uniqueWords[root_index][word] + "}")
            
            for logic_connection in range(1, len(rule)):
                if len(np.unique(branch[i][0])) != 1 and rule[logic_connection].endswith(" AND ") == False  and rule[logic_connection].endswith("}") == True:
                    rule[logic_connection] = rule[logic_connection] + " AND "
        skip_update = False
        i = i + 1
        print("iteration: ", i)
        stop = len(rule)
    
    for i in range(len(rule) - 1, -1, -1):
        if rule[i].endswith(".") == False:
            del rule[i]
    
    rule.append("1) Total Number of Rules: " + str(len(rule)))
    rule.append("2) When No Rule Applies: " + name + " = " + dataset.agg(lambda x:x.value_counts().index[0])[0]) 
    
    return rule

    ############### End of Function ##############

######################## Part 4 - Usage ####################################

df = pd.read_csv('Python-DM-Classification-01-ID3.csv', sep = ';')

X = df.iloc[:, 0:4] # Attributes
y = df.iloc[:, 4]   # Target

dt_id3(Xdata = X, ydata = y)

########################## End of Code #####################################
