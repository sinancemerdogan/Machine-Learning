import numpy as np
import pandas as pd



X = np.genfromtxt("data_points.csv", delimiter = ",", dtype = str)
y = np.genfromtxt("class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# first 50000 data points should be included to train
# remaining 43925 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    
    X_train = X[:50000,:]
    y_train = y[:50000]
    X_test = X[50000:]
    y_test = y[50000:]
    
    
    # your implementation ends above  
    
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    
    class_priors = []
    
    prob_class_1 = np.count_nonzero(y == 1) / len(y)
    prob_class_2 = 1 - prob_class_1
    
    class_priors.append(prob_class_1)
    class_priors.append(prob_class_2)
    # your implementation ends above
    
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 5
# assuming that there are K classes and D features
# should return four numpy arrays with shape (K, D)
def estimate_nucleotide_probabilities(X, y):
    # your implementation starts below
    
    pAcd = np.zeros((2,7))
    pCcd = np.zeros((2,7))
    pGcd = np.zeros((2,7))
    pTcd = np.zeros((2,7))
    
    
    N = len(X)
    NFeateures = len(X[0])
    N1 = np.sum(y == 1)
    N2 = np.sum(y == 2)
    
    for d in range(NFeateures):
        for n in range(N):
            if X[n,d] == 'A':
                if y[n] == 1:
                    pAcd[0,d] += 1
                else:
                    pAcd[1,d] += 1
            
            elif X[n,d] == 'C':
                if y[n] == 1:
                    pCcd[0,d] += 1
                else:
                    pCcd[1,d] += 1
            
            elif X[n,d] == 'G':
                if y[n] == 1:
                    pGcd[0,d] += 1
                else:
                    pGcd[1,d] += 1
                    
            elif X[n,d] == 'T':
                if y[n] == 1:
                    pTcd[0,d] += 1
                else:
                    pTcd[1,d] += 1
                    
    pAcd[0] /= N1
    pAcd[1] /= N2
    
    pCcd[0] /= N1
    pCcd[1] /= N2
    
    pGcd[0] /= N1
    pGcd[1] /= N2
    
    pTcd[0] /= N1
    pTcd[1] /= N2
    
    # your implementation ends above
    return(pAcd, pCcd, pGcd, pTcd)

pAcd, pCcd, pGcd, pTcd = estimate_nucleotide_probabilities(X_train, y_train)
print(pAcd)
print(pCcd)
print(pGcd)
print(pTcd)



# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, pAcd, pCcd, pGcd, pTcd, class_priors):
    # your implementation starts below
    
    N = len(X)
    NFeateures = len(X[0])
    K = 2
    
    score_values = np.zeros((N,K))
    
    for n in range(N):
        for k in range(K):
            tempScore = 1
            for d in range(NFeateures):
                if X[n,d] == 'A':
                        tempScore *= pAcd[k,d]
                if X[n,d] == 'C':
                        tempScore *= pCcd[k,d]
                if X[n,d] == 'G':
                        tempScore *= pGcd[k,d]
                if X[n,d] == 'T':
                        tempScore *= pTcd[k,d]
                        
            score_values[n,k] += np.log(class_priors[k]) + np.log(tempScore)
    
    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_train)

scores_test = calculate_score_values(X_test, pAcd, pCcd, pGcd, pTcd, class_priors)
print(scores_test)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    N = len(y_truth)
    y_pred = np.empty(N)
    
    for n in range(N):
        if scores[n,0] > scores[n,1]:
            y_pred[n] = 1
        else:
            y_pred[n] = 2

    tp = np.sum((y_truth == 2) & (y_pred == 2))
    tn = np.sum((y_truth == 1) & (y_pred == 1))
    fp = np.sum((y_truth == 2) & (y_pred == 1))
    fn = np.sum((y_truth == 1) & (y_pred == 2))

    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, scores_test)
print(confusion_test)
