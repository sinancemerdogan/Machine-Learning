import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X_train = np.genfromtxt(fname = "data_points.csv", delimiter = ",", dtype = float)
y_train = np.genfromtxt(fname = "class_labels.csv", delimiter = ",", dtype = int)



# STEP 3
# assuming that there are K classes
# should return a numpy array with shape (K,)
def estimate_prior_probabilities(y):
    # your implementation starts below
    
    #Reference to the Lab 01 - Parametric Methods, ENGR-421 (Spring 2023)
    K = len(np.unique(y))
    class_priors = np.array([np.mean(y == (c + 1)) for c in range(K)])
    
    # your implementation ends above
    return(class_priors)

class_priors = estimate_prior_probabilities(y_train)
print(class_priors)



# STEP 4
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D)
def estimate_class_means(X, y):
    # your implementation starts below
    K = len(np.unique(y))
    
    #Reference the Lab 01 - Parametric Methods, ENGR-421 (Spring 2023)
    sample_means = np.array([np.mean(X[y == (c + 1)], axis=0) for c in range(K)])
    
    # your implementation ends above
    return(sample_means)

sample_means = estimate_class_means(X_train, y_train)
print(sample_means)



# STEP 5
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_class_covariances(X, y):
    # your implementation starts below
    K = len(np.unique(y))
    D = X.shape[1]
    
    sample_covariances = np.zeros((K,D,D))
    
    sample_covariances = np.array([(np.matmul(np.transpose(X[y == (c + 1)] - sample_means[c]), (X[y == (c + 1)] - sample_means[c])) / np.count_nonzero(y == c + 1))for c in range(K)])
    
    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_class_covariances(X_train, y_train)
print(sample_covariances)

# STEP 6
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def calculate_score_values(X, class_means, class_covariances, class_priors):
    # your implementation starts below
    
    K = len(class_priors)
    N = X.shape[0]
    D = X.shape[1]
    
    score_values = np.zeros((N,K))

    for n in range(N):
        for c in range(K):

            q = -D/2 * np.log(2 * np.pi)
            
            w = -1/2 * np.log(np.linalg.det(sample_covariances[c]))
            
            e1 = np.transpose(X[n] - sample_means[c])
            e2 = np.linalg.inv(sample_covariances[c])
            e3 = X[n] - sample_means[c]
            e4 = np.matmul(e1, e2)
            e5 = np.matmul(e4, e3)
            
            e = -1/2 * e5
            
            r = np.log(class_priors[c]) 
           
            score_values[n][c] = q + w + e + r

    # your implementation ends above
    return(score_values)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)



# STEP 7
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, scores):
    # your implementation starts below
    
    N = len(y_truth)
    y_pred = np.empty(N)

    for n in range(N):
        
        maxClassNumber = np.array(scores[n]).argmax()
        y_pred[n] = maxClassNumber
        
    confusion_matrix = np.array(pd.crosstab(y_pred, y_truth))
    
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)



def draw_classification_result(X, y, class_means, class_covariances, class_priors):
    class_colors = np.array(["#1f78b4", "#33a02c", "#e31a1c", "#ff7f00", "#6a3d9a"])
    K = np.max(y)

    x1_interval = np.linspace(-75, +75, 151)
    x2_interval = np.linspace(-75, +75, 151)
    x1_grid, x2_grid = np.meshgrid(x1_interval, x2_interval)
    X_grid = np.vstack((x1_grid.flatten(), x2_grid.flatten())).T
    scores_grid = calculate_score_values(X_grid, class_means, class_covariances, class_priors)

    score_values = np.zeros((len(x1_interval), len(x2_interval), K))
    for c in range(K):
        score_values[:,:,c] = scores_grid[:, c].reshape((len(x1_interval), len(x2_interval)))

    L = np.argmax(score_values, axis = 2)

    fig = plt.figure(figsize = (6, 6))
    for c in range(K):
        plt.plot(x1_grid[L == c], x2_grid[L == c], "s", markersize = 2, markerfacecolor = class_colors[c], alpha = 0.25, markeredgecolor = class_colors[c])
    for c in range(K):
        plt.plot(X[y == (c + 1), 0], X[y == (c + 1), 1], ".", markersize = 4, markerfacecolor = class_colors[c], markeredgecolor = class_colors[c])
    plt.xlim((-75, 75))
    plt.ylim((-75, 75))
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")
    plt.show()
    return(fig)
    
fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_different_covariances.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are K classes and D features
# should return a numpy array with shape (K, D, D)
def estimate_shared_class_covariance(X, y):
    # your implementation starts below

    N = X.shape[0]
    D = X.shape[1]
    K = len(np.unique(y))
    u = np.mean(X, axis=0)
    sample_covariances = np.zeros((K,D,D))
     
    sample_covariances = np.array([(np.matmul(np.transpose(X - u), (X - u))) / N for c in range(K)])

    # your implementation ends above
    return(sample_covariances)

sample_covariances = estimate_shared_class_covariance(X_train, y_train)
print(sample_covariances)

scores_train = calculate_score_values(X_train, sample_means,
                                      sample_covariances, class_priors)
print(scores_train)

confusion_train = calculate_confusion_matrix(y_train, scores_train)
print(confusion_train)

fig = draw_classification_result(X_train, y_train, sample_means, sample_covariances, class_priors)
fig.savefig("hw02_result_shared_covariance.pdf", bbox_inches = "tight")
