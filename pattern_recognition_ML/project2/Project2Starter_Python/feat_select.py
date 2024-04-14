'''
Start code for Project 2
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: A. Burak Gulhan
    PSU Email ID: abg6029@psu.edu
    Description: 
        forward_selection: Perform forward selection using F1-score citerion. Parameter 'classifier' allows for choosing what type of classifier to use.
        filter_method: Calculates variance ratio for features
}
'''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# TODO: Implement the sequential forward selection algorithm.
def forward_selection(train_data, train_labels, select=20, classifier="linear"):
    """
    Performs forward selection.
    Args:
        train_data (numpy array): [NxM] vector of training data, where N is the number of samples and M is the number of features.
        train_labels (numpy array): [Nx1] vector of training labels.
        select (int): how many features to select
        classifier ("linear", "knn", "tree"): which classifer to use
    Returns:
        selected_inds (numpy array): a [1xK] vector containing the indices of features
            selected in forward selection. K is the # of feats choosen before selection was terminated.
    """

    # choose a classifier
    if classifier == "linear": 
        clf = LinearDiscriminantAnalysis()
    elif classifier == "knn":
        clf = KNeighborsClassifier()
    elif classifier == "tree":
        clf = DecisionTreeClassifier()
    elif classifier == "randomforest":
        clf = RandomForestClassifier()
    else:
        print("using default classifier")
        clf = LinearDiscriminantAnalysis()
    
    '''
    # perform forward feature selection
    sfs = SequentialFeatureSelector(estimator=clf, n_features_to_select=select, direction="forward", n_jobs=-1)
    sfs.fit(train_data, train_labels)

    # get indices of best featuress
    selected_inds = sfs.get_support(indices=True)
    '''
    
    selected_inds = []
    M = train_data.shape[1] # number of features
    for n in range(select):
        feat_scores = {} # store scores of each feature
        for i in range(M):
            if i in selected_inds: # ignore any feature we selected before
                continue
            train_data_ = train_data[:,selected_inds+[i]] # extract required columns from data
            # split train & test data
            X_train, X_test, y_train, y_test = train_test_split(train_data_, train_labels, test_size=0.66, stratify=train_labels, random_state=i) # use stratify to ensure even balance of classes in each split

            clf.fit(X_train, y_train)
            pred = clf.predict(X_test)
            score = f1_score(y_true=y_test, y_pred=pred, average="micro")
            feat_scores[i] = score
        # get the highest scoring feature for this iteration
        selected_inds.append(max(feat_scores, key=feat_scores.get))
        #print(feat_scores)
        print(f"selected feature {max(feat_scores, key=feat_scores.get)} with score {feat_scores[max(feat_scores, key=feat_scores.get)]}")

        
            
    '''        
    ## Select 20 features at random as an example
    selected_inds = np.random.choice(train_data.shape[1], 20, replace=False)
    '''
    
    return np.array(selected_inds)

# TODO: Implement the filtering method.
def filter_method(train_data, train_labels):
    """
    Performs filter method.
    Args:
        train_data (numpy array): [NxM] vector of training data, where N is the number of samples and M is the number of features.
        train_labels (numpy array): [Nx1] vector of training labels.
    Returns:
        selected_inds (numpy array): a [1xM] vector sorted in descending order of feature importance.
        scores (numpy array): a [1xM] vector containing the scores of the corresponding features.
    """
    # The current method just returns a random selection of features. 
    selected_inds = np.zeros(train_data.shape[1], dtype=np.int32)
    scores = np.zeros(train_data.shape[1])

    # Caluclate variances for all features
    variances = np.var(train_data, axis=0)
    #print(variances.shape)
    #print(scores.shape)
    assert(variances.shape == scores.shape)

    # calculate variances for each class
    features = np.unique(train_labels)
    C = features.size # number of classes
    M = train_data.shape[1] # number of features
    class_variances = np.zeros((C, M)) # stores all class variances
    for f in features:
        ind = np.where(train_labels == f)[0]
        class_data = train_data[ind]
        class_variances[f] = np.var(class_data)

    # sum up all class variances
    class_var_sum = np.sum(class_variances, axis=0)

    # calculate feature scores, using variance ratio
    scores = C*variances/class_var_sum
    selected_inds = np.flip(np.argsort(scores))

    #selected_inds = np.random.choice(train_data.shape[1], train_data.shape[1], replace=False)
    #scores = np.random.uniform(0, 100, train_data.shape[1])
    #print(f"scores {scores}")
    #print(f"selected inds {selected_inds}")
    return selected_inds, scores
    
