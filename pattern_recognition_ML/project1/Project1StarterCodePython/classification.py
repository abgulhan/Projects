'''
Start code for Project 1-Part 2: Classification
CSE583/EE552 PRML
TA: Keaton Kraiger, 2023
TA: Shimian Zhang, 2023

Your Details: (The below details should be included in every python 
file that you add code to.)
{
    Name: A. Burak Gulhan
    PSU Email ID: abg6029@psu.edu
    Description: (A short description of what each of the functions you've written does.).
}
'''

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, classification_report
from sklearn.neighbors import KNeighborsClassifier

# TODO: import your implemented model here or use one of sklearn's classifiers 
# from models import Classifier
import models

# Possible classifiers include LDA, KNN, etc. or any other classifier 
# from sklearn
# from sklearn.neighbors import KNeighborsClassifier, ...

def viz_desc_bounds(classifier, feats, labels, idxA, idxB):
    """
    Visualizes the decision boundaries of a classifier trained on two features of the dataset.
    Args:
        classifier: linear classifier trained on 2 features.
        feats: features to be used for visualization.
        labels: labels to be used for visualization.
        idxA & idxB: indices of the features to be used for visualization. 
    """
    if not os.path.exists('results'):
        os.makedirs('results')

    ys = np.sort(np.unique(labels))
    y_ind = np.searchsorted(ys, labels)

    fig, ax = plt.subplots()

    x0, x1 = feats[:, 0], feats[:, 1]
    all_feats = np.concatenate((x0, x1))
    pad = np.percentile(all_feats, 60)

    x_min, x_max = x0.min() - pad, x0.max() + pad
    y_min, y_max = x1.min() - pad, x1.max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    preds = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    preds = preds.reshape(xx.shape)

    lut = np.sort(np.unique(labels)) 
    ind = np.searchsorted(lut,preds)

    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    ax.contourf(xx, yy, preds, cmap=plt.cm.Pastel1, alpha=0.8)
    for i in range(len(lut)):
        ax.scatter(x0[y_ind == i], x1[y_ind == i], color=plt.cm.jet(i/len(lut)), s=50, edgecolors='k', marker=markers[i])

    ax.set_xlabel(f'Feature {idxA}')
    ax.set_ylabel(f'Feature {idxB}')
    ax.set_title('Decision Boundary')

    handles = []
    markers = ["o", "v", "P", "X", "s", "p", "h", "d"]
    handles = [plt.plot([],[],color=plt.cm.jet(i/len(lut)), ls="", marker=markers[i])[0] for i in range(len(lut))]
    labels = [f'Class {i}' for i in lut]
    ax.legend(handles, labels, loc='upper right')
    plt.show()
    plt.savefig('results/decision_boundary.png')



def split_data(data, label, sub_info, i):
    """
    Splits the data in a LOSO fashion with the i-th subject as test data.
    Args:
        data (numpy array): [NxM] vector of data to be split, where N is the number of samples and M is the number of features.
        label (numpy array): [N] vector of labels of the data.
        sub_info (numpy array): [sxN] vector of subject information of the data where s is the number of subjects and N is the number of samples.
        i (int): Test subject.
    Returns:
        train_data: Training data.
        train_label: Training labels.
        test_data: Test data.
        test_label: Test labels.
    """
    test_inds = np.where(sub_info == i)[0]
    train_inds = np.where(sub_info != i)[0]
    train_data = data[train_inds, :]
    train_label = label[train_inds]
    test_data = data[test_inds, :]
    test_label = label[test_inds]
    return train_data, train_label, test_data, test_label


def load_dataset(dataset='taiji', verbose=False, subject_index=3):
    '''
    Loads the taiji dataset.
    Args:
        dataset: name of the dataset to load. Currently only taiji is supported.
        verbose: print dataset information if True.
        subject_index: subject index to use for LOSO. The subject with this index will be used for testing. 

    Returns (all numpy arrays):
        train_feats: training features.
        train_labels: training labels.
        test_feats: testing features.
        test_labels: testing labels.
    '''

    if dataset == 'taiji':
        labels = np.loadtxt("data/taiji/taiji_labels.csv", delimiter=",", dtype=int)
        person_idxs = np.loadtxt("data/taiji/taiji_person_idx.csv", delimiter=",", dtype=int)
        feats = np.loadtxt("data/taiji/taiji_quat.csv", delimiter=",", dtype=float)

        # Combine repeated positions
        labels[labels == 4] = 2
        labels[labels == 8] = 6

        # Remove static dimensions. Get mask of all features with zero variance
        feature_mask = np.var(feats, axis=1) > 0

        # Train mask
        train_mask = person_idxs != subject_index

        train_feats = feats[feature_mask, :][:, train_mask].T
        train_labels = labels[train_mask].astype(int)
        test_feats = feats[feature_mask, :][:, ~train_mask].T
        test_labels = labels[~train_mask].astype(int)


    if verbose:
        print(f'{dataset} Dataset Loaded')
        print(f'\t# of Classes: {len(np.unique(train_labels))}')
        print(f'\t# of Features: {train_feats.shape[1]}')
        print(f'\t# of Training Samples: {train_feats.shape[0]}')
        print('\t# per Class in Train Dataset:')
        for cls in np.unique(train_labels):
            print (f'\t\tClass {cls}: {np.sum(train_labels == cls)}')
        print(f'\t# of Testing Samples: {test_feats.shape[0]}')
        print('\t# per Class in Test Dataset:')
        for clas in np.unique(test_labels):
            print(f'\t\tClass {clas}: {np.sum(test_labels == clas)}')
        
    return train_feats, train_labels, test_feats, test_labels

def plot_conf_mats(dataset, **kwargs):
    """
    Plots the confusion matrices for the training and testing data.
    Args:
        dataset: name of the dataset.
        train_labels: training labels.
        pred_train_labels: predicted training labels.
        test_labels: testing labels.
        pred_test_labels: predicted testing labels.
    """

    train_labels = kwargs['train_labels']
    pred_train_labels = kwargs['pred_train_labels']
    test_labels = kwargs['test_labels']
    pred_test_labels = kwargs['pred_test_labels']

    train_confusion = confusion_matrix(train_labels, pred_train_labels)
    test_confusion = confusion_matrix(test_labels, pred_test_labels)

    # calculate f1 scores
    f1_train = f1_score(y_true=train_labels, y_pred=pred_train_labels, average='micro')
    f1_test = f1_score(y_true=test_labels, y_pred=pred_test_labels, average='micro')

    # Plot the confusion matrices as seperate figures
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=train_confusion, display_labels=np.unique(train_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title(f'Training Confusion Matrix, F1 score: {f1_train:.4}')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_train_confusion.png', bbox_inches='tight', pad_inches=0)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=test_confusion, display_labels=np.unique(test_labels))
    disp.plot(ax=ax, xticks_rotation='vertical')
    ax.set_title(f'Testing Confusion Matrix, F1 score: {f1_test:.4}')
    plt.tight_layout()
    plt.savefig(f'results/{dataset}_test_confusion.png', bbox_inches='tight', pad_inches=0)

def example_decision_boundary(dataset='taiji', indices=[0, 6]):
    """
    An example of how to visualize the decision boundary of a classifier.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    dc_train_feats = train_feats[:, indices]
    dc_test_feats = test_feats[:, indices]

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(dc_train_feats, train_labels)

    # Visualize the decision boundary
    viz_desc_bounds(clf, dc_test_feats, test_labels, indices[0], indices[1])

def example_classification(dataset='taiji'):
    """
    An example of performing classification. Except you will need to first project the data.
    """
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)

    # Example Linear Discriminant Analysis classifier with sklearn's implemenetation
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_feats, train_labels)

    # Predict the labels of the training and testing data
    pred_train_labels = clf.predict(train_feats)
    pred_test_labels = clf.predict(test_feats)

    # Get statistics
    plot_conf_mats(dataset, train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)

# TODO: Implement your Fisher projection
# result dims is the number of dimensions to return. 
# That is, it determines how many times the data is projected.
def fisher_projection(train_feats, train_labels, result_dims=None):

    D = train_feats.shape[1] # feature dimensions
    N = train_feats.shape[0] # number of data points
    classes = np.unique(train_labels) # vector of classes found in train_labels
    S_W = np.zeros((D, D)) # within class covariance matrix
    S_B = np.zeros((D, D)) # between class covariance matrix
    S_T = np.zeros((D, D)) # total covariance matrix

    # Calculate S_W, within class covariance matrix
    for k in range(classes.size):
        class_name = classes[k]
        ind = np.where(train_labels == class_name)[0] # find indices corresponding to class_name
        x = train_feats[ind]
        mean_k = x.sum(axis=0)/ind.size # calculate mean vector of class_name
        S_k = (x-mean_k).T @ (x-mean_k) # subtract mean from each row in x. Numpy automatically broadcasts matrix vector subtraction.
                                        # Note: (x-mean) in this code is a row vector, whereas in the textbook equation it is a column vector
        S_W += S_k # update S_w matrix
        
    # Calculate S_T to find S_B using formula 4.46: S_T = S_W + S_B
    # I do this instead of directly calculating S_B, because calculating S_T is easier to code
    mean = train_labels.sum(axis=0)/N # total mean of dataset
    S_T = (x-mean).T @ (x-mean) # Formula 4.43 from textbook. Note that (x-mean) is a row vector, whereas in the textbook it is a column vector

    # Calculate S_B using formula 4.46 from textbook: S_T = S_W + S_B
    S_B = S_T - S_W

    # round so that np.eig does not return complex numbers
    #S_W = np.round(S_W, decimals=3)
    #S_W_inv = np.round(np.linalg.inv(S_W), decimals=3)
    S_B = np.round(S_B, decimals=3)
    

    # From class slides: S_B*w = lambda*S_W*w -> a generalized eigenvalue problem
    # can convert into standard eigenvalue problem: (S_W)^-1 *S_B*w = lambda*w

    eig_mat = np.linalg.inv(S_W)@S_B # matrix to find eigenvectors

    w, v = np.linalg.eig(eig_mat) # w=eigenvalues, v=eigenvectors

    # sort eigenvectors from largest to smallest
    # since there are negative eigenvalues resulting from np.eig, sort by magnitude
    sort_ind = np.flip(np.absolute(w).argsort())
    w = w[sort_ind]
    v = v[sort_ind]

    if result_dims == None:
        return v
    else:
        return v[:result_dims]

# TODO: Use the exisintg functions load_dataset and plot_conf_mats. Using a classifier (either one you write
# or an imported sklearn function), perform classification on the fisher projected data.
def classification(dataset='taiji'):
    train_feats, train_labels, test_feats, test_labels = load_dataset(dataset=dataset)
    K = np.unique(train_labels).size # number of classes
    result_dims = 7 # determines final dimensions of the weight vector
    W = fisher_projection(train_feats, train_labels, result_dims=result_dims)

    #print(f"Weights shape: {W.shape}")
    #print(f"train features shape: {train_feats.shape}")

    train_feats_proj = train_feats@W.T
    test_feats_proj = test_feats@W.T

    #train_feats_proj = train_feats
    #test_feats_proj = test_feats
    #print(f"train features original: {train_feats.shape}")
    #print(f"train features projected: {train_feats_proj.shape}")

    # use KNN to classify
    KNN = KNeighborsClassifier(n_neighbors=K)
    KNN.fit(train_feats_proj, train_labels)
    pred_train_labels = KNN.predict(train_feats_proj)
    pred_test_labels = KNN.predict(test_feats_proj)

    f1_train = f1_score(y_true=train_labels, y_pred=pred_train_labels, average='micro')
    f1_test = f1_score(y_true=test_labels, y_pred=pred_test_labels, average='micro')

    print(f"train f1: {f1_train}")
    print(f"test f1: {f1_test}")

    #indices=[0, 6]  
    #viz_desc_bounds(KNN, test_feats_proj, pred_test_labels, indices[0], indices[1])
    report = classification_report(y_true=test_labels, y_pred=pred_test_labels)
    print(report)


    plot_conf_mats(dataset='taiji', train_labels=train_labels, pred_train_labels=pred_train_labels, test_labels=test_labels, pred_test_labels=pred_test_labels)

    return pred_train_labels, pred_test_labels

def main():
    classification()
    example_decision_boundary()
    

if __name__ == '__main__':
    main()

