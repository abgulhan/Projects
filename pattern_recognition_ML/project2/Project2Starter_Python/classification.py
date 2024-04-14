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
        train(): added alternative normalization code and code for making histograms
}
'''

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier


from feat_select import *
from util import *


def train(args):
    """
    Perform the feature selection
    """
    data, labels, sub_info, form_names, feat_names  = load_dataset()
    num_subs = len(np.unique(sub_info[:, 0]))
    num_feats = data.shape[1]
    num_forms = len(form_names) + 1
    filter_feat_count = args.filter_feat_count

    sub_train_acc = np.zeros(num_subs)
    sub_class_train = np.zeros((num_subs, num_forms))
    sub_test_acc = np.zeros(num_subs)
    sub_class_test = np.zeros((num_subs, num_forms))
    overall_train_mat = np.zeros((num_forms, 1))
    overall_test_mat = np.zeros((num_forms, 1))

    if args.save_results:
        if not os.path.exists(os.path.join(args.save_dir, 'stats')):
            os.makedirs(os.path.join(args.save_dir, 'stats'))

    # data structures for plotting histograms
    feature_hist_filter = {}
    feature_hist_wrapper = {}
    
    for i in range(0, num_subs):
        print(f'Starting training and evaluation for subject {i+1}...')
        start = time.time()

        # Split and normalize data

        train_data, train_label, test_data, test_label = split_data(data, labels, sub_info, i+1)
        #train_data, min_value, max_value = normalize_data_column(train_data, interval_start=0, interval_end=52)
        #train_data, min_value2, max_value2 = normalize_data(train_data, interval_start=52, interval_end=-1)
        train_data, min_value, max_value = normalize_data_column(train_data)
        
        #print(train_data)
        #print(train_data.shape)
        #print(f"min value: {min_value}")
        #print(f"max value: {max_value}")
        #print(f"min value2: {min_value2}")
        #print(f"max value2: {max_value2}")



        # TODO: Add data normalization here. Implement the code in the normalize_data function in util.py 
        # Apply normalization to test data
        #test_data, _, _ = normalize_data_column(test_data, min_value, max_value, interval_start=0, interval_end=52)
        #test_data, _, _ = normalize_data(test_data, min_value2, max_value2, interval_start=52, interval_end=-1)
        test_data, _, _ = normalize_data(test_data, min_value, max_value)

        # Perform feature filtering 
        if num_feats < filter_feat_count:
            filter_feat_count = num_feats

        # TODO: Implement the filter method. 
        
        filter_inds, filter_scores = filter_method(train_data, train_label)
        print("filtering done")
        train_data = train_data[:, filter_inds[:filter_feat_count]]
        test_data = test_data[:, filter_inds[:filter_feat_count]]
        
        # save data for histogram
        for k in range(num_feats):
            feat_ind = filter_inds[k]
            if feat_ind not in feature_hist_filter:
                feature_hist_filter[feat_ind] = [filter_scores[k]]
            else:
                feature_hist_filter[feat_ind].append(filter_scores[k])
        

        # TODO: Implement the forward selection method.
        # Perform feature selection
        
        selected_inds = forward_selection(train_data, train_label, select=20, classifier="linear")
        print("forward selection done")
        
        # save data for histogram
        for k in range(len(selected_inds)):
            feat_ind = selected_inds[k]
            if feat_ind not in feature_hist_wrapper:
                feature_hist_wrapper[feat_ind] = 1
            else:
                feature_hist_wrapper[feat_ind]+=1


        train_data = train_data[:, selected_inds]
        test_data = test_data[:, selected_inds]
        
        # Train and evaluate the model
        #model = KNN(n_neighbors=10) # You are free to use any classifier/classifier configuration
        model = RandomForestClassifier()
        model.labels_ = np.unique(train_label)
        model.fit(train_data, train_label)
        print("Model done")

        sub_pred_train = model.predict(train_data)
        sub_pred_test = model.predict(test_data)

        # Get model evaluation metrics
        train_acc = np.sum(sub_pred_train == train_label) / len(train_label)
        sub_train_acc[i] = train_acc
        test_acc = np.sum(sub_pred_test == test_label) / len(test_label)
        sub_test_acc[i] = test_acc

        # Get subject-wise statistics
        sub_classes_train, sub_conf_mat_train = sub_stats(sub_pred_train, train_label, num_forms)
        sub_classes_test, sub_conf_mat_test = sub_stats(sub_pred_test, test_label, num_forms)
        sub_class_train[i, :] = sub_classes_train
        sub_class_test[i, :] = sub_classes_test

        # Get overall statistics
        overall_train_mat = overall_train_mat + (1/num_subs) * sub_conf_mat_train
        overall_test_mat = overall_test_mat + (1/num_subs) * sub_conf_mat_test

        # Print results
        print(f'Training accuracy for subject {i+1}: {train_acc}')
        print(f'Testing accuracy for subject {i+1}: {test_acc}')

        # Save results
        
        if args.save_results:
            sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i+1}.npz')
            np.savez(sub_file, filter_feat_count=filter_feat_count, filter_inds=filter_inds, filter_scores=filter_scores,  
                selected_inds=selected_inds, train_acc=train_acc, test_acc=test_acc, sub_classes_train=sub_classes_train,
                sub_classes_test=sub_classes_test, sub_conf_mat_train=sub_conf_mat_train, sub_conf_mat_test=sub_conf_mat_test)
        
        print(f"Time taken for subject {i+1}: {time.time() - start} seconds")
        print('------------------------------------')
        

    # overall statistics
    overall_train_acc = np.mean(sub_train_acc)
    overall_per_class_train = np.mean(sub_class_train, axis=0)
    overall_test_acc = np.mean(sub_test_acc)
    overall_per_class_test = np.mean(sub_class_test, axis=0)

    print(f'Overall training accuracy: {overall_train_acc}, std: {np.std(sub_train_acc)}')
    print(f'Overall testing accuracy: {overall_test_acc}, std: {np.std(sub_test_acc)}')

    if args.save_results:
        overall_file = os.path.join(args.save_dir, 'stats', 'overall.npz')
        np.savez(overall_file, overall_train_acc=overall_train_acc, overall_per_class_train=overall_per_class_train,
            overall_test_acc=overall_test_acc, overall_per_class_test=overall_per_class_test, overall_train_mat=overall_train_mat,
            overall_test_mat=overall_test_mat, sub_train_acc=sub_train_acc, sub_test_acc=sub_test_acc, sub_class_train=sub_class_train,
            sub_class_test=sub_class_test)
        
        # make histograms
        # filter histogram
        
        for key in feature_hist_filter: # average each score
            feature_hist_filter[key] = sum(feature_hist_filter[key]) / float(len(feature_hist_filter[key]))
        keys = np.array(list(feature_hist_filter.keys()))
        values = np.array(list(feature_hist_filter.values()))       
        filter_plot_inf = np.zeros((num_feats, 2))
        filter_plot_inf[:num_feats, 0] = keys
        filter_plot_inf[:num_feats, 1] = values
        ind_ = np.argsort( filter_plot_inf[:,1] )[::-1]  # sort by scores
        filter_plot_inf = filter_plot_inf[ind_]
        plot_feat(filter_plot_inf, feat_names, 20, f'Most Discriminative Features (Filter) Histogram', 'Variance Ratio Score', f'filter_histogram', args.save_dir)
        
        # wrapper histogram
        keys = np.array(list(feature_hist_wrapper.keys()))
        values = np.array(list(feature_hist_wrapper.values()))       
        wrapper_plot_inf = np.zeros((len(keys), 2))
        wrapper_plot_inf[:len(keys), 0] = keys
        wrapper_plot_inf[:len(keys), 1] = values
        ind_ = np.argsort(wrapper_plot_inf[:,1] )[::-1]  # sort by number of selections
        wrapper_plot_inf = wrapper_plot_inf[ind_]
        plot_feat(wrapper_plot_inf, feat_names, 20, f'Most Commonly Selected Features (Wrapper) Histogram', 'Times Selected', f'wrapper_histogram', args.save_dir)
        
    return

# Feel free to edit/add to this function in any way you see fit. This provides the minimum
# functionality required for the assignment.
def visualize(args):
    """
    Visualize the results
    """
    data, labels, sub_info, form_names, feat_names = load_dataset()
    num_subs = len(np.unique(sub_info[:, 0])) 
    num_feats = data.shape[1]
    num_forms = len(form_names) + 1
    form_names = np.insert(form_names, 0, 'Transition')
    form_names = np.arange(num_forms)

    # Load results
    overall_file = os.path.join(args.save_dir, 'stats', 'overall.npz')
    overall_results = np.load(overall_file)

    if not os.path.exists(os.path.join(args.save_dir, 'plots')):
        os.makedirs(os.path.join(args.save_dir, 'plots'))

    # Overall subject training and testing rates as a grouped bar chart
    sub_train_acc = overall_results['sub_train_acc']
    sub_test_acc = overall_results['sub_test_acc']
    fig, ax = plt.subplots()
    ax.bar(np.arange(num_subs), sub_train_acc, width=0.35, label='Training')
    ax.bar(np.arange(num_subs)+0.35, sub_test_acc, width=0.35, label='Testing')
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Subject')
    ax.set_title('Subject-wise training and testing accuracies')
    # Make the x-axis labels start from 1
    ax.set_xticks(np.arange(num_subs))
    ax.set_xticklabels(np.arange(1, num_subs+1))
    ax.legend()
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'subject_wise_acc.png'))
    plt.close()


    # Overall per class training data. Tilt the x-axis labels by 45 degrees
    overall_per_class_train = overall_results['overall_per_class_train']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_forms), overall_per_class_train)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Form')
    ax.set_title('Overall per class training accuracy')
    ax.set_xticks(np.arange(num_forms))
    ax.set_xticklabels(form_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_per_class_train.png'))
    plt.close()

    # Overall per class testing data
    overall_per_class_test = overall_results['overall_per_class_test']
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(np.arange(num_forms), overall_per_class_test)
    ax.set_ylabel('Accuracy')
    ax.set_xlabel('Form')
    ax.set_title('Overall per class testing accuracy')
    ax.set_xticks(np.arange(num_forms))
    ax.set_xticklabels(form_names, rotation='vertical')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_per_class_test.png'))

    # Overall training confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_train_mat = overall_results['overall_train_mat']
    disp = ConfusionMatrixDisplay(overall_train_mat, display_labels=form_names)
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    ax.set_title('Overall training confusion matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_train_conf_mat.png'))

    # Overall testing confusion matrix with sklearns display
    fig, ax = plt.subplots(figsize=(10, 10))
    overall_test_mat = overall_results['overall_test_mat']
    disp = ConfusionMatrixDisplay(overall_test_mat, display_labels=form_names)
    disp.plot(include_values=False, xticks_rotation='vertical', ax=ax, cmap=plt.cm.plasma)
    ax.set_title('Overall testing confusion matrix')
    fig.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'plots', 'overall_test_conf_mat.png'))

    # Most commonly selected features
    selected_feats = np.zeros(num_feats)
    for i in range(0, num_subs):
        sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i+1}.npz')
        sub_results = np.load(sub_file)
        selected_feats[sub_results['selected_inds']] += 1

    filter_plot_info = np.zeros((num_feats, 2))
    filter_plot_info[:, 0] = np.arange(num_feats)
    filter_plot_info[:, 1] = selected_feats
    filter_plot_file_name = "selected_feats.png"
    plot_feat(filter_plot_info, feat_names, args.num_vis_feat, 'Most Commonly Selected Features', 'Num. Times Selected', filter_plot_file_name, args.save_dir)

    # Average filter scores
    avg_filter_scores = np.zeros(num_feats)
    for i in range(0, num_subs):
        sub_file = os.path.join(args.save_dir, 'stats', f'subject_{i+1}.npz')
        sub_results = np.load(sub_file)
        avg_filter_scores += sub_results['filter_scores']

    avg_filter_scores /= num_subs
    avg_filter_plot_info = np.zeros((num_feats, 2))
    avg_filter_plot_info[:, 0] = np.arange(num_feats)
    avg_filter_plot_info[:, 1] = avg_filter_scores
    avg_filter_plot_filename = 'avg_filter_scores.png'
    plot_feat(avg_filter_plot_info, feat_names, args.num_vis_feat, 'Average Selected Features', 'Average Score.', avg_filter_plot_filename, args.save_dir)
    

    return

def main():
    args = arg_parse()

    if args.do_train:
        train(args)

    if args.do_plot:
        visualize(args)

    return

if __name__ == '__main__':
    main()
