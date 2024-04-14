# PRML Project 2, Spring 2023

We provide some starter code for this project. Some functions are fully implemented, and there are some others that you will need to implement yourself. See below for file descriptions. Read the project description on CANVAS for more details on the project.

## Quickstart
To main file, `classification.py`, takes in 7 arguments:
- `--filter_feat_count`: the number of features to select using the filter method
- `--num_vis_feat`: the number of features to visualize
- `--do_train`: whether or not to train the model
- `--do_plot`: whether or not to plot the results (visualization)
- `--save_results`: whether or not to save the results (not including the trained model)
- `--save_dir`: path to save the results

You may leave the default values for all arguments if you choose to, or you may call with any values you see fit. For example, to run the code with the default values, you may run the following command:

```
python classification.py --filter_feat_count 20 --num_vis_feat 10 --do_train --do_plot --save_results --save_dir train_results
```

The code as is will run, though with arbitray selection of features. You will need to implement the remaining functions. To see the code run, run from the command line:

```
python classification.py --do_train --do_plot --save_results 
```


## Data
- data/data.npz
  - The data you will be working with
  - Variables (np arrays):
    - **feature_names**: the names of all 1961 features
    - **form_names**: the names of all 45 forms you will be trying to classify
    - **labels**: provided class labels corresponding to each data point
    - **sub_info**: information about subject number and performance number corresponding to each data point
    - **data**: the actual data you will be working with


## Provided Code
Note this data is just to get you started and you may edit it according to your needs so long as the basic functionality is preserved.
- **normalize_data**: normalizes provided training data such that each feature is on the range [0, 1] or applies a previously applied transformation to a set of testing data
- **load_dataset**: loads the dataset from the npz file.
- **split_data**: splits the dataset into training and testing data based on subject number
- **plot_feat**: a function for visualizing values that correspond to features
- **sub_stats**: gets subject-wise statistics
- **visualize**: overall classification visualization 

## Code you need to edit/implement
Note that most (if not all) places you need to edit the code are marked with "TODO" markers that you can search for.
- **train**: this is the training loop for LOSO classification. Most of it is complete with some excpetions.
- **filter_method**: function for the feature filtering
- **forward_selection**: forward selection for selecting features.


## Packages
This project will use three primary packages: numpy, sklearn, and matplotlib. You will likely need to install such packages via pip if you don't already have them installed.

We recommend creating a virtual environment using Anaconda for this project.

## Package Imports
You may not add any additional packages to the code. You must also write the pertinant parts of the code (filter + forward selectiong) from scratch and may only make use of numpy.

For performing classification, if you choose to use an imported classifier, you may only select from those available in sklearn (e.g. KNN, LDA, etc.).

Read the project description on CANVAS for more details on the project.

Good luck!
