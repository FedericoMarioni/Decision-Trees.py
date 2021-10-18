# Classification Trees to predict whether or not a person has heart disease
# Classification Trees are very good when you need to know how the decisions are being made

# Missing data: Identifying it and dealing with it
# Formatting data for Decisions Trees: Splitting it into Dependent and Independent variables, One-Hot Encoding
# Building a preliminary Classification Tree
# Optimizing the Tree with Cost Complexity Pruning: Visualizing Alpha, Using Cross Validation to find the best alpha
# Building, Drawing, Interpreting and Evaluating the Final Decision Tree

import pandas as pd  # to load and manipulate data and for One-Hot Encoding
import numpy as np   # data manipulation, calculate the mean and std.dv
import matplotlib.pyplot as plt  # graphs
from sklearn.tree import DecisionTreeClassifier  # to build a decision tree classifier
from sklearn.tree import plot_tree  # to draw a classification tree
from sklearn.model_selection import train_test_split  # to split the data into training and testing sets
from sklearn.model_selection import cross_val_score   # for cross validation
from sklearn.metrics import confusion_matrix  # to create a confusion matrix for evaluation
from sklearn.metrics import plot_confusion_matrix  # to draw a confusion matrix

# 1. Import the data
df = pd.read_csv('processed.cleveland.data', header=None)
df.columns = ['age', 'sex', 'cp', 'restbp', 'chol', 'fbs', 'restecg', 'thalach', 'exang',
              'oldpeak', 'slope', 'ca', 'thal', 'hd']

# 2. Identifying missing data.  We can either remove the missing data point or impute it
# print(df.dtypes)
# print(df['ca'].unique())
# print(df.['thal'].unique())

# Dealing with missing data
# print(len(df.loc[(df['ca'] == '?')
#                 |
#                 (df['thal'] == '?')]))

# print(df.loc[(df['ca'] == '?')
#             |
#             (df['thal'] == '?')])

# print(len(df))
# Only 6 out of 303 rows ( 2 % ) contain missing values. Since 297 is plenty of data to build a decision tree
# we will remove the rows

df_no_missing = df.loc[(df['ca'] != '?')
                       &
                       (df['thal'] != '?')]

# Verify that the missing rows have been deleted
# print(len(df_no_missing))
# print(df_no_missing['ca'].unique())
# print(df_no_missing['thal'].unique())

# 3. Formatting the data to make a Classification Tree
# Split the data into dependent variable and independent variables

X = df_no_missing.drop('hd', axis=1).copy()
Y = df_no_missing['hd'].copy()

# One-Hot Encoding. Transforming the categorical data variables into dummies
# We do not treat categorical data like continuous numerical data
# print(X.dtypes)
# print(pd.get_dummies(X, columns=['cp']))

X_encoded = pd.get_dummies(X, columns=['cp',
                           'restecg',
                           'slope',
                           'thal'])

# print(Y.unique())
# We see that Y contains 5 different discrete values. Ranging from 0 to 4 if the person has heart disease or not, and
# if it does what is the severity of it. In this example, we are only interested in Y containing two values:
# Y = O person does not have heart disease, Y = 1 person has heart disease.

Y_not_zero_index = Y > 0
Y[Y_not_zero_index] = 1
# print(Y.unique())

# 4 Building a preliminary Decision Tree
# Split the data into training and testing sets

X_train, X_test, Y_train, Y_test = train_test_split(X_encoded, Y, random_state=42)

# Create a decision tree and fit it to the training data

clf_dt = DecisionTreeClassifier(random_state=42)
clf_dt = clf_dt.fit(X_train, Y_train)

# Plot the Decision Tree

plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns);

# plt.show()

# Evaluating on the testing data sets and plotting a confusion matrix

plot_confusion_matrix(clf_dt, X_test, Y_test, display_labels=["Does not have HD", "Has HD"])

# plt.show()
# Percentage of Y = 0 predicted as Y = 0 - 74%
# Percentage of Y = 1 predicted as Y = 1 - 79%

# 5 Optimizing. Decision Trees are notorious for being over fit to the training data set
# Pruning a tree with cost complexity pruning can improve the accuracy with the testing data set

path = clf_dt.cost_complexity_pruning_path(X_train, Y_train)  # Determine values for alpha
ccp_alphas = path.ccp_alphas  # extract the different values for alpha
ccp_alphas = ccp_alphas[:-1]  # exclude the maximum value for alpha

clf_dts = []  # create an array that we will put decision trees into

# now create one decision tree per value for alpha and store it in the array
for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf_dt.fit(X_train, Y_train)
    clf_dts.append(clf_dt)

# now let´s graph the accuracy of the trees using the Training Dataset and the Testing Dataset as a function of alpha

train_scores = [clf_dt.score(X_train, Y_train) for clf_dt in clf_dts]
test_scores = [clf_dt.score(X_test, Y_test) for clf_dt in clf_dts]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test", drawstyle="steps-post")
ax.legend()
# plt.show()

# Cross Validation for finding the best Alpha

# create an array to store the results of each fold during cross validation
alpha_loop_values = []

# For each candidate value for alpha we will run a 5-fold cross validation
# Then we will store the mean and standard deviation of the scores (the accuracy) for each call
# to cross_val_score in alpha_loops_values

for ccp_alpha in ccp_alphas:
    clf_dt = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    scores = cross_val_score(clf_dt, X_train, Y_train, cv=5)
    alpha_loop_values.append([ccp_alpha, np.mean(scores), np.std(scores)])

# Now we can draw a graph of the means and standard deviations of the scores
# For each candidate value for alpha

alpha_results = pd.DataFrame(alpha_loop_values,
                             columns=['alpha', 'mean_accuracy', 'std'])

alpha_results.plot(x='alpha',
                   y='mean_accuracy',
                   yerr='std',
                   marker='o',
                   linestyle='--')
# plt.show()

# Using cross validation we can see that instead of setting ccp_alpha=0.016, we need to set it
# to something closer to 0.014. We can find the exact value with:
# print(alpha_results[(alpha_results['alpha'] > 0.014 ) & (alpha_results['alpha'] < 0.015)])
# Now let´s store the ideal value for alpha so that we can use it to build the best tree

ideal_ccp_alpha = alpha_results[(alpha_results['alpha'] > 0.014)
                                &
                                (alpha_results['alpha'] < 0.015)]['alpha']

# print(ideal_ccp_alpha)

# convert idea_ccp_alpha from a series to a float
ideal_ccp_alpha = float(ideal_ccp_alpha)

# 6. Build, evaluate, draw the final Classification Tree

clf_dt_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=ideal_ccp_alpha)
clf_dt_pruned = clf_dt_pruned.fit(X_train, Y_train)

plot_confusion_matrix(clf_dt_pruned,
                      X_test,
                      Y_test,
                      display_labels=["Does not have HD", "Has HD"])
# plt.show()
# Percentage of Y = 0 predicted as Y = 0 - 81%
# Percentage of Y = 1 predicted as Y = 1 - 85%

plt.figure(figsize=(15, 7.5))
plot_tree(clf_dt_pruned,
          filled=True,
          rounded=True,
          class_names=["No HD", "Yes HD"],
          feature_names=X_encoded.columns)

# plt.show()

# This Tree is smaller than the first one and still it performs much better. That is because
# the first tree over fits the training data set.











