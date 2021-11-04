import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

# pd.set_option('display.max_columns', 100)
# pd.set_option('display.width', 500)

# loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";", low_memory=False)
# loan_test_csv = pd.read_csv('../ficheiros_competicao/loan_test.csv', delimiter=";", low_memory=False)

# loan_train_csv.columns.drop('account_id')

feature_cols = ['loan_id', 'date', 'amount', 'duration', 'payments']
# # X = loan_train_csv[feature_cols] # Features
# # y = loan_train_csv.status # Target variable
# # Split dataset into training set and test set
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# # decision_tree_classifier = DecisionTreeClassifier()

# # parameter_grid = {'criterion': ['gini', 'entropy'],
# #                   'splitter': ['best', 'random'], 
# #                   'max_depth': [5],
# #                   'max_features': [5],
# #                   'min_samples_leaf': [1, 10] }

# # cross_validation = StratifiedKFold(n_splits=10,  shuffle=True)

# # grid_search = GridSearchCV(decision_tree_classifier, param_grid=parameter_grid, cv=cross_validation)

# # grid_search.fit(X_train, y_train)

# # print('Best score: {}'.format(grid_search.best_score_))
# # print('Best parameters: {}'.format(grid_search.best_params_))

# # y_pred = grid_search.predict(X_test)

# # print("Accuracy: {}".format(accuracy_score(y_test, y_pred)))
# # print()
# # cfmx = confusion_matrix(y_test, y_pred)
# # print(cfmx)
# # print()
# # print("Report: ")
# # print(classification_report(y_test, y_pred))

# # decision_tree_classifier.fit(X_train, y_train)

# # y_pred_proba = decision_tree_classifier.predict_proba(X_test)[::,1]
# # print(y_pred)

# # calculate AUC of model
# # auc = metrics.roc_auc_score(y_test, y_pred_proba)

# # print AUC score
# # print(auc)

# # Predict

# # test_data = loan_test_csv[feature_cols]

# # test_df = pd.DataFrame(data=[[test_data]], columns=feature_cols)

# # prediction = decision_tree_classifier.predict_proba(test_data)[::,1]

# # print(prediction)

####### Logistic Regression

loan_train_csv = pd.read_csv('../ficheiros_competicao/loan_train.csv', delimiter=";", low_memory=False)
data = pd.read_csv(loan_train_csv)

#define the predictor variables and the response variable
X = data[feature_cols]
y = data['status']

#split the dataset into training (70%) and testing (30%) sets
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0) 

#instantiate the model
log_regression = LogisticRegression()

#fit the model using the training data
log_regression.fit(X_train,y_train)

#use model to predict probability that given y value is 1
y_pred_proba = log_regression.predict_proba(X_test)[::,1]

#calculate AUC of model
auc = metrics.roc_auc_score(y_test, y_pred_proba)

#print AUC score
print(auc)