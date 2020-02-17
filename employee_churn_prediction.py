#!/bin/python3
# importing necessary packages
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline 
from sklearn.feature_selection import RFE
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import matplotlib.pyplot as plt # for plotting graphs
import seaborn as sns # for plotting graphs

# Data loading and understanding feature
def load_dataset(dataframe):
    left = dataframe.groupby('left')

    # Analyze the target variable

    print (left.mean())
    # print (dataframe.describe())

    # Loop through columns and visualize using matplotlib
    # Uncomment when needed

    # for feature in features:
    #     visualize (dataframe, feature) # visualize using matplotlib
    
    # for feature in features:
    #     visualize_seaborn(dataframe, feature) # visualize using seaborn

# Exploratory data analysis and Data visualization
def visualize (dataframe, column_name):
    y_axis = dataframe.groupby(column_name).count()
    plt.bar(y_axis.index.values, y_axis['satisfaction_level'])
    plt.xlabel(column_name)
    plt.ylabel('Number of Employees')
    plt.show()

def visualize_seaborn (dataframe, feature):
    sns.countplot(x=feature,data = dataframe)
    plt.xticks(rotation=90)
    plt.title("No. of employee")
    plt.show()

def elbow_method(dataframe, column_name):
    sse={}
    df_cluster = dataframe[[column_name]]
    for k in range(1, 10):
        kmeans = KMeans(n_clusters=k, max_iter=1000).fit(df_cluster)
        df_cluster["clusters"] = kmeans.labels_
        sse[k] = kmeans.inertia_ 
    plt.figure()
    plt.plot(list(sse.keys()), list(sse.values()))
    plt.xlabel("Number of cluster")
    plt.title(column_name)
    plt.show()

# Feature Selection using RFE
def select_features(dataframe):

    X = dataframe.loc[:, dataframe.columns != 'left'] # Select all columns except target column
    y = dataframe['left'] # Select only the target column
    
    model = LinearRegression()#Initializing RFE model
    rfe = RFE(model, 7)#Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y)  #Fitting the data to model
    model.fit(X_rfe,y)
    # print("RFE SUPPORT " + str(rfe.support_))
    # print("RFE RANKING " + str(rfe.ranking_))

    #no of features
    nof_list=np.arange(1,9)            
    high_score=0
    #Variable to store the optimum features
    nof=0           
    score_list =[]
    for n in range(len(nof_list)):
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)
        model = LinearRegression()
        rfe = RFE(model,nof_list[n])
        X_train_rfe = rfe.fit_transform(X_train,y_train)
        X_test_rfe = rfe.transform(X_test)
        model.fit(X_train_rfe,y_train)
        score = model.score(X_test_rfe,y_test)
        score_list.append(score)
        if(score>high_score):
            high_score = score
            nof = nof_list[n]
    # print("OPTIMUM NUMBER OF FEATURES: %d" %nof) # Optimal number of features
    # print("SCORE WITH %d FEATURES: %f" % (nof, high_score))

    cols = list(X.columns)
    model = LinearRegression() #Initializing RFE model
    rfe = RFE(model, nof) #Transforming data using RFE
    X_rfe = rfe.fit_transform(X,y) #Fitting the data to model
    model.fit(X_rfe,y)              
    temp = pd.Series(rfe.support_,index = cols)
    selected_features_rfe = temp[temp==True].index
    # print("SELECTED FEATURES " + str(selected_features_rfe))

# Cluster analysis
def cluster_analysis(dataframe):

    # Filter data
    left_emp =  dataframe[['satisfaction_level', 'last_evaluation']][dataframe.left == 1]
    # Create groups using K-means clustering.
    kmeans = KMeans(n_clusters = 3, random_state = 0).fit(left_emp)

    # Add new column "label" annd assign cluster labels.
    left_emp['label'] = kmeans.labels_
    # Draw scatter plot
    plt.scatter(left_emp['satisfaction_level'], left_emp['last_evaluation'], c=left_emp['label'],cmap='Accent')
    plt.xlabel('Satisfaction Level')
    plt.ylabel('Last Evaluation')
    plt.title('3 Clusters of employees who left')
    plt.show()

    X = dataframe.loc[:, dataframe.columns != 'left'] # Select all columns except target column
    y = dataframe['left'] # Select only the target column
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # 70% training and 30% test

    # Find the optimal algorithm for building the model
    # Uncomment if needed
    # grid_search(X_train, y_train)

    # Build the model
    model_building(dataframe, X_train, X_test, y_train, y_test)

def grid_search(features, target):
    pipe = Pipeline([("classifier", RandomForestClassifier())])

    # Create dictionary with candidate learning algorithms and their hyperparameters
    search_space =  [
        {
            'classifier': [DecisionTreeClassifier()],
            'classifier__criterion': ['gini', 'entropy'],
            'classifier__splitter': ['best', 'random'],
            'classifier__min_samples_leaf': [1],
            'classifier__min_samples_split': [2],
            'classifier__max_features': ['auto', 'sqrt', 'log2']
        }, 
        { 
            'classifier': [GradientBoostingClassifier()],
            'classifier__loss': ['deviance'],
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [3, 5],
            'classifier__min_samples_leaf': [1],
            'classifier__min_samples_split': [2]
        },
        {  
            'classifier':[SVC()],
            'classifier__gamma': ['scale', 'auto']
        },
        {
            "classifier": [LogisticRegression()],
            "classifier__penalty": ['l2','l1'],
            "classifier__C": np.logspace(0, 4, 10),
            "classifier__fit_intercept":[True, False],
            "classifier__solver":['saga','liblinear']
        },
        {
            "classifier": [LogisticRegression()],
            "classifier__penalty": ['l2'],
            "classifier__C": np.logspace(0, 4, 10),
            "classifier__solver":['newton-cg','saga','sag','liblinear'], ##These solvers don't allow L1 penalty
            "classifier__fit_intercept":[True, False]
        },
        {
            "classifier": [xgb.XGBClassifier()],
            "classifier__learning_rate": [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
            "classifier__max_depth": [ 3, 4, 5, 6, 8, 10, 12, 15],
            "classifier__min_child_weight": [ 1, 3, 5, 7 ],
            "classifier__gamma": [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
            "classifier__colsample_bytree": [ 0.3, 0.4, 0.5 , 0.7 ]
        },
        {   
            "classifier": [RandomForestClassifier()],
            "classifier__n_estimators": [10, 100, 1000],
            "classifier__max_depth":[5,8,15,25,30,None],
            "classifier__min_samples_leaf":[1,2,5,10,15,100],
            "classifier__max_leaf_nodes": [2, 5,10]
        }
    ]

    # create a gridsearch of the pipeline, the fit the best model
    gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
    best_model = gridsearch.fit(features, target)

    print(best_model.best_estimator_)
    print("The mean accuracy of the model is:",best_model.score(features, target))  

# Building prediction model using Gradient Boosting Tree.
def model_building(dataframe, X_train, X_test, y_train, y_test):
    
    #Create the best model after hyper-parameter tuning from grid search
    model = xgb.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
       colsample_bynode=1, colsample_bytree=0.7, gamma=0.0,
       learning_rate=0.25, max_delta_step=0, max_depth=15,
       min_child_weight=1, missing=None, n_estimators=100, n_jobs=1,
       nthread=None, objective='binary:logistic', random_state=0,
       reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
       silent=None, subsample=1, verbosity=1)

    #Train the model using the training sets
    model.fit(X_train, y_train)

    #Predict the response for test dataset
    y_pred = model.predict(X_test)
    dataframe['proba'] = model.predict_proba(dataframe[X_train.columns])[:,1]
    # print (dataframe[['proba']].sort_values(by=['proba'], ascending=False))

    # Export to csv
    dataframe[['proba']].sort_values(by=['proba'], ascending=False).to_csv('churn_probability.csv')

    # Evaluating model performance
    evaluate(y_test, y_pred)

def evaluate(y_test, y_pred):
    print ("Confusion Matrix")
    print (confusion_matrix(y_test, y_pred))
    # Model Accuracy, how often is the classifier correct?
    print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
    # Model Precision
    print("Precision:",metrics.precision_score(y_test, y_pred))
    # Model Recall
    print("Recall:",metrics.recall_score(y_test, y_pred))

# The main function
def main():
    dataset = "./data/HR_comma_sep.csv"
    features = ['number_project','time_spend_company','Work_accident','left', 'promotion_last_5years','Departments ','salary']
    
    dataframe = pd.read_csv(dataset)
    #creating labelEncoder
    le = preprocessing.LabelEncoder()
    # Converting string labels into numbers.
    dataframe['salary']=le.fit_transform(dataframe['salary'])
    dataframe['Departments ']=le.fit_transform(dataframe['Departments '])
    
    load_dataset(dataframe)

    # Use this to find the optimal value of k for cluster analysis
    # Uncomment when needed
    # for feature in dataframe.columns:
    #     elbow_method(dataframe, feature)
    
    select_features(dataframe)
    feature_selected_dataframe = dataframe.drop(columns=['average_montly_hours']) # Remove columns with less importance
    cluster_analysis(feature_selected_dataframe)    
    
if __name__ == "__main__":
    main()