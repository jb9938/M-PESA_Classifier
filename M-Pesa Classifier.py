# -*- coding: utf-8 -*-
"""
ECO481: Assignment 1


@author: Joonbum Yang
"""
import pandas as pd 
import numpy as np 
import sklearn as sk


# Read Data
mobile_money = pd.read_csv("C:/Users/User/Downloads/mobile_money.csv")
mobile_money = mobile_money.replace({"yes":1, "no": 0})

# Descriptive Statistics
summary1 = mobile_money.describe(include = 'all').loc[["min", "max", "mean", "std"]]
outcome_stat = summary1["mpesa_user"]


"""
The mean of response variable is 0.73443 while std is 0.4417. Meaning, 
about 73% of response variable has recorded to be a mpesa_user with variance of approximately
0.195
"""

# Data Cleaning with interested varaibles
df1=mobile_money[mobile_money['mpesa_user']==1] 
df2=mobile_money[mobile_money['mpesa_user']==0] 

summary_yes = df1.describe(include = 'all').loc[["min", "max", "mean", "std"]]
summary_no = df2.describe(include = 'all').loc[["min", "max", "mean", "std"]]

predictor_stat_yes = summary_yes[["cellphone","totexppc","hhid","wkexppc",
                                  "wealth","education_years",
                                  "pos","neg","ag","sick","sendd","recdd",
                                  "bank_acct","mattress","sacco","merry",
                                  "occ_farmer","occ_public","occ_prof","occ_help",
                                  "occ_bus","occ_sales","occ_ind","occ_other","occ_ue"]]

predictor_stat_no = summary_no[["cellphone","totexppc","hhid","wkexppc",
                                  "wealth","education_years",
                                  "pos","neg","ag","sick","sendd","recdd",
                                  "bank_acct","mattress","sacco","merry",
                                  "occ_farmer","occ_public","occ_prof","occ_help",
                                  "occ_bus","occ_sales","occ_ind","occ_other","occ_ue"]]

from sklearn.model_selection import train_test_split 
from sklearn.utils import shuffle 

# Storing response variable
response = mobile_money["mpesa_user"]
# Only keeping variables from above
mobile_money = mobile_money[predictor_stat_yes.columns]
# dropping Nan values
mobile_money = mobile_money.dropna()

# Keeping predictor variables as features
features = list(mobile_money.columns)
# adding response variables
mobile_money["mpesa_user"] = response

# First, shuffle the rows of the dataframe.
mobile_money_shuffled = shuffle(mobile_money, random_state = 7)
 
# Split into training and testing data 
X_train, X_test, y_train, y_test = train_test_split(mobile_money_shuffled[features], mobile_money_shuffled['mpesa_user'],
test_size=0.2, random_state = 2)

# Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
x_train_scaled = scaler.transform(X_train)
x_test_scaled = scaler.transform(X_test)

# Logistic Classifier
from sklearn.linear_model import LogisticRegression
logit = LogisticRegression()
logit.fit(x_train_scaled, y_train)

# Decision Tree Classifier
from sklearn import tree
dt_ = tree.DecisionTreeClassifier(random_state = 3)
dt_ = dt_.fit(x_train_scaled, y_train)

# Random Forest 
from sklearn.ensemble import RandomForestClassifier 
rf = RandomForestClassifier(max_depth = 7, random_state=5)
rf = rf.fit(x_train_scaled, y_train)


# Accuracy rate
Scores ={} 
Scores['decision tree'] = dt_.score(x_test_scaled, y_test)
Scores['Random Forrest'] = rf.score(x_test_scaled, y_test)
Scores['Logistic'] = logit.score(x_test_scaled, y_test)

from sklearn.metrics import roc_curve, RocCurveDisplay, roc_auc_score

# ROC curve logistic regression
y_score_logit = logit.predict_proba(x_test_scaled)[:,1]
fpr, tpr, _  = roc_curve(y_test,y_score_logit, pos_label=logit.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
auc_logit = roc_auc_score(y_test, y_score_logit)

# ROC curve decision tree
y_score = dt_.predict_proba(x_test_scaled)[:,1]
fpr, tpr, _  = roc_curve(y_test,y_score, pos_label=dt_.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
auc = roc_auc_score(y_test, y_score)

# ROC curve random forest
y_score_rf = rf.predict_proba(x_test_scaled)[:,1]
fpr, tpr, _  = roc_curve(y_test,y_score_rf, pos_label=rf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
auc_rf = roc_auc_score(y_test, y_score_rf)

# Collecting auc scores
auc_scores = {}
auc_scores['decision tree'] = auc
auc_scores['Random Forrest'] = auc_rf
auc_scores['Logistic'] = auc_logit

"""
AUC score close to 1 indicate almost perfect trade off between TPR and FPR
Acurracy rate close to 1 indicate almost perfect preidction of resposne variable.
While Logistic classifier has the highest auc_scores, the difference between logistic AUC 
and random forest auc is miniscule compared to difference between logistic accuracy rate
and random forest accuracy rate. Therefore, random forest is the best classifier. 
Decision tree seems to be the worst performing for both rate.
"""

# Random Forrest 
mdi_importances = pd.Series(
    rf[-1].feature_importances_, index=features
).sort_values(ascending=True)
print('Random Forrest')
print(mdi_importances[-3:])


"""
To find best predictors in random forest, we measure how much each predictors 
have been used in each tree. The output gives wkexppc, totexppc, cellphone to 
be the best predictors.
"""


#KNN Classifier.

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=10)

neigh.fit(x_train_scaled, y_train)

neigh.score(x_test_scaled, y_test) 

#Determine the optimal 'k' 

Scores_KNN = {} 

for k in range(1, 11):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(x_train_scaled, y_train)
    Scores_KNN[str(k)] = neigh.score(x_test_scaled, y_test)

print(f'Optimal k is equal to {max(Scores_KNN, key=Scores_KNN.get)} which returns a score of {Scores_KNN[str(5)]}.')    

   
"""
Since the accuracy rate for knn clasifier is smaller than random forest,
we cannot say KNN classifire performs better than random forest. 
As we see per capita consumption, per capita food consumption and cell phone to be the 
best predictor, if government wishes to increase the usage of M-pesa, increasing the percentage
of population with cell phone and consumption per capita can be useful.
"""


