import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

'''Notes: 
1. Goal -> Produce range of mobile prices given features
2. Clean data'''


####### CLEANING DATA ##########


''' Attempt 1: Chi-squared ~ Univariate Selection

NOTE: Chi-squared test assumes independence, given alpha
calculates probability variables are independent. 

Performs classification for EACH feature, 
selects the 10 most dependent features.

If a feature is mostly independent with the output 
variable then it doesn't have a strong influence on
the model & can be ignored.
'''

data = pd.read_csv("train.csv")
X = data.iloc[:, 0:20]
y = data.iloc[:, -1]

bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(X.columns)

# concat two dataframes so they are easier to visualize
featureScores = pd.concat([dfcolumns, dfscores], axis=1)
featureScores.columns = ['Specs', 'Score']
print(featureScores.nlargest(10, 'Score'))

'''
Attempt 2: Permutation ~ Feature Importance

NOTE: Permutation assumes in an accurate model importance 
features will heavily effect on reducing error.

We go through every feature and randomize ONLY that feature
in the original matrix. Run the model on this new matrix,
and the difference or ratio. Sort by most relevant. 

Important features will have the largest effect on "guiding"
prediction to correct classifications. If we shuffle those
values, the most important features should have the biggest 
impact on increasing error. 
'''

# Load in ExtraTreeClassifier (has built-in feature importance)
from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt
model = ExtraTreesClassifier()
model.fit(X,y)

# print "feature_importances_"
print(model.feature_importances_)

#plot resutls
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(10).plot(kind="barh")
plt.show()

