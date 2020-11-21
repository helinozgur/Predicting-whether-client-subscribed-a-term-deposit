import numpy as np
import pandas as pd
import matplotlib as plt
# preprocessing data:
# X covers all columns from 0 to 12
# y only includes the result column, which corresponds to column 13 in this database.
dataset = pd.read_csv('data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
#Encoding data
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
for i in range(12):
    X[:,i]= encoder.fit_transform(X[:,i])
pass
y= encoder.fit_transform(y)

# spliting data to train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# training model
# I preferred the Gaussian Naive Bayes algorithm as a model,
# because this model gave the highest accuracy rate among the machine learning algorithms I tried.
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# prediction
y_pred = classifier.predict(X_test)
np.set_printoptions(precision=2)
# print("prediction of the test set:")
# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
# Some sample data are entered below. These data are randomly generated.
print("has the client subscribed to a term deposit?")
result=encoder.inverse_transform(classifier.predict(sc.transform([[30,4,2,2,0,3000,1,0,2,4,8,198,1]])))
print(result)
# confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
print("confusion matrix:\n", cm)

# 5-fold cross validation and reporting the average performance score.
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))
# BONUS PARTS 1
column_values= ["age","job","marital","education","defult","balance","housing","loan","contact","day","month","duration","campaign"]
df = pd.DataFrame(data = X[:,:13],
                  columns = column_values)
y= encoder.fit_transform(y)
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
#apply SelectKBest class to extract top 10 best features
bestfeatures = SelectKBest(score_func=chi2, k=10)
fit = bestfeatures.fit(X,y)
dfscores = pd.DataFrame(fit.scores_)
dfcolumns = pd.DataFrame(df.columns)
#concat two dataframes for better visualization
featureScores = pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns = ['Specs','Score']  #naming the dataframe columns
print(featureScores.nlargest(10,'Score'))  #print 10 best features

# BONUS PARTS 2

new_data =pd.DataFrame(dataset[y== 1])
# print("                  Database\n", new_data)
new_data.loc[new_data['y'] == 'yes', 'y'] = 1
X_new= new_data.iloc[:,:-1].values
y_new= new_data.iloc[:,-1].values
# Encoding data
for i in range(12):
    X_new[:,i]= encoder.fit_transform(X_new[:,i])
pass

y_new=y_new.astype('float')
y_new = y_new.reshape(len(y_new),1)
# print(y_new)
# print(X_new)
column_values= ["age","job","marital","education","defult","balance","housing","loan","contact","day","month","duration","campaign"]
df = pd.DataFrame(data = X_new[:,:13],
                  columns = column_values)
# We created two different summary types to look at the summary of
# our data.We used the first to see a summary of our data of type object,
# and the second to view the summary of our data, including numeric expressions.
summary= new_data.describe(include=[object])
summary_int= new_data.describe()
print("                                 SUMMARY FOR NUMERİCAL VALUES\n",summary_int)
print("                                 SUMMARY FOR OBJECT VALUES\n",summary)