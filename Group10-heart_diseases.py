import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# reading the csv file and loading it into dataframe
data=pd.read_csv("C:\\Users\\vedan\\Desktop\\heart.csv")
#getting statistical values like count mean max min etc without categorical variables
print(data.drop(['Chestpain','Resting Cardiographic result','FBS','sex','Exang','slope','oldpeak','ca'],axis = 1).describe())
#checking for null values
print(data.isnull().sum())

lk_cat=[]
lk_FBS=[]
lk_exang=[]
lk_thal=[]
# creating dummy variables
for label, content in data.items():
    if(label=='sex'):
        for i in content:
            if(i=="Male"):
                lk_cat.append(1);
            else:
                lk_cat.append(0);
    if(label=='FBS'):
        for i in content:
            if(i=="True"):
                lk_FBS.append(1);
            else:
                lk_FBS.append(0);
    if(label=='Exang'):
        for i in content:
            if(i=="Yes"):
                lk_exang.append(1);
            else:
                lk_exang.append(0);
    if(label=='thal'):
        for i in content:
            temp=int(i)
            if(temp<=3):
                lk_thal.append(0);
            elif(temp>3 and temp<=6):
                lk_thal.append(1);
            else:
                lk_thal.append(2)

#adding dummy varibales to database
data['sex_cat']=lk_cat
data['FBS_cat']=lk_FBS
data['exang_cat']=lk_exang
data['thal_cat']=lk_thal
#veiwing dummy variables
print("categorical variable")
print(data.iloc[:,[1,14,5,15,8,16,12,17]])



#viewing all columns
print(data.columns)
# correlation
corr=data.drop(['sex', 'FBS','Exang','thal','sex_cat','FBS_cat','exang_cat','thal_cat'], axis = 1).corr()
print(corr)
corrlt=sn.heatmap(corr,cmap='coolwarm')
plt.title("Correlation")
plt.show()
#scatter plot 
ax = sn.scatterplot(x='age',y='Cholestrol',data=data,hue='sex')
sn.lmplot(x='age', y='Cholestrol',  data=data,  height = 8,hue='sex')
ax.set_title('Scatter plot of age and cholestrol level')
plt.show()
#box plot for outlier detection

X = data.boxplot( column =['Cholestrol','Max Heart Rate'])
plt.title("Outliers-Cholestrol, Max Heart Rate")
plt.show()
plt.figure(3)
Y=data.boxplot( column =['Resting BP'])
plt.title("Outliers-Resting BP")
plt.show()

#count plot to se difference between male and female
ax = sn.countplot(x=data['sex'])

#see plot difference in male and female heart diseases
plt.figure(4)
sn.catplot(x='AHD', kind="count",hue = 'sex', palette="Blues", data=data)
plt.subplots_adjust(top=0.9)
plt.title("Number of Male /Female with diagnosis of HD")
plt.show()

#dividing dataset into train and test

X_train, X_test, y_train, y_test = train_test_split(data.drop(['AHD','sex','FBS','Exang','thal'],axis=1),data['AHD'], test_size=0.30, random_state=101)

#building logistic regression model
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)

#prediction the output variables
Predictions = logmodel.predict(X_test)

#creating the classification report
print(classification_report(y_test,Predictions))

#creating the confusion matrix
print(confusion_matrix(y_test, Predictions))

#plotting the confusion matrix
cm=confusion_matrix(y_test, Predictions)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='green',size='20')
plt.title("Confusion Matrix")
plt.show()







