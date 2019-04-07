
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

#import dataset 
dataset = pd.read_csv('Social_Network_Ads.csv')
X = dataset.iloc[:, [2,3]].values  #[2,3] pick the columns in that location
y = dataset.iloc[:, 4].values

#Split the dataset 
from sklearn.cross_validation import train_test_split 
train_X , test_X, train_y, test_y = train_test_split(X, y, test_size = 0.25, random_state = 0)

#feature scaling 
from sklearn.preprocessing import StandardScaler
Scale_X = StandardScaler()
train_X = Scale_X.fit_transform(train_X)
test_X = Scale_X.transform(test_X)

#Create, fit and train logistic regression model 
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(train_X,train_y)

#Making predictions 
predicted_y = logreg.predict(test_X)

#checking the performance of the model 
logreg_pred_test_df = pd.DataFrame({'Expected':test_y, 'Predicted Score':predicted_y })
logreg_pred_test_df.head(20)

#Making confusion matrix 
from sklearn.metrics import confusion_matrix
con_matrix = confusion_matrix(test_y,predicted_y)

#Visualizing the training set
from matplotlib.colors import ListedColormap
X_set,y_set = train_X, train_y
X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                    np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01))

plt.contourf(X1,X2, logreg.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape), 
             alpha = 0.75, cmap = ListedColormap(('red','green')))

plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j,0], X_set[y_set == j,1],
                c = ListedColormap(('red','green'))(i), label = j)
    
plt.title('Logistic Regression for training set')
plt.xlabel('Age')
plt.ylabel('Estimated salary')
plt.legend()
plt.show()



