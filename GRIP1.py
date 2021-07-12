print("Shaikh Malaika Begum ")

#Step1-Import required libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



#STEP 2: Loading data
newd = "http://bit.ly/w-data"
n = pd.read_csv(newd)
print("Data imported successfully")
n.head()

#STEP 3: Information about data
n.info()


#STEP 4: Plotting the data in 2D graph
n.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  


#STEP 5: Plotting regression line
ax = sns.regplot(x="Hours", y="Scores", data =n)
plt.title('Hours vs Percentage', fontsize=25)
plt.xlabel('Hours Studied')
plt.ylabel('Percentage Score')
plt.show()

#STEP 6: Training the data
X = n.iloc[:, :-1].values  
y = n.iloc[:, 1].values 
from sklearn.model_selection import train_test_split  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression  

regressor = LinearRegression()  
regressor.fit(X_train, y_train) 
print("Training complete.")

print(X_test)
y_pred = regressor.predict(X_test)


#STEP 7: Making prediction
df1 = pd.DataFrame({'Hours':[1.5,3.2,7.4,2.5,5.9], 'Actual': y_test, 'Predicted': y_pred})  
df1
df1.plot(x= "Hours", y=["Actual", "Predicted"], kind="bar")
plt.grid(linewidth='1')
plt.title(" Actual and predicted Percentage comparison") 
plt.ylabel('Percentage Score')  
plt.show()

#STEP 8: What will be predicted score if a student studies for 9.25 hrs/ day?
hour = 9.25
own_pred = regressor.predict([[hour]])
print("No of Hours = {}".format(hour))
print("Predicted Score = {}".format(own_pred[0]))


#STEP 9: Evaluating the model
from sklearn import metrics 
print('Mean Absolute Error:', 
      metrics.mean_absolute_error(y_test, y_pred))

print("Thanks for watching!!!")

