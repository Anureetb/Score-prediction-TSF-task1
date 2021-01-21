#!/usr/bin/env python
# coding: utf-8

# # Prediction using Supervised ML
# #Name: ANUREET BANSAL
# #GRIPJAN21
# #Task 1

# Predict the percentage of an student based on the no. of study hours. What will be predicted score if a student studies for 9.25 hrs/ day?

# # importing libraries

# In[20]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# # importing dataset

# In[25]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
data.head(5)


# In[37]:


data.shape


# In[35]:


data.describe()


# # Visual representation of data

# In[42]:


data.plot(x='Hours',y="Scores",style ='o')
plt.xlabel("Number of hours studied")
plt.ylabel("Percentage Scores")
plt.title("Actual values")
plt.show()


# Interpretation: According to the graph as shown above, we can see high positive correlation between number of hours studied and percentage score of a student.

# In[18]:


x = data.iloc[:, :-1].values
y = data.iloc[:,1].values


# # Training and testing model

# In[5]:


from sklearn.model_selection import train_test_split
x_train, x_test, Y_train, Y_test = train_test_split(x,y, test_size =0.2, random_state = 123)


# In[23]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(x_train, Y_train)


# In[8]:


y_pred = lr.predict(x_test)
y_pred


# In[45]:


difference = pd.DataFrame({"Actual": Y_test, "Predicted": y_pred})
difference


# In[40]:


plt.scatter(x_train, Y_train, color = 'red')
plt.plot(x_train, lr.predict(x_train), color = 'blue')
plt.ylabel("Percentage scores")
plt.xlabel("Number of hours studied")
plt.title("Actual values vs Predicted values")


# # Predicted score if a student studies for 9.25 hrs/ day

# In[11]:


result = lr.predict([[9.25]])
result


# Interpretation: The predicted score of the student according to our model will be 91.511 if the student studies for 9.25 hrs/day.

# # Evaluation metric (MAE & MSE)

# In[14]:


from sklearn import metrics
MAE = metrics.mean_absolute_error(Y_test, y_pred)
MAE


# Interpretation: The mean absolute error of our model is 4.97 i.e.deviation of predicted values on an average is 4.97. Since it is very low, our model performed well.

# In[15]:


MSE = metrics.mean_squared_error(Y_test, y_pred)
MSE


# In[47]:


RMSE = np.sqrt(metrics.mean_squared_error(Y_test, y_pred))


# Interpretation: The root mean squared error is 5.15 which is less than 10% of mean value which is 51.48 in our dataset, therefore the model performed well.
