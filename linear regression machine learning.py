#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[24]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


customers = pd.read_csv(r"Downloads/Ecommerce Customers")


# In[26]:


customers.head()


# In[27]:


customers.info()


# In[28]:


sns.set_palette("GnBu_d")
sns.set_style('whitegrid')


# In[29]:


sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)


# In[30]:


sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)


# In[31]:


sns.jointplot(x='Time on App',y='Length of Membership',kind="hex",data=customers)


# In[32]:


sns.pairplot(customers)


# In[33]:


# Length of Membership 


# In[34]:


sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)


# In[35]:


X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]


# In[36]:


y = customers['Yearly Amount Spent']


# In[37]:


from sklearn.model_selection import train_test_split


# In[38]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# In[39]:


from sklearn.linear_model import LinearRegression


# In[40]:


lm = LinearRegression()


# In[41]:


lm.fit(X_train,y_train)


# In[42]:


lm.coef_


# In[43]:


predictions = lm.predict(X_test)


# In[44]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# In[45]:


from sklearn import metrics


# In[46]:


print('MAE :'," ", metrics.mean_absolute_error(y_test,predictions))
print('MSE :'," ", metrics.mean_squared_error(y_test,predictions))
print('RMAE :'," ", np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:




