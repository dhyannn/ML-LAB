#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# In[20]:


df=pd.read_csv("Market_Basket.csv")
df.head()


# In[21]:


x=df['Item 1'].value_counts().sort_values(ascending=False)[:10]
x


# In[22]:


plt.figure(figsize=(15, 10))
sns.barplot(x=x.index,y=x.values)
plt.show()


# In[23]:


df.describe()


# In[24]:


df.value_counts()


# In[25]:


df.isna()


# In[26]:


df.fillna(value=0)


# In[27]:


data=df.drop(columns = ['Item 20'])
data


# In[28]:


transactions = []

for i in range(0, data.shape[0]):
    transaction = [str(data.values[i, j]) for j in range(0, data.shape[1]) if str(data.values[i, j]) != '0']
    transactions.append(transaction)

print(transactions)


# In[29]:


unique_items_set = set()
for transaction in transactions:
    unique_items_set.update(item for item in transaction)
unique_items_list = list(unique_items_set)


# In[30]:


binary_matrix = pd.DataFrame(columns=unique_items_list)
data = []
for transaction in transactions:
    row = [1 if item in transaction else 0 for item in unique_items_list]
    data.append(row)

binary_matrix = pd.concat([binary_matrix, pd.DataFrame(data, columns=unique_items_list)], ignore_index=True)

print(binary_matrix)


# In[31]:


top10=binary_matrix.sum().sort_values(ascending=False)[:10]
top10


# In[32]:


frequent_itemset = apriori(binary_matrix,min_support=0.06,use_colnames=True)
rules = association_rules(frequent_itemset,metric='lift',min_threshold=1)


# In[33]:


basket=pd.DataFrame(binary_matrix)
frequent_itemset = apriori(basket,min_support=0.03,use_colnames=True)
rules = association_rules(frequent_itemset,metric='lift',min_threshold=1)
rules.head()
rules[(rules['confidence']>0.2) & (rules['lift']>1)]


# In[34]:


worst_choice=rules.sort_values(by='lift',ascending=True)
print('Worst choice to buy items together')
print(worst_choice[['antecedents','consequents','lift']])


# In[35]:


best_choice = rules.sort_values(by=['confidence', 'lift'], ascending=[False, False])
print('Best choice to buy items together')
print(best_choice[['antecedents', 'consequents', 'confidence','lift']])


# In[36]:


top=best_choice.head(10)
print(top)


# In[38]:


top10 = best_choice.head(10)
combinations = [' => '.join(map(str, combination)) for combination in zip(top10['antecedents'], top10['consequents'])]
support_values = top10['support']

# Create the pie chart
plt.figure(figsize=(10, 10))
plt.pie(support_values, labels=combinations, autopct='%1.1f%%', startangle=140)
plt.axis('equal')  
plt.title('Top 10 Most Frequent Bought-Together Combinations (Pie Chart)')
plt.show()


# In[ ]:




