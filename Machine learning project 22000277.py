#!/usr/bin/env python
# coding: utf-8

# ## 1ST EVALUATION

# In[134]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
dataset = pd.read_csv('C:\\Users\\kamal\\OneDrive\\Documents\\Machine Learning\\ML project\\german_credit.csv')


# In[135]:


dataset


# In[136]:


missing = dataset.isnull().sum()


# In[137]:


missing


# In[138]:


dataset.describe()


# In[139]:


print(dataset.dtypes)


# In[140]:


box_plot = dataset.drop(columns=['Creditability', 'Account Balance', 'Telephone'])


# In[141]:


plt.figure(figsize=(20, 10))
for i, col in enumerate(box_plot):
    plt.subplot(3, 7, i + 1)
    sns.boxplot(y=dataset[col])
    plt.title(col)
plt.tight_layout()
plt.show()


# In[142]:


attributes = ['Creditability', 'Account Balance', 'Duration of Credit (month)', 'Payment Status of Previous Credit', 'Purpose', 'Credit Amount', 'Value Savings/Stocks',
       'Length of current employment', 'Instalment per cent', 'Sex & Marital Status', 'Duration in Current address', 'Most valuable available asset', 'Age (years)', 'Concurrent Credits',
       'Type of apartment', 'No of Credits at this Bank', 'Occupation','No of dependents','Telephone','Foreign Worker']


# In[143]:


import pandas as pd
from scipy.stats import zscore

z_scores = dataset[attributes].apply(zscore)
threshold = 3

outliers_mask = abs(z_scores) > threshold
outliers = dataset[outliers_mask]

sum_of_outliers = outliers_mask.sum()

print("Sum of outliers in each column:")
print(sum_of_outliers)


# In[175]:


#outlier_columns = ['Duration of Credit (month)', 'Credit Amount', 'Age (years)', 'No of Credits at this Bank', 'Foreign Worker']


#def remove_outliers(df, columns):
    #for column in columns:
        #mean = dataset[column].mean()
        #std = dataset[column].std()
        #threshold = 3
        #df = df[(df[column] - mean).abs() < threshold * std]
    #return df


#df = remove_outliers(dataset, outlier_columns)


# In[144]:


for column, value in outliers.items():
    dataset.loc[dataset[column] == value, column] = dataset[column].mean()

print(dataset)


# In[145]:


plt.figure(figsize=(15, 10))
for i, col in enumerate(['Account Balance', 'Duration of Credit (month)', 'Payment Status of Previous Credit', 'Credit Amount', 'Value Savings/Stocks', 'No of dependents', 'Most valuable available asset', 'Instalment per cent', 'Concurrent Credits']):
    plt.subplot(3, 3, i + 1)
    sns.histplot(data=dataset, x=col, hue='Creditability', kde=True, bins=20)
    plt.title(f'Histogram of {col} by class')
    plt.xlabel(col)
    plt.ylabel('Frequency')
plt.suptitle('Histograms')
plt.tight_layout()
plt.show()


# In[146]:


numeric_cols = dataset.select_dtypes(include=[np.number]).columns
numeric_cols


# In[147]:


attributes = ['Creditability', 'Account Balance', 'Duration of Credit (month)', 'Payment Status of Previous Credit', 'Purpose', 'Credit Amount', 'Value Savings/Stocks',
       'Length of current employment', 'Instalment per cent', 'Sex & Marital Status', 'Duration in Current address', 'Most valuable available asset', 'Age (years)', 'Concurrent Credits',
       'Type of apartment', 'No of Credits at this Bank', 'Occupation','No of dependents','Telephone','Foreign Worker']

correlation_matrix = dataset[attributes].corr()

print("Correlation Matrix:")
print(correlation_matrix)


# In[148]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[149]:


correlation_matrix = dataset.corr()
correlation_matrix


# ## 2ND EVALUATION

# In[165]:


x = dataset[['Account Balance','Duration of Credit (month)','Payment Status of Previous Credit','Credit Amount',
       'Value Savings/Stocks','Age (years)','Occupation','Instalment per cent',
     'Most valuable available asset','Concurrent Credits','No of Credits at this Bank','Duration in Current address',
       'No of dependents','Foreign Worker']]
y = dataset['Creditability']


# In[166]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)


# In[167]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)


# # Naive Bayes Model

# In[168]:


from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
gnb = GaussianNB()


# In[169]:


gnb.fit(x_train_scaled, y_train)
y_pred = gnb.predict(x_test_scaled)
nb_accuracy = accuracy_score(y_test, y_pred)


# ## Decision tree classifier

# In[170]:


from sklearn.tree import DecisionTreeClassifier
dr = DecisionTreeClassifier()
dr.fit(x_train_scaled,y_train)
predict = dr.predict(x_test_scaled)
dr_accuracy = accuracy_score(y_test,predict)


# ## Logistic Regression Model

# In[171]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train_scaled,y_train)
predict_lr = lr.predict(x_test_scaled)
lr_accuracy = accuracy_score(y_test,predict_lr)


# ## Comparing accuracy

# In[172]:


print("Accuracy of naive bayes :", nb_accuracy)
print("Accuracy of Decision tree: ",dr_accuracy)
print("Accuracy based on logistic regression model: ",lr_accuracy)


# ## Comparing recall

# In[173]:


from sklearn.metrics import recall_score

recall_naivebayes = recall_score(y_test, y_pred)
recall_dr = recall_score(y_test,predict)
recall_lr = recall_score(y_test,predict_lr)

print("Recall for naive bayes:", recall_naivebayes)
print("Recall for Decision Tree:", recall_dr)
print("Recall for Logistic Regression :", recall_lr)


# ## Comparing f1 score

# In[174]:


from sklearn.metrics import f1_score

f1_naivebayes = f1_score(y_test, y_pred)
f1_dr = f1_score(y_test,predict)
f1_lr = f1_score(y_test,predict_lr)

print("f1 score for naive bayes :", f1_naivebayes)
print("f1 score for decision tree :", f1_dr)
print("f1 score for logistic regression :", f1_lr)


# ## COMPARING RESULTS

# 
# ### 1) The decision tree model performs the worst among the three models as it has an accuracy of 0.71, which is the lowest. Its recall and F1 score are also lower than those of the other models. This indicates that the decision tree model may not be the best choice for this dataset compared to logistic regression and naive Bayes.
# 
# ### 2) The naive Bayes model might be an appropriate choice for this this dataset with an accuracy of 0.75
# 
# ### 3) The logistic regression model outperforms both the naive Bayes and decision tree models in terms of accuracy, recall, and F1 score. It has the highest accuracy of 0.76, recall - 0.90 and f1 score - 0.84

# ## From the above observations, a logistic regression model is the most suitable model for this dataset

# In[160]:


import pandas as pd
filtered_data = dataset[dataset['Creditability'] == 1]


# In[161]:


from sklearn.cluster import KMeans

wcss = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(filtered_data)
    wcss.append(kmeans.inertia_)  

plt.plot(range(1, 11), wcss)
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()


# In[162]:


k = 2
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(filtered_data)


# In[163]:


from sklearn.metrics import silhouette_score

silhouette_avg = silhouette_score(filtered_data, kmeans.labels_)

# Print or interpret the score
print("Silhouette Score (k =", k, "):", silhouette_avg)


# ## CONCLUSIONS

# # Before treating outliers

# Accuracy of logistic regression before treating outliers- 0.76
# 
# recall of logistic regression before treating outliers - 0.90  
# 
# f1 score of logistic regression before treating outliers - 0.84

# ## After replacing outliers with mean

# Accuracy based on logistic regression model:  0.72
# 
# Recall for Logistic Regression : 0.875
# 
# f1 score for logistic regression : 0.8095238095238096

# ## After removing outliers 

# Accuracy based on logistic regression model:  0.7248677248677249
#     
# Recall for Logistic Regression : 0.889763779527559
#     
# f1 score for logistic regression : 0.8129496402877697

# 1)The accuracy of the logistic regression model decreased slightly after replacing outliers with the mean and after removing outliers compared to before treating outliers. This suggests that treating outliers did not significantly improve the overall accuracy of the model.
# 
# However, the recall of the logistic regression model improved after both outlier treatment methods. This indicates that the model became better at correctly identifying positive cases (creditable applicants in this context) after outlier treatment.
# 
# The F1 score, which is a measure of a model's accuracy, also showed improvement after outlier treatment, indicating a better balance between precision and recall.
# 
# ###  This suggests that the outliers in the dataset are not due to errors but rather because the dataset includes values at a higher range that are important for the prediction model.

# 2)K-means clustering is a method used to partition a dataset into groups (clusters) based on similarities in the data.
# 
# The Silhouette Score is a measure of how similar an object is to its own cluster compared to other clusters. A score close to 1 indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters, suggesting a good clustering. In this case, a score of 0.701 suggests that the clustering with 2 clusters is reasonably good

# ## RECOMMENDATIONS

# These are the features which will have a greater or a higher impact on the target variable
# 
# Payment Status of Previous Credit
# 
# Credit Amount
# 
# Duration of Credit (month)
# 
# Account Balance
# 
# Value Savings/Stocks
# 
# No of Credits at this Bank
# 
# No of dependents
# 
# Foreign Worker

# These are the features which can be removed as they do not affect the target variable to a greater extent-
# 
# Purpose
# 
# Sex & Marital Status
# 
# Type of apartment
# 
# Telephone

# In[ ]:




