#!/usr/bin/env python
# coding: utf-8

# In[5]:


get_ipython().system('pip install pandas plotly')


# In[6]:


get_ipython().system('pip install nbformat --upgrade')


# In[7]:


import pandas as pd
import plotly as pl


# In[8]:


from datetime import datetime as dt, timedelta
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors


# In[9]:


df = pd.read_csv(r'\Users\binde\Documents\Python Projects\amazon.csv')


# In[10]:


df.head()


# In[11]:


df.info()
df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[14]:


df['discounted_price'] = df['discounted_price'].str.replace('₹', '')
df['actual_price'] = df['actual_price'].str.replace('₹', '')


# In[15]:


df.head()


# In[16]:


df.dropna(subset=['rating_count'], inplace=True)
df['discounted_price'] = pd.to_numeric(df['discounted_price'], errors='coerce')
df['actual_price'] = pd.to_numeric(df['actual_price'], errors='coerce')
df.dropna(subset=['discounted_price', 'actual_price'], inplace=True)


# In[17]:


df.head()


# In[18]:


# Calculate purchase frequency
df['purchase_frequency'] = df.groupby('user_id')['product_id'].transform('count')

# Calculate average spend
df['actual_spend'] = df.apply(
    lambda x:
        x ['discounted_price'] if x ['discounted_price'] > 0 else x['actual_price'], axis=1)
df['average_spend'] = df.groupby('user_id')['actual_spend'].transform('mean')

# Calculate discount utilization
df['discount_utilization'] = df.apply(lambda x: 1 if x['discounted_price'] < x['actual_price'] else 0, axis=1)
df['discount_usage'] = df.groupby('user_id')['discount_utilization'].transform('mean')


# In[19]:


df.head()


# In[20]:


df.info()


# In[21]:


df.rename(columns={'discount_utilization':'Discount Utilization', 'discount_usage': 'Discount Usage', 'actual_spend':'Actual Spending', 'average_spend':'Average Spending', 'purchase_frequency': 'Purchase Frequency' , 'user_id':'User ID'}, inplace=True)


# In[22]:


df.head()


# In[23]:


df.to_csv('df.csv', index=False)


# In[24]:


df.info()


# # Customer Lifetime Value

# In[25]:


total_purchase_frequency = df['Purchase Frequency'].sum()

# Count of unique User IDs
num_users = df['User ID'].nunique()

# Calculate Customer Lifetime
customer_lifetime = total_purchase_frequency / num_users

df['CLV'] = df['Average Spending'] * df['Purchase Frequency'] * customer_lifetime


# Discount Adjustment :  Calculate how discounts impact the revenue. 

# In[26]:


df['Discount Adjustment'] = 1 - df['Discount Utilization'] * 0.1  # Example adjustment factor, adjust as needed
df['adjusted_CLV'] = df['CLV'] * df['Discount Adjustment']
df.head()


# In[27]:


clv_analysis = df.groupby('User ID').agg({
    'CLV': 'mean',
    'adjusted_CLV': 'mean'
}).reset_index()


# In[28]:


get_ipython().system('pip install matplotlib')
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.hist(df['CLV'], bins=50, alpha=0.5, label='CLV')
plt.hist(df['adjusted_CLV'], bins=50, alpha=0.5, label='Adjusted CLV')
plt.xlabel('Customer Lifetime Value')
plt.ylabel('Frequency')
plt.title('Distribution of CLV and Adjusted CLV')
plt.legend()
plt.show()


# In[29]:


#Segmenting
segment_labels=['Low-Value', 'Mid-Value', 'High-Value']

def assign_segment(score):
    if score < 1000:
        return 'Low_Value'
    elif score < 3000:
        return 'Mid_Value'
    else:
        return 'High-Value'

df['Customer Lifetime Segment']= df['CLV'].apply(assign_segment)



# In[30]:


# Count the number of customers in each segment for both CLV and adjusted_CLV
segment_counts = df['Customer Lifetime Segment'].value_counts()
# Create a DataFrame for plotting
segment_df = pd.DataFrame({
    'Segment': segment_counts.index,
    'Count': segment_counts.values,
})
segment_df.head()


# In[31]:


# Plotting the bar chart
plt.figure(figsize=(12, 8))
plt.bar(segment_df['Segment'], segment_df['Count'], color='lightblue', label='CLV Segment', width=0.4)
plt.ylabel('Number of Customers')
plt.title('Customer Lifetime Value Segments')
plt.show()


# # RFM Segmentation

# In[32]:


#RFM Segmentation without Recency
rfm = df[['User ID', 'Purchase Frequency', 'Actual Spending', 'product_id']].copy()
rfm=df.groupby('User ID').agg({
    'Actual Spending': 'sum' ,
    'Purchase Frequency': 'count',
})
print(rfm.dtypes)  
rfm.head()


# In[33]:


# Defining quantiles
quantiles = rfm.quantile(q=[0.25, 0.5, 0.75])

# Defining the scoring function
def Rscore(x, p, d):
    if x <= d[p][0.25]:
        return 1
    elif x <= d[p][0.5]:
        return 2
    elif x <= d[p][0.75]:
        return 3
    else:
        return 4

# Applying the scoring function using a lambda function
rfm['Frequency'] = rfm['Purchase Frequency'].apply(lambda x: Rscore(x, 'Purchase Frequency', quantiles))
rfm['Value'] = rfm['Actual Spending'].apply(lambda x: Rscore(x, 'Actual Spending', quantiles))



# In[34]:


rfm.head()


# In[54]:


# Segmenting the RFM scores
segment_labels=['Low-Value', 'Mid-Value', 'High-Value']

def assign_segment(score):
    if score < 4:
        return 'Low_Value'
    elif score < 8:
        return 'Mid_Value'
    else:
        return 'High-Value'

rfm['RFM_Segment_Label']= rfm['Value'].apply(assign_segment)


# In[55]:


rfm.tail()


# In[56]:


rfm.to_csv('rfm_data.csv', index=False)


# In[57]:


#Visualization
# Count unique customer IDs
num_customers = df['User ID'].nunique()

fig1= px.scatter(rfm, x='Actual Spending', y='Purchase Frequency',
                 size='Actual Spending', color='Value',
                 size_max=20,
                 title='Purchase Frequency vs Actual Spending')



# In[58]:


fig1.show()


# In[59]:


fig = px.density_heatmap(rfm, x='Frequency', y='Value',
                         title='Heatmap of Frequency vs. Value',
                         labels={'Frequency': 'Frequency Score', 'Value': 'Value Score'},
                         color_continuous_scale='Blues')
fig.show()


# In[60]:


# Count the number of customers in each RFM segment
rfm_segment_count = rfm['RFM_Segment_Label'].value_counts().reset_index()

rfm_segment_count.columns = ['RFM_Segment_Label', 'Count']

fig = px.bar(rfm_segment_count,
             x='RFM_Segment_Label', y='Count',
             title='Customer Distribution by RFM Segment',
             labels={'RFM_Segment_Label': 'RFM Segment', 'Count': 'Number of Customers'},
             color='RFM_Segment_Label',
             color_discrete_sequence=px.colors.qualitative.Pastel)

fig.show()

