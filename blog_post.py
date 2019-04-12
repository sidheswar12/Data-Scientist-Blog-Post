#!/usr/bin/env python
# coding: utf-8

# This solution is described about the Data Scientist Blog Post. CRISP-DM process will be applied.

# In[1]:


# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, median_absolute_error
from sklearn.ensemble import RandomForestRegressor


# 
# # 1. Business Understanding
# 
# It would be quite interesting to apply data analysis skills here as a football fan. I chose FIFA 19 complete player dataset.
# I will fouc on mentioned below questions:
# 
# Q1. What is the ratio of total wages/ total potential for clubs. Which clubs are the most economical？
# 
# Q2. How is nation team player in total market value distributed? 
# 
# Q3. Which player skils set influence potential/wage? Can we predict player/player's potential based on his skills set?
# 

# # 2. Data Understanding and Exploration

# In[2]:


# Load dataset
fifa19_player_data_frame = pd.read_csv('data.csv')
fifa19_player_data_frame.head()


# In[3]:


# Number of players
fifa19_player_data_frame.shape[0]


# In[4]:


# Data format for each column
fifa19_player_data_frame.info()


# In[5]:


# Types of informations in data set
fifa19_player_data_frame.columns


# In[6]:


# Missing values
fifa19_player_data_frame.isnull().sum()


# # 3. Prepare Data
# 
# As per data exploration in above section, tthere are some necessary steps to be applied before preparing data:
# 
#     1. Unused column to be dropted
# 
#     2. String to be converted to number
# 
#     3. Handle missing values, if necessary drop them
# 

# In[7]:


# Drop unused columns
columns_to_drop = ['Unnamed: 0', 'ID', 'Photo', 'Flag','Club Logo', 'Preferred Foot', 
                   'Body Type', 'Real Face', 'Jersey Number', 'Joined', 'Loaned From',
                   'Contract Valid Until', 'Height', 'Weight','LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
                   'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
                   'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB', 'Release Clause']

fifa19_player_data_frame.drop(columns_to_drop, axis=1, inplace=True)


# In[8]:


# Display data after dropped column
fifa19_player_data_frame.head()


# In[9]:


# Convert value and wage columns string to number
# Example: €110.5M = 110.5 * 1000000
def string2number(amount_str):
    """
    This function convert value and wage string to floating point number 
    
    Parameter:
    amount(str): Amount string with M & K as Abbreviation for Million and Thousands
    
    Returns:
    float: A float number represents the numerical value of the input parameter amount(str)
    """
    if amount_str[-1] == 'M':
        return float(amount_str[1:-1])*1000000
    elif amount_str[-1] == 'K':
        return float(amount_str[1:-1])*1000
    else:
        return float(amount_str[1:])


# In[10]:


# First convert value, wage string to actual amount, then divide by 1 million and 1 thousand. 
# Assigned to new columns "Value_M and Wage_K respectively"
fifa19_player_data_frame['Value_M'] = fifa19_player_data_frame['Value'].apply(lambda x: string2number(x) / 1000000)
fifa19_player_data_frame['Wage_K'] = fifa19_player_data_frame['Wage'].apply(lambda x: string2number(x) / 1000)

# Drop original value & wage column
fifa19_player_data_frame.drop(['Value', 'Wage'], axis=1, inplace=True)


# In[11]:


# Display data set ater string to number conversion
fifa19_player_data_frame.describe()


# In[12]:


# Find player Name who's Value is highest
fifa19_player_data_frame.loc[fifa19_player_data_frame['Value_M'].idxmax()]


# In[13]:


# Find player Name who's Wage is highest
fifa19_player_data_frame.loc[fifa19_player_data_frame['Wage_K'].idxmax()]


# In[14]:


# Missing value handling
missing_player_data_frame = fifa19_player_data_frame[fifa19_player_data_frame['Agility'].isnull()]


# In[15]:


missing_player_data_frame.describe()


# ### 3.1 Fifa19 data observation
# From above analysis result there are 48 missing values that quite a few columns which are related to player's skills.
# 
# So there were 48 players that simply missing those values. But, to answer Question-1 and Question-2 we will reserve those players since there were no missing value in Value_M and Wage_K column.
# 
# To explain Question-3, we will drop those player rows since there are many missing values.
# 

# # 4. Answer Questions base on dataset
# #### Q1. What is the ratio of total wages/ total potential for clubs. Which clubs are the most economical？

# In[17]:


club_wages = fifa19_player_data_frame.groupby('Club').sum()


# In[18]:


club_player_count = fifa19_player_data_frame.groupby('Club').count()


# In[20]:


# Total Number of clubs and average number of players in each club
print('Total Number of clubs is {}'.format(club_player_count.shape[0]))
print('Players average in each club is {}'.format(round(club_player_count['Age'].mean(),2)))
print('Total Average wage(K) potential ratio is {}'
      .format(round(club_wages['Wage_K'].sum() / club_wages['Potential'].sum(), 2)))


# In[21]:


club_wages['Wage/Potential'] = club_wages['Wage_K'] / club_wages['Potential']
club_wages['Player Number'] = club_player_count['Age']
club_wages['Player Average Age'] = club_wages['Age'] / club_wages['Player Number']


# In[22]:


club_wages.sort_values('Wage/Potential', ascending=False, inplace=True)


# In[23]:


club_wages.head()


# In[36]:


club_wages['Wage/Potential'].head(10).plot(kind='bar', color='Blue')
plt.title('Top 10 clubs spending wage on players potential')


# In[37]:


club_wages['Wage/Potential'].tail(10).plot(kind='bar', color='Blue')
plt.title('Top 10 economical clubs ')


# From the above analysis and plot, the Real Madrid, Barcelona, and Juventus club are willing to spend more wage for high potential players than other clubs.
# 
# The economical clubs are not famous and from nowhere that we heard about. Few of them are quite famous like AEK Athens, Dynamo Kyiv may be more. This conclude that those club's players are potiential but underpayed. It would be good approach for 'Giant' clubs to bring more econimical players to reduce their overall wage spent.

# #### Q2. How is nation team player in total market value distributed?

# In[39]:


# Age count
age_count = fifa19_player_data_frame['Age'].value_counts()
age_count.sort_index(ascending=True, inplace=True)


# In[41]:


# Calculate average overall rating
age_mean = fifa19_player_data_frame.groupby('Age').mean()


# In[43]:


# Collect age distribuion and overall rating together
age_count_list = age_count.values.tolist()
age_overall_rating_list = age_mean['Overall'].values.tolist()


# In[47]:


# Plot age distribution and overall rating together
age = age_count.index.values.tolist()
figure = plt.figure()
axis_1 = figure.add_subplot(111)
axis_1.plot(age,age_overall_rating_list, color = 'red', label='Average Rating')
axis_1.legend(loc=1)
axis_1.set_ylabel('Average Rating')

axis_2 = axis_1.twinx()
plt.bar(ages, age_count_list, label='Age Count')
axis_2.legend(loc=2)
axis_2.set_ylabel('Age Count')
plt.show()


# In figure above, we can see that most of the players are between 20-26 years age. The number of player's start decreases after 26 years age and much more decreases after 30. The main reasons could be that, many young player didn't get enough opportunities to prove themselves as a football player.
# 
# In ideal scenario, When a football player reaches their age of 20 years, they must have gain enough experience and reaches peak of their rating. The golden era for most of the football player starts 20 years of there age and ends when age reaches 35 years. Most of the football playes physical body condition drops quickly after their 35 years of age and rating quite low.
# 
# But there are also set of player's rating can remain quite high with age over 37, 38 years.

# #### Q3. Which player skils set influence potential/wage? Can we predict player/player's potential based on his skills set?

# In[49]:


# Drop unused columns to answers Question-3
columns_to_drop_q3 = ['Name', 'Nationality', 'Club']
fifa19_player_data_frame.drop(columns_to_drop_q3, axis=1, inplace=True)


# In[50]:


# Drop players whose skill set is missing.
fifa19_player_data_frame.dropna(axis=0, how='any', inplace=True)


# In[51]:


# Split Work Rate is in format of attack work rate and defence work rate
# Create two new columns here.
fifa19_player_data_frame['Work Rate Attack'] = fifa19_player_data_frame['Work Rate'].map(lambda x: x.split('/')[0])
fifa19_player_data_frame['Work Rate Defence'] = fifa19_player_data_frame['Work Rate'].map(lambda x: x.split('/')[1])


# In[52]:


#Drop origin Work Rate column
fifa19_player_data_frame.drop('Work Rate', axis=1, inplace=True)


# In[53]:


fifa19_player_data_frame.head()


# In[54]:


# One Hot Encoding for Position, Work Rate Attack and Work Rate Defence
one_hot_columns = ['Position', 'Work Rate Attack', 'Work Rate Defence']
fifa19_player_data_frame = pd.get_dummies(fifa19_player_data_frame, columns=one_hot_columns, prefix = one_hot_columns)


# In[55]:


fifa19_player_data_frame.shape


# # 5. Train model and Performance Evaluation

# In[56]:


y = fifa19_player_data_frame['Potential']
X = fifa19_player_data_frame.drop(['Value_M', 'Wage_K', 'Potential', 'Overall'], axis=1)


# In[57]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)


# In[58]:


forest_regressor = RandomForestRegressor(n_estimators=500)
forest_regressor.fit(X_train, y_train)
y_test_preds = forest_regressor.predict(X_test)
print(r2_score(y_test, y_test_preds))
print(mean_squared_error(y_test, y_test_preds))


# In[59]:


coefs_df = pd.DataFrame()

coefs_df['Features'] = X_train.columns
coefs_df['Coefs'] = forest_regressor.feature_importances_
coefs_df.sort_values('Coefs', ascending=False).head(10)


# As a football fan we know, ball control, reactions, and age are the main three features that describe player's potential and performance. In this analysis our perception is also same.
# 
# Players with excellent ball control and fast reactions tends to give us an outstanding performance in football match.

# In[60]:


coefs_df.set_index('Features', inplace=True)
coefs_df.sort_values('Coefs', ascending=False).head(5).plot(kind='bar', color='Blue')


# In[ ]:




