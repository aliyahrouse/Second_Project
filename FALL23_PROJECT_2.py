#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Capstone Project - Solutions

# For this capstone project we will be analyzing some 911 call data from [Kaggle](https://www.kaggle.com/mchirico/montcoalert). The data contains the following fields:
# 
# * lat : String variable, Latitude
# * lng: String variable, Longitude
# * desc: String variable, Description of the Emergency Call
# * zip: String variable, Zipcode
# * title: String variable, Title
# * timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# * twp: String variable, Township
# * addr: String variable, Address
# * e: String variable, Dummy variable (always 1)
# 
# Just go along with this notebook and try to complete the instructions or answer the questions in bold using your Python and Data Science skills!

# ## Data and Setup

# ____
# ** Import numpy and pandas **

# In[1]:


import numpy as np
import pandas as pd


# ** Import visualization libraries and set %matplotlib inline. **

# In[2]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# ** Read in the csv file as a dataframe called df **

# In[3]:


df = pd.read_csv('911.csv')
df.drop(labels = 'e',axis=1,inplace=True)


# ** Check the info() of the df **

# In[4]:


df.info()


# In[ ]:





# ** Check the head of df **

# In[5]:


df.head()


# In[ ]:





# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[6]:


df['zip'].value_counts().iloc[:5]


# In[ ]:





# ** What are the top 5 townships (twp) for 911 calls? **

# In[7]:


df['twp'].value_counts().iloc[:5]


# In[ ]:





# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[8]:


df['title'].nunique()


# In[ ]:





# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.** 
# 
# **For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. **

# In[9]:


df.info()


# In[10]:


df['timeStamp']=pd.to_datetime(df['timeStamp'])


# In[ ]:





# In[15]:


df.head()


# In[11]:


df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day_of_Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df['Reason'] = df['title'].apply(lambda s:s.split(':')[0])
df['Reason'].head()


# In[ ]:





# ** What is the most common Reason for a 911 call based off of this new column? **

# In[12]:


df['Reason'].value_counts()


# In[ ]:





# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[13]:


sns.countplot(x='Reason', data=df)


# In[ ]:





# ** Now use seaborn to create a barplot of 911 calls by Reason. **

# In[14]:


df.head(1)


# In[16]:


sns.barplot(x='Reason', y='Hour', data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


type(df['timeStamp'].iloc[0])


# In[ ]:





# ** You should have seen that these timestamps are still strings. Use [pd.to_datetime](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.to_datetime.html) to convert the column from strings to DateTime objects. **

# In[ ]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[ ]:





# In[ ]:


type(df['timeStamp'].iloc[0])


# ** You can now grab specific attributes from a Datetime object by calling them. For example:**
# 
#     time = df['timeStamp'].iloc[0]
#     time.hour
# 
# **You can use Jupyter's tab method to explore the various attributes you can call. Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week. You will create these columns based off of the timeStamp column, reference the solutions if you get stuck on this step.**

# In[17]:


df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)
df['Month'] = df['timeStamp'].apply(lambda time:time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)


# In[ ]:


df.sample()


# ** Notice how the Day of Week is an integer 0-6. Use the .map() with this dictionary to map the actual string names to the day of the week: **
# 
#     dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

# In[18]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[19]:


df['Day of Week'] = df['Day of Week'].apply(lambda int:dmap[int])


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[20]:


sns.countplot(x='Day of Week', hue='Reason', data=df)
plt.legend(bbox_to_anchor=(1,1))


# In[ ]:





# In[ ]:





# ** Now do the same for Month:**

# In[21]:


sns.countplot(x='Month', hue='Reason', data=df)
plt.legend(bbox_to_anchor=(1,1))


# In[ ]:





# ** Did you notice something strange about the Plot? **

# In[ ]:


# It is missing some months! 9,10, and 11 are not there.


# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas...**

# ** Now create a gropuby object called byMonth, where you group the DataFrame by the month column and use the count() method for aggregation. Use the head() method on this returned DataFrame. **

# In[22]:


byMonth = df.groupby(by='Month').count()


# In[23]:


byMonth


# In[ ]:





# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[24]:


byMonth['lat'].plot()


# In[25]:


sns.pointplot(x=byMonth.index, y = 'lat', data=byMonth, markers='.')


# In[ ]:


# Could be any column


# ** Now see if you can use seaborn's lmplot() to create a linear fit on the number of calls per month. Keep in mind you may need to reset the index to a column. **

# In[26]:


byMonth['Month'] = byMonth.index
byMonth


# In[27]:


sns.lmplot(x='Month', y='lat', data=byMonth)


# **Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[28]:


df['Date'] = df['timeStamp'].apply(lambda time:time.date())


# In[29]:


df.head()


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[30]:


df.groupby(by='Date').count()['lat'].plot()
plt.tight_layout()


# In[ ]:





# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[31]:


df[df['Reason']=='Traffic'].groupby(by='Date').count()['lat'].plot()
plt.title('Traffic')


# In[32]:


df[df['Reason']=='Fire'].groupby(by='Date').count()['lat'].plot()
plt.title('Fire')


# In[33]:


df[df['Reason']=='EMS'].groupby(by='Date').count()['lat'].plot()
plt.title('EMS')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ____
# ** Now let's move on to creating  heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. There are lots of ways to do this, but I would recommend trying to combine groupby with an [unstack](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.unstack.html) method. Reference the solutions if you get stuck on this!**

# In[34]:


dfGrid = df.groupby(by=['Day of Week','Hour']).count()['lat'].unstack()
dfGrid = dfGrid.loc[['Sun','Mon','Tue','Wed','Thu','Fri','Sat']]
dfGrid


# In[ ]:





# ** Now create a HeatMap using this new DataFrame. **

# In[35]:


plt.figure(figsize=(12,6))
sns.heatmap(dfGrid, cmap='winter')


# In[ ]:





# ** Now create a clustermap using this DataFrame. **

# In[36]:


sns.clustermap(dfGrid, cmap='Greens')


# In[ ]:





# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[37]:


dfMonth = df.groupby(['Day of Week','Month']).count()['lat'].unstack()
dfMonth = dfMonth.loc[['Sun','Mon','Tue','Wed','Thu','Fri','Sat']]
dfMonth


# In[ ]:





# In[38]:


plt.figure(dpi=100)
sns.heatmap(dfMonth, cmap='plasma')


# In[ ]:





# In[39]:


plt.figure(dpi=100)
sns.clustermap(dfMonth, cmap='coolwarm')


# In[ ]:





# # Below, find the list of possible values for the seabor palette

# 'gnuplot1' is not a valid value for name; supported values are 'Accent', 'Accent_r', 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r', 'CMRmap', 'CMRmap_r', 'Dark2', 'Dark2_r', 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r', 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r', 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd', 'PuRd_r', 'Purples', 'Purples_r', 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r', 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'Spectral', 'Spectral_r', 'Wistia', 'Wistia_r', 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r', 'afmhot', 'afmhot_r', 'autumn', 'autumn_r', 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r', 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'crest', 'crest_r', 'cubehelix', 'cubehelix_r', 'flag', 'flag_r', 'flare', 'flare_r', 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar', 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r', 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r', 'hot', 'hot_r', 'hsv', 'hsv_r', 'icefire', 'icefire_r', 'inferno', 'inferno_r', 'jet', 'jet_r', 'magma', 'magma_r', 'mako', 'mako_r', 'nipy_spectral', 'nipy_spectral_r', 'ocean', 'ocean_r', 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r', 'rainbow', 'rainbow_r', 'rocket', 'rocket_r', 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r', 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted', 'twilight_shifted_r', 'viridis', 'viridis_r', 'vlag', 'vlag_r', 'winter', 'winter_r'

# **Continue exploring the Data however you see fit!**
# # Great Job!
