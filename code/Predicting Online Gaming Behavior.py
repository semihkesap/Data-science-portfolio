#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('gaming_behavior.csv')
print("Data Preview:\n", df.head())
print("Data Info:\n", df.info())


# In[3]:


# 1. Numeric Variables Analysis
numeric_vars = ['PlayerID', 'Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
                'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']


# In[4]:


# Histograms
plt.figure(figsize=(15, 10))
for i, var in enumerate(numeric_vars, 1):
    plt.subplot(4, 2, i)
    sns.histplot(df[var], bins=30, kde=True)
    plt.title(f'Histogram of {var}')
plt.tight_layout()
plt.show()


# In[5]:


# Box Plots (Outlier Check)
plt.figure(figsize=(15, 10))
for i, var in enumerate(numeric_vars, 1):
    plt.subplot(4, 2, i)
    sns.boxplot(x=df[var])
    plt.title(f'Box Plot of {var}')
plt.tight_layout()
plt.show()


# In[6]:


# Handle Outliers (IQR Method)
for var in numeric_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]
    print(f"{var} - Outliers: {len(outliers)} (Range: {lower_bound:.2f} to {upper_bound:.2f})")


# In[7]:


# QQ Plots (Key Numerics)
key_numerics = ['PlayTimeHours', 'AvgSessionDurationMinutes', 'SessionsPerWeek', 'AchievementsUnlocked']
plt.figure(figsize=(12, 8))
for i, var in enumerate(key_numerics, 1):
    plt.subplot(2, 2, i)
    stats.probplot(df[var], dist="norm", plot=plt)
    plt.title(f'QQ Plot of {var}')
plt.tight_layout()
plt.show()

print("\nQQ Plots Explained: These compare your data to a normal distribution. If points follow the red line, it’s normal. Deviations suggest skewness/outliers.")


# In[8]:


# Kolmogorov-Smirnov Tests (Normality Check)
for var in numeric_vars:
    stat, p = stats.kstest(df[var], 'norm', args=(df[var].mean(), df[var].std()))
    print(f"{var} - KS Stat: {stat:.4f}, p-value: {p:.4f}")
    if p > 0.05:
        print(f"  -> {var} looks normally distributed (p > 0.05)")
    else:
        print(f"  -> {var} is not normal (p <= 0.05)")


# In[9]:


# Missing Value Check & Drop (Quantitative Only)
quant_vars = ['Age', 'PlayTimeHours', 'InGamePurchases', 'SessionsPerWeek', 
              'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']
print("\nMissing Values in Quantitative Vars:")
for var in quant_vars:
    missing = df[var].isna().sum()
    print(f"{var}: {missing} missing")
df = df.dropna(subset=quant_vars)
print(f"Rows after drop: {len(df)}")


# In[10]:


# 2. Calculate KPIs
# Retention Rate
retention_rate = (df['EngagementLevel'] != 'Low').mean() * 100
print(f"Retention Rate: {retention_rate:.2f}%")


# In[11]:


# Monetization Rate
monetization_rate = (df['InGamePurchases'] == 1).mean() * 100
print(f"Monetization Rate: {monetization_rate:.2f}%")


# In[12]:


# Avg Session Duration
avg_session = df['AvgSessionDurationMinutes'].mean()
print(f"Avg Session Duration: {avg_session:.2f} minutes")


# In[13]:


# Engagement Frequency
avg_sessions = df['SessionsPerWeek'].mean()
print(f"Avg Sessions per Week: {avg_sessions:.2f}")


# In[14]:


# Progression Rate
df['progression_rate'] = df['AchievementsUnlocked'] / df['PlayerLevel'].replace(0, 1)
progression_avg = df['progression_rate'].mean()
print(f"Avg Progression Rate: {progression_avg:.2f} achievements/level")


# In[15]:


# 3. Explore Patterns
# Engagement vs. Session Duration
sns.boxplot(x='EngagementLevel', y='AvgSessionDurationMinutes', data=df)
plt.title('Session Duration by Engagement Level')
plt.show()


# In[16]:


# Purchases vs. Play Time
sns.scatterplot(x='PlayTimeHours', y='InGamePurchases', hue='EngagementLevel', data=df)
plt.title('Play Time vs. Purchases by Engagement')
plt.show()


# In[17]:


# 4. Predict Engagement
# Encode categorical
le = LabelEncoder()
df['EngagementLevel'] = le.fit_transform(df['EngagementLevel'])  # High=2, Medium=1, Low=0
df['Gender'] = le.fit_transform(df['Gender'])  # Male/Female to 0/1
df['GameDifficulty'] = le.fit_transform(df['GameDifficulty'])  # Easy/Med/Hard to 0/1/2


# In[18]:


# Features 
X = df[['Age', 'Gender', 'PlayTimeHours', 'InGamePurchases', 'GameDifficulty', 
        'SessionsPerWeek', 'AvgSessionDurationMinutes', 'PlayerLevel', 'AchievementsUnlocked']]
y = df['EngagementLevel']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[19]:


# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)
print(f"Engagement Prediction Accuracy: {accuracy:.2f}")


# In[20]:


# Feature Importance
sns.barplot(x=model.feature_importances_, y=X.columns)
plt.title('What Drives Engagement?')
plt.show()


# In[21]:


# 5. Insight Checks
# Sessions by Engagement
print("\nSessions per Week by Engagement:")
print(df.groupby('EngagementLevel')['SessionsPerWeek'].mean())


# In[22]:


print(df['EngagementLevel'].value_counts())


# In[23]:


# Session Duration by Engagement & Level
df['AboveMedianLevel'] = df['PlayerLevel'] > df['PlayerLevel'].median()  # Boolean column
sns.boxplot(x='EngagementLevel', y='AvgSessionDurationMinutes', hue='AboveMedianLevel', data=df)
plt.title('Session Duration by Engagement (High vs. Low Level)')
plt.legend(title='Above Median Level', labels=['No', 'Yes'])
plt.show()


# In[24]:


# Purchases vs. Sessions
print("\nSessions per Week by Purchases:")
print(df.groupby('InGamePurchases')['SessionsPerWeek'].mean())


# In[25]:


# Validate Low Engagement Sessions
low_sessions = df[df['EngagementLevel'] == 1]['SessionsPerWeek']
print("\nLow Engagement Sessions - Median:", low_sessions.median())
print("Low Engagement Sessions - 75th Percentile:", low_sessions.quantile(0.75))

sns.boxplot(x='GameDifficulty', y='AvgSessionDurationMinutes', data=df)
plt.title('Session Duration by Difficulty')
plt.show()

df['AboveMedianLevel'] = df['PlayerLevel'] > df['PlayerLevel'].median()
sns.boxplot(x='EngagementLevel', y='AvgSessionDurationMinutes', hue='AboveMedianLevel', data=df)
plt.title('Session Duration by Engagement (High vs. Low Level)')
plt.legend(title='Above Median Level', labels=['No', 'Yes'])
plt.show()


# In[26]:


# Save
df.to_csv('semih_gaming_analytics.csv', index=False)
print("Saved for your portfolio!")

