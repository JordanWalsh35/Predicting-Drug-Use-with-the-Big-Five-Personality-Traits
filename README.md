# Predicting-Drug-Use-with-the-Big-Five-Personality-Traits
Using machine learning classification models to predict a person's willingness to try illegal drugs based on quantified scores of their personality traits.
  
  
# Overview

The objective of this project was to study the relationship between the Big Five personality traits and the use of illegal drugs. This was done by testing the significance of trait differences between users and non-users and by creating several machine learning classification models to test the predictive power of the Big Five traits and some other features. In addition, correlations between the use of different drugs were also examined in order to discover any interesting relationships and test the ‘Gateway drug’ theory. 


# Background

The Big Five personality trait model was initially developed in 1949 by D. W. Fiske and has grown in popularity over time. The model breaks personality down into five character traits: Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism – often abbreviated as OCEAN. The Big Five model has been studied extensively in relation to its predictive power (successfully predicts outcomes relating to personal, interpersonal, social, health etc.), cultural differences, gender differences and personality disorders. 

Openness describes a person’s willingness to try new experiences, their intellectually curiosity, openness to emotion and creativity. An individual who has high openness is also more likely to engage in risky behaviour. Conscientiousness is represented by high levels of thoughtfulness, impulse control and organization. Extraversion is displayed by those who enjoy interacting with other people, are full of energy, talkative and assertive. Agreeableness reflects an individual’s concern for social harmony, optimistic thinking and their willingness to compromise their own interests for those of others. Neuroticism is associated with emotional instability and those who are high in this trait often experience high levels of sadness, moodiness, anxiety and any other negative feelings. 


# Data

The dataset was downloaded from the UCI (University of California Irvine) machine learning repository (link here). The data was recorded for 1885 individuals who responded to online surveys. The test for the OCEAN model involves answering a series of questions (typically 40-60 questions) using a scale from 1-5 (1 representing ‘strongly disagree’ and 5 representing strongly agree). The answers are then used to calculate an unitless score for each personality trait. 

In addition to scores for the Big Five personality traits, the data also contained values for age (ranged values, e.g. 18-24), highest educational status, gender, country, ethnicity and two other personality measures labelled as ‘Impulsivity’ and ‘Sensation Seeking’. There are also an additional 18 columns, one for each of the various legal and illegal substances where participants gave one of the following answers: CL0 = Never Used, CL1 = Used over a Decade Ago, CL2 = Used in Last Decade, CL3 = Used in Last Year, CL4 =  Used in Last Month, CL5 = Used in Last Week, CL6 = Used in Last Day.

These substances were as follows (ordered): alcohol, amphetamines, amyl nitrate, benzodiazepines, caffeine, cannabis, chocolate, cocaine, crack, ecstasy, heroin, ketamine, legal highs, LSD, Methamphetamine, mushrooms, nicotine and volatile substance abuse. 

The data was imported into Visual Studio Code using Python and the head of the dataset was shown with the following: 

```python
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statistics as stat
from scipy import stats

missing_values = ['na','NA','N/A','n/a']
data = pd.read_csv("drug_consumption.data",header=None, na_values=missing_values)
header_cols = ['ID','Age','Gender','Education','Country','Ethnicity','Nscore','Escore','Oscore','AScore','Cscore',
    'Impulsive','SS','Alcohol','Amphet','Amyl','Benzos','Caff','Cannabis','Choc','Coke','Crack','Ecstasy','Heroin',
    'Ketamine','Legalh','LSD','Meth','Mushrooms','Nicotine','Semer','VSA'
    ]
data.columns = header_cols

data.head()

```
![](Images/data_head.PNG)

# Data Pre-Processing
The first pre-processing task that was completed was checking for missing values. This was done with the following code which found zero missing values.

```python
# Check for missing values
chk = 0
for head in header_cols:
    null_val = data[head].isnull()
    for j in null_val:
        if j == True:
            chk = chk + 1
print(chk)

```

It should be noted that some data pre-processing was already performed by the original author before the dataset was made available for download. This may or may not have included some data cleaning. What is known for sure is that the pre-processing performed by the author did include at least two steps: handling of categorical variables and feature scaling:


### 1)	Handling of Categorical Variables:
There are a number of categorical variables in the dataset, namely: age (range), gender, education, country and ethnicity. These were all given numerical values. This was presumably done in a way that intended to avoid one-hot encoding (creating binary data for each categorical option, i.e. 1 or 0) given the number of possible values for each category. 
For example age has six levels (possible values), education has nine, country has seven and ethnicity has seven. If these variables were one-hot encoded there would be 29 columns for just four variables. When dealing with categorical variables that have multiple options like this, having this many dummy columns is often avoided as it may impact the results of the study. 

### 2)	Feature Scaling
It can be seen by looking the dataset above that some sort of feature scaling was also performed on the data. This appears to be a form of standardization scaling as the data is distributed about a mean of approximately zero. This applies to both categorical and numerical data. Therefore, no feature scaling was needed to be done as part of this project in order to facilitate the predictive modelling section later on. 

### Converting Back to Original Data
The keys were provided by the authors so that the scaled data can be converted back to the original data. This was partially done for visual/aesthetic purposes but mostly because the original data was used to calculate the descriptive statistics for the Big Five traits. This was not a complete necessity but it was preferred to show the descriptive statistics of the Big Five in a form that was consistent with typical scores. The keys were saved in a file called Dictionaries.xlsx. The converted (original) data can be seen below.

```python
# Dictionaries for categorical variables
Nscore_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Nscores')
Nscore_xl = Nscore_xl.iloc[:, 1:]
Nscore_dict = Nscore_xl.set_index('Score').to_dict()['Value']

Escore_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Escores')
Escore_xl = Escore_xl.iloc[:, 1:]
Escore_dict = Escore_xl.set_index('Score').to_dict()['Value']

Oscore_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Oscores')
Oscore_xl = Oscore_xl.iloc[:, 1:]
Oscore_dict = Oscore_xl.set_index('Score').to_dict()['Value']

Ascore_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Ascores')
Ascore_xl = Ascore_xl.iloc[:, 1:]
Ascore_dict = Ascore_xl.set_index('Score').to_dict()['Value']

Cscore_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Cscores')
Cscore_xl = Cscore_xl.iloc[:, 1:]
Cscore_dict = Cscore_xl.set_index('Score').to_dict()['Value']

Age_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Age')
Age_xl = Age_xl.iloc[:, 1:]
Age_dict = Age_xl.set_index('Score').to_dict()['Value']

Edu_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Education')
Edu_xl = Edu_xl.iloc[:, 1:]
Edu_dict = Edu_xl.set_index('Score').to_dict()['Value']

Country_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Country')
Country_xl = Country_xl.iloc[:, 1:]
Country_dict = Country_xl.set_index('Score').to_dict()['Value']

Eth_xl = pd.read_excel('Dictionaries.xlsx', sheet_name = 'Ethnicity')
Eth_xl = Eth_xl.iloc[:, 1:]
Eth_dict = Eth_xl.set_index('Score').to_dict()['Value']
```

```python
# Convert Scores via Dictionaries to calculate Descriptive Stats
orig_data = data.copy()
for i in range(0, len(data)):
    for keys, values in Nscore_dict.items():
        if orig_data['Nscore'][i] == values:
            orig_data['Nscore'].iat[i] = keys
    for keys, values in Escore_dict.items():
        if orig_data['Escore'][i] == values:
            orig_data['Escore'].iat[i] = keys
    for keys, values in Ascore_dict.items():
        if orig_data['AScore'][i] == values:
            orig_data['AScore'].iat[i] = keys
    for keys, values in Oscore_dict.items():
        if orig_data['Oscore'][i] == values:
            orig_data['Oscore'].iat[i] = keys
    for keys, values in Cscore_dict.items():
        if orig_data['Cscore'][i] == values:
            orig_data['Cscore'].iat[i] = keys
    for keys, values in Age_dict.items():
        if orig_data['Age'][i] == values:
            orig_data['Age'].iat[i] = keys
    for keys, values in Edu_dict.items():
        if orig_data['Education'][i] == values:
            orig_data['Education'].iat[i] = keys
    for keys, values in Country_dict.items():
        if orig_data['Country'][i] == values:
            orig_data['Country'].iat[i] = keys
    for keys, values in Eth_dict.items():
        if orig_data['Ethnicity'][i] == values:
            orig_data['Ethnicity'].iat[i] = keys
    if orig_data['Gender'][i] == 0.48246:
            orig_data['Gender'].iat[i] = 'F'
    else:
         orig_data['Gender'].iat[i] = 'M'
            
orig_data.head()
```

![](Images/orig_data.PNG)

### Splitting the Data & Creating the Response Variable

The dataset was then copied so that three distinct scenarios could be studied, based on the type of drug user. In each of these three datasets a new variable was created which will be used as the response variable in the predictive modelling section later on. Given the nature of the answers that the respondents gave for each substance (CL0 to CL6 mentioned earlier, i.e. Never tried, Used in Last Decade etc.), it was determined that this would have to be a binary variable (0 or 1) for classification modelling. 
The method used for calculating the output variable was therefore the defining feature that separates the datasets. This was done to investigate any possible differences in predictive power of the Big Five traits for those who are more active users and for those who use more dangerous drugs.

The three datasets were created with the following logic and code:   
### 1) A participant who has tried at least one illegal drug at any point in their lives.   
The list of illegal substances used can be seen in the code below. The binary variable (‘Try_Drug’) was set to 1 if a participant answered CL1, CL2, CL3, CL4, CL5 or CL6 for at least one drug in the dataset and was therefore 0 if they answered CL0 for all illegal drugs.

```python
# Dataset 1: Tried illegal drugs at least once

df1 = data.copy()
df1 = df1.drop(columns=['ID','Country','Ethnicity'])
df1['Try_Drug'] = ''
TryD = df1['Try_Drug']

drugs = ['Amphet','Amyl','Benzos','Cannabis','Coke','Crack','Ecstasy','Heroin','Ketamine','LSD','Meth','Mushrooms','VSA']

for i in range(0, len(df1)):
    tot = 0
    for j in drugs:
        if df1[j][i] != "CL0":
            tot = tot + 1
        if tot > 0:
            TryD.iat[i] = 1
        else:
            TryD.iat[i] = 0       

# Separate data for t-test
Nvr_Tried = pd.DataFrame().assign(Nscore = orig_data['Nscore'], Escore = orig_data['Escore'], Oscore = orig_data['Oscore'], Ascore = orig_data['AScore'], Cscore = orig_data['Cscore'], 
            Impulsive = orig_data['Impulsive'], SS = orig_data['SS'] ,Try_Drug = df1['Try_Drug'])

Tried = Nvr_Tried.loc[Nvr_Tried['Try_Drug'] == 1]
Tried_Samp = Tried.sample(n=300)
Nvr_Tried = Nvr_Tried.loc[Nvr_Tried['Try_Drug'] == 0]
```

### 2) A participant who has used any illegal drug in the past year. 
This dataset was for more recent (active) users and also has the benefit of being more balanced. Dataset 1 had 1585 positives (1) and only 300 negatives (0). The benefit of having a more balanced dataset will be seen in the modelling section later on. This dataset on the other hand has 1174 positives and 711 negatives.

```python
# Dataset 2: Used illegal drugs in past year

df2 = data.copy()
df2 = df2.drop(columns=['ID','Country','Ethnicity'])
df2['Try_Drug'] = ''
TryD2 = df2['Try_Drug']

for i in range(0, len(df2)):
    tot = 0
    for j in drugs:
        if df2[j][i] == "CL3" or df2[j][i] == "CL4" or df2[j][i] == "CL5" or df2[j][i] == "CL6":
            tot = tot + 1
        if tot > 0:
            TryD2.iat[i] = 1
        else:
            TryD2.iat[i] = 0    

# Separate data for t-test
Recent_N = pd.DataFrame().assign(Nscore = orig_data['Nscore'], Escore = orig_data['Escore'], Oscore = orig_data['Oscore'], Ascore = orig_data['AScore'], Cscore = orig_data['Cscore'], 
            Impulsive = orig_data['Impulsive'], SS = orig_data['SS'] ,Try_Drug = df2['Try_Drug'])

Recent_Y = Recent_N.loc[Recent_N['Try_Drug'] == 1]
Recent_Y_Samp = Recent_Y.sample(n=711)
Recent_N = Recent_N.loc[Recent_N['Try_Drug'] == 0]
```

### 3) A participant who has tried ‘hard’ drugs (heroin, crack, meth) at least once in their lives. 
The final dataset was for participants who engaged in very risky behaviour. While many drugs in this study can be viewed as party drugs that are often used infrequently – heroin, crack and methamphetamine are extremely addictive and dangerous substances. The dataset has 579 positives and 1306 negatives. 

```python
# Dataset 3: Tried hard drugs (heroin, crack or meth) at least once

df3 = data.copy()
df3 = df3.drop(columns=['ID','Country','Ethnicity'])
df3['Try_Drug'] = ''
TryD3 = df3['Try_Drug']

for i in range(0, len(df3)):
    if (df3['Heroin'][i] != 'CL0' or df3['Meth'][i] != 'CL0' or df3['Crack'][i] != 'CL0'):
        TryD3.iat[i] = 1
    else:
        TryD3.iat[i] = 0 

# Separate data for t-test
Heavy_N = pd.DataFrame().assign(Nscore = orig_data['Nscore'], Escore = orig_data['Escore'], Oscore = orig_data['Oscore'], Ascore = orig_data['AScore'], Cscore = orig_data['Cscore'], 
            Impulsive = orig_data['Impulsive'], SS = orig_data['SS'] ,Try_Drug = df3['Try_Drug'])

Heavy_Y = Heavy_N.loc[Heavy_N['Try_Drug'] == 1]
Heavy_N = Heavy_N.loc[Heavy_N['Try_Drug'] == 0] 
Heavy_N_Samp = Heavy_N.sample(n=579)
```

As can be seen in the code above, in addition to creating the response variable, two other actions were performed for each dataset. Firstly, some columns were removed, namely: ID, country and ethnicity. Ethnicity was removed as over 90% of the respondents happened to be white, meaning that this couldn’t be fairly analysed as an unbiased predictor variable. Country was removed for the same reason, as UK and USA alone account for 85% of the data.

The second action that was performed was to create new subsets of each dataset, to be used for the t-tests that will be carried out in the next section. These new subsets are simply dataframes that contain the Big 5 scores for each participant, with one dataframe being for users and the other for non-users (i.e. add row to dataframe conditional on whether Try_Drug is equal to 1 or 0). 


# Exploratory Data Analysis
Several aspects of the data were examined before the machine learning models were applied (next section). This section highlights some key insights about the data, including descriptive statistics, checks for normality, t-tests, a graphical representation of the popularity of each substance and correlations between the use of each substance.

### Descriptive Statistics  
As mentioned previously the overall dataset contains responses from 1885 survey participants. Based on dataset 1 we can see that 84.1% of the participants (1585) have tried at least one illegal drug. This is quite high and may represent a self-selection bias in the people who participated in the survey. Further analysis shows that approximately 15% of participants have tried heroin (280 people) and dataset 3 shows that over 30% of the sample have tried either heroin, meth or crack. 

The mean values and standard deviations for each of the Big Five personality traits were calculated in python and are shown below. These stats are for the entire sample of 1885 participants.

```python
# Descriptive Statistics of Personality Traits

# Means
print("Means:")
print("Nscore: {:.2f}".format(orig_data['Nscore'].mean()))
print("Escore: {:.2f}".format(orig_data['Escore'].mean()))
print("Oscore: {:.2f}".format(orig_data['Oscore'].mean()))
print("Ascore: {:.2f}".format(orig_data['AScore'].mean()))
print("Cscore: {:.2f}".format(orig_data['Cscore'].mean()))

# Standard Devs
print("\nStandard Deviations:")
print("Nscore: {:.2f}".format(orig_data['Nscore'].std()))
print("Escore: {:.2f}".format(orig_data['Escore'].std()))
print("Oscore: {:.2f}".format(orig_data['Oscore'].std()))
print("Ascore: {:.2f}".format(orig_data['AScore'].std()))
print("Cscore: {:.2f}".format(orig_data['Cscore'].std()))
```

Means:  
Nscore: 35.92  
Escore: 39.58  
Oscore: 45.76  
Ascore: 42.87  
Cscore: 41.44  

Standard Deviations:  
Nscore: 9.14  
Escore: 6.77  
Oscore: 6.58  
Ascore: 6.44  
Cscore: 6.97  

### Checking for normality  
In order to examine whether there are significant differences in the five traits between groups of users and non-users we need to perform t-tests. These will show whether the differences between the means are statistically significant. Before performing a t-test we need to assess whether the data follows a normal distribution as this is one of the assumptions of the standard Student T-test.  

Multiple tests were carried out to check for normality. Firstly, some histograms and Q-Q plots were created to make an approximate judgement the normality of the data by eye.  



