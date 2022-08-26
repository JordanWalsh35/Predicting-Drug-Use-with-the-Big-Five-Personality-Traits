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


# Data Pre-Processing
The first pre-processing task that was completed was checking for missing values. This was done with the following code which found zero missing values.


