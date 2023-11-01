

## Problem Formulation

This project focuses on classifying Reddit posts as either real or fake news based on their titles. The rise of fake news on social networks, such as Reddit, has had significant societal impacts, especially in the political domain. This project aims to predict whether a Reddit post is real or fake by analyzing its title.

## Input

- Text Column/Post (Title of Reddit Posts)

## Output

- Binary Classification: Whether the Post is Real or Fake

## Challenges

- Data Cleaning: Removing whitespaces and special characters.
- Classification: Identifying patterns to distinguish between real and fake news.

## Data Mining Function

- Text Classification (Supervised Learning)

## Impact

- Detecting and mitigating the spread of false information and fake news on social platforms.

## Optimal Solution

The optimal solution for this task was achieved using Logistic Regression with text embedding and Random Search, delivering a public score of 0.85532.

## Importing Libraries

```python
# Useful Imports
%matplotlib inline
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from google.colab import drive
```

## Loading The Data

```python
# Connect to Google Drive
drive.mount('/content/drive')

# Load the Training and Test Data
train_data = pd.read_csv("/content/drive/MyDrive/fake_reddit/xy_train.csv")
test_data = pd read_csv("/content/drive/MyDrive/fake_reddit/x_test.csv")
```

## Data Preprocessing and Cleaning

### Dropping Outliers

```python
# Dropping rows where the label == 2 (considered as outliers/noise)
train_data = train_data[train_data.label != 2]
```

### Cleaning and Preprocessing

```python
# Useful imports for word preprocessing
import re
import pickle
import sklearn
import pandas as pd
import numpy as np
import holoviews as hv
import nltk
from bokeh.io import output_notebook
output_notebook()

from pathlib import Path

pd.options.display.max_columns = 100
pd.options.display.max_rows = 300
pd.options.display.max_colwidth = 100
np.set_printoptions(threshold=2000
```

### Installing scikit-optimize

```python
# Install the required package
pip install scikit-optimize
```

### More Data Cleaning and Preprocessing

```python
# Additional data cleaning and preprocessing steps are performed.
```

## Model Training

The README can include details of different model training trials, such as Logistic Regression and XGBoost, with information about the hyperparameter search, scoring, and results. This section can be structured similarly to the following examples:

### Trial 1

- Classifier: Logistic Regression
- Search: Random Search CV
- Preprocessing: TfidfVectorizer with word n-grams
- Public Score: 0.81155

### Trial 2

- Classifier: XGBoosting
- Search: Random Search CV
- Preprocessing: TfidfVectorizer with word n-grams
- Public Score: 0.78612

### Trial 3 (Optimal Solution)

- Classifier: Logistic Regression
- Search: Grid Search CV
- Preprocessing: TfidfVectorizer with word n-grams
- Public Score: 0.83756

### Trial 4 (Optimal Solution with Text Embedding)

- Classifier: Logistic Regression
- Search: Random Search CV
- Preprocessing: Text embedding
- Public Score: 0.85532

### Trial 5

- Classifier: XGBoosting
- Search: Random Search CV
- Preprocessing: TfidfVectorizer with character n-grams
- Public Score: 0.77701

## Questions

The README includes answers to specific questions related to the project, such as the differences between character n-grams and word n-grams, stop word removal and stemming, tokenization techniques, and the differences between count vectorizer and TF-IDF vectorizer. This section is structured as follows:

- Question 1: What is the difference between Character n-gram and Word n-gram?
- Question 2: What is the difference between stop word removal and stemming?
- Question 3: Is tokenization techniques language dependent?
- Question 4: What is the difference between count vectorizer and TF-IDF vectorizer? Would it be feasible to use all possible n-grams?

---

This README file provides an overview of the code, its setup instructions, data preprocessing, model training, and answers to specific questions. It serves as a guide to understand and use the provided code effectively.