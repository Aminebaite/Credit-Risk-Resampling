## Introduction :

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. In this project, we’ll use various techniques to train and evaluate models with imbalanced classes. we’ll use a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.
Using our knowledge of the imbalanced-learn library, we’ll use a logistic regression model to compare two versions of the dataset. First, we’ll use the original dataset. Second, we’ll resample the data by using the RandomOverSampler module from the imbalanced-learn library.
For both cases, we’ll get the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

## Technologies : 
This project leverages Anaconda and Jupyterlab with Python 3.7.
We need also to import the following libraries and dependencies :

```
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from imblearn.metrics import classification_report_imbalanced

import warnings
warnings.filterwarnings('ignore')
````
## Contributor:

Amine Baite
