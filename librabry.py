import numpy as np
import pandas as pd
from itertools import groupby
from sklearn.model_selection import train_test_split
from pandas.api.types import is_datetime64_ns_dtype

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import lightgbm as lgb
from imblearn.under_sampling import RandomUnderSampler
from joblib import Parallel, delayed
import gc
import plotly.express as px
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score , auc

import warnings
warnings.filterwarnings("ignore")

from metric import score # Import event detection ap score function