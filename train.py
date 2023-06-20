import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyClassifier

# Load data
data = pd.read_csv("dataset/train.csv")
# print(data.info())

target = "price_range"

x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

x_train = transformer.fit_transform(x_train)
x_test = transformer.transform(x_test)

cls = LazyClassifier(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = cls.fit(x_train, x_test, y_train, y_test)
print(models)
print(predictions)

plt.figure(figsize=(10, 5))
sns.set_theme(style="whitegrid")
ax = sns.barplot(x=models.index, y="Accuracy", data=models)
plt.xticks(rotation=90)
plt.show()