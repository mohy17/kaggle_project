#---------------------import librarie--------------------#

import pandas as pd
import os
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

#-----------------------read data--------------------------------#

df=pd.read_csv(os.path.join(os.getcwd(),"Mall_Customers.csv"))

df.drop(['CustomerID'],axis=1,inplace=True)

#-------------------------prepare for transform -----------------#

numeric_data = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_data = df.select_dtypes(exclude=[np.number]).columns.tolist()

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Handle missing values
    ('scaler', StandardScaler())  # Scale numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('ordinal', OrdinalEncoder())  # Encode categorical features
])


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numeric_data),
        ('cat', categorical_transformer, categorical_data)
    ]
)

X_processed = preprocessor.fit_transform(df)


kmeans = KMeans(n_clusters=6, random_state=42)
kmean=kmeans.fit(X_processed)
df['label']=kmeans.labels_

#------------------------------------add cluster ---------------------#

def new_instance(x_new):
    x=pd.DataFrame([x_new])
    x=preprocessor.transform(x)
    x=kmeans.predict(x)
    return f"The cluster is: {x[0]}"