import pandas as  pd
import numpy as np
import os
#------------------------------------------------------------------#

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder,OrdinalEncoder
from sklearn.impute import SimpleImputer

#------------------------------------------------------------------#

from sklearn.ensemble import RandomForestClassifier
#------------------------------------------------------------------#

from sklearn.metrics import accuracy_score

#-------------------------------------------------#

# READ AND HANDLE DATA


df=pd.read_csv(os.path.join(os.getcwd(),'WA_Fn-UseC_-Telco-Customer-Churn.csv'))

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df.dropna(subset=['TotalCharges'], inplace=True)
df.drop(['customerID'],axis=1,inplace=True)

df['SeniorCitizen'] = df['SeniorCitizen'].map({1:'yes', 0:'no'})


#-----------------------------------------------#

#BUILD MODEL

X = df.drop('Churn', axis=1)
y = df['Churn']

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42,stratify=y)


#SPLIT NUMERIC AND CATEGORIC
numeric_data = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_data = df.select_dtypes(exclude=[np.number]).columns.tolist()
if 'Churn' in categorical_data:
    categorical_data.remove('Churn')


#---------------PIPELINE ------------------------------------#
# Create numerical pipeline
numerical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values
    ('scaler', StandardScaler())  # Scale features
])

# Create categorical pipeline 
categorical_pipeline = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Impute missing values
    ('ordinal_encoder', OrdinalEncoder())  # Use Ordinal Encoding directly
])

# Combine pipelines using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_pipeline, numeric_data),
        ('cat', categorical_pipeline, categorical_data)
    ]
)

# Transform training and test data
X_train_full = preprocessor.fit_transform(X_train)
X_test_full = preprocessor.transform(X_test)


bagging_forest = RandomForestClassifier(bootstrap=True, n_estimators=10, max_depth=7, random_state=41)

# Fit the model
bagging_forest.fit(X_train_full, y_train)

# Make predictions
y_pred_bagging = bagging_forest.predict(X_test_full)

# Calculate accuracy
accuracy_bagging = accuracy_score(y_test, y_pred_bagging)

#-----------------add new instance-----------------------------------#

def predict_new_instance(instance):
 
    instance_df = pd.DataFrame([instance])

    instance_df['SeniorCitizen'] = instance_df['SeniorCitizen'].map({1:'yes', 0:'no'})

    instance_transformed = preprocessor.transform(instance_df)

    prediction = bagging_forest.predict(instance_transformed)

    return label_encoder.inverse_transform(prediction)[0]


