
import numpy as np
import pandas as pd
import os


#for prepare data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#models
from sklearn.ensemble import RandomForestClassifier

#metric
from sklearn.metrics import  accuracy_score


#----------------------------------------------------------#

#read file 

df =pd.read_csv(os.path.join(os.getcwd(),'train.csv'))


#--------------------------------------------------------_#


#extract fearure and handle data 


df.drop(['PassengerId','Name','Cabin','Ticket'],axis=1,inplace=True)



df['Age']=df['Age'].fillna(df['Age'].mean())
df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode().iloc[0])


e=df['SibSp']+df['Parch']



def family_size(size):
    if size == 0:
        return 0
    elif size <= 3:
        return 1
    elif size <= 6:
        return 2
    else:
        return 3

df['family_size']=df['SibSp']+df['Parch']


df['family_size']=df['family_size'].apply(family_size)


df['Fare_new'] = np.log(df['Fare'] + 1)


df['Age']=df['Age'].round(1)


df.drop(['SibSp', 'Parch','Fare'],inplace=True,axis=1)


#------------------------------------------------------------#


#prepare for model 

num_cols = df.drop(columns=['Survived']).select_dtypes(include=['number']).columns
categ_cols=df.select_dtypes(exclude=['number']).columns

#split data
y = df['Survived']
X=df.drop(['Survived'],axis=1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),  # Scale numerical features
        ('cat', OrdinalEncoder(), categ_cols)   # One-hot encode categorical features
    ]
)

# Create the pipeline with preprocessing only
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Apply the pipeline to transform the training data
X_train_full = pipeline.fit_transform(X_train)

# Optionally, apply the same transformation to the test data
X_test_full = pipeline.transform(X_test)


bagging_model = RandomForestClassifier(bootstrap=False, max_depth=8, n_estimators=100, random_state=44)
bagging_model.fit(X_train_full, y_train)

# Evaluate the model
y_pred = bagging_model.predict(X_test_full)
accuracy = accuracy_score(y_test, y_pred)
print(f"bagging Random Forest Accuracy: {accuracy:.4f}")


#-----------------------------------------------------------------------#


#new instance

def x_new(x):
    x = pd.DataFrame([x])
    
    x['Fare_new'] = np.log(x['Fare'] + 1)
    x['family_size'] = x['SibSp'] + x['Parch']
    x['family_size']=x['family_size'].apply(family_size)
    x.drop(['SibSp', 'Parch', 'Fare'], inplace=True, axis=1)
    
    # Use the preprocessing pipeline
    x_preprocessed = pipeline.transform(x)
    
    # Predict using the trained model
    prediction = bagging_model.predict(x_preprocessed)
    
    
    if prediction==0:
        
        return 'Not Survive'
    else:
        
        return 'survive'
    
