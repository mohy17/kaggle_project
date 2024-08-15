#------------------------------import libraries-----------------------------------#
import pandas as pd
import os
import mlflow
import argparse


#import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR


#import metric
from sklearn.metrics import r2_score

#------------------------------------------------------------------#


#------------------------------------------------------------#
#read train 
train_df=pd.read_csv(os.path.join(os.getcwd(),'train.csv'))

#change datetime[table] to type datetime in pandas to handle it


train_df['datetime'] = pd.to_datetime(train_df['datetime'])
train_df['year'] = train_df['datetime'].dt.year
train_df['month'] = train_df['datetime'].dt.month
train_df['hour'] = train_df['datetime'].dt.hour


#drop umwanted colums after extract specific feature

train_df.drop(columns=['datetime'],axis=1,inplace=True)
train_df

#---------------------------------------------------------------------#

#convert hours from ot 0 --> 23 to specific period

def custom_hours(hours):
    if 6 <= hours <12:
        return 'morning'
    elif 12 <= hours < 18:
        return 'afternoon'
    elif 18 <= hours < 23:
        return 'night'
    else :
        return 'overnight'

train_df['custom_hour']=train_df['hour'].apply(custom_hours)

#drop table after make many visulization in it to not confuss model


train_df.drop(['month','hour','temp'],axis=1,inplace=True)



#-------------------------------------------------------------------------#

#convert to numeric and categorial 


num_col = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categ_col = train_df.select_dtypes(include=['object']).columns.tolist()
ready_col=['holiday','workingday']
num_col=[num for num in num_col if num not in ready_col]
if 'count' in num_col:
    num_col.remove('count')


# Create transformers
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  # Handle missing values
    ('scaler', StandardScaler())  # Standardize numerical features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
    ('encoder', OrdinalEncoder())  # Ordinal encode categorical features
])

ready_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Handle missing values
])

# Create the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, num_col),
        ('cat', categorical_transformer, categ_col),
        ('ready',ready_transformer,ready_col)

    ])

X = train_df.drop('count', axis=1)
y = train_df['count']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)


#------------------------------------------------------------------------------#


def models(max_depth: int, n_estimators: int):
    mlflow.set_experiment('bike')
    with mlflow.start_run() as run:
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree Regression': DecisionTreeRegressor(max_depth=max_depth, random_state=42, criterion='absolute_error'),
            'Random Forest bagging': RandomForestRegressor(n_estimators=n_estimators, bootstrap=True, max_depth=max_depth, criterion='squared_error'),
            'Random Forest pasting': RandomForestRegressor(n_estimators=n_estimators, bootstrap=False, max_depth=max_depth, criterion='squared_error'),
            'Gradient Boosting Regression': GradientBoostingRegressor(n_estimators=n_estimators),
            'Support Vector Regression': SVR(kernel='rbf', C=1.0, epsilon=0.2)
        }

        # Evaluate each model
        for name, model in models.items():
            with mlflow.start_run(nested=True) as run:
                # Create and fit the pipeline
                pipeline = Pipeline(steps=[
                    ('regressor', model)
                ])
                
                # Fit the model
                pipeline.fit(X_train_transformed, y_train)
                
                # Predict and evaluate
                y_pred = pipeline.predict(X_test_transformed)
                r2 = r2_score(y_test, y_pred)
                
                # Log model and metrics in MLflow
                mlflow.log_param('model_name', name)
                mlflow.log_param('max_depth', max_depth)
                mlflow.log_param('n_estimators', n_estimators)
                mlflow.log_metric('r2_score', r2)
                
                # Log the model
                mlflow.sklearn.log_model(pipeline, "model")

                print(f"{name} R^2 Score: {r2:.3f}")




if __name__ == '__main__':
    ## Take input from user via CLI using argparser library
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_estimators', '-n', type=int, default=1)
    parser.add_argument('--max_depth', '-d', type=int, default=3)
    args = parser.parse_args()

    ## Call the main function
    models(n_estimators=args.n_estimators, max_depth=args.max_depth)


                            