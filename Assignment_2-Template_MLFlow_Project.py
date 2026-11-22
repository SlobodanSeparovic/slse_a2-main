########################################################################################################################
# IMPORTS
# You absolutely need these
from influxdb import InfluxDBClient
import mlflow

# You will probably need these
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

# This are for example purposes. You may discard them if you don't use them.
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error

### TODO -> HERE YOU CAN ADD ANY OTHER LIBRARIES YOU MAY NEED ###
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from mlflow.models import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec
import mlflow.sklearn
from mlflow.tracking import MlflowClient


########################################################################################################################

## Step 1: The Data

### Getting the data with InfluxDB

"""
The data is stored in an InfluxDB (https://www.influxdata.com/), which is a non-relational time-series database. InfluxDB can be queried using InfluxQL
(https://docs.influxdata.com/influxdb/v1.8/query_language/spec/), a "SQL-like" query language for time-series data. InfluxDB does not have tables with rows and columns,
instead data is stored in measurements with fields and tags.

NOTE: You don't need to know much about InfluxDB syntax, but if you are interested, feel free to browse around the documentation (https://docs.influxdata.com/).
The data for this assignment is stored in a database, with one table for the weather data and another for the power generation data. To do this, we first need to
create an instance of the InfluxDB Client, that will allow us to query the needed data. Let's see how this is done.

"""

# Set the needed parameters to connect to the database
### THIS SHOULD NOT BE CHANGED ###
settings = {
    'host': 'influxus.itu.dk',
    'port': 8086,
    'username': 'lsda',
    'password': 'icanonlyread'
    }

# Create an InfluxDB Client instance and select the orkney database
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
client = InfluxDBClient(host=settings['host'], port=settings['port'], username=settings['username'], password=settings['password'])
client.switch_database('orkney')

## Function to tranform the InfluxDB resulting set into a Dataframe
### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
def set_to_dataframe(resulting_set):
    
    values = resulting_set.raw["series"][0]["values"]
    columns = resulting_set.raw["series"][0]["columns"]
    df = pd.DataFrame(values, columns=columns).set_index("time")
    df.index = pd.to_datetime(df.index) # Convert to datetime-index

    return df

# Retrieve the data
days = 90 # -> You can change this to get any other range of days

### YOU DON'T NEED TO CHANGE ANYTHING HERE ###
power_set = client.query(
    "SELECT * FROM Generation where time > now()-"+str(days)+"d"
    ) # Query written in InfluxQL. We are retrieving all generation data from 90 days back.

# Get the last 90 days of weather forecasts with the shortest lead time
wind_set  = client.query(
    "SELECT * FROM MetForecasts where time > now()-"+str(days)+"d and time <= now() and Lead_hours = '1'"
    ) # Query written in InfluxQL. We are retrieving all weather forecast data from 90 days back and with 1 lead hour.

power_df = set_to_dataframe(power_set)
wind_df = set_to_dataframe(wind_set)

### TODO -> JOIN THE TWO DATASETS ###
# Keep all the power_df measures (target), and only keep from wind_df (features) those that match the timestamp of power_df.
# Thus ensuring that the target variable is available for every row in the features. I will transform features in the pipeline later on.
joined_dfs = power_df.join(wind_df, how="left")

# Splitting the joined_dfs into X that holds all possible features and y that holds only the target.
# As you may see later in the model's signature I defined, not all columns from X are needed for the model to work, 
# but I defined there which are needed.
X = joined_dfs.drop(["Total"], axis=1)
y = joined_dfs["Total"]


# TODO: Handle missing data
# Preprocessing of features to use for training the model and prediction.
# Among one of many different tries, I thought to include handling of the Total (target variable) outliers
# by merging X and y, rounding the Speed to 1 decimal and grouping by rounded speed then removing Total outliers
# using IQR. I also tried grouping by speed and categorical wind direction and then remove the outliers in Total,
# but this did not have noticeable effect when visually inspecting the data.
# Ultimately, I decided not to do it as this may have introduced bias, i.e. model may lose the oportunity to 
# learn some paterns that result in lower Total production. Also, as I incorporate the DataPreprocessor in the pipeline
# as the first step, and as the next step I use StandardScaler, that would also scale the target variable, 
# and this is not something I intend to do.
# in my opinion, the correct approach to handling outliers in the target variable would be to make aditional custom transformer
# that would transform the target data after the proprocessing of the features and scaling them.

class DataPreprocessor(BaseEstimator, TransformerMixin):
    
    #Constructor
    def __init__(self):
        pass
    
    # Making fit method to meet the requirements of fit and transform
    def fit(self, X, y=None):
        return self
    
    # Applying all preprocessing steps sequentially
    def transform(self, X, y=None):
        X = self.drop_not_needed_at_start(X)
        X = self.interpolate_missing_speed(X)
        X = self.fill_speed_nan(X)
        X = self.convert_direction_to_linear_components(X)
        X = self.interpolate_missing_direction_sin_cos(X)
        X = self.fill_sin_cos_nan(X)
        X = self.drop_not_needed_at_the_end(X)
        return X
    
    # Dropping not needed attributes at start from the dataframe
    def drop_not_needed_at_start(self, X, y=None):
        # Columns to drop
        columns_to_drop = ['Lead_hours', 'Source_time', 'ANM', 'Non-ANM']
        # Drop the columns if they exist
        X=X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors='ignore')              
        return X

    # Assuming the linear change of wind speed, I am interpolating the values in between the available entries.
    # This is clearly a weakness of the model, as it does not take into consideration 
    # wind gusts (sudden change of the wind speed and direction), but I needed to accept it
    # to build a model from what I have available
    def interpolate_missing_speed(self, X, y=None):
        X['Speed'] = X['Speed'].interpolate(method='linear')
        return X
    
    # Filling the rest of the NaN values in Speed column with 0 if all are NaN, else with median.
    # Not ideal, but deals with whole possible NaN columns in early partitions/folds of TimeSeriesSplits
    # and keeps the features df at the same lenght as the target df
    def fill_speed_nan(self, X, y=None):
        if X['Speed'].isna().all():
            X['Speed'] = X['Speed'].fillna(0)
        else:
            X['Speed'] = X['Speed'].fillna(X['Speed'].median())
        return X    
       
    # Transforming Direction to linear components
    def convert_direction_to_linear_components(self, X, y=None):
        # Mapping wind directions to angles in degrees
        direction_map = {
            'NNE': 22.5, 'NE': 45, 'ENE': 67.5, 'E': 90,
            'ESE': 112.5, 'SE': 135, 'SSE': 157.5, 'S': 180,
            'SSW': 202.5, 'SW': 225, 'WSW': 247.5, 'W': 270,
            'WNW': 292.5, 'NW': 315, 'NNW': 337.5, 'N': 0
        }

        # Stripping leading and trailing spaces and standardise everything to upper case
        X['Direction'] = X['Direction'].str.strip().str.upper()
        
        # Converting directions to angles as this is needed due to the sin and cos implementations.
        X['Direction_Angle'] = X['Direction'].map(direction_map)

        # Converting angles to radians
        X['Direction_rad'] = np.radians(X['Direction_Angle'])

        # Calculating sine and cosine
        X['Direction_sin'] = np.sin(X['Direction_rad'])
        X['Direction_cos'] = np.cos(X['Direction_rad'])
        return X

    # Interpolating missing values in sine and cosine using linear interpolation as now the information 
    # on degrees is captured in two linear component
    def interpolate_missing_direction_sin_cos(self, X, y=None): 
        X['Direction_sin'] = X['Direction_sin'].interpolate(method='linear')
        X['Direction_cos'] = X['Direction_cos'].interpolate(method='linear')
        return X
    
    
    # Filling the rest of the NaN values in Direction_sin and Direction_cos columns with 0 if all are NaN, else with median.
    # Not ideal, but deals with whole possible NaN columns in early TimeSeriesSplits
    def fill_sin_cos_nan(self, X, y=None):
        if X['Direction_sin'].isna().all():  
            X['Direction_sin'] = X['Direction_sin'].fillna(0)
        else:
            X['Direction_sin'] = X['Direction_sin'].fillna(X['Direction_sin'].median())

        if X['Direction_cos'].isna().all(): 
            X['Direction_cos'] = X['Direction_cos'].fillna(0)
        else:
            X['Direction_cos'] = X['Direction_cos'].fillna(X['Direction_cos'].median())
        return X
        
    # Cleaning the features dataframe and returning only the features we need for use in the pipeline
    def drop_not_needed_at_the_end(self, X, y=None):        
        columns_to_drop = ['Direction_Angle', 'Direction_rad', 'Direction']
        # Drop the columns if they exist
        X=X.drop(columns=[col for col in columns_to_drop if col in X.columns], errors='ignore')
        return X




# Do not use below line once you have created a MLproject, since it will probably throw an error. This is because below line tries to create a new experiment.
# mlflow.set_experiment("<ITU Username> - <Descriptive experiment name>")
# When you have an MLproject, you should define name using the command in a terminal:
# mlflow run <folder> --experiment-name <experiment_name>

### TODO -> SET YOUR OWN EXPERIMENT SETUP ###
# Defining the experiment and enabling automatic logging for sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("slse - Completely_automated_experiment - Metric_average_r2_score")

# Defining model hyperparameters grid for tuning
param_grids = {
    "HistGradientBoostingRegressor": {
    'model__n_iter_no_change': [150, 300],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 6, 10],
    'model__min_samples_leaf': [10, 20],
    'model__max_leaf_nodes': [31, 50]
},
    "RandomForestRegressor": {
        'model__n_estimators': [150, 300],
        'model__max_depth': [10, 20],
        'model__min_samples_split': [2, 5],
        'model__min_samples_leaf': [1, 2], 
        'model__max_features': ['auto', 'sqrt']
    },
    "KNeighborsRegressor": {
        'model__n_neighbors': [3, 5, 7],
        'model__weights': ['uniform', 'distance'],
        'model__metric': ['euclidean', 'manhattan'],
        'model__leaf_size': [20, 30]
    },    
    "XGBoostRegressor": {
        'model__n_estimators': [150, 300],
        'model__learning_rate': [0.01, 0.05],
        'model__max_depth': [3, 6],        
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0],
        'model__gamma': [0, 1]           
    }
}

# Defining models. For the efficiency, I decided to exclude MLPRegressor (Multi-Layer Perceptron Regressor) from the original
# experiment. Its training lasted almost for 45 minutes, but it was not more efficient than other regressors, and they are much faster.
# I decided to exclude LinearRegression as we obviously deal with power-curve data here, i.e. linear regression does not make sense.
models = {
    "HistGradientBoostingRegressor": HistGradientBoostingRegressor(random_state=42),
    "RandomForestRegressor": RandomForestRegressor(random_state=42),
    "KNeighborsRegressor": KNeighborsRegressor(), # No need for random state as it does not rely on randomness
    "XGBoostRegressor": XGBRegressor(random_state=42)
}

# Creating variables to store the overall best model
global_best_model_name = None
global_best_avg_r2_score = -np.inf
global_best_model = None
global_best_hyperparams = None
global_best_run_id = None
   
# Defining a function to perform hyperparameter tuning and log results
def run_experiment_with_timeseries_split(model_name, model, X, y):             
       
    # Declaring the variables as global inside the function
    global global_best_model_name, global_best_avg_r2_score, global_best_model, global_best_hyperparams, global_best_run_id, tr_i, te_i
    
    # Creating a pipeline with DataPreprocessor
    pipe = Pipeline([
        ('preprocessor', DataPreprocessor()),
        ('scaler', StandardScaler()),
        ('model', model)
    ])
           
    # Initialising TimeSeriesSplit for outer loop
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initializing TimeSeriesSplit for inner loop (3 folds)
    inner_tscv = TimeSeriesSplit(n_splits=3)
    
    # Starting MLflow run
    with mlflow.start_run(run_name=model_name):                     
                
        # Initiating best hyperparameters to track them across all partitions/folds for each model
        best_hyperparams = None
        
        # Tracking the best_r2 score for the model as it looks as how well the model fits the data.
        # It informs about how much of variance in power generation the model can explain
        # The higher the score, the better the model     
        best_r2_score = -np.inf                     
        
        # Initialising variables to accumulate the performance metrics for all partitions        
        all_mae = []
        all_mape = []
        all_r2 = []
        all_rmse = []
        all_mse = []                
                
        # For each train-test split in TimeSeriesSplit
        fold = 1
        for train_index, test_index in tscv.split(X):
            # Splitting the data into train and test based on the time-series split
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            
            # If in the fold 5 of the last model in the models, save the indexes to split data for the final model training
            if fold == 5 and model_name == list(models.keys())[-1]:
                tr_i, te_i = train_index, test_index
                                               
            # Starting a new nested run for each partition
            with mlflow.start_run(nested=True, run_name=f"fold_{fold}"):                                                                            
                                
                # Performing hyperparameter tuning for this specific fold/partition
                search = GridSearchCV(pipe, param_grid=param_grids.get(model_name, {}), cv=inner_tscv, n_jobs=-1)
                               
                # Fit the GridSearchCV with the training data (which internally fits the pipeline)
                search.fit(X_train, y_train)
                
                # Extract the best hyperparameters for this fold
                best_fold_hyperparams = search.best_params_
                
                # Remove the 'model__' prefix from the best_hyperparams 
                cleaned_fold_hyperparams = {key.replace('model__', ''): value for key, value in best_fold_hyperparams.items()}
                                
                # Logging the best hyperparameters for this fold
                mlflow.log_param(f"{model_name}_best_hyperparameters_in_fold_{fold}", cleaned_fold_hyperparams)
                
                # Using the best model for predictions
                best_model_fold = search.best_estimator_
                
                # Making predictions
                predictions = best_model_fold.predict(X_test)
                
                # Calculating metrics
                mae = mean_absolute_error(y_test, predictions)
                mape = mean_absolute_percentage_error(y_test, predictions)
                mse = mean_squared_error(y_test, predictions)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, predictions) 
                              
                # Logging metrics for the fold
                mlflow.log_param("Fold", fold)
                mlflow.log_metric("MAE", mae)
                mlflow.log_metric("MAPE", mape)
                mlflow.log_metric("R2", r2)
                mlflow.log_metric("RMSE", rmse)
                mlflow.log_metric("MSE", mse)                
                                               
                # Append metrics to accumulate for assessment of model's performance across all folds
                all_mae.append(mae)
                all_mape.append(mape)
                all_mse.append(mse)
                all_rmse.append(rmse)
                all_r2.append(r2)
                
                # Update the overall best hyperparameters and r2 score if r2 in this fold is better than the best_r2_score
                if r2 > best_r2_score:
                    best_r2_score = r2
                    best_hyperparams = cleaned_fold_hyperparams
                                               
                fold += 1        
                      
        # Logging the best hyperparameters once for each MLflow outer run/regressor model
        mlflow.log_param(f"Best_hyperparameters_for_{model_name}_found_by_GridSearchCV", best_hyperparams)
                       
        # Logging the overall average performance metrics after all folds        
        avg_mae = np.mean(all_mae)
        avg_mape = np.mean(all_mape)
        avg_mse = np.mean(all_mse)
        avg_rmse = np.mean(all_rmse)
        avg_r2 = np.mean(all_r2)

        # Logging average metrics after all folds
        mlflow.log_metric("MAE_Avg_All_Folds", avg_mae)
        mlflow.log_metric("MAPE_Avg_All_Folds", avg_mape)
        mlflow.log_metric("R2_Avg_All_Folds", avg_r2)
        mlflow.log_metric("RMSE_Avg_All_Folds", avg_rmse)
        mlflow.log_metric("MSE_Avg_All_Folds", avg_mse)
        
        # Checking if the current model's avg_r2 score is better than the global best
        # If yes, update it to be the best global model
        if avg_r2 > global_best_avg_r2_score:
            global_best_model_name = model_name
            global_best_avg_r2_score = avg_r2
            global_best_model = model
            global_best_hyperparams = best_hyperparams
            global_best_run_id = mlflow.active_run().info.run_id


# Running the experiment for all models
for model_name, model in models.items():
    run_experiment_with_timeseries_split(model_name, model, X, y)
    
    # After the experiment, retraining the model on the whole dataset with best hyperparameters.
    # Then checking for the best global model. If not there, registering the first or updating the registered version if avg_r2
    # is better than those of the registered version.   
    if model_name == list(models.keys())[-1]:
        with mlflow.start_run(run_id=global_best_run_id):           
                 
            # Setting the best hyperparameters for the global_best_model
            global_best_model.set_params(**global_best_hyperparams)            
            
            # Creating the pipeline with the global_best_model as the last step           
            pipeline = Pipeline([
                ('preprocessor', DataPreprocessor()),
                ('scaler', StandardScaler()),
                ('model', global_best_model)
            ])            
                       
            # Splitting X and y for the training of the final model using indexes from the last fold of TimeSeriesSplit
            X_train, X_test = X.iloc[tr_i], X.iloc[te_i]
            y_train, y_test = y.iloc[tr_i], y.iloc[te_i]         

            # Retrain the model using the pipeline
            pipeline.fit(X_train, y_train)
            
            # Predict
            actual_predictions = pipeline.predict(X_test)
            
            # Calculating metrics that will be reported as the objective performance metrics
            reported_mae = mean_absolute_error(y_test, actual_predictions)
            reported_mape = mean_absolute_percentage_error(y_test, actual_predictions)
            reported_mse = mean_squared_error(y_test, actual_predictions)
            reported_rmse = np.sqrt(reported_mse)
            reported_r2 = r2_score(y_test, actual_predictions)          
                
            # Logging final model's performance (with best hyperparameters) on the whole dataset
            # (using indexes of the last TimeSeriesSplit) and logging the pipeline
            mlflow.log_metric("Best_Model_MAE_Whole_Dataset", reported_mae)
            mlflow.log_metric("Best_Model_MAPE_Whole_Dataset", reported_mape)
            mlflow.log_metric("Best_Model_MSE_Whole_Dataset", reported_mse)
            mlflow.log_metric("Best_Model_RMSE_Whole_Dataset", reported_rmse)
            mlflow.log_metric("Best_Model_R2_Whole_Dataset", reported_r2)
            for i, (step_name, step_object) in enumerate(pipeline.steps, start=1):
                mlflow.log_param(f"Best_Model_Pipeline_Step_{i}_{step_name}", str(step_object)) 
                
            # Initialize the MLflow client
            mfclient = MlflowClient()
            
            # Defining the model registry name
            best_global_run_name = "Best_Registered_Model"       

            # Defining filter string to use with MLflow client
            filter_string = "name='Best_Registered_Model'"

            # Checking if the model exists in the registry
            existing_model_versions = mfclient.search_model_versions(filter_string)
            existing_versions = [version.version for version in existing_model_versions]
            
            # Defining input and output schemas manually as the model otherwise registers wrong data types in the input_schema        
            input_schema = Schema([ColSpec(DataType.string, 'Direction'), ColSpec(DataType.double, 'Speed')])
        
            # Defining output schema 
            output_schema = Schema([TensorSpec(np.dtype(np.float64), shape=[-1], name='Power_generation_prediction')])
                        
            # Creating a model signature with the input and output schemas
            signature = ModelSignature(inputs=input_schema, outputs=output_schema)
            
            # Registering the first version if there are no versions registered
            if not existing_versions:
                print(f"Registering the first version of the model: run_id={global_best_run_id}")     
                mlflow.sklearn.log_model(global_best_model, artifact_path="model", registered_model_name=best_global_run_name, signature=signature)
                

            else:
                # Fetching the latest registered model version
                latest_version = max(existing_model_versions, key=lambda x: x.version)                
                                                
                # Extract the run_id from the latest model version
                latest_run_id = latest_version.run_id

                # Fetching the avg_r2 metric for the latest registered model
                metrics = mfclient.get_metric_history(latest_run_id, "R2_Avg_All_Folds")
                
                # Assuming the metric list is not empty and retrieving the first metric value
                registered_model_avg_r2_score = metrics[0].value

                # Compare avg_r2 of the current best_global_model with those of the registered model and update if better.
                if global_best_avg_r2_score > registered_model_avg_r2_score:
                    print(f"New model has higher average R2 score. "
                        f"Registering new model: run_id={global_best_run_id}, "
                        f"New_global_best_avg_r2_score={global_best_avg_r2_score}, "
                        f"Old_registered_model_r2_score={registered_model_avg_r2_score}")   
                    mlflow.sklearn.log_model(global_best_model, artifact_path="model", registered_model_name=best_global_run_name, signature=signature)
                
                else:
                    print("New model does not have higher average R2 score. The best model from current experiment will not be registered.")               