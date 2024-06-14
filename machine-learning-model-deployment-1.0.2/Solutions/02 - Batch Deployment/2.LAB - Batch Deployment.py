# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC <div style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://databricks.com/wp-content/uploads/2018/03/db-academy-rgb-1200px.png" alt="Databricks Learning">
# MAGIC </div>
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC # LAB - Batch Deployment
# MAGIC
# MAGIC Welcome to the "Batch Deployment" lab! This lab focuses on batch deployment of machine learning models using Databricks. you will engage in tasks related to model inference, model registry, and explore performance results for feature such as Liquid Clustering using `CLUSTER BY`.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC By the end of this lab, you will be able to;
# MAGIC
# MAGIC + **Task 1: Load Dataset**
# MAGIC     + Load Dataset
# MAGIC     + Split the dataset to features and response sets
# MAGIC
# MAGIC + **Task 2: Inference with feature table**
# MAGIC
# MAGIC     + Create Feature Table
# MAGIC     + Setup Feature Lookups
# MAGIC     + Fit and Register a Model with UC using Feature Table
# MAGIC     + Perform batch inference using Feature Engineering's  **`score_batch`** method.
# MAGIC
# MAGIC + **Task 3: Assess Liquid Clustering:**
# MAGIC
# MAGIC     + Evaluate the performance results for specific optimization techniques:
# MAGIC         + Liquid Clustering
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12**

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-02Lab

# COMMAND ----------

# MAGIC %md
# MAGIC **Other Conventions:**
# MAGIC
# MAGIC Throughout this demo, we'll refer to the object `DA`. This object, provided by Databricks Academy, contains variables such as your username, catalog name, schema name, working directory, and dataset locations. Run the code block below to view these details:

# COMMAND ----------

print(f"Username:          {DA.username}")
print(f"Catalog Name:      {DA.catalog_name}")
print(f"Schema Name:       {DA.schema_name}")
print(f"Working Directory: {DA.paths.working_dir}")
print(f"User DB Location:  {DA.paths.datasets}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 1: Load Dataset
# MAGIC
# MAGIC + Load a dataset:
# MAGIC   + Define the dataset path
# MAGIC   + Define the primary key (`customerID`), response variable (`Churn`), and feature variables (`SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`) for further processing.
# MAGIC   + Read the dataset, casting relevant columns to the correct data types, and drop any rows with missing values
# MAGIC + Split the dataset into training and testing sets
# MAGIC   + Separate the features and the response for the training set
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col

# dataset path
dataset_p_telco = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"

# features to use
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]

# Read dataset (and drop nan)
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", col("TotalCharges").cast('double'))\
            .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
            .withColumn("Tenure", col("tenure").cast('double'))\
            .na.drop(how='any')

# Split with 80 percent of the data in train_df and 20 percent of the data in test_df
train_df, test_df = telco_df.randomSplit([.8, .2], seed=42)

# Separate features and ground-truth
features_df = train_df.select(primary_key, *features)
response_df = train_df.select(primary_key, response)

# COMMAND ----------

# MAGIC %md
# MAGIC ##Task 2: Inference with feature table
# MAGIC In this task, you will perform batch inference using a feature table. Follow the steps below:
# MAGIC
# MAGIC + **Step 1: Create Feature Table**
# MAGIC
# MAGIC + **Step 2: Setup Feature Lookups**
# MAGIC
# MAGIC + **Step 3: Fit and Register a Model with UC using Feature Table**
# MAGIC
# MAGIC + **Step 4: Use the Model for Inference**
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 1: Create Feature Table**
# MAGIC   + Begin by creating a feature table that incorporates the relevant features for inference. This involves selecting the appropriate columns, performing any necessary transformations, and storing the resulting data in a feature table.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# Prepare feature set
features_df_all = telco_df.select(primary_key, *features)

# Feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"

# Drop table if exists
try:
    fe.drop_table(name=feature_table_name)
except:
    pass

# Create feature table
fe.create_table(
    name=feature_table_name,
    df=features_df_all,
    primary_keys=[primary_key],
    description="Lab feature table"
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 2: Setup Feature Lookups**
# MAGIC   + Set up a feature lookup to create a training set from the feature table. 
# MAGIC   + Specify the `lookup_key` based on the columns that uniquely identify records in your feature table.

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup

fl_handle = FeatureLookup(
    table_name=feature_table_name,
    lookup_key=[primary_key]
)

#  Create a training set based on feature lookup
training_set_spec = fe.create_training_set(
    df=response_df,
    label=response,
    feature_lookups=[fl_handle],
    exclude_columns=[primary_key]
)

# Load training dataframe based on defined feature-lookup specification
training_df = training_set_spec.load_df()

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 3: Fit and Register a Model with UC using Feature Table**
# MAGIC   + Fit and register a Machine Learning Model using the created training set.
# MAGIC   + Train a model on the training set and register it in the model registry.

# COMMAND ----------

import mlflow
import warnings
from mlflow.types.utils import _infer_schema

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# Train a sklearn Decision Tree Classification model
from sklearn.tree import DecisionTreeClassifier

# Covert data to pandas dataframes
X_train_pdf = training_df.drop(primary_key, response).toPandas()
Y_train_pdf = training_df.select(response).toPandas()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# End the active MLflow run before starting a new one
mlflow.end_run()

with mlflow.start_run(run_name="Model-Batch-Deployment-lab-With-FS") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(X_train_pdf, Y_train_pdf)

    # Infer output schema
    try:
        output_schema = _infer_schema(Y_train_pdf)
    except Exception as e:
        warnings.warn(f"Could not infer model output schema: {e}")
        output_schema = None

    model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model"
    
    # Log using feature engineering client and push to registry
    fe.log_model(
        model=clf,
        artifact_path="decision_tree",
        flavor=mlflow.sklearn,
        training_set=training_set_spec,
        output_schema=output_schema,
        registered_model_name= model_name
    )

    # Set model alias (i.e. Champion)
    client.set_registered_model_alias(model_name, "Champion", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC **Step 4: Use the Model for Inference**
# MAGIC   + Utilize the feature engineering client's `score_batch()` method for inference.
# MAGIC   + Provide the model URI and a dataframe containing primary key information for the inference.

# COMMAND ----------

# Load the model
model_uri = f"models:/{model_name}@champion"
# Define the test dataset
test_features_df = test_df.select("customerID")

# Perform batch inference using Feature Engineering's score_batch method
result_df = fe.score_batch(
    model_uri=model_uri,
    df=test_features_df,
    result_type='string'  # Update with the desired result type
)

# Display the inference results
display(result_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Task 3: Assess Liquid Clustering:
# MAGIC
# MAGIC Evaluate the performance results for specific optimization techniques, such as: Liquid Clustering Follow the step-wise instructions below:  
# MAGIC + **Step 1:** Create `batch_inference_liquid_clustering` table and import the following columns: `customerID`, `Churn`, `SeniorCitizen`, `tenure`, `MonthlyCharges`, `TotalCharges`, and `prediction`.
# MAGIC + **Step 2:**  Begin by assessing Liquid Clustering, an optimization technique for improving performance by physically organizing data based on a specified clustering column.
# MAGIC + **Step 3:**  Optimize the target table for Liquid Clustering.
# MAGIC + **Step 4:** Specify the `CLUSTER BY` clause with the desired columns (e.g., (customerID, tenure)) to enable Liquid Clustering on the table.

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE batch_inference_liquid_clustering(
# MAGIC   customerID STRING,
# MAGIC   Churn STRING,
# MAGIC   SeniorCitizen DOUBLE,
# MAGIC   tenure DOUBLE,
# MAGIC   MonthlyCharges DOUBLE,
# MAGIC   TotalCharges DOUBLE,
# MAGIC   prediction STRING
# MAGIC   )

# COMMAND ----------

# MAGIC %sql
# MAGIC OPTIMIZE batch_inference_liquid_clustering;
# MAGIC ALTER TABLE batch_inference_liquid_clustering
# MAGIC CLUSTER BY (customerID, tenure);

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Clean up Classroom
# MAGIC
# MAGIC Run the following cell to remove lessons-specific assets created during this lesson.

# COMMAND ----------

DA.cleanup()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Conclusion
# MAGIC
# MAGIC This lab provides you with hands-on experience in batch deployment, covering model inference, Model Registry usage, and the impact of features like Liquid Clustering on performance. you will gain practical insights into deploying models at scale in a batch-oriented environment.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>