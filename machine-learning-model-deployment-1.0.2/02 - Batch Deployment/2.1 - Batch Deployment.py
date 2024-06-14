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
# MAGIC # Batch Deployment
# MAGIC
# MAGIC Batch inference is the most common way of deploying machine learning models.  This lesson introduces various strategies for deploying models using batch including Spark. In addition, we will show how to enable optimizations for Delta tables.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Load a logged Model Registry model using `pyfunc`.
# MAGIC
# MAGIC * Compute predictions using `pyfunc` APIs.
# MAGIC
# MAGIC * Perform batch inference using Feature Engineering's `score_batch` method.
# MAGIC
# MAGIC * Materialize predictions into inference tables (Delta Lake).
# MAGIC
# MAGIC * Perform common write optimizations like liquid clustering, predictive optimization to maximize data skipping and on inference tables.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12**
# MAGIC
# MAGIC **🚨 Prerequisites:** 
# MAGIC * **Feature Engineering** and **Feature Store** are not focus of this lesson. This course expect that you already know these topics. If not, you can check the **Data Preparation for Machine Learning** course.
# MAGIC
# MAGIC * Model development with MLFlow is not in the scope of this course. If you need to refresh your knowledge about model tracking and logging, you can check the **Machine Learning Model Development** course.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %run ../Includes/Classroom-Setup-01

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
# MAGIC ## Data Preparation
# MAGIC
# MAGIC For this demonstration, we will utilize a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including gender, as well as internet subscription details such as subscription plans and payment methods.
# MAGIC
# MAGIC After load the dataset, we will perform simple **data cleaning and feature selection**. 
# MAGIC
# MAGIC In the final step, we will split the dataset to **features** and **response** sets.

# COMMAND ----------

from pyspark.sql.functions import col

# dataset path
dataset_p_telco = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"

# features to use
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes

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

# review the features dataset
display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Batch Deployment - Without Feature Store
# MAGIC
# MAGIC This demo will cover two main batch deployment methods. The first method is deploying models without a feature table. For the second method, we will use a feature table to train the model and later use the feature table for inference.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this demo, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# helper function that we will use for getting latest version of a model
def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC

# COMMAND ----------

# Train a sklearn Decision Tree Classification model
from sklearn.tree import DecisionTreeClassifier
from mlflow.models import infer_signature

# Covert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()
clf = DecisionTreeClassifier(max_depth=3, random_state=42)

# Use 3-level namespace for model name
model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model" 

with mlflow.start_run(run_name="Model-Batch-Deployment-Demo") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf.fit(X_train_pdf, Y_train_pdf)

    # Log model and push to registry
    signature = infer_signature(X_train_pdf, Y_train_pdf)
    mlflow.sklearn.log_model(
        clf,
        artifact_path="decision_tree",
        signature=signature,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Baseline)
    client.set_registered_model_alias(model_name, "Baseline", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Use the Model for Inference 
# MAGIC
# MAGIC Now that our model is ready in model registry, we can use it for inference. In this section we will use the model for inference directly on a spark dataframe, which called **batch inference**.

# COMMAND ----------

# MAGIC %md
# MAGIC #### Load the Model
# MAGIC
# MAGIC Loading a model from UC-based model registry is done by getting a model using **alias** and **version**. 
# MAGIC
# MAGIC After loading the model, we will create a **`spark_udf`** from the model.

# COMMAND ----------

latest_model_version = client.get_model_version_by_alias(name=model_name, alias="baseline").version
model_uri = f"models:/{model_name}/{latest_model_version}" # Should be version 1
# model_uri = f"models:/{model_name}@baseline # uri can also point to @alias
predict_func = mlflow.pyfunc.spark_udf(
    spark,
    model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Inference
# MAGIC
# MAGIC Next, we will simply use the created function for inference.

# COMMAND ----------

# prepare test dataset
test_features_df = test_df.select(primary_key, *features)

# make prediction
prediction_df = test_features_df.withColumn("prediction", predict_func(*test_features_df.drop(primary_key).columns))

display(prediction_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Deployment - With Feature Store 
# MAGIC
# MAGIC In the previous section we trained and registered a model using Spark dataframe. In some cases, you will need to use features from a feature store for training and inference. 
# MAGIC
# MAGIC In this section we will demonstrate how to train and deploy a model using Feature Store.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create Feature Table
# MAGIC
# MAGIC Let's create a feature table based on the `features_df` that we create before. Please note that we will be using **Feature Store with Unity Catalog**, which means we need to use **`FeatureEngineeringClient`**.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

# prepare feature set
features_df_all = telco_df.select(primary_key, *features)

# feature table definition
fe = FeatureEngineeringClient()
feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"

#drop table if exists
try:
    fe.drop_table(name=feature_table_name)
except:
    pass

# Create feature table
fe.create_table(
    name=feature_table_name,
    df=features_df_all,
    primary_keys=[primary_key],
    description="Example feature table"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Feature Lookups
# MAGIC
# MAGIC In order to create a training set from the feature table, we need to define a *feature lookup*. This will be used for creating training set from the feature table. 
# MAGIC
# MAGIC Note that the **`lookup_key`** is used for matching records in feature table.

# COMMAND ----------

# Create training set based on feature lookup
from databricks.feature_engineering import FeatureLookup

fl_handle = FeatureLookup(
    table_name=feature_table_name,
    lookup_key=[primary_key]
)

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
# MAGIC ### Fit and Register a Model with UC using Feature Table
# MAGIC
# MAGIC After creating the training set, **model training and registering is the same as the previous step**.

# COMMAND ----------

import warnings
from mlflow.types.utils import _infer_schema
    
    
# Covert data to pandas dataframes
X_train_pdf2 = training_df.drop(primary_key, response).toPandas()
Y_train_pdf2 = training_df.select(response).toPandas()
clf2 = DecisionTreeClassifier(max_depth=3, random_state=42)


with mlflow.start_run(run_name="Model-Batch-Deployment-Demo-With-FS") as mlflow_run:

    # Enable automatic logging of input samples, metrics, parameters, and models
    mlflow.sklearn.autolog(
        log_input_examples=True,
        log_models=False,
        log_post_training_metrics=True,
        silent=True)
    
    clf2.fit(X_train_pdf, Y_train_pdf)

    # Infer output schema
    try:
      output_schema = _infer_schema(Y_train_pdf)
    except Exception as e:
      warnings.warn(f"Could not infer model output schema: {e}")
      output_schema = None
    
    # Log using feature engineering client and push to registry
    fe.log_model(
        model=clf2,
        artifact_path="decision_tree",
        flavor=mlflow.sklearn,
        training_set=training_set_spec,
        output_schema=output_schema,
        registered_model_name=model_name
    )

    # Set model alias (i.e. Champion)
    client.set_registered_model_alias(model_name, "Champion", get_latest_model_version(model_name))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Use the Model for Inference
# MAGIC
# MAGIC Inference for models that are registered with a Feature Store table is different than inference with Spark dataframe. For inference, we will use **feature engineering client's `.score_batch()` method**. This method takes **a model URI** and **dataframe with primary key info**.
# MAGIC
# MAGIC **So how does the function know which feature table to use?** If you visit **Artifacts** section of registered model, you will see a **`data`** folder is registered with the model. Also, model file includes **`data: data/feature_store`** statement to define feature data.
# MAGIC

# COMMAND ----------

champion_model_uri = f"models:/{model_name}@champion"

# COMMAND ----------

# prepare lookup dataset
lookup_df = test_df.select("customerID")

# predict in batch using lookup df
prediction_fe_df = fe.score_batch(
    model_uri=champion_model_uri,
    df=lookup_df,
    result_type='string')

# COMMAND ----------

display(prediction_fe_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Considerations
# MAGIC
# MAGIC There are many possible (write) optimizations that Delta Lake can offer such as:
# MAGIC - Partitioning: stores data associated with different categorical values in different directories.
# MAGIC - Z-Ordering: colocates related information in the same set of files.
# MAGIC - **Liquid Clustering:** replaces both above-mentioned  methods to simplify data layout decisions and optimize query performance.
# MAGIC - **Predictive Optimizations:** removes the need to manually manage maintenance operations for Delta tables on Databricks.
# MAGIC
# MAGIC In this demo, we will show the last two options; liquid clustering and predictive optimization.

# COMMAND ----------

spark.sql(f"USE CATALOG {DA.catalog_name}")
spark.sql(f"USE SCHEMA {DA.schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC **Enable Predictive Optimization** at schema level (can also be done at catalog level)

# COMMAND ----------

spark.sql(f"ALTER SCHEMA {DA.catalog_name}.{DA.schema_name} ENABLE PREDICTIVE OPTIMIZATION;")

# COMMAND ----------

# MAGIC %md
# MAGIC Create inference table (where batch scoring jobs would materialized) and enable liquid clustering on using `CLUSTER BY`

# COMMAND ----------

# MAGIC %sql
# MAGIC CREATE OR REPLACE TABLE batch_inference(
# MAGIC   customerID STRING
# MAGIC  ,Churn STRING
# MAGIC  ,SeniorCitizen DOUBLE
# MAGIC  ,tenure DOUBLE
# MAGIC  ,MonthlyCharges DOUBLE
# MAGIC  ,TotalCharges DOUBLE
# MAGIC  ,prediction STRING)
# MAGIC CLUSTER BY (customerID, tenure)

# COMMAND ----------

(
  prediction_fe_df.write
  .mode("append")
  .option("mergeSchema", True)
  .saveAsTable(f"{DA.catalog_name}.{DA.schema_name}.batch_inference")
)

# COMMAND ----------

# MAGIC %md
# MAGIC Manually optimize table

# COMMAND ----------

# MAGIC %sql
# MAGIC ANALYZE TABLE batch_inference COMPUTE STATISTICS FOR ALL COLUMNS;
# MAGIC OPTIMIZE batch_inference

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
# MAGIC In this demo, we presented two main batch deployment methods using MLflow for model tracking and logging with Unity Catalog. In the first approach, we trained and registered a model without a feature table, reloading it for inference through a Spark UDF. The second method involved training a model with a feature table, registering it in the model registry, and using a look-up key for data retrieval during inference.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>