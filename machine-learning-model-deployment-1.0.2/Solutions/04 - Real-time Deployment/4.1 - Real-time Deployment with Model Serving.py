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
# MAGIC # Real-time Deployment with Model Serving
# MAGIC
# MAGIC In this demo, we will focus on real-time deployment of machine learning models. ML models can be deployed using various technologies. Databricks' Model Serving is an easy to use serverless infrastructure for serving the models in real-time.
# MAGIC
# MAGIC First, we will fit a model **without using a feature store**. Then, we will serve the model via Model Serving. Model serving **supports both the API and the UI**. To demonstrate both methods, we will, first, serve the model using the UI and then server the model using **Databricks' Python SDK**.
# MAGIC
# MAGIC In the second section, we will fit a model **with feature store and we will use online features during the inference.** For online features, **Databricks' Online Tables** can be used.
# MAGIC
# MAGIC **Learning Objectives:**
# MAGIC
# MAGIC *By the end of this demo, you will be able to;*
# MAGIC
# MAGIC * Implement a real-time deployment REST API using Model Serving.
# MAGIC
# MAGIC * Serve multiple model versions to a Serverless Model Serving endpoint.
# MAGIC
# MAGIC * Set up an A/B testing endpoint by splitting the traffic.
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## Requirements
# MAGIC
# MAGIC Please review the following requirements before starting the lesson:
# MAGIC
# MAGIC * To run this notebook, you need to use one of the following Databricks runtime(s): **13.3.x-cpu-ml-scala2.12**
# MAGIC
# MAGIC * Online Tables must be enabled for the workspace.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Classroom Setup
# MAGIC
# MAGIC Before starting the demo, run the provided classroom setup script. This script will define configuration variables necessary for the demo. Execute the following cell:

# COMMAND ----------

# MAGIC %pip install databricks-sdk --upgrade
# MAGIC
# MAGIC dbutils.library.restartPython()

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
# MAGIC For this demonstration, we will use a fictional dataset from a Telecom Company, which includes customer information. This dataset encompasses **customer demographics**, including internet subscription details such as subscription plans, monthly charges and payment methods.
# MAGIC
# MAGIC After load the dataset, we will perform simple **data cleaning and feature selection**. 
# MAGIC
# MAGIC In the final step, we will split the dataset to **features** and **response** sets.

# COMMAND ----------

from pyspark.sql.functions import col

# dataset path
dataset_p_telco = f"{DA.paths.datasets}/telco/telco-customer-churn.csv"

# Dataset specs
primary_key = "customerID"
response = "Churn"
features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"] # Keeping numerical only for simplicity and demo purposes


# Read dataset (and drop nan)
# Convert all fields to double for spark compatibility
telco_df = spark.read.csv(dataset_p_telco, inferSchema=True, header=True, multiLine=True, escape='"')\
            .withColumn("TotalCharges", col("TotalCharges").cast('double'))\
            .withColumn("SeniorCitizen", col("SeniorCitizen").cast('double'))\
            .withColumn("Tenure", col("tenure").cast('double'))\
            .na.drop(how='any')

# Separate features and ground-truth
features_df = telco_df.select(primary_key, *features)
response_df = telco_df.select(primary_key, response)

# Covert data to pandas dataframes
X_train_pdf = features_df.drop(primary_key).toPandas()
Y_train_pdf = response_df.drop(primary_key).toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Fit and Register Models
# MAGIC
# MAGIC Before we start model deployment process, we will **fit and register two models**. These models are called **"Champion"** and **"Challenger"** and they will be served later on using Databricks Model Serving.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Setup Model Registry with UC
# MAGIC
# MAGIC Before we start model deployment, we need to fit and register a model. In this demo, **we will log models to Unity Catalog**, which means first we need to setup the **MLflow Model Registry URI**.

# COMMAND ----------

import mlflow

# Point to UC model registry
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Register a Model with UC

# COMMAND ----------

import time
import warnings
from mlflow.types.utils import _infer_schema
from mlflow.models import infer_signature
from sklearn.tree import DecisionTreeClassifier
from databricks.feature_engineering import FeatureEngineeringClient

model_name = f"{DA.catalog_name}.{DA.schema_name}.ml_model" # Use 3-level namespace

def get_latest_model_version(model_name):
    """Helper function to get latest model version"""
    model_version_infos = client.search_model_versions("name = '%s'" % model_name)
    return max([model_version_info.version for model_version_info in model_version_infos])

def fit_and_register_model(X, Y, model_name_=model_name, random_state_=42, model_alias=None, log_with_fs=False, training_set_spec_=None):
    """Helper function to train and register a decision tree model"""

    clf = DecisionTreeClassifier(random_state=random_state_)
    with mlflow.start_run(run_name="Demo4_1-Real-Time-Deployment") as mlflow_run:

        # Enable automatic logging of input samples, metrics, parameters, and models
        mlflow.sklearn.autolog(
            log_input_examples=True,
            log_models=False,
            log_post_training_metrics=True,
            silent=True)
        
        clf.fit(X, Y)

        # Log model and push to registry
        if log_with_fs:
            # Infer output schema
            try:
                output_schema = _infer_schema(Y)
            except Exception as e:
                warnings.warn(f"Could not infer model output schema: {e}")
                output_schema = None
            
            # Log using feature engineering client and push to registry
            fe = FeatureEngineeringClient()
            fe.log_model(
                model=clf,
                artifact_path="decision_tree",
                flavor=mlflow.sklearn,
                training_set=training_set_spec_,
                output_schema=output_schema,
                registered_model_name=model_name_
            )
        
        else:
            signature = infer_signature(X, Y)
            mlflow.sklearn.log_model(
                clf,
                artifact_path="decision_tree",
                signature=signature,
                registered_model_name=model_name_
            )

        # Set model alias
        if model_alias:
            time.sleep(20) # Wait 20secs for model version to be created
            client.set_registered_model_alias(model_name_, model_alias, get_latest_model_version(model_name_))

    return clf

# COMMAND ----------

model_champion   = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, 42, "Champion")

# COMMAND ----------

model_challenger = fit_and_register_model(X_train_pdf, Y_train_pdf, model_name, 10, "Challenger")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Real-time A/B Testing Deployment with Model Serving
# MAGIC
# MAGIC Let's serve the two models we logged in the previous step using Model Serving. Model Serving supports endpoint management via the UI and the API. 
# MAGIC
# MAGIC Below you will find instructions for using the UI and it is simpler method compared to the API. **In this demo, we will use the API to configure and create the endpoint**.
# MAGIC
# MAGIC **Both the UI and the API support querying created endpoints in real-time**. We will use the API to query the endpoint using a test-set.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 1: Serve model(s) using UI
# MAGIC
# MAGIC After registering the (new version(s) of the) model to the model registry. To provision a serving endpoint via UI, follow the steps below.
# MAGIC
# MAGIC 1. In the left sidebar, click **Serving**.
# MAGIC
# MAGIC 2. To create a new serving endpoint, click **Create serving endpoint**.   
# MAGIC   
# MAGIC     a. In the **Name** field, type a name for the endpoint.  
# MAGIC   
# MAGIC     b. Click in the **Entity** field. A dialog appears. Select **Unity catalog model**, and then select the catalog, schema, and model from the drop-down menus.  
# MAGIC   
# MAGIC     c. In the **Version** drop-down menu, select the version of the model to use.  
# MAGIC   
# MAGIC     d. Click **Confirm**.  
# MAGIC   
# MAGIC     e. In the **Compute Scale-out** drop-down, select Small, Medium, or Large. If you want to use GPU serving, select a GPU type from the **Compute type** drop-down menu.
# MAGIC   
# MAGIC     f. *[OPTIONAL]* to deploy another model (e.g. for A/B testing) click on **+Add Served Entity** and fill the above mentioned details.
# MAGIC   
# MAGIC     g. Click **Create**. The endpoint page opens and the endpoint creation process starts.   
# MAGIC   
# MAGIC See the Databricks documentation for details ([AWS](https://docs.databricks.com/machine-learning/model-serving/create-manage-serving-endpoints.html#ui-workflow)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/create-manage-serving-endpoints#--ui-workflow)).

# COMMAND ----------

# MAGIC %md
# MAGIC ### Option 2: Serve Model(s) Using the *Databricks Python SDK*
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC #### Get Models to Serve
# MAGIC
# MAGIC We will serve two models, therefore, we will get model version of the two models (**Champion** and **Challenger**) that we registered in the previous step.

# COMMAND ----------

model_version_champion = client.get_model_version_by_alias(name=model_name, alias="Champion").version # Get champion version
model_version_challenger = client.get_model_version_by_alias(name=model_name, alias="Challenger").version # Get challenger version

# COMMAND ----------

# MAGIC %md
# MAGIC #### Configure and Create Serving Endpoint

# COMMAND ----------

from databricks.sdk.service.serving import EndpointCoreConfigInput


# Parse model name from UC namespace
served_model_name =  model_name.split('.')[-1]

endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": model_version_champion,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
        {
            "model_name": model_name,
            "model_version": model_version_challenger,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        },
    ],
    "traffic_config": {
        "routes": [
            {"served_model_name": f"{served_model_name}-{model_version_champion}", "traffic_percentage": 50},
            {"served_model_name": f"{served_model_name}-{model_version_challenger}", "traffic_percentage": 50},
        ]
    },
    "auto_capture_config":{
        "catalog_name": DA.catalog_name,
        "schema_name": DA.schema_name,
        "table_name_prefix": "db_academy"
    }
}


endpoint_config = EndpointCoreConfigInput.from_dict(endpoint_config_dict)

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointTag


# Create/Update endpoint and deploy model+version
w = WorkspaceClient()

# COMMAND ----------

endpoint_name = f"ML_AS_03_Demo4_{DA.unique_name('_')}"

try:
  w.serving_endpoints.create_and_wait(
    name=endpoint_name,
    config=endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_model_example"})]
  )
  
  print(f"Creating endpoint {endpoint_name} with models {model_name} versions {model_version_champion} & {model_version_challenger}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Verify Endpoint Creation
# MAGIC
# MAGIC Let's verify that the endpoint is created and ready to be used for inference.

# COMMAND ----------

endpoint = w.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)

assert endpoint.state.config_update.value == "NOT_UPDATING" and endpoint.state.ready.value == "READY" , "Endpoint not ready or failed"

# COMMAND ----------

# MAGIC %md
# MAGIC #### Query the Endpoint
# MAGIC
# MAGIC Here we will use a very simple `test-sample` to use for inference. In a real-life scenario, you would typically load this set from a table or a streaming pipeline.

# COMMAND ----------

# Hard-code test-sample
dataframe_records = [
    {"SeniorCitizen": 0, "tenure":12, "MonthlyCharges":65, "TotalCharges":800},
    {"SeniorCitizen": 1, "tenure":24, "MonthlyCharges":40, "TotalCharges":500}
]

# COMMAND ----------

print("Inference results:")
query_response = w.serving_endpoints.query(name=endpoint_name, dataframe_records=dataframe_records)
print(query_response.predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Real-time Deployment with Online Features
# MAGIC
# MAGIC In the previous section we deployed a model without using feature tables. In this section **we will register and deploy a model for real-time inference with feature tables.** First, we will **deploy a model with online store integration** and then we will demonstrate **inference with online store integration**.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fit and Log the Model with Feature Table

# COMMAND ----------

from databricks.feature_engineering import FeatureLookup, FeatureEngineeringClient


feature_table_name = f"{DA.catalog_name}.{DA.schema_name}.features"
fe = FeatureEngineeringClient()

# Create feature table
fe.create_table(
    name=feature_table_name,
    df=features_df,
    primary_keys=[primary_key],
    description="Example feature table"
)

# Create training set based on feature lookup
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

# Covert data to pandas dataframes
X_train_pdf2 = training_df.drop(primary_key, response).toPandas()
Y_train_pdf2 = training_df.select(response).toPandas()

# COMMAND ----------

model_fe = fit_and_register_model(X_train_pdf2, Y_train_pdf2, model_name, 20, log_with_fs=True, training_set_spec_=training_set_spec)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set Up Databricks Online Tables
# MAGIC
# MAGIC In this section, we will create an online table to serve feature table for real-time inference. Databricks Online Tables can be created and managed via the UI and the SDK. While we provided instructions for both of these methods, you can pick one option for creating the table.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC
# MAGIC #### OPTION 1: Create Online Table via the UI
# MAGIC
# MAGIC You create an online table from the Catalog Explorer. The steps are described below. For more details, see the Databricks documentation ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#create)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#create)). For information about required permissions, see Permissions ([AWS](https://docs.databricks.com/en/machine-learning/feature-store/online-tables.html#user-permissions)|[Azure](https://learn.microsoft.com/azure/databricks/machine-learning/feature-store/online-tables#user-permissions)).
# MAGIC
# MAGIC
# MAGIC In **Catalog Explorer**, navigate to the source table that you want to sync to an online table. 
# MAGIC
# MAGIC From the kebab menu, select **Create online table**.
# MAGIC
# MAGIC * Use the selectors in the dialog to configure the online table.
# MAGIC   
# MAGIC   * `Name`: Name to use for the online table in Unity Catalog.
# MAGIC   
# MAGIC   * `Primary Key`: Column(s) in the source table to use as primary key(s) in the online table.
# MAGIC   
# MAGIC   * Timeseries Key: (Optional). Column in the source table to use as timeseries key. When specified, the online table includes only the row with the latest timeseries key value for each primary key.
# MAGIC   
# MAGIC   * `Sync mode`:  Select **`Snapshot`** for Sync mode. Please refer to the documentation for more details about available options.
# MAGIC   
# MAGIC   * When you are done, click Confirm. The online table page appears.
# MAGIC
# MAGIC The new online table is created under the catalog, schema, and name specified in the creation dialog. In Catalog Explorer, the online table is indicated by online table icon.

# COMMAND ----------

# MAGIC %md
# MAGIC #### OPTION 2: Use the Databricks SDK 
# MAGIC
# MAGIC The first option for creating an online table the UI. The other alternative is the Databricks' [python-sdk](https://databricks-sdk-py.readthedocs.io/en/latest/workspace/catalog/online_tables.html). Let's  first define the table specifications, then create the table.
# MAGIC
# MAGIC **🚨 Note:** The workspace must be enabled for using the SDK for creating and managing online tables. You can run following code blocks in your workspace is enabled for this feature.

# COMMAND ----------

# DBTITLE 1,Create Online Table Specifications
# MAGIC %md
# MAGIC
# MAGIC **Step1: Define table configuration:**
# MAGIC
# MAGIC ```
# MAGIC from databricks.sdk.service.catalog import OnlineTableSpec
# MAGIC
# MAGIC online_table_spec = OnlineTableSpec().from_dict({
# MAGIC     "source_table_full_name": feature_table_name,
# MAGIC     "primary_key_columns": [primary_key],
# MAGIC     "perform_full_copy": True
# MAGIC })
# MAGIC ```
# MAGIC
# MAGIC **Step2: Create the table**
# MAGIC
# MAGIC ```
# MAGIC from databricks.sdk.service.catalog import OnlineTablesAPI
# MAGIC
# MAGIC # Create online table
# MAGIC w = WorkspaceClient()
# MAGIC online_table = w.online_tables.create(
# MAGIC     name=f"{DA.catalog_name}.{DA.schema_name}.online_features",
# MAGIC     spec=online_table_spec
# MAGIC )
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Deploy the Model with Online Features
# MAGIC
# MAGIC Now that we have a model registered with feature table and we created an online feature table, we can deploy the model with Model Serving and use the online table during inference.
# MAGIC
# MAGIC **💡 Note:** The Model Serving **endpoint configuration and creation process is the same for serving models with or without feature tables**. The registered model metadata handles feature lookup during inference.

# COMMAND ----------

fs_model_version = get_latest_model_version(model_name)

fs_endpoint_config_dict = {
    "served_models": [
        {
            "model_name": model_name,
            "model_version": fs_model_version,
            "scale_to_zero_enabled": True,
            "workload_size": "Small"
        }
    ]
}

fs_endpoint_config = EndpointCoreConfigInput.from_dict(fs_endpoint_config_dict)

fs_endpoint_name = f"ML_AS_03_Demo4_FS_{DA.unique_name('_')}"

try:
  w.serving_endpoints.create_and_wait(
    name=fs_endpoint_name,
    config=fs_endpoint_config,
    tags=[EndpointTag.from_dict({"key": "db_academy", "value": "serve_fs_model_example"})]
  )
  
  print(f"Creating endpoint {fs_endpoint_name} with models {model_name} versions {fs_model_version}")

except Exception as e:
  if "already exists" in e.args[0]:
    print(f"Endpoint with name {fs_endpoint_name} already exists")

  else:
    raise(e)

# COMMAND ----------

# Hard-code test-sample
dataframe_records_lookups_only = [
    {"customerID": "0002-ORFBO"},
    {"customerID": "0003-MKNFE"}
]

# COMMAND ----------

print("FS Inference results:")
query_response = w.serving_endpoints.query(name=fs_endpoint_name, dataframe_records=dataframe_records_lookups_only)
print(query_response.predictions)

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
# MAGIC In this demo, we covered how to serve ML models in real-time using Databricks Model Serving. In the first part, we demonstrated how to serve models without feature store integration. Furthermore, we showed how to deploy two models on the same endpoint to conduct an A/B testing scenario. In the second section of the demo, we deployed a model with feature store integration using Databricks Online Tables. Additionally, we demonstrated how to use the endpoint for inference with Online Tables integration.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC &copy; 2024 Databricks, Inc. All rights reserved.<br/>
# MAGIC Apache, Apache Spark, Spark and the Spark logo are trademarks of the 
# MAGIC <a href="https://www.apache.org/">Apache Software Foundation</a>.<br/>
# MAGIC <br/><a href="https://databricks.com/privacy-policy">Privacy Policy</a> | 
# MAGIC <a href="https://databricks.com/terms-of-use">Terms of Use</a> | 
# MAGIC <a href="https://help.databricks.com/">Support</a>