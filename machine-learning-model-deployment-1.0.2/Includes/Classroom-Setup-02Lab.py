# Databricks notebook source
# MAGIC %run ./_common

# COMMAND ----------

def create_features_table(self):
    spark.sql(f"USE CATALOG {DA.catalog_name}")
    spark.sql(f"USE SCHEMA {DA.schema_name}")
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
DBAcademyHelper.monkey_patch(create_features_table)

# COMMAND ----------

DA = DBAcademyHelper(course_config, lesson_config)  # Create the DA object
DA.reset_lesson()                                   # Reset the lesson to a clean state
DA.init()                                           # Performs basic intialization including creating schemas and catalogs
DA.create_features_table()                         
DA.conclude_setup()                                 # Finalizes the state and prints the config for the student

# COMMAND ----------

