import os
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType, TimestampType


def get_spark():
    return (
        SparkSession.builder
        .appName("AML-Detection")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )


def load_transactions(spark, path):
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    df = df.toDF(
        "timestamp", "from_bank", "from_account",
        "to_bank", "to_account",
        "amount_received", "receiving_currency",
        "amount_paid", "payment_currency",
        "payment_format", "is_laundering"
    )
    df = (
        df
        .withColumn("amount_received", F.col("amount_received").cast(DoubleType()))
        .withColumn("amount_paid", F.col("amount_paid").cast(DoubleType()))
        .withColumn("is_laundering", F.col("is_laundering").cast(IntegerType()))
        .withColumn("timestamp", F.to_timestamp("timestamp", "yyyy/MM/dd HH:mm"))
        .filter(F.col("amount_paid") > 0)
        .filter(F.col("is_laundering").isNotNull())
    )
    return df


def load_accounts(spark, path):
    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
    )
    return df


def summarize(df, label):
    total = df.count()
    laundering = df.filter(F.col("is_laundering") == 1).count()
    print(f"[{label}] rows={total:,} | laundering={laundering:,} ({laundering/total*100:.2f}%)")
    return total, laundering


if __name__ == "__main__":
    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    trans = load_transactions(spark, "data/HI-Small_Trans.csv")
    accounts = load_accounts(spark, "data/HI-Small_Accounts.csv")

    total, laundering = summarize(trans, "Transactions")

    print("\nSchema:")
    trans.printSchema()

    print("\nSample:")
    trans.show(3, truncate=False)

    spark.stop()
