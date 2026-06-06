from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql import Window


def build_transaction_features(df):
    w_from = Window.partitionBy("from_account")
    w_to   = Window.partitionBy("to_account")

    df = (
        df
        .withColumn("is_cross_currency",
            (F.col("payment_currency") != F.col("receiving_currency")).cast(DoubleType()))
        .withColumn("amount_ratio",
            F.col("amount_received") / (F.col("amount_paid") + F.lit(1e-9)))
        .withColumn("hour", F.hour("timestamp"))
        .withColumn("is_night", ((F.col("hour") < 6) | (F.col("hour") > 22)).cast(DoubleType()))
        .withColumn("payment_format_idx",
            F.dense_rank().over(Window.orderBy("payment_format")).cast(DoubleType()))
        .withColumn("from_tx_count",   F.count("*").over(w_from))
        .withColumn("from_tx_avg_amt", F.avg("amount_paid").over(w_from))
        .withColumn("from_tx_max_amt", F.max("amount_paid").over(w_from))
        .withColumn("from_unique_destinations",
            F.approx_count_distinct("to_account").over(w_from))
        .withColumn("to_tx_count",   F.count("*").over(w_to))
        .withColumn("to_tx_avg_amt", F.avg("amount_received").over(w_to))
        .withColumn("to_unique_sources",
            F.approx_count_distinct("from_account").over(w_to))
        .withColumn("from_currency_diversity",
            F.approx_count_distinct("payment_currency").over(w_from))
    )

    return df


FEATURE_COLS = [
    "amount_paid",
    "amount_received",
    "amount_ratio",
    "is_cross_currency",
    "is_night",
    "payment_format_idx",
    "from_tx_count",
    "from_tx_avg_amt",
    "from_tx_max_amt",
    "from_unique_destinations",
    "to_tx_count",
    "to_tx_avg_amt",
    "to_unique_sources",
    "from_currency_diversity",
]


if __name__ == "__main__":
    from src.data_pipeline import get_spark, load_transactions

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_transactions(spark, "data/HI-Small_Trans.csv")
    df = build_transaction_features(df)

    print(f"Feature matrix: {df.count():,} rows x {len(FEATURE_COLS)} features")
    df.select(FEATURE_COLS + ["is_laundering"]).show(3, truncate=True)

    spark.stop()
