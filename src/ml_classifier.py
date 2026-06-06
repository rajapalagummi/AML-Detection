from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, Imputer
from pyspark.ml.classification import RandomForestClassifier, GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator
from pyspark.sql import functions as F
from src.feature_engineering import FEATURE_COLS


def train_and_evaluate(df):
    df = df.select(FEATURE_COLS + ["is_laundering"]).dropna(subset=["is_laundering"])

    for col in FEATURE_COLS:
        df = df.withColumn(col, F.col(col).cast("double"))

    imputer = Imputer(inputCols=FEATURE_COLS, outputCols=[c + "_imp" for c in FEATURE_COLS])
    imputed_cols = [c + "_imp" for c in FEATURE_COLS]

    assembler = VectorAssembler(inputCols=imputed_cols, outputCol="features")

    df_labeled = df.withColumnRenamed("is_laundering", "label")
    train, test = df_labeled.randomSplit([0.8, 0.2], seed=42)

    auc_eval = BinaryClassificationEvaluator(metricName="areaUnderROC")
    f1_eval  = MulticlassClassificationEvaluator(metricName="f1")
    acc_eval = MulticlassClassificationEvaluator(metricName="accuracy")

    results = {}

    for name, clf in [
        ("RandomForest", RandomForestClassifier(
            numTrees=100, maxDepth=6, seed=42,
            labelCol="label", featuresCol="features"
        )),
        ("GBT", GBTClassifier(
            maxIter=50, maxDepth=5, seed=42,
            labelCol="label", featuresCol="features"
        )),
    ]:
        pipeline = Pipeline(stages=[imputer, assembler, clf])
        model    = pipeline.fit(train)
        preds    = model.transform(test)

        auc = auc_eval.evaluate(preds)
        f1  = f1_eval.evaluate(preds)
        acc = acc_eval.evaluate(preds)

        tp = preds.filter((F.col("label") == 1) & (F.col("prediction") == 1)).count()
        fp = preds.filter((F.col("label") == 0) & (F.col("prediction") == 1)).count()
        fn = preds.filter((F.col("label") == 1) & (F.col("prediction") == 0)).count()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)

        results[name] = {
            "auc":       round(float(auc), 4),
            "f1":        round(float(f1), 4),
            "accuracy":  round(float(acc), 4),
            "precision": round(float(precision), 4),
            "recall":    round(float(recall), 4),
            "tp": int(tp), "fp": int(fp), "fn": int(fn),
        }

        print(f"[{name}] AUC={auc:.4f} | F1={f1:.4f} | "
              f"Precision={precision:.4f} | Recall={recall:.4f}")

    return results


if __name__ == "__main__":
    from src.data_pipeline import get_spark, load_transactions
    from src.feature_engineering import build_transaction_features

    spark = get_spark()
    spark.sparkContext.setLogLevel("ERROR")

    df = load_transactions(spark, "data/HI-Small_Trans.csv")
    df = build_transaction_features(df)
    results = train_and_evaluate(df)

    spark.stop()
