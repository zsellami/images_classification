from pyspark.ml.evaluation import MulticlassClassificationEvaluator
df = p_model.transform(test_df)
df.show()

predictionAndLabels = df.select("prediction", "label")
evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
print("Training set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))