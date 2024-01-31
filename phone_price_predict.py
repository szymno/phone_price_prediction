import numpy as np
import matplotlib.pyplot as plt

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, FloatType

from pyspark.ml import feature, classification, Pipeline
from pyspark.ml.stat import Correlation
from pyspark.ml.evaluation import MulticlassClassificationEvaluator


structure_train = StructType(
    [
        StructField('battery_power', IntegerType(), True),
        StructField('blue', IntegerType(), True),
        StructField('clock_speed', FloatType(), True),
        StructField('dual_sim', IntegerType(), True),
        StructField('fc', IntegerType(), True),
        StructField('four_g', IntegerType(), True),
        StructField('int_memory', IntegerType(), True),
        StructField('m_dep', FloatType(), True),
        StructField('mobile_wt', IntegerType(), True),
        StructField('n_cores', IntegerType(), True),
        StructField('pc', IntegerType(), True),
        StructField('px_height', IntegerType(), True),
        StructField('px_width', IntegerType(), True),
        StructField('ram', IntegerType(), True),
        StructField('sc_h', IntegerType(), True),
        StructField('sc_w', IntegerType(), True),
        StructField('talk_time', IntegerType(), True),
        StructField('three_g', IntegerType(), True),
        StructField('touch_screen', IntegerType(), True),
        StructField('wifi', IntegerType(), True),
        StructField('price_range', IntegerType(), True)
    ]
)

seed = None
spark = SparkSession.builder.appName("phone_price_prediction").getOrCreate()

# structure_test = StructType([StructField('id', IntegerType(), True)] + [struct for struct in structure_train[0:-1]])
# raw_test_data = spark.read.csv("./archive/test.csv", header=True, schema=structure_test)


raw_data = spark.read.csv("./archive/train.csv", header=True, schema=structure_train)
raw_train_data, raw_test_data = raw_data.randomSplit([0.8, 0.2], seed=seed)


vectorize = feature.VectorAssembler(inputCols=raw_data.columns[:-1], outputCol="features_")
scaler = feature.StandardScaler(inputCol="features_", outputCol="features")


#  CORRELATION

data = vectorize.transform(raw_data)
scaler_fit = scaler.fit(data)
data = scaler_fit.transform(data.select("price_range", "features_"))

labels = ["battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g", "int_memory", "m_dep", "mobile_wt",
          "n_cores", "pc", "px_height", "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g", "touch_screen",
          "wifi"]
corr_matrix = Correlation.corr(data.select("features_"), "features_").collect()[0][0].toArray()

fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(corr_matrix)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)
plt.setp(ax.get_xticklabels(), rotation=90)
fig.tight_layout()
plt.show()


#  MODELS
model_RF = classification.RandomForestClassifier(
    maxDepth=8,
    minInstancesPerNode=3,
    labelCol="price_range",
    seed=seed
)
model_NB = classification.NaiveBayes(labelCol="price_range")
model_LR = classification.LogisticRegression(labelCol="price_range")

evaluators = {
    "accuracy": MulticlassClassificationEvaluator(labelCol="price_range", metricName="accuracy"),
    "recall": MulticlassClassificationEvaluator(labelCol="price_range", metricName="weightedRecall"),
    "f1": MulticlassClassificationEvaluator(labelCol="price_range", metricName="f1"),
    "precision": MulticlassClassificationEvaluator(labelCol="price_range", metricName="weightedPrecision")
}
pipelines = {
    "LR": Pipeline(stages=[vectorize, scaler, model_LR]),
    "NB": Pipeline(stages=[vectorize, scaler, model_NB]),
    "RF": Pipeline(stages=[vectorize, scaler, model_RF])
}


predictions = dict()
for key, pipeline in pipelines.items():
    prediction = pipeline.fit(raw_train_data).transform(raw_test_data).select("price_range", "prediction", "probability")
    predictions[key] = prediction

values = dict()
for key, prediction in predictions.items():
    values[key] = {name: evaluator.evaluate(prediction) for name, evaluator in evaluators.items()}
    print(key, values[key])



