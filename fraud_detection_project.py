# Importar las librerías necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder

# Fase 1: Comprensión del Negocio
"""
El objetivo de este proyecto es desarrollar un modelo de predicción de fraude para una institución financiera como un banco.
El fraude representa una pérdida significativa de ingresos para los bancos y es importante detectarlo de manera oportuna.
El modelo se construirá usando datos históricos de transacciones y se aplicará a nuevas transacciones para identificar posibles fraudes.
"""

# Fase 2: Comprensión de los Datos
"""
Los datos que se utilizarán provienen de Kaggle y contienen información de transacciones bancarias, incluyendo características
como el tipo de transacción, el monto, la ubicación geográfica, entre otras. Además, cada transacción está etiquetada como
fraude o no fraude.
"""

# Cargar los datos desde el archivo local
spark = SparkSession.builder.appName("FraudDetection").getOrCreate()
data = spark.read.csv("creditcard.csv", header=True, inferSchema=True)

# Fase 3: Preparación de los Datos
# Dividir los datos en conjunto de entrenamiento y prueba
(train_data, test_data) = data.randomSplit([0.8, 0.2], seed=42)

# Vectorizar las características
assembler = VectorAssembler(inputCols=data.columns[:-1], outputCol="features")
scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures", withStd=True, withMean=False)

# Fase 4: Modelado
# Definir el pipeline de preprocesamiento y modelo
lr = LogisticRegression(featuresCol="scaledFeatures", labelCol="Class")
pipeline = Pipeline(stages=[assembler, scaler, lr])

# Entrenar el modelo
model = pipeline.fit(train_data)

# Fase 5: Evaluación
# Evaluar el modelo en el conjunto de prueba
predictions = model.transform(test_data)
evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="Class")
auc = evaluator.evaluate(predictions)
print(f"Area Under ROC: {auc}")

# Generar la curva ROC
fpr, tpr, thresholds = evaluator.roc_curve(predictions)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % auc)
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# Fase 6: Despliegue
"""
Para desplegar el modelo en una página web, se puede utilizar un framework como Flask o Django en Python.
La página web tendrá un formulario donde el usuario ingresará los valores de las características de una nueva transacción.
Estos valores se procesarán usando el pipeline entrenado y se obtendrá la predicción de fraude o no fraude.
El resultado se mostrará al usuario en la página web.
"""