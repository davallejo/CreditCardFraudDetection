# Detección de Fraude con Tarjetas de Crédito
## Elaborado por: Diego Armando Vallejo Vinueza

## Descripción del Proyecto
El objetivo del proyecto es desarrollar un modelo de predicción de fraude para una institución financiera como un banco. El fraude representa una pérdida significativa de ingresos para los bancos y es importante detectarlo de manera oportuna. El modelo se construirá usando datos históricos de transacciones y se aplicará a nuevas transacciones para identificar posibles fraudes.

## Comprensión de los Datos
Los datos que se utilizarán provienen de Kaggle y contienen información de transacciones bancarias, incluyendo características como el tipo de transacción, el monto, la ubicación geográfica, entre otras. Además, cada transacción está etiquetada como fraude o no fraude.

Los datos pueden ser revisados desde del enlace: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

## Preparación de los Datos
Se importan las librerías necesarias y se cargan los datos desde el archivo local creditcard.csv. Luego, se dividen los datos en características (X) y etiquetas (y), se normalizan las características y se dividen en conjuntos de entrenamiento y prueba.
![image](https://github.com/davallejo/CreditCardFraudDetection/assets/45080339/d1ea4ad7-62d7-4892-9d41-e687ce1b33e7)

## Modelado y Evaluación de 4 Modelos
Se entrenan y evalúan cuatro modelos de clasificación: Regresión Logística, Árbol de Decisión, Bosque Aleatorio y Red Neuronal. Se calculan métricas de evaluación como precision, recall, área bajo la curva ROC, y se visualizan las curvas ROC y las matrices de confusión.
![image](https://github.com/davallejo/CreditCardFraudDetection/assets/45080339/556f951d-ced6-44c4-af24-baa95d5e2b8a)

## Análisis de Resultados
Se analizan los coeficientes del modelo de Regresión Logística para determinar las variables del conjunto de datos que tienen mayor relevancia en la predicción de fraude. Las características con los coeficientes más grandes indican una mayor influencia en la predicción de fraude, lo que proporciona información valiosa para la toma de decisiones y la prevención de pérdidas financieras.
![image](https://github.com/davallejo/CreditCardFraudDetection/assets/45080339/763680ac-4d59-4945-837f-b1fb986827e4)
![Sin título](https://github.com/davallejo/CreditCardFraudDetection/assets/45080339/598b48a9-e723-40be-a3a8-d73bb0caa292)

## Conclusiones
Los coeficientes representan la influencia relativa de cada característica en la predicción del fraude, donde un valor positivo indica que la característica aumenta la probabilidad de fraude, y un valor negativo indica que la característica disminuye la probabilidad de fraude.
Las características con los coeficientes más grandes son:

- **V4:** Este es el coeficiente más grande y positivo, lo que sugiere que esta característica (desconocida) tiene una fuerte influencia en aumentar la probabilidad de fraude.
- **V10:** El segundo coeficiente más grande, es negativo, lo que indica que esta característica no contribuye significativamente a incrementar la probabilidad de fraude.
- **V14:** Esta característica también tiene un coeficiente negativo grande, sugiriendo que no aumenta la probabilidad de fraude.
- **V22:** Tiene un valor positivo menor a v4, lo que significa que su presencia puede aumentar la probabilidad de fraude.
- **V20**, **V13**, **V27**, **V9**: Estas características tienen coeficientes moderadamente negativos, lo que significa que su presencia no tiende a aumentar, la probabilidad de fraude.
- **V21** y **Amount**: Estas características tienen coeficientes positivos pero no son significativamente grandes, lo que implica que su presencia puede influir en la probabilidad de fraude aunque no en gran medida.



