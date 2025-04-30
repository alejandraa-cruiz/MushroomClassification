# Clasificación de hongos por medio de redes neuronales

## Abstract

La clasificación precisa de hongos en comestibles y venenosos es fundamental para la seguridad alimentaria
y para el desarrollo de nuevos tratamientos médicos. Este proyecto presenta un modelo de clasificación binaria
basado en redes neuronales artificiales de retropropagación (Backpropagation Neural Network, BPNN) para identificar
hongos comestibles y venenosos a partir de un conjunto de datos del UC Irvine Machine Learning, que incluye 61,069
observaciones y 20 variables.

Se realizó un preprocesamiento exhaustivo, que incluyó la imputación de valores faltantes mediante K-Nearest Neighbors (KNN), técnica seleccionada por su alta precisión en conjuntos de datos con más del 50 % de datos ausentes (Memon et al., 2023). Dado que KNN opera únicamente con representaciones numéricas, se aplicó codificación de etiquetas (Label Encoding) previa a la imputación y, posteriormente, los valores fueron transformados nuevamente a sus etiquetas categóricas originales. La reducción de dimensionalidad fue descartada debido a la baja multicolinealidad entre los atributos.

El modelo se implementó mediante una BPNN secuencial en Keras, utilizando una arquitectura simple con una capa oculta de 128 neuronas y activación ReLU, seguida de una capa de salida con activación sigmoide. En la evaluación inicial utilizando Label Encoding, el modelo alcanzó una precisión del 99 %. Al aplicar One-Hot Encoding sobre las variables categóricas, se obtuvo una precisión del 100 %, con solo cinco errores de clasificación en el conjunto de prueba. El modelo es capaz de recibir entradas codificadas en arreglos de 125 valores (equivalentes a las 20 variables originales transformadas), lo que lo hace apto para una implementación práctica.

Los resultados evidencian que, si bien es posible desarrollar modelos precisos utilizando un subconjunto de variables, la utilización del total de atributos garantiza un desempeño superior, crítico para aplicaciones donde los errores pueden tener consecuencias significativas. Este enfoque demuestra el potencial de las redes neuronales como herramientas eficaces y seguras para tareas de clasificación en contextos sensibles como la micología aplicada.

## Introducción

La clasificación de los hongos en comestibles y venenosos constituye un aspecto fundamental de la micología,
ya que permite prevenir intoxicaciones, especialmente en comunidades indígenas donde su consumo es frecuente.
Los hongos silvestres comestibles representan una fuente valiosa de nutrientes y poseen un alto potencial para
mejorar la diversidad y calidad de la alimentación (Singh et al., 2025). Además, un estudio reciente identificó
a los hongos como una fuente rica en metabolitos secundarios con propiedades antivirales, destacando su
potencial en el desarrollo de nuevos tratamientos contra el SARS-CoV-2 (Patni et al., 2025).

En este contexto, el objetivo del presente proyecto es **desarrollar un clasificador binario que permita
distinguir entre hongos comestibles y venenosos**, utilizando para ello redes neuronales artificiales como
herramienta principal de modelado.

### Descripción del Dataset

El conjunto de datos proviene del [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/848/secondary+mushroom+dataset)
contiene 61,069 observaciones de hongos hipotéticos, clasificados según 173 especies, con un total de 353 ejemplares
por especie. Cada hongo se etiqueta como comestible, venenoso o de comestibilidad desconocida (esta última categoría
se combina con la de hongos venenosos). De las 20 variables descritas, 17 son de tipo nominal y 3 de tipo métrico.

Los hongos están clasificados de forma **binaria** en venenoso `p` y comestible `e`. Los atributos están clasificados
como nominales `n` y métricas `m`. A continuación se presenta una tabla con los atributos y sus respectivas descripciones.

| **Atributo**          | **Tipo** | **Descripción/Valores**                                                                                                        |
| --------------------- | -------- | ------------------------------------------------------------------------------------------------------------------------------ |
| **cap-diameter**      | m        | Float number in cm                                                                                                             |
| **cap-shape**         | n        | bell=b, conical=c, convex=x, flat=f, sunken=s, spherical=p, others=o                                                           |
| **cap-surface**       | n        | fibrous=i, grooves=g, scaly=y, smooth=s, shiny=h, leathery=l, silky=k, sticky=t, wrinkled=w, fleshy=e                          |
| **cap-color**         | n        | brown=n, buff=b, gray=g, green=r, pink=p, purple=u, red=e, white=w, yellow=y, blue=l, orange=o, black=k                        |
| **does-bruise-bleed** | n        | bruises-or-bleeding=t, no=f                                                                                                    |
| **gill-attachment**   | n        | adnate=a, adnexed=x, decurrent=d, free=e, sinuate=s, pores=p, none=f, unknown=?                                                |
| **gill-spacing**      | n        | close=c, distant=d, none=f                                                                                                     |
| **gill-color**        | n        | see cap-color + none=f                                                                                                         |
| **stem-height**       | m        | Float number in cm                                                                                                             |
| **stem-width**        | m        | Float number in mm                                                                                                             |
| **stem-root**         | n        | bulbous=b, swollen=s, club=c, cup=u, equal=e, rhizomorphs=z, rooted=r                                                          |
| **stem-surface**      | n        | see cap-surface + none=f                                                                                                       |
| **stem-color**        | n        | see cap-color + none=f                                                                                                         |
| **veil-type**         | n        | partial=p, universal=u                                                                                                         |
| **veil-color**        | n        | see cap-color + none=f                                                                                                         |
| **has-ring**          | n        | ring=t, none=f                                                                                                                 |
| **ring-type**         | n        | cobwebby=c, evanescent=e, flaring=r, grooved=g, large=l, pendant=p, sheathing=s, zone=z, scaly=y, movable=m, none=f, unknown=? |
| **spore-print-color** | n        | see cap color                                                                                                                  |
| **habitat**           | n        | grasses=g, leaves=l, meadows=m, paths=p, heaths=h, urban=u, waste=w, woods=d                                                   |
| **season**            | n        | spring=s, summer=u, autumn=a, winter=w                                                                                         |

## Metodología

### Preprocesamiento

El conjunto de datos tiene un tamaño de 61,069 filas y 21 columnas. Está compuesto por un total de 20 variables, de las cuales 3 son de tipo métrico `float64`:
_cap-diameter, stem-height y stem-width_. Las 17 variables restantes, incluida la variable objetivo class,
son de tipo categórico `object`. Estas variables describen características morfológicas del hongo, como forma,
color, superficie, presencia de anillo, tipo de velo, hábitat y estación de aparición, entre otras.

De los 61,069 registros, 33,888 corresponden a hongos venenosos
(etiquetados como `p`) y 27,181 a hongos comestibles (etiquetados como `e`). Aunque existe un leve desbalance entre las clases
—con un **56% de hongos venenosos y un 44% de comestibles**—, la diferencia no es lo suficientemente pronunciada como para requerir
técnicas específicas de balanceo. Se considera que el ligero desbalance no compromete la capacidad del modelo para aprender de
ambas clases.

<p align="center">
  <img src="./images/classDistribution.png" alt="class distribution" width="30%" />
  <br>
  <em>Gráfica 1. Distribución de los datos en las clases venenoso (p) y comestible (e)</em>
</p>

#### Codificación de Etiquetas

Se transformaron las variables categóricas en valores númericos, lo que se conoce como
_Label Encoding_ o _Codificación de Etiquetas_. Parte fundamental del preprocesamiento
de los datos ya que el modelo no puede procesar datos categóricos directamente.

Inicialmente se aplico la técnica de _Label Encoding_ a todos los atributos. No obstante, esto
no evitaba que el modelo interpetrara relaciones entre los valores o que aprendiera un orden
no existente porque las categorías son nominales. Por ello, se aplicó _One-Hot Encoding_ a las columnas
con más de dos clases; conviritendo cada categoría a una columna binaria. Se mantuvo el _Label Encoding_
para las columnas binarias: _class, does-bruise-or-bleed y has-ring_.

#### Multicolinealidad

A través del análisis de la matriz de correlación y su correspondiente mapa de calor, se observó que los atributos
del conjunto de datos presentan niveles bajos de correlación entre sí, lo que indica una relativa independencia entre las
características. Esta ausencia de correlaciones fuertes sugiere que **no existen variables redundantes ni multicolinealidad
significativa que justifique, por el momento, la aplicación de técnicas de reducción de dimensionalidad**. Por lo tanto,
se decidió conservar todas las variables para el modelado inicial.

<p align="center">
  <img src="./images/correlationHeatmap.png" alt="correlation heatmap" width="40%" />
  <br>
  <em>Gráfica 2. Mapa de calor de la matriz de correlación</em>
</p>

#### Chi-Square Score

Se evaluó el Chi-Square Score de cada variable con el objetivo de medir su grado de asociación con la variable
dependiente (la clase objetivo `class`). Esta prueba estadística permite evaluar que atrbibutos tiene una mayor
influencia en la clasificación. A mayor puntuación, mayor es la relación entre la variable independiente y la variable
de salida. Aunque en la etapa inicial no se busca la reducción de dimensionalidad, los resultados serán considerados
en una etapa posterior, para simplificar el modelo una vez que se cuente con un buen desempeño.

Las variables que mostraron una mayor asociación con la clase objetivo, según el valor del Chi-Square Score,
fueron: _stem-width, cap-diameter, stem-height, stem-surface, cap-shape, stem-color, y gill-color_.

### Modelo

La red neuronal definida en el modelo esta entrenada con un algoritmo de entrenamiento BPNN (Backpropagation Neural Network). Es una red neuronal que entrena con retropropagación. Jeatrakul y Wong (2009), al evaluar distintas arquitecturas para la clasificación binaria
encontraron que BPNN se comportaba de manera robusta en cada caso de prueba (diferentes conjuntos de datos). Lo que representaba una ventaja sin añadir la complejidad de una red CMTNN (Convolutional Multiscale Twin Neural Network).

Por lo tanto, en este proyecto se define una red neuronal simple utilizando Keras. Es un modelo secuencial, donde las capas van una detrás de otra. El modelo consiste de dos capas.
La primera capa es una capa densa que tiene 128 neuronas, se utiliza la función de activación `relu` para permitirle aprender relaciones complejas.
La segunda capa es la capa salida, tiene una sola neurona de salida debido a que es una clasificación binaria. La función de activación `sigmoid` asigna el valor de salida entre 0 y 1.
Para compilar la función se utiliza el optimizador Adam y se añade la función de pérdida `binary crossentropy`. Para evaluar el rendimiento del modelo se utiliza `accuracy`.

Otros estudios (Shujaaddeen et al., 2024) han encontrado que en problemas de clasificación binaria una arquitectura MLP obtiene resultados superiores al 99% en las métricas de `accuracy`, `precision`,
`f1-score` y `recall`. La arquitectura carece de capas ocultas y cuenta únicamente con la capa de entrada y la capa de salida. Entrenada con la función de activación ReLu y
alrededor de 300 épocas.

### Modelo Reducido

Utilizando las variables que superaron el umbral (threshold) de 300 en el chi-score se mantuvieron las siete variables previamente mencionadas y se redujo la dimensionalidad del modelo.
La variable `stem-surface` presentaba 38,124 valores faltantes de un total de 61, 069 registros. Al reducir la dimensionalidad, estos valores faltantes afectan el desempeño del modelo.
Al tratar con valores faltantes, el estado del arte ha establecido dos enfoques principales.
El primero consiste en ignorar los valores faltantes; este enfoque omite o elimina los casos que presentan datos faltantes
en al menos una variable. Sin embargo, cuando la pérdida de datos supera el 50%, la precisión del modelo se ve afectada negativamente.
El segundo enfoque es la imputación de valores faltantes, que consiste en reemplazar dichos valores por otros plausibles utilizando
técnicas estadísticas (Shaheen MZ. Memon et al., 2023). En el caso del presente conjunto de datos, ignorar los valores faltantes implicaba una pérdida del 62%
de los datos, por lo que se optó por la imputación.

Para la imputación de variables categóricas existen diversos métodos, entre ellos: imputación por moda,
K-Nearest Neighbors (KNN), imputación mediante Random Forest e Imputación Múltiple mediante Ecuaciones Encadenadas (MICE).
Entre estos, KNN presenta la mayor precisión en la imputación de datos cuando más del 50% de los valores están ausentes (MShaheen MZ. Memon et al., 2023).
En el presente conjunto de datos se aplicó la imputación mediante KNN; primero se realizó una codificación de etiquetas
(label encoding), dado que el imputador KNN solo opera con representaciones numéricas. Posteriormente, los valores faltantes
fueron imputados utilizando los 30 vecinos más cercanos. Finalmente, los datos fueron transformados nuevamente a sus etiquetas
categóricas originales para continuar con las etapas subsiguientes del modelo.

Estas etapas incluyeron la codificación one-hot y la normalización de características mediante escalado min-max.
Asimismo, las etiquetas de la variable objetivo fueron representadas de forma binaria, asignando el valor 1 a las instancias
clasificadas como venenosas y 0 a las no venenosas.

Para evitar el sobreajuste en una MLP, un problema común en el aprendizaje automático que puede surgir con la reducción de dimensionalidad, se añadió L2 como método de regularización. Esta técnica agrega una penalización a la función de pérdida para forzar al modelo a aprender el patrón (Abu-Doush et al., 2023). L2 es una técnica común para prevenir el sobreajuste en MLP y, como se muestra en la evaluación, fue suficiente para equilibrar el rendimiento de ambas clases.

## Resultados

### Matriz de Confusión

La matriz de confusión es una herramienta fundamental en la evaluación del desempeño de los modelos de clasificación,
ya que proporciona una visualización detallada de las predicciones realizadas por el modelo en comparación con los valores reales
en el conjunto de datos de prueba. Según Wahyudi y Andrian (2021), la matriz de confusión permite calcular métricas clave como
la precisión y el recall. Estas métricas son esenciales para evaluar no solo la exactitud global del modelo, sino también su
capacidad para identificar correctamente ambas clases, minimizando falsos positivos y falsos negativos.

### Evaluación Inicial

El modelo alcanzó una precisión del 99 % en el conjunto de prueba utilizando _Label Enconding_ para todas las variables, lo que indica un desempeño altamente efectivo.
Las métricas de evaluación muestran resultados equilibrados entre ambas clases, lo que sugiere que el modelo no está
sesgado hacia ninguna de ellas. Además, la matriz de confusión evidenció un bajo número de errores, con únicamente 180
clasificaciones incorrectas sobre un total de 12,000 muestras. Se observó una ligera tendencia a generar más falsos
positivos que falsos negativos, aunque la diferencia es mínima.

<p align="center">
  <table>
    <tr>
      <th>Clase</th>
      <th>Precisión</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Soporte</th>
    </tr>
    <tr>
      <td>0</td>
      <td>0.99</td>
      <td>0.98</td>
      <td>0.98</td>
      <td>5405</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.98</td>
      <td>0.99</td>
      <td>0.99</td>
      <td>6809</td>
    </tr>
  </table>
  <br>
  <em>Tabla 2. Resultados de evaluación del modelo de clasificación inicial.</em>
</p>

<p align="center">
  <img src="./images/initialConfussionMatrix.png" alt="initial confussion matrix" width="40%" />
  <br>
  <em>Gráfica 3. Matriz de confusión del modelo inicial</em>
</p>

### Usando One-Hot Encoding

Al cambiar la codificación de etiquetas para usar _One-Hot Encoding_ para las columnas con múltiples categorías.
El modelo obtuvo una precisión del 100%, lo que sugiere que esta técnica permitió una representación más adecuada
de las variables categóricas y, por lo tanto, una mayor capacidad de generalización. Se mantuvo la tendencia de
generar más falsos positivos, pero únicamente con 3 resultados erróneos.

<p align="center">
  <table>
    <tr>
      <th>Clase</th>
      <th>Precisión</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Soporte</th>
    </tr>
    <tr>
      <td>0</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>5405</td>
    </tr>
    <tr>
      <td>1</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>1.00</td>
      <td>6809</td>
    </tr>
  </table>
  <br>
  <em>Tabla 3. Resultados de evaluación del modelo con One-Hot Encoding.</em>
</p>

<p align="center">
  <img src="./images/oneHotEncodingConfussionMatrix.png" alt="one-hot confussion matrix" width="40%" />
  <br>
  <em>Gráfica 4. Matriz de confusión del modelo utilizando One-Hot Encoding</em>
</p>

### Evaluación del modelo reducido

Desde la perspectiva de generalización del modelo, los resultados indican un desempeño robusto y equilibrado entre
ambas clases, con una precisión general del 95%. La similitud entre las métricas de precisión, recall y F1 para
ambas clases, así como la ausencia de valores extremos o desbalance marcados, sugiere que el modelo no está sobreajustado
(overfitted) a los datos de entrenamiento. En particular, el hecho de que tanto la clase mayoritaria (venenosa) como la
clase minoritaria (no venenosa) presenten métricas F1 muy similares (ambas de 0.95) refuerza la idea de una buena capacidad
de generalización.

Considerando la reducción de dimensionalidad de 20 variables para predecir a 7 únicamente, el comportamiento del modelo
es mejor de lo esperado. Reforzando que estás 7 variables tienen el mayor peso en la predicción de la clase.

<p align="center">
  <table>
    <tr>
      <th>Clase</th>
      <th>Precisión</th>
      <th>Recall</th>
      <th>F1-score</th>
      <th>Soporte</th>
    </tr>
    <tr>
      <td>0</td>
      <td>0.92</td>
      <td>0.97</td>
      <td>0.95</td>
      <td>5469</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.98</td>
      <td>0.93</td>
      <td>0.95</td>
      <td>6743</td>
    </tr>
  </table>
  <br>
  <em>Tabla 4. Resultados de evaluación del modelo reducido a 7 variables predictoras.</em>
</p>

<p align="center">
  <img src="./images/reducedConfussionMatrix.png" alt="reduced confussion matrix" width="40%" />
  <br>
  <em>Gráfica 5. Matriz de confusión del modelo reducido a 7 variables predictoras</em>
</p>

## Conclusiones

Es factible construir un modelo clasificador de hongos utilizando al menos siete variables predictoras,
mediante una red neuronal entrenada con el algoritmo de retropropagación y una arquitectura del tipo MLP
(Perceptrón Multicapa). Este modelo logra una precisión del 95%, lo cual representa un desempeño sólido.
No obstante, si bien la reducción de variables puede conllevar beneficios en términos de eficiencia computacional,
la clasificación precisa de hongos como venenosos o no venenosos tiene implicaciones directas en la seguridad alimentaria,
así como en la investigación de posibles aplicaciones terapéuticas. Por esta razón, se recomienda la utilización del
conjunto completo de 20 variables, con el cual el modelo alcanza una precisión del 100%, cometiendo solo cinco errores
en el conjunto de datos de prueba. Además, el modelo está preparado para realizar predicciones sobre arreglos de entrada
compuestos por 125 valores, correspondientes a las 20 variables transformadas mediante codificación one-hot.

En síntesis, este enfoque no solo ofrece un alto rendimiento, sino que también sienta las bases para soluciones confiables
y escalables en el ámbito de la micología aplicada y la inteligencia artificial.

## Referencias

Abu-Doush, I., Ahmed, B., Awadallah, M. A., Al-Betar, M. A., & Rababaah, A. R. (2023). Enhancing multilayer perceptron neural network using archive-based Harris Hawks optimizer to predict gold prices. Journal of King Saud University-Computer and Information Sciences. https://doi.org/10.1016/j.jksuci.2023.101557

A. A. Shujaaddeen, F. M. Ba-Alwi, A. T. Zahary, A. S. Alhegami, A. Alsabry and A. M. Al-Badani, "A Binary and Multi Classification Model on Tax Evasion: A Comparative Study," 2024 1st International Conference on Emerging Technologies for Dependable Internet of Things (ICETI), Sana'a, Yemen, 2024, pp. 1-9, doi: 10.1109/ICETI63946.2024.10777224.

Shaheen MZ. Memon, Robert Wamala, Ignace H. Kabano. A comparison of imputation methods for categorical data. Informatics in Medicine Unlocked, Volume 42, 2023, 101382. ISSN 2352-9148. https://doi.org/10.1016/j.imu.2023.101382

Singh A, Singh G, Kapoor R, Dhasmana A, Jerath S. G. Wild Edible Mushrooms of Jharkhand: Nutrient-Dense Seasonal Foods to Improve Dietary Diversity Among Indigenous Communities. Nutr Food Sci 2025; 13(1). doi : http://dx.doi.org/10.12944/CRNFSJ.13.1.4

Patni, B., Bhattacharyya, M., Pokhriyal, A. et al. Remedying SARS-CoV-2 through nature: a review highlighting the potentiality of herbs, trees, mushrooms, and endophytic microorganisms in controlling Coronavirus. Planta 261, 89 (2025). doi: https://doi.org/10.1007/s00425-025-04647-8

P. Jeatrakul and K. W. Wong, "Comparing the performance of different neural networks for binary classification problems," 2009 Eighth International Symposium on Natural Language Processing, Bangkok, Thailand, 2009, pp. 111-115, doi: 10.1109/SNLP.2009.5340935.

Wahyudi, M., & Andriani, A. (2021). Application of C4.5 and Naïve Bayes Algorithm for Detection of Potential Increased Case Fatality Rate Diarrhea. Journal of Physics: Conference Series, 1830(1), 012016. https://doi.org/10.1088/1742-6596/1830/1/012016
