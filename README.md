# 🚗 Predicción de Heridos en Accidentes Viales - Regresión Lineal

Este repositorio contiene la implementación end-to-end de un modelo de **Regresión Lineal** en Python para predecir la cantidad de heridos en accidentes de tránsito utilizando datos abiertos de Colombia.

El proyecto enfatiza un **flujo de trabajo robusto y libre de fuga de información (*data leakage*)**, estructurado con Pipelines de Scikit-Learn para preprocesamiento, imputación, escalado y codificación categórica.



## 📋 Índice
* [Descripción del Proyecto](#-descripción-del-proyecto)
* [Prevención de Fuga de Información](#-prevención-de-fuga-de-información)
* [Estructura del Repositorio](#-estructura-del-repositorio)
* [Tecnologías Utilizadas](#-tecnologías-utilizadas)
* [Resultados y Evaluación del Modelo](#-resultados-y-evaluación-del-modelo)
* [Instrucciones de Ejecución](#-instrucciones-de-ejecución)
* [Autor](#-autor)

---

## 🎯 Descripción del Proyecto

El objetivo es predecir la variable continua `Heridos` (número de personas lesionadas por evento) en función de variables temporales, geográficas y de magnitud del incidente.

* **Variable Objetivo ($y$):** `Heridos`.
* **Variables Predictoras ($X$):**
  * **Numéricas:** `Vehiculos Involucrados`, `Mes`, `Dia_Mes`, `Es_Fin_Semana`.
  * **Categorización:** `Clase de Accidente`, `Barrio`, `Dia_Semana`.
* **Variables Excluidas:** `Muertes` y `Accidente con` (se excluyen del modelo por ser consecuencias posteriores al hecho que provocarían *data leakage*), así como identificadores únicos `Codigo_Accidente` y `Direccion`.

---

## 🛡️ Prevención de Fuga de Información (*Data Leakage*)

Para garantizar la validez metodológica:
1. **División Train/Test Temprana:** La partición del dataset (80% train / 20% test) se realiza **antes** de cualquier cálculo de estadísticas (medias, medianas o modas).
2. **Encapsulamiento dentro del Pipeline:** La imputación (`SimpleImputer`), la estandarización (`StandardScaler`) y la codificación one-hot (`OneHotEncoder`) se entrenan **exclusivamente** con el conjunto de datos de entrenamiento.
3. **Manejo de Categorías Desconocidas:** `OneHotEncoder(handle_unknown="ignore")` evita fallos o fugas cuando aparecen nuevos barrios o categorías durante la prueba.

---

## 📁 Estructura del Repositorio

```text
prediccion-heridos-accidentes/
├── Accidentes_Viales_20261009.csv   # Dataset original de Datos Abiertos Colombia
├── main.py                          # Script principal de ejecución y evaluación
├── requirements.txt                 # Dependencias del proyecto
├── resultados/                      # Gráficos exportados (PNG) y reportes (CSV/TXT)
│   ├── metricas_modelo.csv
│   ├── predicciones_test.csv
│   ├── resumen.txt
│   └── *.png                        # Visualizaciones del EDA y del modelo
└── README.md                        # Documentación técnica del proyecto
```

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.x
* **Análisis y Manipulación de Datos:** Pandas, NumPy
* **Visualización:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn (Pipelines, ColumnTransformer, LinearRegression, DummyRegressor)

---

## 📊 Resultados y Evaluación del Modelo

El modelo de Regresión Lineal se evalúa frente a un predictor base (`DummyRegressor` que estima la media de heridos):

| Métrica | Entrenamiento | Prueba (Modelo) | Prueba (Base) |
| :--- | :---: | :---: | :---: |
| **MAE** | 0.5898 | 0.7175 | 0.6882 |
| **RMSE** | 0.7816 | 0.8893 | 0.8481 |
| **$R^2$** | 0.1827 | -0.1031 | -0.0034 |

### 🔍 Conclusiones Académicas
1. **Comportamiento en Test:** El coeficiente de determinación negativo en el conjunto de prueba ($R^2 = -0.1031$) y el error absoluto ($MAE = 0.7175$) indican que el modelo no logra superar al estimador simple de la media con las variables disponibles.
2. **Diagnóstico:** Los factores que provocan lesiones en un accidente dependen en gran medida de variables no disponibles en el dataset (velocidad del impacto, tipo de vehículo, uso de elementos de protección).
3. **Tratamiento de Datos:** Se preservó el 100% de los registros reales de accidentes sin filtrar *outliers* de heridos elevados, ya que corresponden a situaciones reales de tráfico.

---

## 🔧 Instrucciones de Ejecución

### 1. Clonar el repositorio
```bash
git clone https://github.com/JuanAnd-coder/prediccion-heridos-accidentes-viales.git
cd prediccion-heridos-accidentes-viales
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Ejecutar el pipeline completo y el menú interactivo
```bash
python main.py
```

Al finalizar la ejecución, el script guardará todos los artefactos en la carpeta `resultados/` y abrirá un menú interactivo en consola para explorar las gráficas generadas.

---

## 👤 Autor

Desarrollado por **Juan Andrés Cárdenas** ([@JuanAnd-coder](https://github.com/JuanAnd-coder)).
