# -*- coding: utf-8 -*-
"""
================================================================================
 TAREA II - MODELOS DE REGRESION | REGRESION LINEAL (main.py)
 Dataset: Accidentes Viales (Datos Abiertos Colombia - https://datos.gov.co/)
 Archivo: Accidentes_Viales_20261009.csv (misma carpeta que este script)
 Objetivo: Predecir el numero de HERIDOS por accidente vial.

 VARIABLE OBJETIVO (y): 'Heridos' (verificada como apta en PASO 2j)
 VARIABLES PREDICTORAS (X):
   - Numericas: 'Vehiculos Involucrados', 'Mes', 'Dia_Mes', 'Es_Fin_Semana'
   - Categoricas: 'Clase de Accidente', 'Barrio' (completo), 'Dia_Semana'
 EXCLUIDAS (fuga de informacion: son CONSECUENCIA, no causa):
   - 'Muertes' y 'Accidente con' solo se usan en limpieza/exploracion.
   - 'Codigo_Accidente' y 'Direccion' son identificadores (no predicen).

 MODELO PRINCIPAL (unico): Regresion Lineal por minimos cuadrados.
 REFERENCIA OPCIONAL: DummyRegressor (predice la media) solo para comparar.

 REQUISITOS CUBIERTOS:
   1. Limpieza: duplicados, textos, tipos, fechas (sin suposiciones injustificadas)
   2. Encoding: OneHotEncoder(handle_unknown="ignore") DENTRO del Pipeline
   3. Escalado: StandardScaler DENTRO del Pipeline
   4. Imputacion: mediana (numericas) / mas frecuente (categoricas), en Pipeline
   5. Outliers: solo visualizacion; ningun accidente real se borra por infrecuente
   6. Division train/test ANTES de cualquier transformacion aprendida
   7. Evaluacion: MAE, RMSE, R2 (train/test) + CV 5 folds + comparacion con base
   8. Resultados guardados en carpeta "resultados" (CSVs, PNGs, resumen.txt)

 Uso:  python main.py
================================================================================
"""

# =============================================================================
# PASO 0. IMPORTS Y CONFIGURACION
# =============================================================================
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyRegressor  # solo referencia base (predice la media)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")

# Estilo global de graficos (limpio y legible para exposicion)
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110
plt.rcParams["axes.titlesize"] = 12
plt.rcParams["axes.labelsize"] = 10

RANDOM_STATE = 42          # Semilla para reproducibilidad
TEST_SIZE = 0.20           # 80% entreno / 20% prueba

# Rutas automaticas desde la carpeta del script (no rutas absolutas fijas)
CARPETA = Path(__file__).resolve().parent
RUTA_CSV = CARPETA / "Accidentes_Viales_20261009.csv"
CARPETA_RES = CARPETA / "resultados"
CARPETA_RES.mkdir(exist_ok=True)

print("=" * 78)
print(" MODELO DE REGRESION LINEAL - ACCIDENTES VIALES (Datos Abiertos Colombia)")
print("=" * 78)


def guardar_fig(nombre):
    """Guarda la figura activa en resultados/ como PNG (antes de mostrarla)."""
    plt.savefig(CARPETA_RES / nombre, bbox_inches="tight")


# =============================================================================
# PASO 1. CARGA DEL DATASET + COMPROBACION DE COLUMNAS
# =============================================================================
print("\n[PASO 1] Carga del dataset...")

# Columnas necesarias para el flujo (si falta alguna, mensaje claro y salida)
REQUERIDAS = ["Fecha_Ocurrencia", "Barrio", "Vehiculos Involucrados",
              "Heridos", "Clase de Accidente"]
OPCIONALES = ["Muertes", "Accidente con", "Codigo_Accidente", "Direccion"]

if not RUTA_CSV.exists():
    print(f"  ERROR: no se encontro el archivo CSV en:\n  {RUTA_CSV}")
    sys.exit(1)

df_raw = pd.read_csv(RUTA_CSV, encoding="utf-8")
print(f"  -> Filas: {df_raw.shape[0]} | Columnas: {df_raw.shape[1]}")

faltantes = [c for c in REQUERIDAS if c not in df_raw.columns]
if faltantes:
    print(f"  ERROR: faltan columnas requeridas en el CSV: {faltantes}")
    print(f"  Columnas encontradas: {list(df_raw.columns)}")
    sys.exit(1)
print("  -> Columnas requeridas presentes OK.")
for col in OPCIONALES:
    if col not in df_raw.columns:
        print(f"  Aviso: columna opcional '{col}' no encontrada; "
              f"se omitira su uso exploratorio.")

# Copia de trabajo (nunca se modifica df_raw, buena practica)
df = df_raw.copy()
TIENE_MUERTES = "Muertes" in df.columns
TIENE_ACC_CON = "Accidente con" in df.columns


# =============================================================================
# PASO 2. LIMPIEZA DE DATOS (solo operaciones deterministas, nada aprendido)
# -----------------------------------------------------------------------------
# Nada de lo que sigue "aprende" de los datos (sin medias, medianas ni top-N
# calculados aqui): por eso es seguro hacerlo antes de dividir en train/test.
# La imputacion de faltantes SI se aprende -> vive dentro del Pipeline (PASO 6).
# =============================================================================
print("\n[PASO 2] Limpieza de datos...")

# --- 2a. Reporte inicial de nulos ---
print("\n  [2a] Nulos por columna (antes):")
print(df.isnull().sum().to_string())

# --- 2b. Duplicados exactos ---
n_dup = df.duplicated().sum()
print(f"\n  [2b] Filas duplicadas exactas: {n_dup} (se eliminan: son registros repetidos)")
df = df.drop_duplicates().reset_index(drop=True)

# --- 2c. Normalizar texto de columnas categoricas ---
# Problema: la misma categoria viene en minuscula/MAYUSCULA ("Choque"/"CHOQUE",
# "Con Heridos"/"HERIDOS") y con tildes danadas ("Da�os"). Operacion por fila,
# determinista: no aprende nada del conjunto (sin fuga).
def normalizar_texto(s: pd.Series) -> pd.Series:
    """Mayusculas, sin espacios sobrantes, unifica caracteres danados."""
    s = s.astype("object")
    s = s.str.upper().str.strip()
    s = s.str.replace("�", "N", regex=False)
    s = s.str.replace(r"\s+", " ", regex=True)
    return s

for col in ["Barrio", "Accidente con", "Clase de Accidente", "Direccion"]:
    if col in df.columns:
        df[col] = normalizar_texto(df[col])

# Unificar "Accidente con" a 3 categorias canonicas (solo exploracion)
if TIENE_ACC_CON:
    mapa_accidente = {
        "HERIDOS": "HERIDOS",
        "CON HERIDOS": "HERIDOS",
        "MUERTOS": "MUERTOS",
        "CON MUERTOS": "MUERTOS",
        "DANOS MATERIALES": "DANOS MATERIALES",
        "DAÑOS MATERIALES": "DANOS MATERIALES",
        "SOLO DANOS": "DANOS MATERIALES",
        "SOLO DAÑOS": "DANOS MATERIALES",
        "SOLO DANOS MATERIALES": "DANOS MATERIALES",
    }
    df["Accidente con"] = df["Accidente con"].replace(mapa_accidente)

# Unificar "Clase de Accidente"
mapa_clase = {
    "CHOQUE": "CHOQUE",
    "ATROPELLO": "ATROPELLO",
    "OTRO": "OTRO",
    "VOLCAMIENTO": "VOLCAMIENTO",
}
df["Clase de Accidente"] = df["Clase de Accidente"].replace(mapa_clase)

# --- 2d. 'Muertes' (SOLO exploracion, NO entra al modelo) ---
# Viene como texto mixto ("0","1","NO APLICA"). NO se convierte "NO APLICA" a 0:
# el dataset no define su significado y asumirlo seria inventar datos. Como la
# columna no alimenta al modelo, se conserva como numerica con NaN (desconocido)
# unicamente para describirla y graficarla.
if TIENE_MUERTES:
    print("\n  [2d] Distribucion de 'Muertes' (texto original, solo exploracion):")
    print(("  " + df["Muertes"].astype(str).str.upper().str.strip()
           .value_counts(dropna=False).to_string()).replace("\n", "\n  "))
    df["Muertes_num"] = pd.to_numeric(
        df["Muertes"].astype(str).str.upper().str.strip()
        .replace({"NO APLICA": np.nan, "NAN": np.nan, "NONE": np.nan}),
        errors="coerce",
    )
    print("  -> 'NO APLICA' se deja como desconocido (NaN), NO como 0: "
          "no se justifica asumirlo y la columna no entra al modelo.")

# --- 2e. 'Heridos' (objetivo): solo se excluyen registros sin objetivo valido ---
df["Heridos"] = pd.to_numeric(df["Heridos"], errors="coerce")
n_sin_objetivo = int(df["Heridos"].isna().sum() + (df["Heridos"] < 0).sum())
print(f"\n  [2e] Registros sin objetivo valido (nulo o Heridos<0, imposible): "
      f"{n_sin_objetivo} (se excluyen: sin 'y' no se puede entrenar ni evaluar)")
df = df[df["Heridos"].notna() & (df["Heridos"] >= 0)].reset_index(drop=True)
# OJO: valores ALTOS de Heridos (4-5) NO se tocan: son accidentes reales
# poco frecuentes y se conservan integramente (ver PASO 4).

# --- 2f. 'Vehiculos Involucrados': imposibles -> NaN (imputa el Pipeline) ---
df["Vehiculos Involucrados"] = pd.to_numeric(df["Vehiculos Involucrados"], errors="coerce")
n_veh_inv = int((df["Vehiculos Involucrados"] < 1).sum())
print(f"  [2f] 'Vehiculos Involucrados' < 1 (imposible: todo accidente implica "
      f"al menos 1): {n_veh_inv} -> pasan a faltante y los imputa el Pipeline "
      f"(mediana del train). La fila se conserva.")
df.loc[df["Vehiculos Involucrados"] < 1, "Vehiculos Involucrados"] = np.nan

# --- 2g. 'Barrio': relleno CONSTANTE (no aprende nada -> sin fuga) ---
df["Barrio"] = df["Barrio"].replace({
    "NO REGISTRA": "SIN INFORMACION",
    "SIN INFORMACIÓN DEL BARRIO": "SIN INFORMACION",
    "SIN INFORMACION DEL BARRIO": "SIN INFORMACION",
    "NAN": "SIN INFORMACION",
    "NONE": "SIN INFORMACION",
    "": "SIN INFORMACION",
})
df["Barrio"] = df["Barrio"].fillna("SIN INFORMACION")
# NOTA: NO se agrupa en Top-N antes de dividir (eso aprenderia del conjunto
# completo). Se usa 'Barrio' completo con OneHotEncoder(handle_unknown="ignore").

# --- 2h. 'Fecha_Ocurrencia' -> variables temporales (determinista por fila) ---
# Formato origen: "2025 Jan 10 12:00:00 AM" (mes en ingles abreviado).
# Los faltantes por fecha invalida se dejan como NaN: los imputa el Pipeline
# (mediana / mas frecuente del train). Ninguna fecha invalida hace perder filas.
df["Fecha_Ocurrencia"] = pd.to_datetime(
    df["Fecha_Ocurrencia"], format="%Y %b %d %I:%M:%S %p", errors="coerce"
)
n_fechas_nat = int(df["Fecha_Ocurrencia"].isna().sum())
print(f"  [2h] Fechas no interpretables (NaT): {n_fechas_nat} "
      f"(NO se borran filas; sus derivados quedan faltantes para el Pipeline)")
df["Mes"] = df["Fecha_Ocurrencia"].dt.month
df["Dia_Mes"] = df["Fecha_Ocurrencia"].dt.day
df["Dia_Semana"] = df["Fecha_Ocurrencia"].dt.day_name()  # Monday..Sunday en ingles
semana = df["Fecha_Ocurrencia"].dt.weekday
df["Es_Fin_Semana"] = np.where(semana.isna(), np.nan, (semana >= 5).astype(float))

# --- 2i. Nulos tras limpieza (los de predictores los resuelve el Pipeline) ---
print("\n  [2i] Nulos por columna (despues; faltantes de predictores -> Pipeline):")
print(df.isnull().sum().to_string())

print("\n  Limpieza terminada. Muestra final:")
print(df[["Fecha_Ocurrencia", "Barrio", "Vehiculos Involucrados",
          "Heridos", "Clase de Accidente", "Mes", "Dia_Semana"]].head().to_string())

# --- 2j. COMPROBACION PREVIA de 'Heridos' como variable objetivo ---
print("\n  [2j] Comprobacion de 'Heridos' como variable objetivo:")
print(f"    - Tipo: {df['Heridos'].dtype} | Nulos: {df['Heridos'].isnull().sum()} | "
      f"Filas: {len(df)}")
print(f"    - Min={df['Heridos'].min():.0f} Max={df['Heridos'].max():.0f} "
      f"Media={df['Heridos'].mean():.2f} Desv={df['Heridos'].std():.2f}")
print("    - Distribucion de valores:")
print(("      " + df["Heridos"].value_counts().sort_index().to_string()).replace("\n", "\n      "))
n_ceros = int((df["Heridos"] == 0).sum())
print(f"    - Casos con 0 heridos: {n_ceros} ({n_ceros/len(df)*100:.1f}%) -> "
      f"hay ceros y valores >0: la variable TIENE variabilidad.")
print("    - VEREDICTO: 'Heridos' es numerica, completa y con rango acotado: "
      "APTA como objetivo de regresion.")
print("    - EXCLUIDAS del modelo (revelan la consecuencia -> fuga): "
      "['Muertes', 'Accidente con']. Excluidos por ser IDs: "
      "['Codigo_Accidente', 'Direccion'].")


# =============================================================================
# PASO 3. ANALISIS EXPLORATORIO (EDA) - FUNCIONES DE GRAFICOS
# -----------------------------------------------------------------------------
# Cada grafico es una funcion: se llama desde el MENU del PASO 8. Todas las
# figuras se guardan en resultados/ (PNG) antes de mostrarse.
# df_eda es solo para explorar (explorar no ajusta nada -> sin fuga).
# =============================================================================
print("\n[PASO 3] EDA preparado (los graficos se ven desde el menu final).")
df_eda = df.copy()  # foto de los datos limpios para explorar

def graf_1_distribucion_heridos():
    """1. Distribucion de la variable objetivo 'Heridos'."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    orden = sorted(df_eda["Heridos"].unique())
    sns.countplot(data=df_eda, x="Heridos", order=orden, palette="Blues_d", ax=ax)
    ax.set_title("Distribución del número de HERIDOS por accidente")
    ax.set_xlabel("N.º de heridos (variable objetivo)")
    ax.set_ylabel("N.º de accidentes")
    for p in ax.patches:
        ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()),
                    ha="center", va="bottom", fontsize=9)
    plt.tight_layout(); guardar_fig("01_distribucion_heridos.png"); plt.show()

def graf_2_heridos_por_clase():
    """2. Heridos segun Clase de Accidente."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.boxplot(data=df_eda, x="Clase de Accidente", y="Heridos", palette="Set2", ax=ax)
    ax.set_title("Heridos según CLASE de accidente")
    ax.set_xlabel("Clase de accidente"); ax.set_ylabel("Heridos")
    plt.xticks(rotation=15); plt.tight_layout()
    guardar_fig("02_heridos_por_clase.png"); plt.show()

def graf_3_vehiculos_vs_heridos():
    """3. Vehiculos involucrados vs Heridos (medias con matplotlib)."""
    fig, ax = plt.subplots(figsize=(7, 4.2))
    sns.stripplot(data=df_eda, x="Vehiculos Involucrados", y="Heridos",
                  jitter=0.25, alpha=0.5, palette="deep", ax=ax)
    # Medias por grupo calculadas con pandas y dibujadas con matplotlib
    # (se evita sns.pointplot: su parametro join esta obsoleto en seaborn 0.13+).
    medias = df_eda.groupby("Vehiculos Involucrados")["Heridos"].mean()
    ax.scatter(medias.index, medias.values, color="red", marker="D", s=80,
               label="Media", zorder=3, edgecolors="k", linewidths=0.5)
    ax.set_title("¿Más vehículos implicados = más heridos?")
    ax.set_xlabel("Vehículos involucrados"); ax.set_ylabel("Heridos")
    ax.legend(); plt.tight_layout()
    guardar_fig("03_vehiculos_vs_heridos.png"); plt.show()

def graf_4_accidentes_por_mes():
    """4. Accidentes por mes (estacionalidad)."""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    por_mes = df_eda["Mes"].value_counts().sort_index()
    sns.barplot(x=por_mes.index.astype(int), y=por_mes.values, palette="viridis", ax=ax)
    ax.set_title("N.º de accidentes por MES")
    ax.set_xlabel("Mes (1=Ene ... 12=Dic)"); ax.set_ylabel("N.º de accidentes")
    plt.tight_layout(); guardar_fig("04_accidentes_por_mes.png"); plt.show()


# =============================================================================
# PASO 4. VALORES ATIPICOS: VISUALIZACION SIN ELIMINAR REGISTROS REALES
# -----------------------------------------------------------------------------
# Regla IQR para DETECTAR (Q1, Q3, limites [Q1-1.5*IQR, Q3+1.5*IQR]) + boxplots.
# Decision: NO se borra ni se recorta ningun registro por tener Heridos o
# Muertes altos: son accidentes reales aunque sean poco frecuentes. Los unicos
# valores corregidos fueron los IMPOSIBLES del PASO 2 (Heridos<0, Vehiculos<1),
# con justificacion explicita. No hay transformacion adicional que ajustar.
# =============================================================================
print("\n[PASO 4] Deteccion de atipicos (IQR, solo visualizacion)...")

NUMERICAS_BOX = ["Vehiculos Involucrados", "Heridos", "Mes"]

def limites_iqr(serie: pd.Series, k: float = 1.5):
    """Calcula (li, ls). Si no hay dispersion devuelve (None, None)."""
    q1, q3 = serie.quantile(0.25), serie.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0 or pd.isna(iqr):
        print(f"    {serie.name}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} -> sin dispersion")
        return None, None
    li, ls = q1 - k * iqr, q3 + k * iqr
    n_fuera = int(((serie < li) | (serie > ls)).sum())
    print(f"    {serie.name}: Q1={q1:.2f} Q3={q3:.2f} IQR={iqr:.2f} -> limites "
          f"[{li:.2f}, {ls:.2f}] ({n_fuera} valores fuera; SE CONSERVAN)")
    return li, ls

print("  Limites IQR (informativos, no se filtra nada):")
for col in NUMERICAS_BOX:
    limites_iqr(df[col])
print(f"  Filas conservadas: {len(df)} (0 eliminadas por atipicos)")

def graf_5_boxplots():
    """5. Boxplots para visualizar valores atipicos (sin filtrar)."""
    cols = [c for c in NUMERICAS_BOX if c in df.columns]
    fig, axes = plt.subplots(1, len(cols), figsize=(4 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.boxplot(data=df, y=col, color="skyblue", ax=ax)
        ax.set_title(f"{col}")
    plt.suptitle("Detección de atípicos (visualización; registros conservados)", fontsize=13)
    plt.tight_layout(); guardar_fig("05_boxplots_atipicos.png"); plt.show()

def graf_6_barrios_y_muertes():
    """6. Top-10 barrios + distribucion de Muertes (ambos solo exploratorios)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    top10 = df_eda["Barrio"].value_counts().head(10)
    sns.barplot(x=top10.values, y=top10.index, palette="viridis", ax=axes[0])
    axes[0].set_title("Top 10 barrios con más accidentes (exploratorio)")
    axes[0].set_xlabel("N.º de accidentes"); axes[0].set_ylabel("Barrio")
    if TIENE_MUERTES and df_eda["Muertes_num"].notna().any():
        m = df_eda["Muertes_num"].dropna().astype(int)
        sns.countplot(x=m, order=sorted(m.unique()), palette="Reds_d", ax=axes[1])
        axes[1].set_title("Distribución de MUERTES (solo exploratorio)")
        axes[1].set_xlabel("Muertes por accidente ('NO APLICA' excluido)")
        axes[1].set_ylabel("N.º de accidentes")
    else:
        axes[1].text(0.5, 0.5, "Columna 'Muertes' no disponible", ha="center")
        axes[1].set_title("Muertes (no disponible)")
    plt.tight_layout(); guardar_fig("06_barrios_y_muertes.png"); plt.show()


# =============================================================================
# PASO 5. VARIABLES DEL MODELO + DIVISION TRAIN/TEST (ANTES DE LO APRENDIDO)
# -----------------------------------------------------------------------------
# La division se hace AHORA: todo lo que aprenda de los datos (imputadores,
# scaler, one-hot y el propio modelo) se ajusta SOLO con train. 'Muertes',
# 'Accidente con', IDs y el objetivo quedan fuera de X (ver PASO 2j).
# =============================================================================
print("\n[PASO 5] Variables del modelo y division train/test...")

TARGET = "Heridos"   # <-- variable a predecir (verificada en PASO 2j)
FEATURES_NUM = ["Vehiculos Involucrados", "Mes", "Dia_Mes", "Es_Fin_Semana"]
FEATURES_CAT = ["Clase de Accidente", "Barrio", "Dia_Semana"]

X = df[FEATURES_NUM + FEATURES_CAT]
y = df[TARGET].astype(float)

print(f"  Objetivo (y): {TARGET}")
print(f"  Numericas (imputacion mediana + escalado): {FEATURES_NUM}")
print(f"  Categoricas (imputacion moda + one-hot): {FEATURES_CAT}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)
print(f"  Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
print(f"  Media de Heridos en train: {y_train.mean():.2f} "
      f"(referencia de la base) | en test: {y_test.mean():.2f}")


# =============================================================================
# PASO 6. PIPELINE: IMPUTACION + ESCALADO + ENCODING + REGRESION LINEAL
# -----------------------------------------------------------------------------
# Todo lo aprendido vive DENTRO del Pipeline (requisito antifuga):
#   - Numericas: SimpleImputer(mediana del train) + StandardScaler del train
#   - Categoricas: SimpleImputer(mas frecuente del train) + OneHotEncoder
#     (handle_unknown="ignore"; salida dispersa por defecto, sin densificar)
# Modelo principal (UNICO): LinearRegression por minimos cuadrados.
# Referencia opcional: DummyRegressor(media) con el MISMO preprocesador.
# La validacion cruzada (PASO 7) usa el pipeline COMPLETO: cada fold ajusta
# su propio preprocesamiento (sin fuga entre particiones).
# =============================================================================
print("\n[PASO 6] Entrenando Regresion Lineal (Pipeline)...")

preprocesador = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ]), FEATURES_NUM),
        ("cat", Pipeline(steps=[
            ("imputar", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), FEATURES_CAT),
    ]
)

modelo = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("regresion", LinearRegression()),
])

modelo.fit(X_train, y_train)
print("  -> Modelo entrenado OK.")

# Predictor base: siempre predice la media de heridos del train
base = Pipeline(steps=[
    ("preprocesamiento", preprocesador),
    ("media", DummyRegressor(strategy="mean")),
])
base.fit(X_train, y_train)
print("  -> Predictor base (media) ajustado OK (solo referencia).")

# Nombres de columnas tras el one-hot (para interpretar coeficientes)
nombres_features = modelo.named_steps["preprocesamiento"].get_feature_names_out()
print(f"  -> N. de features tras encoding: {len(nombres_features)}")


# =============================================================================
# PASO 7. EVALUACION: MAE, RMSE, R2 (TRAIN/TEST) + CV + COMPARACION CON BASE
# -----------------------------------------------------------------------------
# MAE: error absoluto promedio en numero de heridos. RMSE: penaliza mas los
# errores grandes. R2: mejora frente a predecir la media (puede ser negativo).
# Una diferencia grande train vs test sugiere sobreajuste.
# =============================================================================
print("\n[PASO 7] Evaluacion (modelo vs base)...")

y_pred_train = modelo.predict(X_train)
y_pred_test = modelo.predict(X_test)
y_base_test = base.predict(X_test)

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)
rmse_train = rmse(y_train, y_pred_train)
rmse_test = rmse(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
mae_base = mean_absolute_error(y_test, y_base_test)
rmse_base = rmse(y_test, y_base_test)
r2_base = r2_score(y_test, y_base_test)
cv_r2 = cross_val_score(modelo, X, y, cv=5, scoring="r2")

print("\n  ============ RESULTADOS (TEST) ============")
print("  Metrica    Lineal     Base  (base = predecir la media)")
print(f"  MAE        {mae_test:.4f}   {mae_base:.4f}")
print(f"  RMSE       {rmse_test:.4f}   {rmse_base:.4f}")
print(f"  R2         {r2_test:.4f}   {r2_base:.4f}")
print(f"  (train: MAE={mae_train:.4f} RMSE={rmse_train:.4f} R2={r2_train:.4f})")
print(f"  R2 CV (5 folds, pipeline completo): {cv_r2.mean():.4f} +- {cv_r2.std():.4f}")
print("  ===========================================")

# --- Lectura basada SOLO en los numeros calculados (objetiva y academica) ---
# MAE: error absoluto medio en numero de heridos (promedio, no intervalo
# garantizado). RMSE: penaliza mas los errores grandes. R2: mejora frente a
# predecir la media; si es negativo, el modelo rinde PEOR que esa referencia.
dif_mae_pct = (mae_base - mae_test) / mae_base * 100 if mae_base else 0.0
if dif_mae_pct > 0:
    texto_mae_base = f"supera a la base en un {dif_mae_pct:.1f}%"
elif dif_mae_pct < 0:
    texto_mae_base = f"queda un {-dif_mae_pct:.1f}% por debajo de la base (no la mejora)"
else:
    texto_mae_base = "iguala a la base"
dif_rmse_pct = (rmse_base - rmse_test) / rmse_base * 100 if rmse_base else 0.0
if dif_rmse_pct > 0:
    texto_rmse_base = f"mejor que la base en un {dif_rmse_pct:.1f}%"
elif dif_rmse_pct < 0:
    texto_rmse_base = f"peor que la base en un {-dif_rmse_pct:.1f}%"
else:
    texto_rmse_base = "igual que la base"
if r2_test < 0:
    veredicto_r2 = (f"R2 test {r2_test:.4f} (negativo): rendimiento INFERIOR al de "
                    f"un predictor que simplemente estima la media de la variable "
                    f"objetivo. El modelo presenta limitaciones claras con estas "
                    f"variables.")
elif r2_test > r2_base:
    veredicto_r2 = (f"R2 test {r2_test:.4f} > R2 base {r2_base:.4f}: el modelo explica "
                    f"el {r2_test*100:.1f}% de la varianza de Heridos en test. "
                    f"Calificarlo como 'bueno' exige ademas valorar MAE, RMSE y CV.")
else:
    veredicto_r2 = (f"R2 test {r2_test:.4f} <= R2 base {r2_base:.4f}: el modelo NO "
                    f"supera a la referencia en test; su aporte es nulo con estas "
                    f"variables.")
print("\n  LECTURA BASADA EN RESULTADOS:")
print(f"  - MAE test {mae_test:.2f}: error absoluto medio de {mae_test:.2f} heridos "
      f"por prediccion; la base obtiene {mae_base:.2f}. En MAE el modelo "
      f"{texto_mae_base}.")
print(f"  - RMSE test {rmse_test:.2f} frente a {rmse_base:.2f} de la base "
      f"(penaliza mas los errores grandes): el modelo resulta {texto_rmse_base}.")
print(f"  - {veredicto_r2}")
if abs(r2_train - r2_test) < 0.10:
    print("  - R2 train y test cercanos: no hay senal de sobreajuste fuerte.")
else:
    print("  - R2 train y test distantes: hay indicios de sobreajuste o inestabilidad.")
print(f"  - Media de heridos en test: {y_test.mean():.2f} -> MAE {mae_test:.2f} "
      f"equivale a ~{mae_test/y_test.mean()*100:.0f}% de la media.")

# --- Coeficientes (numericas estandarizadas + dummies 0/1) ---
coefs = pd.Series(
    modelo.named_steps["regresion"].coef_, index=nombres_features
).sort_values(key=abs, ascending=False).head(15)
intercepto = float(modelo.named_steps["regresion"].intercept_)
residuos_test = y_test - y_pred_test

print("\n  Top coeficientes (asociacion aprendida, no causalidad):")
print(coefs.to_string())
print(f"\n  Intercepto: {intercepto:.4f} (prediccion base con las numericas en su "
      f"media de train; NO se lee como 'todas las variables en cero' porque las "
      f"categoricas usan dummies).")
print(f"  Residuo medio (test): {residuos_test.mean():+.4f} (ideal aprox 0) | "
      f"Desv. residuos: {residuos_test.std():.4f}")

# --- Ejemplo de prediccion concreta ---
ejemplo = X_test.iloc[[0]]
print("\n  Ejemplo de prediccion:")
print(f"  Datos: {ejemplo.to_dict(orient='records')[0]}")
print(f"  Heridos reales: {y_test.iloc[0]:.0f} | Predichos: {modelo.predict(ejemplo)[0]:.2f}")

# --- Funciones de graficos de resultados (menu + PNG) ---
def graf_7_correlacion():
    """7. Matriz de correlacion (numericas del modelo vs HERIDOS)."""
    cols = [c for c in FEATURES_NUM + [TARGET] if c in df.columns]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(df[cols].corr(numeric_only=True), annot=True, fmt=".2f",
                cmap="coolwarm", vmin=-1, vmax=1, square=True, ax=ax)
    ax.set_title("Correlación numéricas ↔ HERIDOS (exploratorio)")
    plt.tight_layout(); guardar_fig("07_correlacion.png"); plt.show()

def graf_8_real_vs_predicho():
    """8. Real vs Predicho train/test (grafica principal del modelo)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, yt, yp, titulo in [
        (axes[0], y_train, y_pred_train, "ENTRENAMIENTO"),
        (axes[1], y_test, y_pred_test, "PRUEBA"),
    ]:
        ax.scatter(yt, yp, alpha=0.55, edgecolors="k", linewidths=0.4)
        mn, mx = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([mn, mx], [mn, mx], "r--", lw=2, label="Ideal (y = y_pred)")
        ax.set_title(f"Real vs Predicho - {titulo}")
        ax.set_xlabel("Heridos reales"); ax.set_ylabel("Heridos predichos")
        ax.legend()
    plt.suptitle("¿Qué tan cerca predice? (puntos sobre la diagonal = acierto)",
                 fontsize=12)
    plt.tight_layout(); guardar_fig("08_real_vs_predicho.png"); plt.show()

def graf_9_residuos():
    """9. Residuos vs Predicho (nube sin patron = supuesto lineal razonable)."""
    residuos = y_test - y_pred_test
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(y_pred_test, residuos, alpha=0.6, edgecolors="k", linewidths=0.4)
    ax.axhline(0, color="red", ls="--", lw=2)
    ax.set_title("Residuos vs Predicho (test)")
    ax.set_xlabel("Heridos predichos"); ax.set_ylabel("Residuo (real - predicho)")
    plt.tight_layout(); guardar_fig("09_residuos.png"); plt.show()

def graf_10_coeficientes():
    """10. Top 15 coeficientes de la regresion lineal."""
    fig, ax = plt.subplots(figsize=(8.5, 5))
    coefs_plot = coefs.sort_values()
    coefs_plot.plot(kind="barh",
                    color=["crimson" if v < 0 else "steelblue" for v in coefs_plot.values],
                    ax=ax)
    ax.set_title("Top 15 coeficientes de la Regresión Lineal")
    ax.set_xlabel("Coeficiente (numéricas estandarizadas; dummies 0/1)")
    plt.tight_layout(); guardar_fig("10_coeficientes.png"); plt.show()

def graf_11_modelo_vs_base():
    """11. Modelo vs Base: MAE/RMSE (menor mejor) y R2 (mayor mejor)."""
    comp = pd.DataFrame(
        {"Lineal": [mae_test, rmse_test, r2_test],
         "Base (media)": [mae_base, rmse_base, r2_base]},
        index=["MAE (menor mejor)", "RMSE (menor mejor)", "R2 (mayor mejor)"],
    )
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, metrica in zip(axes, comp.index):
        comp.loc[metrica].plot(kind="bar", color=["steelblue", "gray"], ax=ax)
        ax.set_title(metrica)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    plt.suptitle("Lineal vs Base: el modelo debe ganar en las 3 barras", fontsize=12)
    plt.tight_layout(); guardar_fig("11_modelo_vs_base.png"); plt.show()


# =============================================================================
# PASO 7b. GUARDAR RESULTADOS EN carpeta resultados/
# -----------------------------------------------------------------------------
# metricas_modelo.csv, predicciones_test.csv y resumen.txt. Los PNG se guardan
# al mostrar cada grafico (ver guardar_fig en cada funcion graf_*).
# =============================================================================
print("\n[PASO 7b] Guardando resultados en carpeta 'resultados'...")

metricas_df = pd.DataFrame([
    {"metrica": "MAE", "conjunto": "entrenamiento", "valor": mae_train},
    {"metrica": "MAE", "conjunto": "prueba", "valor": mae_test},
    {"metrica": "MAE_base", "conjunto": "prueba", "valor": mae_base},
    {"metrica": "RMSE", "conjunto": "entrenamiento", "valor": rmse_train},
    {"metrica": "RMSE", "conjunto": "prueba", "valor": rmse_test},
    {"metrica": "RMSE_base", "conjunto": "prueba", "valor": rmse_base},
    {"metrica": "R2", "conjunto": "entrenamiento", "valor": r2_train},
    {"metrica": "R2", "conjunto": "prueba", "valor": r2_test},
    {"metrica": "R2_base", "conjunto": "prueba", "valor": r2_base},
    {"metrica": "R2_CV_media", "conjunto": "validacion_cruzada_5folds", "valor": cv_r2.mean()},
    {"metrica": "R2_CV_std", "conjunto": "validacion_cruzada_5folds", "valor": cv_r2.std()},
])
metricas_df.to_csv(CARPETA_RES / "metricas_modelo.csv", index=False)

pred_df = pd.DataFrame({
    "real": y_test.values,
    "predicho": y_pred_test,
    "residuo": (y_test - y_pred_test).values,
})
pred_df.to_csv(CARPETA_RES / "predicciones_test.csv", index=False)
print(f"  -> metricas_modelo.csv ({len(metricas_df)} filas) y "
      f"predicciones_test.csv ({len(pred_df)} filas) guardados.")

lineas_resumen = [
    "RESUMEN DE RESULTADOS - Regresion Lineal de Heridos (accidentes viales)",
    f"Filas utiles: {len(df)} | Train: {len(X_train)} | Test: {len(X_test)}",
    f"MAE train={mae_train:.4f} | MAE test={mae_test:.4f} | MAE base={mae_base:.4f}",
    f"RMSE train={rmse_train:.4f} | RMSE test={rmse_test:.4f} | RMSE base={rmse_base:.4f}",
    f"R2 train={r2_train:.4f} | R2 test={r2_test:.4f} | R2 base={r2_base:.4f}",
    f"R2 CV 5 folds: media={cv_r2.mean():.4f} std={cv_r2.std():.4f}",
    f"MAE vs base: el modelo {texto_mae_base}. RMSE vs base: {texto_rmse_base}.",
    f"Mayor peso: '{coefs.index[0]}' ({coefs.iloc[0]:+.3f}); "
    f"segundo: '{coefs.index[1]}' ({coefs.iloc[1]:+.3f})",
    f"Registros eliminados por atipicos: 0 (casos reales conservados).",
    f"Conclusion: {veredicto_r2}",
]
(CARPETA_RES / "resumen.txt").write_text("\n".join(lineas_resumen), encoding="utf-8")
print("  -> resumen.txt guardado (con conclusiones calculadas).")


# =============================================================================
# PASO 8. MENU INTERACTIVO DE GRAFICOS (para la exposicion)
# -----------------------------------------------------------------------------
# Escribe el numero y pulsa Enter. Puedes ver un grafico, cerrarlo y elegir
# otro sin reiniciar el programa.
# =============================================================================
def mostrar_menu():
    print("\n" + "=" * 78)
    print(" MENU DE GRAFICOS - Elige cual ver (puedes cambiar entre ellos)")
    print("=" * 78)
    print("  EDA (datos):")
    print("   1. Distribucion de HERIDOS (variable objetivo)")
    print("   2. Heridos segun CLASE de accidente")
    print("   3. Vehiculos involucrados vs Heridos")
    print("   4. Accidentes por MES")
    print("  ATIPICOS / EXPLORACION:")
    print("   5. Boxplots de atipicos (registros conservados)")
    print("   6. Top barrios + Muertes (exploratorio)")
    print("  MODELO:")
    print("   7. Matriz de correlacion")
    print("   8. Real vs Predicho (train/test)  <-- grafica principal")
    print("   9. Residuos vs Predicho")
    print("  10. Top 15 coeficientes del modelo")
    print("  11. Modelo vs Base (comparativa)")
    print("  12. Ver TODOS seguidos")
    print("   0. Salir")
    print("=" * 78)

# Diccionario numero -> (funcion, descripcion corta para confirmar)
OPCIONES = {
    "1": (graf_1_distribucion_heridos, "Distribucion de HERIDOS"),
    "2": (graf_2_heridos_por_clase, "Heridos por CLASE"),
    "3": (graf_3_vehiculos_vs_heridos, "Vehiculos vs Heridos"),
    "4": (graf_4_accidentes_por_mes, "Accidentes por MES"),
    "5": (graf_5_boxplots, "Boxplots atipicos"),
    "6": (graf_6_barrios_y_muertes, "Barrios + Muertes"),
    "7": (graf_7_correlacion, "Correlacion"),
    "8": (graf_8_real_vs_predicho, "Real vs Predicho"),
    "9": (graf_9_residuos, "Residuos"),
    "10": (graf_10_coeficientes, "Coeficientes"),
    "11": (graf_11_modelo_vs_base, "Modelo vs Base"),
}

while True:
    mostrar_menu()
    try:
        eleccion = input("  Tu eleccion (0-12): ").strip()
    except EOFError:
        print("\n  Entrada terminada (EOF). Saliendo del menu.")
        break
    if eleccion == "0":
        print("\n  Saliendo del menu. Exito en tu exposicion!")
        break
    elif eleccion == "12":
        print("\n  Mostrando TODOS los graficos seguidos (cierra cada uno para ver el siguiente)...")
        for num in ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]:
            print(f"  ... grafico {num}: {OPCIONES[num][1]}")
            OPCIONES[num][0]()
    elif eleccion in OPCIONES:
        print(f"\n  Mostrando grafico {eleccion}: {OPCIONES[eleccion][1]} "
              f"(cierra la ventana para volver al menu)...")
        OPCIONES[eleccion][0]()
    else:
        print("  Opcion no valida. Escribe un numero del 0 al 12.")

print("\n" + "=" * 78)
print(" CONCLUSIONES (generadas desde los resultados obtenidos):")
print(f" 1. Con estas variables, la Regresion Lineal logra MAE={mae_test:.2f}, "
      f"RMSE={rmse_test:.2f} y R2={r2_test:.4f} en test, frente a la base "
      f"(MAE={mae_base:.2f}, RMSE={rmse_base:.2f}, R2={r2_base:.4f}).")
print(f" 2. La variable con mayor peso aprendido fue '{coefs.index[0]}' "
      f"({coefs.iloc[0]:+.3f}), seguida de '{coefs.index[1]}' ({coefs.iloc[1]:+.3f}).")
print(f" 3. Se conservaron las {len(df)} filas utiles: ningun accidente real fue "
      f"eliminado por ser poco frecuente (atipicos solo visualizados).")
print(f" 4. Valoracion: {veredicto_r2}")
print("=" * 78)
