import streamlit as st
import duckdb
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import shapiro
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="Práctica 4.2 - Regresión Lineal", layout="wide")

st.title("Práctica 4.2: Regresión Lineal y Correlación")
st.subheader("Predicción del Consumo Per Cápita de Agua Potable (CPC)")

# PASO 1: DUCKDB
con = duckdb.connect(database=':memory:')
con.execute("SELECT setseed(0.42);")

con.execute("""
CREATE TABLE agua_potable AS
WITH base AS (
    SELECT
        range AS residencia_id,
        ROUND(random() * 15 + 20, 2) AS temperatura_max,
        ROUND(random() * 40 + 50, 2) AS humedad_max,
        ROUND(random() * 3 + 2, 1)  AS personas_familia,
        ROUND(random() * 1.5 + 0.5, 2) AS consumo_bidon_pc,
        (random() * 10 - 5) AS ruido
    FROM range(1, 101)
)
SELECT
    residencia_id,
    temperatura_max,
    humedad_max,
    personas_familia,
    consumo_bidon_pc,
    ROUND(
        94.49 
        + (2.48 * temperatura_max) 
        - (0.72 * humedad_max) 
        - (5.10 * personas_familia) 
        + (14.10 * consumo_bidon_pc) 
        + ruido, 
        2
    ) AS cpc_consumo
FROM base;
""")

df_agua = con.execute("SELECT * FROM agua_potable").df()

# CÁLCULOS
variables = ['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc', 'cpc_consumo']
matriz_correlacion = df_agua[variables].corr(method='pearson')

X = df_agua[['temperatura_max', 'humedad_max', 'personas_familia', 'consumo_bidon_pc']]
Y = df_agua['cpc_consumo']
X_const = sm.add_constant(X)
modelo_ols = sm.OLS(Y, X_const).fit()

residuos = modelo_ols.resid
stat_shapiro, p_val_shapiro = shapiro(residuos)
stat_dw = durbin_watson(residuos)

y_pred = modelo_ols.predict(X_const)
r2 = r2_score(Y, y_pred)
rmse = np.sqrt(mean_squared_error(Y, y_pred))

# MOSTRAR MÉTRICAS EN INTERFAZ
col1, col2, col3 = st.columns(3)
col1.metric("Coeficiente R²", f"{r2 * 100:.2f}%")
col2.metric("RMSE", f"{rmse:.4f} lt/hab-día")
col3.metric("Shapiro-Wilk (p-val)", f"{p_val_shapiro:.4f}")

# PANEL GRÁFICO
st.write("### Panel de Diagnóstico Visual")
sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

sns.heatmap(matriz_correlacion, annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=axes[0, 0])
axes[0, 0].set_title('1. Matriz de Correlación de Pearson (r)')

sns.regplot(data=df_agua, x='humedad_max', y='cpc_consumo', color='b', ax=axes[0, 1], line_kws={'color': 'red', 'linewidth': 2})
axes[0, 1].set_title('2. Tendencia: Humedad Máx vs Consumo')

axes[0, 2].scatter(y_pred, Y, alpha=0.7, color='teal')
axes[0, 2].plot([Y.min(), Y.max()], [Y.min(), Y.max()], 'r--', lw=2)
axes[0, 2].set_title(f'3. Val. Reales vs Predichos (R² = {r2*100:.1f}%)')

axes[1, 0].scatter(y_pred, residuos, color='purple', alpha=0.6)
axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
axes[1, 0].set_title('4. Residuos vs Ajustados (Homocedasticidad)')

sm.qqplot(residuos, line='45', fit=True, ax=axes[1, 1])
axes[1, 1].set_title('5. Gráfico Q-Q Normal (Residuos)')

axes[1, 2].axis('off')
plt.tight_layout()

st.pyplot(fig)