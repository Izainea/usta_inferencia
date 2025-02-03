---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# 📘 Introducción a los Modelos Lineales

## 🌟 Objetivos del cuaderno

En este cuaderno, exploraremos los fundamentos de los modelos lineales, su importancia en estadística y cómo pueden aplicarse para modelar relaciones entre variables. Además, configuraremos nuestro entorno de trabajo en **Python**, utilizando las bibliotecas `statsmodels` y `sklearn`.

## 📖 ¿Qué es un modelo lineal?

Un **modelo lineal** es una representación matemática que describe la relación entre una variable respuesta (dependiente) y una o más variables predictoras (independientes) a través de una combinación lineal de estas últimas. 

La ecuación general de un modelo de regresión lineal múltiple es:

```{math}
Y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p + \varepsilon
```

donde:
- $ Y $ es la variable dependiente.
- $ X_1, X_2, \dots, X_p $ son las variables independientes.
- $ \beta_0 $ es el intercepto.
- $ \beta_1, \beta_2, \dots, \beta_p $ son los coeficientes de regresión.
- $ \varepsilon $ es el término de error.

## 📌 Supuestos del Modelo Lineal

Para que un modelo de regresión lineal sea válido, se deben cumplir ciertos supuestos fundamentales:

1. **Linealidad**: La relación entre las variables explicativas y la variable respuesta es lineal.
2. **Independencia**: Los errores $ \varepsilon $ son independientes entre sí.
3. **Homoscedasticidad**: La varianza de los errores es constante en todos los niveles de las variables explicativas.
4. **Normalidad**: Los errores se distribuyen de manera normal.
5. **No multicolinealidad**: No existe una alta correlación entre las variables explicativas.

```{note}
Estos supuestos deben verificarse antes de interpretar los resultados de un modelo de regresión lineal. Existen pruebas estadísticas y visualizaciones que permiten diagnosticar si estos supuestos se cumplen.
```

## 🛠️ Configuración del Entorno

Antes de empezar con los análisis, debemos asegurarnos de tener instaladas las librerías necesarias. Para instalar las dependencias, ejecuta:

```python
!pip install numpy pandas matplotlib seaborn statsmodels scikit-learn
```

Luego, importamos las bibliotecas clave:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
```

## 📊 Exploración de Datos

Antes de construir un modelo, es esencial explorar los datos para comprender su estructura y distribución. Veamos un ejemplo con un dataset simulado:

```{code-cell} ipython3
# Generamos datos simulados
def generar_datos(n: int = 100):
    np.random.seed(42)
    X = np.random.rand(n, 1) * 10  # Variable independiente
    Y = 2.5 + 1.8 * X + np.random.randn(n, 1) * 2  # Variable dependiente con ruido
    return X, Y

X, Y = generar_datos()

# Convertimos a DataFrame
data = pd.DataFrame({'X': X.flatten(), 'Y': Y.flatten()})

# Visualización
plt.figure(figsize=(8, 5))
sns.scatterplot(x='X', y='Y', data=data)
plt.xlabel('Variable Independiente (X)')
plt.ylabel('Variable Dependiente (Y)')
plt.title('Relación entre X e Y')
plt.show()
```

## 📈 Ajuste de un Modelo de Regresión Lineal con `statsmodels`

Ahora, ajustemos un modelo de regresión lineal usando `statsmodels`:

```{code-cell} ipython3
# Agregamos una constante para el intercepto
X_sm = sm.add_constant(X)

# Ajustamos el modelo
modelo = sm.OLS(Y, X_sm).fit()

# Resumen del modelo
modelo.summary()
```

## 🚀 Conclusión

En este cuaderno hemos introducido los conceptos clave de los modelos lineales, configurado nuestro entorno de trabajo y ajustado un primer modelo de regresión simple. En los siguientes cuadernos, profundizaremos en técnicas avanzadas como diagnóstico del modelo, transformaciones y modelos más sofisticados.

```{note}
Recuerda que los supuestos del modelo deben verificarse antes de interpretar los resultados. En el próximo cuaderno, abordaremos cómo diagnosticar estos supuestos con técnicas gráficas y pruebas estadísticas.
