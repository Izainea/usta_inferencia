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

# **Introducción**

## **Contenidos del Curso**

Durante este semestre, abordaremos los siguientes temas clave en inferencia estadística:

1. **Distribuciones de probabilidad y su importancia en inferencia.**
   - Concepto de variable aleatoria.
   - Distribuciones discretas: Bernoulli, Binomial, Poisson.
   - Distribuciones continuas: Normal, Exponencial, Chi-cuadrado.
   - Propiedades de las distribuciones y su relevancia en inferencia.

2. **Distribuciones muestrales y su uso en estimación.**
   - Distribución de la media y varianza muestral.
   - Distribución t de Student y su aplicación en muestras pequeñas.
   - Distribución F y Chi-cuadrado en pruebas de varianza.
   - Aplicaciones prácticas en simulaciones y datos reales.

3. **Estimación puntual y por intervalos.**
   - Propiedades de los estimadores: insesgadez, eficiencia y consistencia.
   - Métodos de estimación: Máxima verosimilitud y Método de los momentos.
   - Construcción de intervalos de confianza para medias y proporciones.
   - Determinación del tamaño de muestra para un nivel de confianza dado.

4. **Pruebas de hipótesis y su aplicación en distintos contextos.**
   - Formulación de hipótesis nula y alternativa.
   - Errores tipo I y II, nivel de significancia y potencia de una prueba.
   - Pruebas para medias y varianzas en muestras grandes y pequeñas.
   - Pruebas no paramétricas y su aplicación cuando no se cumplen supuestos de normalidad.

5. **Modelos de regresión y correlación en inferencia.**
   - Regresión lineal simple y múltiple.
   - Supuestos del modelo de regresión y validación de los mismos.
   - Inferencia en regresión: intervalos de confianza y pruebas sobre coeficientes.
   - Coeficiente de correlación y análisis de correlación parcial.

6. **Métodos avanzados en inferencia estadística.**
   - Inferencia Bayesiana y comparación con métodos frecuentistas.
   - Métodos de remuestreo: Bootstrap y Jackknife.
   - Pruebas de bondad de ajuste y modelos de selección de hipótesis.
   - Aplicaciones avanzadas en aprendizaje automático y modelos probabilísticos.

## **Importancia de la Inferencia Estadística**

La inferencia estadística permite tomar decisiones fundamentadas con datos incompletos. Su aplicación es crucial en diversas disciplinas como economía, biomedicina y aprendizaje automático.

## **Metodología de Calificación**

La evaluación del curso se divide en tres cortes con la siguiente ponderación:

- **Primer corte:** 30%
- **Segundo corte:** 30%
- **Tercer corte:** 40%

Cada corte incluye un examen que representa la mitad del porcentaje del corte:

- **Exámenes:**
  - Primer corte: 15%
  - Segundo corte: 15%
  - Tercer corte: 20%

### **Fechas Importantes**
- **Finalización del Primer Corte:** 21 de marzo
- **Finalización del Segundo Corte:** 5 de mayo
- **Finalización del Tercer Corte:** 31 de mayo

---

# **Introducción a la Inferencia Estadística**



## **Objetivos de la Clase**

En esta sesión, exploraremos los fundamentos de la inferencia estadística, comprendiendo su importancia y aplicaciones en el análisis de datos. Al finalizar esta clase, serás capaz de:

- Diferenciar entre estadística descriptiva e inferencial.
- Entender el papel de las distribuciones de probabilidad en la inferencia.
- Aplicar conceptos clave como el Teorema del Límite Central.

---

## **¿Qué es la Inferencia Estadística?**

La inferencia estadística es el proceso de extraer conclusiones sobre una población a partir de una muestra de datos. Se basa en la teoría de la probabilidad y utiliza herramientas como estimación y pruebas de hipótesis.

```{admonition} Importancia de la Inferencia Estadística
:class: note
Permite tomar decisiones fundamentadas con datos incompletos, siendo clave en disciplinas como economía, biomedicina y aprendizaje automático.
```

---

## **Diferencia entre Estadística Descriptiva e Inferencial**

```{list-table}
:header-rows: 1
:widths: 30 30

* - Estadística Descriptiva
  - Estadística Inferencial
* - Resume y organiza datos.
  - Generaliza resultados a una población.
* - Usa medidas como media y desviación estándar.
  - Usa técnicas como intervalos de confianza y pruebas de hipótesis.
* - No permite hacer predicciones.
  - Se basa en modelos probabilísticos para realizar inferencias.
```

---

## **Distribuciones de Probabilidad en Inferencia**

Antes de profundizar en la inferencia, es fundamental entender las distribuciones de probabilidad más utilizadas:

### **Distribución Binomial**
- Modela el número de éxitos en una serie de ensayos independientes.
- Se define por los parámetros $n$ (número de ensayos) y $p$ (probabilidad de éxito en cada ensayo).
- Ejemplo: Número de veces que un cliente compra en una tienda durante 10 visitas.

```{math}
P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}
```

### **Distribución Normal**
- Es continua y simétrica, modela fenómenos naturales y errores de medición.
- Se define por su media $\mu$ y desviación estándar $\sigma$.
- El **Teorema del Límite Central** explica por qué la normal es crucial en la inferencia.

```{math}
f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}
```

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

x = np.linspace(-4, 4, 1000)
y = norm.pdf(x, loc=0, scale=1)

plt.figure(figsize=(8, 5))
plt.plot(x, y, color='blue', lw=2, label='Distribución Normal estándar')
plt.fill_between(x, y, alpha=0.2, color='blue')
plt.title('Distribución Normal')
plt.xlabel('Valores')
plt.ylabel('Densidad')
plt.legend()
plt.grid()
plt.show()
```

---

## **El Teorema del Límite Central**

El **Teorema del Límite Central (TLC)** establece que, para muestras grandes, la distribución de la media muestral se aproxima a una distribución normal, independientemente de la distribución original de los datos.

```{admonition} Importancia del TLC
:class: tip
Permite justificar el uso de métodos inferenciales basados en la normalidad, incluso cuando los datos originales no son normales.
```

**Ejemplo en Python:**

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

# Simulación del TLC con una distribución uniforme
np.random.seed(42)
muestras = [np.mean(np.random.uniform(0, 1, 30)) for _ in range(1000)]

plt.hist(muestras, bins=30, density=True, alpha=0.7, color='blue')
plt.title('Distribución de la media muestral (TLC)')
plt.xlabel('Media muestral')
plt.ylabel('Densidad')
plt.show()
```

---

## **Conclusión y Próximos Pasos**

En esta clase, revisamos los conceptos fundamentales de la inferencia estadística y la importancia de las distribuciones de probabilidad en el análisis de datos. En la próxima sesión, exploraremos en detalle las **distribuciones muestrales** y su papel en la estimación y pruebas de hipótesis.

```{admonition} Tarea
:class: warning
Reflexiona sobre la importancia del TLC en problemas reales. ¿Cómo crees que se aplica en modelos de predicción?
```