# Challenge Notes

## Competition

- **URL:** https://www.kaggle.com/competitions/cdaw-loan-approval-prediction-in-illinois/overview
- **Main question:** As a bank representative, should I grant a loan to a particular small business (Company X)? Why or why not?

## Project Strategy

Dividir el proyecto en fases operativas: preparación, análisis del dataset, feature engineering, desarrollo de modelos (geométricos / árboles / algoritmo secreto), optimización, competición Kaggle y documentación final.

Mantener tres líneas de trabajo en paralelo (geométrico, árboles, algoritmo secreto) con infraestructura común:

- dataset limpio
- features compartidas
- evaluación común
- pipeline de Kaggle

### Pipeline común

```text
raw data
  ↓
cleaning
  ↓
feature engineering
  ↓
dataset final
```

Todos los modelos deben usar las mismas features base.

- Centralizar experimentos, métricas y versionado para comparar modelos.
- Optimizar el algoritmo secreto como esfuerzo transversal (clave para el informe final).

### Estrategia Kaggle

- En Kaggle suele ganar: **feature engineering > algoritmo**.
- Muchos equipos pierden tiempo cambiando modelo sin mejorar variables.

### Preguntas clave

- ¿Qué variables explican mejor el riesgo de aprobar/denegar un crédito?
- ¿Hay clusters naturales de empresas?
- ¿El dataset está desbalanceado?
- ¿Un ensemble mejora el score?
- ¿Qué hiperparámetros afectan más al algoritmo secreto?

## Herramientas recomendadas

### Experimentos

- MLflow
- Weights & Biases
- Simple CSV logs

### Colaboración

- GitHub
- Notion / Google Docs

### Librerías

- scikit-learn
- XGBoost
- pandas
- seaborn

---

# Plan Extensivo del Challenge

## Fase 0 - Preparación del proyecto

- **Fecha:** _pendiente_

### Tareas

1. Infraestructura del proyecto
   - Crear repositorio Git.
   - Definir estructura recomendada.
2. Pipeline base
   - Carga de datos.
   - Limpieza básica.
   - Split train/validation.
   - Generación de CSV para Kaggle.
   - Pipeline reproducible de entrenamiento.
3. Primer envío a Kaggle
   - Modelo trivial (baseline).
   - Verificar formato de CSV.

### Colaboración

- Definir convenciones de:
  - métricas
  - naming
  - formato de features
  - estructura de notebook
- 1 persona (infraestructura) coordina:
  - repositorio
  - pipeline base
  - integración Kaggle

---

## Fase 1 - Conocimiento del dominio

- **Fecha:** 19 de marzo

### Tareas

1. Analizar dataset
   - variables
   - tipos de datos
   - valores faltantes
   - distribución
2. Hipótesis y preguntas clave
   - ¿Qué variables afectan al riesgo de crédito?
   - ¿Qué variables pueden ser ruido?
   - ¿Hay relaciones relevantes entre variables?
3. Crear documentación inicial en notebook con:
   - descripción de variables
   - hipótesis predictivas
   - top predictors (hipótesis)
   - variables problemáticas
   - ideas de feature engineering
   - problemas detectados

### Colaboración

- Reunión de 1-2h de brainstorming (todo el equipo).

---

## Fase 2 - Análisis exploratorio (EDA)

- **Fecha:** 20 de marzo

### Tareas

1. Estadísticas
   - correlaciones
   - distribución por clase
2. Visualizaciones
   - pairplots
   - PCA
   - scatter/clusters
3. Detección de problemas
   - outliers
   - imbalance
   - colinealidad

### Entregable

Notebook EDA con:

- visualizaciones
- conclusiones
- lista de features útiles
- decisión conjunta sobre:
  - feature set común
  - transformaciones
  - variables eliminadas

---

## División temporal del equipo

| Equipo                                | Trabajo                                     |
| ------------------------------------- | ------------------------------------------- |
| Geométrico                            | Análisis de separabilidad                   |
| Árboles                               | Importancia de variables                    |
| Algoritmo secreto (apoyo transversal) | Investigación y comportamiento con baseline |

---

## Fase 3 - Feature Engineering

- **Fecha:** del 21 al 28 de marzo (descanso para estudiar examen del 26)

Optimizar variables y/o crear mejores variables predictivas. Esto suele mejorar más que cambiar algoritmos.

### Tipos de features

1. Transformaciones
   - normalización
   - log transform
2. Combinaciones
   - ratios
   - interacciones
3. Reducción de dimensionalidad
   - PCA
   - feature selection

Todos prueban nuevas features.

---

## Fase 4 - Desarrollo de modelos

- **Fecha general:** del 30 de marzo al 1 de abril

El trabajo se divide en tres subequipos. Cada subequipo debe:

- entrenar modelos
- registrar resultados
- compartir métricas

Todos colaboran en:

- optimización del algoritmo secreto
- experimentación conjunta

### Equipo 1 - Algoritmos geométricos (3 personas)

Ejemplos: evaluar si los datos son linealmente separables o con kernel.

- SVM
- KNN
- Logistic Regression
- Linear classifier

Experimentos:

- normalización
- kernels SVM
- reducción de dimensionalidad

Métricas:

- accuracy
- F1
- cross-validation

### Equipo 2 - Árboles de decisión (3 personas)

Ejemplos: capturar interacciones no lineales.

- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Experimentos:

- depth
- number of trees
- feature subsampling

### Equipo completo - Algoritmo secreto

- **Fecha:** del 1 al 3 de abril

Este es el punto más importante del proyecto. El informe exige optimización detallada.

Trabajo necesario:

1. Implementación base
2. Hyperparameter tuning
3. Feature tuning
4. Posible ensemble

---

## Fase 5 - Optimización de modelos

- **Fecha:** del 2 al 4 de abril
- **Objetivo:** mejorar resultados

### Técnicas recomendadas

1. Hyperparameter tuning
   - Grid Search
   - Random Search
   - Bayesian optimization
2. Cross-validation
3. Feature selection
4. Ensembles

### Colaboración

Reuniones cortas y frecuentes con seguimiento:

| Modelo | Features usadas | Score local | Score Kaggle |
| ------ | --------------- | ----------: | -----------: |

---

## Fase 6 - Competición Kaggle

- **Fecha:** 5 de abril

### Pipeline

1. Entrenar modelo final
2. Generar predicciones
3. Crear CSV
4. Subir a Kaggle
5. Registrar score

**Nota:** Mantener registro de envíos. Evitar subir modelos aleatorios.

---

## Fase 7 - Documentación final

- **Fecha de redacción:** 6 de abril
- **Fecha de entrega:** 9 de abril

### Estructura recomendada

1. Coordinación del equipo
   - roles
   - flujo de trabajo
   - herramientas
2. Problemas encontrados
   - datos desbalanceados
   - overfitting
3. Técnicas de mejora
   - feature engineering
   - tuning
   - ensembles
4. Optimización del algoritmo secreto (sección más fuerte)
   - experimentos
   - parámetros
   - mejoras obtenidas
