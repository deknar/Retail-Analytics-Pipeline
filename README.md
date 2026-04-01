# Retail Analytics Pipeline - RetailMax

Este proyecto fue desarrollado para el Departamento de Analítica y Machine Learning de **RetailMax**. Tiene como objetivo crear un pipeline de procesos de Big Data y Machine Learning, procesando millones de transacciones, reseñas y actividades de navegación utilizando **Apache Spark**, **Spark SQL** y **Spark MLlib**.

## Objetivos del Proyecto

1. Procesar datos estructurados y no estructurados en entornos distribuidos.
2. Ejecutar transformaciones masivas de datos para generar métricas de negocio.
3. Entrenar y evaluar modelos de Machine Learning (supervisados y no supervisados) para analizar grandes volúmenes de datos usando Spark MLlib.

## Arquitectura y Flujo de Trabajo

El flujo de trabajo automatizado se compone de 5 lecciones, o etapas, abarcando desde la introducción al Big Data hasta la creación y evaluación robusta de modelos predictivos:

1. **Fundamentos de Big Data**: Ingesta de las transacciones (Orders), Items de orden y Reseñas. Se abordan las 5 V's y el soporte arquitectónico del flujo completo de principio a fin.
2. **Apache Spark y Configuración**: Configuración de `SparkSession` y prueba de lectura de los millones de transacciones alojadas de manera remota/local a través del pipeline.
3. **Manejo mediante RDDs**: Transformaciones profundas y funcionales aplicando métodos nativos RDD o equivalentes en DataFrames para agregaciones rápidas de clúster. 
4. **Spark SQL y DataFrames**: Uso exhaustivo de SQL Distribuido para producir agregaciones, identificar tendencias de ventas y optimizar flujos en formato de sistema columnares modernos (Parquet).
5. **Machine Learning Escalable (Spark MLlib)**: Entrenamiento de modelos logísticos de clasificación de satisfacción de usuarios y K-Means para agrupar clientes en base de patrones de consumo.

## Conjunto de Datos (Dataset)

Los datos originales utilizados en este proyecto provienen de la plataforma **Kaggle** y corresponden al dataset público [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

Dado el gran tamaño de los archivos, los datos crudos originales no se incluyen en este repositorio (se recomienda ignorarlos usando `.gitignore`). Para poder ejecutar el pipeline de forma local, es necesario descargar el dataset y ubicar los siguientes archivos CSV específicos dentro de la carpeta `datos/`:

* `olist_orders_dataset.csv`
* `olist_order_items_dataset.csv`
* `olist_order_reviews_dataset.csv`

## Estructura del repositorio

```text
├── main.py                     # Script unificado con el pipeline y lecciones completas.
├── Informe_Final.md            # Documento de reporte final documentando métricas y resultados detallados.
├── datos/                      # Carpeta con los conjuntos de datos en formato CSV (Olist Orders, Items, Reviews).
└── resultados/                 
    ├── graficos/               # Visualizaciones estáticas reportadas desde el pipeline (Distribución, clusters, ventas).
    └── parquet/                # Datos procesados y agregados exportados para máxima velocidad.
```

## Tecnologías Implementadas

* **Python** (Entorno de desarrollo y Data Science)
* **Apache Spark** (Framework clave de motor de procesamiento e inyección a Big Data)
* **PySpark** e interfaces **Spark SQL**
* **Spark MLlib** (Librerías estandarizadas de clustering VectorAssembler, LogisticRegression, KMeans)
* **Matplotlib** (Para la graficación programática automática)

## Instrucciones de Ejecución

Para iniciar el flujo a lo largo de las 5 clases modulares de Spark y Machine Learning, ejecutar:

```bash
python main.py
```

Al terminar su ejecución, podrás encontrar los gráficos, los datos procesados en formato moderno estructurado (parquet) y las métricas en consola.
