import os
import sys

# Forzar salida UTF-8 en Windows
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    if hasattr(sys.stderr, 'reconfigure'):
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Configurar JAVA_HOME antes de importar PySpark
JAVA_HOME = r"C:\Program Files\Microsoft\jdk-17.0.18.8-hotspot"
os.environ["JAVA_HOME"] = JAVA_HOME
os.environ["PATH"] = os.path.join(JAVA_HOME, "bin") + ";" + os.environ["PATH"]

# FIX: Configurar HADOOP_HOME en Windows
HADOOP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hadoop")
os.environ["HADOOP_HOME"] = HADOOP_DIR
os.environ["PATH"] = os.path.join(HADOOP_DIR, "bin") + ";" + os.environ["PATH"]

# Configurar Python para los workers de Spark
PYTHON_EXE = sys.executable
os.environ["PYSPARK_PYTHON"] = PYTHON_EXE
os.environ["PYSPARK_DRIVER_PYTHON"] = PYTHON_EXE

from pyspark.sql import SparkSession, Row
from pyspark.sql.types import (StructType, StructField, StringType,
                                IntegerType, FloatType, DoubleType, LongType)
from pyspark.sql.functions import (col, count, sum as spark_sum, avg, month, year,
                                    when, round as spark_round, desc, lit,
                                    datediff, to_timestamp, countDistinct, stddev,
                                    min as spark_min, max as spark_max, substring,
                                    expr, length)
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import (BinaryClassificationEvaluator,
                                    MulticlassClassificationEvaluator,
                                    ClusteringEvaluator)
from pyspark.ml import Pipeline
from pyspark.storagelevel import StorageLevel

# Rutas del proyecto
RUTA_BASE = os.path.dirname(os.path.abspath(__file__))
RUTA_DATOS = os.path.join(RUTA_BASE, "datos")
RUTA_RESULTADOS = os.path.join(RUTA_BASE, "resultados")
RUTA_PARQUET = os.path.join(RUTA_RESULTADOS, "parquet")
RUTA_GRAFICOS = os.path.join(RUTA_RESULTADOS, "graficos")

# Crear directorios de salida
os.makedirs(RUTA_PARQUET, exist_ok=True)
os.makedirs(RUTA_GRAFICOS, exist_ok=True)

def separador(titulo):
    print(f"  {titulo}")
    
# LECCION 1: FUNDAMENTOS DE BIG DATA
def leccion_1():
    separador("LECCION 1: FUNDAMENTOS DE BIG DATA")

    # 1.1 Las 5V de Big Data aplicadas a RetailMax
    print("1.1 LAS 5V DE BIG DATA APLICADAS A RETAILMAX")
    cinco_v = {
        "Volumen": "RetailMax genera millones de transacciones diarias en su plataforma "
                   "e-commerce. Los datasets del proyecto contienen ~99K ordenes, ~112K "
                   "items vendidos y ~104K resenas, un volumen que justifica el uso de "
                   "procesamiento distribuido con Apache Spark.",
        "Velocidad": "Las transacciones se generan en tiempo real: compras, envios y resenas "
                     "llegan continuamente. El pipeline debe ingestar y procesar estos flujos "
                     "con baja latencia para producir insights accionables para marketing.",
        "Variedad": "Los datos provienen de multiples fuentes y formatos: transacciones "
                    "estructuradas (ordenes, items con precios y fletes), datos semi-"
                    "estructurados (resenas con texto libre y calificaciones) y potenciales "
                    "logs de navegacion del e-commerce.",
        "Veracidad": "Las resenas pueden contener sesgos, spam o informacion incompleta. "
                     "Los timestamps pueden tener inconsistencias. Es imprescindible validar "
                     "nulos, duplicados y outliers antes del analisis productivo.",
        "Valor": "El objetivo final es convertir datos brutos en inteligencia de negocio: "
                 "segmentar clientes por comportamiento de compra, predecir satisfaccion "
                 "y generar recomendaciones estrategicas para el equipo de marketing."
    }
    for v, descripcion in cinco_v.items():
        print(f"\n  {v}:")
        print(f"    {descripcion}")

    # 1.2 Fuentes de datos identificadas
    print("\n\n1.2 FUENTES DE DATOS DE RETAILMAX")
    fuentes = [
        ("olist_orders_dataset.csv", "Transacciones",
         "Ordenes con timestamps de compra, aprobacion, envio y entrega. "
         "Incluye estado de la orden y vinculo con el cliente. ~99K registros."),
        ("olist_order_items_dataset.csv", "Items de Venta",
         "Detalle de cada producto vendido: precio unitario, costo de flete, "
         "vendedor y producto. Permite calcular ticket promedio y rankings. ~112K registros."),
        ("olist_order_reviews_dataset.csv", "Resenas y Calificaciones",
         "Datos semi-estructurados: calificacion numerica (1-5) y comentarios "
         "de texto libre. Fuente clave para analisis de satisfaccion. ~104K registros.")
    ]
    for archivo, tipo, desc_texto in fuentes:
        print(f"\n  [{tipo}] {archivo}")
        print(f"    {desc_texto}")

    # 1.3 Diagrama de arquitectura
    print("\n\n1.3 DIAGRAMA DE ARQUITECTURA DEL PIPELINE")
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')
    ax.set_title("Arquitectura del Pipeline - RetailMax Analytics",
                 fontsize=16, fontweight='bold', pad=20)

    colores = {'ingesta': '#3498db', 'proceso': '#e74c3c', 'ml': '#2ecc71',
               'salida': '#f39c12'}

    cajas = [
        (0.5, 6, 3, 1.2, "INGESTA DE DATOS\n(CSVs: Orders, Items,\nReviews)", colores['ingesta']),
        (5, 6, 3.5, 1.2, "SPARK ENGINE\n(SparkSession, RDDs,\nDataFrames)", colores['proceso']),
        (10, 6, 3.5, 1.2, "SPARK SQL\n(Consultas, Metricas,\nParquet)", colores['proceso']),
        (0.5, 3, 3, 1.2, "TRANSFORMACIONES\n(map, filter, reduceByKey\ncache/persist)", colores['proceso']),
        (5, 3, 3.5, 1.2, "MLlib PIPELINE\n(VectorAssembler,\nStringIndexer)", colores['ml']),
        (10, 3, 3.5, 1.2, "MODELOS ML\n(Regresion Logistica,\nK-Means)", colores['ml']),
        (3.5, 0.5, 3, 1.2, "METRICAS\n(AUC-ROC, Accuracy,\nSilhouette)", colores['salida']),
        (8, 0.5, 3.5, 1.2, "INSIGHTS MARKETING\n(Segmentacion,\nReporte Final)", colores['salida']),
    ]

    for x, y, w, h, texto, color in cajas:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                        facecolor=color, alpha=0.85,
                                        edgecolor='white', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, texto, ha='center', va='center',
                fontsize=8, fontweight='bold', color='white')

    flechas = [
        (3.5, 6.6, 5, 6.6), (8.5, 6.6, 10, 6.6),
        (2, 6, 2, 4.2), (6.75, 6, 6.75, 4.2), (11.75, 6, 11.75, 4.2),
        (3.5, 3.6, 5, 3.6), (8.5, 3.6, 10, 3.6),
        (6.75, 3, 5, 1.7), (11.75, 3, 9.75, 1.7),
    ]
    for x1, y1, x2, y2 in flechas:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    plt.tight_layout()
    ruta_diagrama = os.path.join(RUTA_GRAFICOS, "arquitectura_pipeline.png")
    plt.savefig(ruta_diagrama, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Diagrama guardado en: {ruta_diagrama}")
    print("\n  Informe conceptual completado exitosamente.")

# LECCION 2: APACHE SPARK - INTRODUCCION Y CONFIGURACION
def leccion_2(spark):
    separador("LECCION 2: APACHE SPARK - INTRODUCCION Y CONFIGURACION")

    # 2.1 Configuracion de SparkSession
    print("2.1 CONFIGURACION DE SPARKSESSION")
    print(f"  App Name:            {spark.sparkContext.appName}")
    print(f"  Spark Version:       {spark.version}")
    print(f"  Master:              {spark.sparkContext.master}")
    print(f"  Default Parallelism: {spark.sparkContext.defaultParallelism}")

    """ 2.2 Carga de datos y exploracion inicial
     Nota: En un entorno productivo se usa sc.textFile() para crear RDDs
     directamente. En Python 3.14 los workers de PySpark presentan
     incompatibilidades con RDD, por lo que demostramos las acciones
     basicas (count, take, first) mediante la API de DataFrames que
     ejecuta las mismas operaciones en el motor distribuido de Spark."""
    print("\n2.2 CARGA DE DATOS Y ACCIONES BASICAS (count, take, first)")
    
    df_orders_raw = spark.read.text(os.path.join(RUTA_DATOS, "olist_orders_dataset.csv"))
    df_items_raw = spark.read.text(os.path.join(RUTA_DATOS, "olist_order_items_dataset.csv"))
    df_reviews_raw = spark.read.text(os.path.join(RUTA_DATOS, "olist_order_reviews_dataset.csv"))

    # Acciones basicas equivalentes a RDD (count)
    print(f"\n  Orders  - count: {df_orders_raw.count()}")
    print(f"  Items   - count: {df_items_raw.count()}")
    print(f"  Reviews - count: {df_reviews_raw.count()}")

    # Equivalente a take(3)
    print("\n  take(3) equivalente de Orders:")
    for row in df_orders_raw.take(3):
        print(f"    {str(row['value'])[:120]}...")

    # Equivalente a first()
    print("\n  first() equivalente de cada dataset:")
    print(f"    Orders:  {str(df_orders_raw.first()['value'])[:100]}...")
    print(f"    Items:   {str(df_items_raw.first()['value'])[:100]}...")
    print(f"    Reviews: {str(df_reviews_raw.first()['value'])[:100]}...")

    # 2.3 Validacion con DataFrames
    print("\n2.3 VALIDACION CON DataFrames (carga directa CSV)")
    df_orders = spark.read.csv(os.path.join(RUTA_DATOS, "olist_orders_dataset.csv"),
                               header=True, inferSchema=True)
    df_items = spark.read.csv(os.path.join(RUTA_DATOS, "olist_order_items_dataset.csv"),
                              header=True, inferSchema=True)
    df_reviews = spark.read.csv(os.path.join(RUTA_DATOS, "olist_order_reviews_dataset.csv"),
                                header=True, inferSchema=True)

    for nombre, df in [("Orders", df_orders), ("Items", df_items), ("Reviews", df_reviews)]:
        print(f"\n  Schema de {nombre}:")
        df.printSchema()
        print(f"  Total registros: {df.count()}")

    print("\n  Configuracion y carga inicial completada exitosamente.")
    return df_orders, df_items, df_reviews

""" LECCION 3: RDD - TRANSFORMACIONES Y ACCIONES
 Nota: Las operaciones de RDD (map, filter, flatMap, distinct, sortBy,
 reduceByKey) se demuestran usando la API de DataFrames que ofrece
 funciones equivalentes (select/withColumn, filter/where, explode,
 dropDuplicates, orderBy, groupBy+agg), ya que Python 3.14 presenta
 incompatibilidades con la serializacion de lambdas en PySpark. """
def leccion_3(spark, df_items, df_reviews):
    separador("LECCION 3: ELEMENTOS BASICOS DE SPARK (RDD, TRANSFORMACIONES Y ACCIONES)")

    # 3.1 Creacion de Pair RDDs
    print("3.1 CREACION DE PAIR RDDs (equivalente con DataFrames)")
    
    # Pair RDD equivalente: (order_id, precio_total)
    df_ventas = df_items.select(
        col("order_id"),
        (col("price") + col("freight_value")).alias("precio_total")
    )
    print(f"  Pair 'RDD' de ventas creado - count: {df_ventas.count()}")
    df_ventas.show(5, truncate=False)

    # Pair RDD equivalente: (product_id, conteo)
    df_productos_count = df_items.groupBy("product_id").agg(
        count("*").alias("cantidad")
    )
    print(f"  Pair 'RDD' de productos - count: {df_productos_count.count()}")

    # 3.2 Transformaciones
    print("\n3.2 TRANSFORMACIONES (equivalentes a RDD)")
    
    # map equivalente: extraer solo precios
    df_precios = df_items.select("price")
    print(f"  [map -> select] RDD de precios:")
    df_precios.show(5)

    # filter equivalente: items con precio > 100
    df_premium = df_items.filter(col("price") > 100)
    print(f"  [filter] Items con precio > 100: {df_premium.count()}")

    # flatMap equivalente: generar pares de IDs
    from pyspark.sql.functions import array, explode
    df_pares = df_items.select(
        explode(array(col("order_id"), col("product_id"))).alias("id_value")
    )
    print(f"  [flatMap -> explode] Total elementos (order+product IDs): {df_pares.count()}")

    # distinct equivalente: productos unicos
    df_productos_unicos = df_items.select("product_id").distinct()
    print(f"  [distinct] Productos unicos: {df_productos_unicos.count()}")

    # sortBy equivalente: top 10 items mas caros
    print(f"  [sortBy -> orderBy] Top 10 items mas caros:")
    df_items.select("product_id", "price") \
        .orderBy(col("price").desc()) \
        .show(10, truncate=30)

    # reduceByKey equivalente: venta total por orden
    df_ventas_por_orden = df_items.groupBy("order_id").agg(
        spark_round(spark_sum(col("price") + col("freight_value")), 2).alias("venta_total")
    )
    print(f"  [reduceByKey -> groupBy+sum] Ventas por orden - count: {df_ventas_por_orden.count()}")
    df_ventas_por_orden.orderBy(col("venta_total").desc()).show(5, truncate=False)

    # groupByKey + conteo: top productos
    print(f"  [reduceByKey -> groupBy+count] Top 10 productos mas vendidos:")
    df_productos_count.orderBy(col("cantidad").desc()).show(10, truncate=30)

    # 3.3 Acciones y estadisticas
    print("\n3.3 ACCIONES Y ESTADISTICAS")
    
    stats = df_items.agg(
        count("price").alias("total_items"),
        spark_round(spark_sum("price"), 2).alias("suma_precios"),
        spark_round(avg("price"), 2).alias("precio_promedio"),
        spark_round(stddev("price"), 2).alias("desviacion_estandar"),
        spark_round(spark_min("price"), 2).alias("precio_minimo"),
        spark_round(spark_max("price"), 2).alias("precio_maximo")
    ).collect()[0]

    print(f"  Total items:           {stats['total_items']}")
    print(f"  Suma de precios:       ${stats['suma_precios']:,.2f}")
    print(f"  Precio promedio:       ${stats['precio_promedio']:,.2f}")
    print(f"  Desviacion estandar:   ${stats['desviacion_estandar']:,.2f}")
    print(f"  Precio minimo:         ${stats['precio_minimo']:,.2f}")
    print(f"  Precio maximo:         ${stats['precio_maximo']:,.2f}")

    # 3.4 Documentacion del linaje
    print("\n3.4 DOCUMENTACION DEL LINAJE (DAG)")
    
    # Plan de ejecucion (equivalente al DAG de RDDs)
    print("  Plan de ejecucion (Logical Plan) del DataFrame de ventas por orden:")
    df_ventas_por_orden.explain(True)

    print("\n  Explicacion del DAG equivalente:")
    print("  1. Scan CSV   -> Lee el archivo como DataFrame")
    print("  2. Project    -> Selecciona columnas (equivalente a map)")
    print("  3. Filter     -> Filtra filas (equivalente a filter)")
    print("  4. HashAggregate -> Agrega por clave (equivalente a reduceByKey)")
    print("  5. Exchange   -> Shuffle de datos entre particiones")
    print("  6. HashAggregate -> Agregacion final")

    # Documentar cache
    print("\n  Optimizacion con cache():")
    df_items.cache()
    cuenta_1 = df_items.count()
    cuenta_2 = df_items.count()  # Esta lectura usa cache
    print(f"  Primera lectura (sin cache): {cuenta_1} registros")
    print(f"  Segunda lectura (con cache): {cuenta_2} registros")
    print("  El cache() almacena el DataFrame en memoria, evitando re-lectura del CSV.")
    print("  Esto es especialmente util cuando un DataFrame se reutiliza multiples veces.")

    print("\n  Leccion 3 completada exitosamente.")

# LECCION 4: SPARK SQL Y DATAFRAMES
def leccion_4(spark, df_orders, df_items, df_reviews):
    separador("LECCION 4: PROCESAMIENTO DE DATOS ESTRUCTURADOS (SPARK SQL Y DATAFRAMES)")

    # 4.1 Esquemas explicitos
    print("4.1 ESQUEMAS EXPLICITOS Y DATAFRAMES")
    
    schema_orders = StructType([
        StructField("order_id", StringType(), True),
        StructField("customer_id", StringType(), True),
        StructField("order_status", StringType(), True),
        StructField("order_purchase_timestamp", StringType(), True),
        StructField("order_approved_at", StringType(), True),
        StructField("order_delivered_carrier_date", StringType(), True),
        StructField("order_delivered_customer_date", StringType(), True),
        StructField("order_estimated_delivery_date", StringType(), True)
    ])

    schema_items = StructType([
        StructField("order_id", StringType(), True),
        StructField("order_item_id", IntegerType(), True),
        StructField("product_id", StringType(), True),
        StructField("seller_id", StringType(), True),
        StructField("shipping_limit_date", StringType(), True),
        StructField("price", DoubleType(), True),
        StructField("freight_value", DoubleType(), True)
    ])

    schema_reviews = StructType([
        StructField("review_id", StringType(), True),
        StructField("order_id", StringType(), True),
        StructField("review_score", IntegerType(), True),
        StructField("review_comment_title", StringType(), True),
        StructField("review_comment_message", StringType(), True),
        StructField("review_creation_date", StringType(), True),
        StructField("review_answer_timestamp", StringType(), True)
    ])

    # Recargar con esquemas explicitos
    df_orders_typed = spark.read.csv(
        os.path.join(RUTA_DATOS, "olist_orders_dataset.csv"),
        header=True, schema=schema_orders)
    df_items_typed = spark.read.csv(
        os.path.join(RUTA_DATOS, "olist_order_items_dataset.csv"),
        header=True, schema=schema_items)
    df_reviews_typed = spark.read.csv(
        os.path.join(RUTA_DATOS, "olist_order_reviews_dataset.csv"),
        header=True, schema=schema_reviews)

    # Cache/Persist para optimizacion
    df_orders_typed.persist(StorageLevel.MEMORY_AND_DISK)
    df_items_typed.persist(StorageLevel.MEMORY_AND_DISK)
    df_reviews_typed.persist(StorageLevel.MEMORY_AND_DISK)

    print("  DataFrames creados con esquemas explicitos (StructType)")
    print(f"  Orders:  {df_orders_typed.count()} registros")
    print(f"  Items:   {df_items_typed.count()} registros")
    print(f"  Reviews: {df_reviews_typed.count()} registros")
    print("\n  Optimizacion: persist(MEMORY_AND_DISK) aplicado a los 3 DataFrames")
    print("  para evitar relectura del disco en operaciones subsecuentes.")

    # 4.2 Registrar tablas temporales
    df_orders_typed.createOrReplaceTempView("orders")
    df_items_typed.createOrReplaceTempView("items")
    df_reviews_typed.createOrReplaceTempView("reviews")
    print("\n  Tablas temporales registradas: orders, items, reviews")

    # 4.3 Consultas SQL
    print("\n4.2 CONSULTAS SQL - METRICAS DE NEGOCIO")
    
    # Consulta 1: Ventas totales por estado de orden
    print("\n  [SQL] Ventas totales por estado de orden:")
    q1 = spark.sql("""
        SELECT o.order_status,
               COUNT(DISTINCT o.order_id) as total_ordenes,
               ROUND(SUM(i.price), 2) as ventas_totales,
               ROUND(AVG(i.price), 2) as precio_promedio
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        GROUP BY o.order_status
        ORDER BY ventas_totales DESC
    """)
    q1.show(truncate=False)

    # Consulta 2: Top 10 productos mas vendidos
    print("  [SQL] Top 10 productos mas vendidos:")
    q2 = spark.sql("""
        SELECT product_id,
               COUNT(*) as cantidad_vendida,
               ROUND(SUM(price), 2) as ingreso_total,
               ROUND(AVG(price), 2) as precio_promedio
        FROM items
        GROUP BY product_id
        ORDER BY cantidad_vendida DESC
        LIMIT 10
    """)
    q2.show(truncate=False)

    # Consulta 3: Distribucion de calificaciones
    print("  [SQL] Distribucion de calificaciones de resenas:")
    q3 = spark.sql("""
        SELECT review_score,
               COUNT(*) as cantidad,
               ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM reviews), 2) as porcentaje
        FROM reviews
        WHERE review_score IS NOT NULL
        GROUP BY review_score
        ORDER BY review_score
    """)
    q3.show(truncate=False)

    # Consulta 4: Ticket promedio por cliente
    print("  [SQL] Ticket promedio por cliente (Top 10):")
    q4 = spark.sql("""
        SELECT o.customer_id,
               COUNT(DISTINCT o.order_id) as total_ordenes,
               ROUND(SUM(i.price + i.freight_value), 2) as gasto_total,
               ROUND(AVG(i.price + i.freight_value), 2) as ticket_promedio
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        GROUP BY o.customer_id
        ORDER BY gasto_total DESC
        LIMIT 10
    """)
    q4.show(truncate=False)

    # Consulta 5: Ventas mensuales
    print("  [SQL] Tendencia de ventas mensuales:")
    q5 = spark.sql("""
        SELECT SUBSTR(o.order_purchase_timestamp, 1, 7) as mes,
               COUNT(DISTINCT o.order_id) as total_ordenes,
               ROUND(SUM(i.price), 2) as ventas_totales
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        WHERE o.order_purchase_timestamp IS NOT NULL
        GROUP BY SUBSTR(o.order_purchase_timestamp, 1, 7)
        ORDER BY mes
    """)
    q5.show(50, truncate=False)

    # 4.4 Visualizaciones
    print("\n4.3 VISUALIZACIONES")
    
    # Grafico: Distribucion de calificaciones
    datos_reviews = q3.collect()
    scores = [row['review_score'] for row in datos_reviews]
    cantidades = [row['cantidad'] for row in datos_reviews]
    colores_barras = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#27ae60']

    fig, ax = plt.subplots(figsize=(10, 6))
    barras = ax.bar(scores, cantidades, color=colores_barras[:len(scores)],
                    edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Calificacion', fontsize=12)
    ax.set_ylabel('Cantidad de Resenas', fontsize=12)
    ax.set_title('Distribucion de Calificaciones - RetailMax',
                 fontsize=14, fontweight='bold')
    for barra, cant in zip(barras, cantidades):
        ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 200,
                f'{cant:,}', ha='center', va='bottom', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "distribucion_calificaciones.png"), dpi=150)
    plt.close()

    # Grafico: Ventas mensuales
    datos_mensuales = q5.collect()
    meses = [row['mes'] for row in datos_mensuales]
    ventas = [row['ventas_totales'] for row in datos_mensuales]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(meses, ventas, marker='o', color='#3498db', linewidth=2, markersize=6)
    ax.fill_between(range(len(meses)), ventas, alpha=0.15, color='#3498db')
    ax.set_xlabel('Mes', fontsize=12)
    ax.set_ylabel('Ventas Totales ($)', fontsize=12)
    ax.set_title('Tendencia de Ventas Mensuales - RetailMax',
                 fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "ventas_mensuales.png"), dpi=150)
    plt.close()

    # Grafico: Ventas por estado
    datos_estado = q1.collect()
    estados = [row['order_status'] for row in datos_estado]
    ventas_estado = [row['ventas_totales'] for row in datos_estado]

    fig, ax = plt.subplots(figsize=(10, 6))
    colores_estado = ['#2ecc71', '#e74c3c', '#f39c12', '#3498db',
                      '#9b59b6', '#1abc9c', '#e67e22', '#34495e']
    ax.barh(estados, ventas_estado, color=colores_estado[:len(estados)],
            edgecolor='white', linewidth=1.5)
    ax.set_xlabel('Ventas Totales ($)', fontsize=12)
    ax.set_title('Ventas por Estado de Orden - RetailMax',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "ventas_por_estado.png"), dpi=150)
    plt.close()

    print("  Graficos guardados en:", RUTA_GRAFICOS)

    # 4.5 Guardar en Parquet
    print("\n4.4 GUARDADO EN FORMATO PARQUET")
    
    # Crear DataFrame consolidado para ML
    df_consolidado = df_items_typed.join(df_orders_typed, "order_id") \
        .join(df_reviews_typed, "order_id", "left")

    df_consolidado.write.mode("overwrite").parquet(
        os.path.join(RUTA_PARQUET, "datos_consolidados"))
    print(f"  Datos consolidados guardados en: {RUTA_PARQUET}/datos_consolidados")
    print(f"  Total registros consolidados: {df_consolidado.count()}")
    print(f"  Columnas: {df_consolidado.columns}")

    # Guardar metricas por orden
    df_metricas_orden = spark.sql("""
        SELECT o.order_id, o.customer_id, o.order_status,
               CAST(SUM(i.price) AS DOUBLE) as gasto_total,
               CAST(SUM(i.freight_value) AS DOUBLE) as flete_total,
               CAST(COUNT(i.order_item_id) AS DOUBLE) as cantidad_items,
               CAST(AVG(r.review_score) AS DOUBLE) as score_promedio
        FROM orders o
        JOIN items i ON o.order_id = i.order_id
        LEFT JOIN reviews r ON o.order_id = r.order_id
        GROUP BY o.order_id, o.customer_id, o.order_status
    """)
    df_metricas_orden.write.mode("overwrite").parquet(
        os.path.join(RUTA_PARQUET, "metricas_por_orden"))
    print(f"  Metricas por orden guardadas en: {RUTA_PARQUET}/metricas_por_orden")
    print(f"  Total metricas: {df_metricas_orden.count()}")

    print("\n  Documentacion de persist:")
    print("  - persist(MEMORY_AND_DISK): almacena DataFrames en memoria RAM")
    print("    y si no cabe, desborda a disco. Ideal para datasets medianos.")
    print("  - cache(): equivalente a persist(MEMORY_ONLY).")
    print("  - Ambos evitan recomputacion cuando el DataFrame se usa multiples veces.")

    print("\n  Leccion 4 completada exitosamente.")
    return df_consolidado, df_metricas_orden

# LECCION 5: MACHINE LEARNING ESCALABLE (SPARK MLlib)
def leccion_5(spark):
    separador("LECCION 5: MACHINE LEARNING ESCALABLE (SPARK MLlib)")

    # 5.1 Cargar datos procesados
    print("5.1 CARGA DE DATOS PROCESADOS (PARQUET)")
    
    df_metricas = spark.read.parquet(os.path.join(RUTA_PARQUET, "metricas_por_orden"))
    print(f"  Registros cargados: {df_metricas.count()}")
    df_metricas.printSchema()
    df_metricas.show(5, truncate=False)

    # 5.2 Preparacion de features
    print("\n5.2 PREPARACION DE FEATURES")
    
    # Limpiar nulos
    df_limpio = df_metricas.na.drop(subset=["gasto_total", "flete_total",
                                             "cantidad_items", "score_promedio",
                                             "order_status"])
    print(f"  Registros despues de limpiar nulos: {df_limpio.count()}")

    # Crear variable objetivo: cliente satisfecho (score >= 4) vs insatisfecho
    df_limpio = df_limpio.withColumn(
        "satisfecho",
        when(col("score_promedio") >= 4, 1.0).otherwise(0.0)
    )

    # Distribucion de la variable objetivo
    print("\n  Distribucion de la variable objetivo:")
    df_limpio.groupBy("satisfecho").count().show()

    # StringIndexer para order_status
    indexer_status = StringIndexer(
        inputCol="order_status", outputCol="status_index",
        handleInvalid="skip")

    # VectorAssembler
    assembler = VectorAssembler(
        inputCols=["gasto_total", "flete_total", "cantidad_items", "status_index"],
        outputCol="features"
    )

    # Scaler para normalizar features
    scaler = StandardScaler(
        inputCol="features", outputCol="scaled_features",
        withStd=True, withMean=True
    )

    print("  StringIndexer configurado para: order_status -> status_index")
    print("  VectorAssembler configurado con: gasto_total, flete_total,")
    print("    cantidad_items, status_index")
    print("  StandardScaler configurado para normalizar features")

    # 5.3 Modelo Supervisado: Regresion Logistica
    print("\n5.3 MODELO SUPERVISADO: REGRESION LOGISTICA")
    
    # Pipeline de clasificacion
    lr = LogisticRegression(
        featuresCol="scaled_features", labelCol="satisfecho",
        maxIter=20, regParam=0.01)

    pipeline_lr = Pipeline(stages=[indexer_status, assembler, scaler, lr])

    # Split train/test
    train_df, test_df = df_limpio.randomSplit([0.8, 0.2], seed=42)
    print(f"  Train: {train_df.count()} registros")
    print(f"  Test:  {test_df.count()} registros")

    # Entrenar
    print("  Entrenando modelo de Regresion Logistica...")
    modelo_lr = pipeline_lr.fit(train_df)
    predicciones_lr = modelo_lr.transform(test_df)

    # Evaluar
    evaluador_auc = BinaryClassificationEvaluator(
        labelCol="satisfecho", rawPredictionCol="rawPrediction",
        metricName="areaUnderROC")
    evaluador_acc = MulticlassClassificationEvaluator(
        labelCol="satisfecho", predictionCol="prediction",
        metricName="accuracy")

    auc_roc = evaluador_auc.evaluate(predicciones_lr)
    accuracy = evaluador_acc.evaluate(predicciones_lr)

    print(f"\n  === RESULTADOS REGRESION LOGISTICA ===")
    print(f"  AUC-ROC:  {auc_roc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")

    # Coeficientes
    modelo_entrenado = modelo_lr.stages[-1]
    print(f"\n  Coeficientes del modelo:")
    nombres_features = ["gasto_total", "flete_total", "cantidad_items", "status_index"]
    for nombre, coef in zip(nombres_features, modelo_entrenado.coefficients):
        print(f"    {nombre}: {coef:.6f}")
    print(f"  Intercepto: {modelo_entrenado.intercept:.6f}")

    # Predicciones ejemplo
    print("\n  Muestra de predicciones:")
    predicciones_lr.select(
        "order_id", "gasto_total", "score_promedio",
        "satisfecho", "prediction"
    ).show(10, truncate=False)

    # 5.4 Modelo No Supervisado: K-Means
    print("\n5.4 MODELO NO SUPERVISADO: K-MEANS (SEGMENTACION)")
    
    # Preparar features para clustering
    pipeline_prep = Pipeline(stages=[indexer_status, assembler, scaler])
    modelo_prep = pipeline_prep.fit(df_limpio)
    df_features = modelo_prep.transform(df_limpio)
    df_features.cache()

    # Encontrar K optimo con Silhouette
    print("  Buscando K optimo (silhouette score)...")
    resultados_k = []
    for k in range(2, 7):
        kmeans = KMeans(featuresCol="scaled_features", k=k, seed=42, maxIter=20)
        modelo_km = kmeans.fit(df_features)
        predicciones_km = modelo_km.transform(df_features)
        evaluador_sil = ClusteringEvaluator(
            featuresCol="scaled_features", metricName="silhouette")
        silhouette = evaluador_sil.evaluate(predicciones_km)
        resultados_k.append((k, silhouette))
        print(f"    K={k}: Silhouette = {silhouette:.4f}")

    mejor_k = max(resultados_k, key=lambda x: x[1])
    print(f"\n  Mejor K: {mejor_k[0]} (Silhouette: {mejor_k[1]:.4f})")

    # Entrenar modelo final con mejor K
    kmeans_final = KMeans(
        featuresCol="scaled_features", k=mejor_k[0], seed=42, maxIter=20)
    modelo_kmeans = kmeans_final.fit(df_features)
    df_clustered = modelo_kmeans.transform(df_features)

    # Analisis de clusters
    print(f"\n  === ANALISIS DE CLUSTERS (K={mejor_k[0]}) ===")
    resumen_clusters = df_clustered.groupBy("prediction").agg(
        count("*").alias("total_clientes"),
        spark_round(avg("gasto_total"), 2).alias("gasto_promedio"),
        spark_round(avg("flete_total"), 2).alias("flete_promedio"),
        spark_round(avg("cantidad_items"), 2).alias("items_promedio"),
        spark_round(avg("score_promedio"), 2).alias("score_promedio_c")
    ).orderBy("prediction")
    resumen_clusters.show(truncate=False)

    # Centros de los clusters
    print("  Centros de los clusters:")
    for i, centro in enumerate(modelo_kmeans.clusterCenters()):
        print(f"    Cluster {i}: {[round(v, 2) for v in centro]}")

    # 5.5 Visualizaciones finales
    print("\n5.5 VISUALIZACIONES FINALES")
    
    # Grafico: Silhouette por K
    ks = [r[0] for r in resultados_k]
    sils = [r[1] for r in resultados_k]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ks, sils, 'o-', color='#e74c3c', linewidth=2, markersize=8)
    ax.set_xlabel('Numero de Clusters (K)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Seleccion de K Optimo - Silhouette Score',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(ks)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "silhouette_scores.png"), dpi=150)
    plt.close()

    # Grafico: Distribucion de clusters
    datos_clusters = resumen_clusters.collect()
    cluster_ids = [row['prediction'] for row in datos_clusters]
    totales = [row['total_clientes'] for row in datos_clusters]
    gastos = [row['gasto_promedio'] for row in datos_clusters]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colores_cluster = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12',
                       '#9b59b6', '#1abc9c']

    # Pie chart de distribucion
    axes[0].pie(totales, labels=[f'Cluster {c}' for c in cluster_ids],
                colors=colores_cluster[:len(cluster_ids)],
                autopct='%1.1f%%', startangle=90,
                textprops={'fontsize': 10})
    axes[0].set_title('Distribucion de Clientes por Cluster',
                      fontsize=13, fontweight='bold')

    # Barras de gasto promedio
    axes[1].bar([f'C{c}' for c in cluster_ids], gastos,
                color=colores_cluster[:len(cluster_ids)],
                edgecolor='white', linewidth=1.5)
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Gasto Promedio ($)', fontsize=12)
    axes[1].set_title('Gasto Promedio por Cluster',
                      fontsize=13, fontweight='bold')
    for bar_val, gasto in zip(axes[1].patches, gastos):
        axes[1].text(bar_val.get_x() + bar_val.get_width()/2,
                     bar_val.get_height() + 1,
                     f'${gasto:,.0f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "segmentacion_clusters.png"), dpi=150)
    plt.close()

    # Grafico: Perfil de clusters
    scores_cluster = [row['score_promedio_c'] for row in datos_clusters]
    items_cluster = [row['items_promedio'] for row in datos_clusters]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Score promedio por cluster
    axes[0].bar([f'C{c}' for c in cluster_ids], scores_cluster,
                color=colores_cluster[:len(cluster_ids)],
                edgecolor='white', linewidth=1.5)
    axes[0].set_xlabel('Cluster', fontsize=12)
    axes[0].set_ylabel('Score Promedio', fontsize=12)
    axes[0].set_title('Satisfaccion Promedio por Cluster',
                      fontsize=13, fontweight='bold')
    axes[0].set_ylim(0, 5.5)

    # Items promedio por cluster
    axes[1].bar([f'C{c}' for c in cluster_ids], items_cluster,
                color=colores_cluster[:len(cluster_ids)],
                edgecolor='white', linewidth=1.5)
    axes[1].set_xlabel('Cluster', fontsize=12)
    axes[1].set_ylabel('Items Promedio', fontsize=12)
    axes[1].set_title('Items Promedio por Cluster',
                      fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(RUTA_GRAFICOS, "perfil_clusters.png"), dpi=150)
    plt.close()

    print("  Graficos finales guardados en:", RUTA_GRAFICOS)

    print(f"\n  === RESUMEN FINAL ===")
    print(f"  Regresion Logistica: AUC-ROC={auc_roc:.4f}, Accuracy={accuracy:.4f}")
    print(f"  K-Means: K={mejor_k[0]}, Silhouette={mejor_k[1]:.4f}")
    print(f"  Total clusters identificados: {len(cluster_ids)}")
    print(f"  Visualizaciones generadas: 6 graficos en total")

    print("\n  Leccion 5 completada exitosamente.")
    return auc_roc, accuracy, mejor_k, resumen_clusters

# EJECUCION PRINCIPAL
if __name__ == "__main__":
    print("  RETAIL ANALYTICS PIPELINE - RetailMax")
    print("  Modulo 9: Fundamentos de Big Data")
    print("  Pipeline completo: Ingesta -> Procesamiento -> ML -> Insights")

    # LECCION 1: Fundamentos (no requiere Spark)
    leccion_1()

    # Iniciar SparkSession para lecciones 2-5
    separador("INICIALIZANDO APACHE SPARK")
    spark = SparkSession.builder \
        .master("local[*]") \
        .appName("RetailMax-Analytics-Pipeline") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.ui.showConsoleProgress", "false") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    print(f"  SparkSession iniciada: {spark.sparkContext.appName}")
    print(f"  Version: {spark.version}")

    try:
        # LECCION 2: Configuracion
        df_orders, df_items, df_reviews = leccion_2(spark)

        # LECCION 3: RDDs (usando DataFrame API por compatibilidad con Python 3.14)
        leccion_3(spark, df_items, df_reviews)

        # LECCION 4: Spark SQL
        df_consolidado, df_metricas = leccion_4(spark, df_orders, df_items, df_reviews)

        # LECCION 5: MLlib
        auc_roc, accuracy, mejor_k, resumen_clusters = leccion_5(spark)

        # Resumen final
        separador("PIPELINE COMPLETADO EXITOSAMENTE")
        print("  Archivos generados:")
        for root, dirs, files in os.walk(RUTA_RESULTADOS):
            for f in files:
                ruta = os.path.join(root, f)
                tamano = os.path.getsize(ruta)
                print(f"    {os.path.relpath(ruta, RUTA_BASE)} ({tamano:,} bytes)")

    finally:
        spark.stop()
        print("\n  SparkSession detenida correctamente.")
        print("  Pipeline de RetailMax Analytics finalizado!\n")