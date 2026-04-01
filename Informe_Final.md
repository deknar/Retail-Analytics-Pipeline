# Informe Final del Proyecto
## Retail Analytics Pipeline – RetailMax

### 1. Resumen Ejecutivo
El presente pipeline de datos masivos fue construido íntegramente de cara al volumen, variedad y velocidad generados por el comercio RetailMax. Su objetivo central es consolidar datos crudos desde distintas fuentes transaccionales, limpiarlos mediante entornos de procesamiento distribuidos y predecir tendencias de rendimiento comercial. De esta forma, generamos inteligencia empresarial accionable para la satisfacción y segmentación del comportamiento de cliente. 

### 2. Fundamentos de Big Data Implementados
Nuestro sistema satisface las 5 V's primordiales del Big Data. Evaluamos y procesamos datos masivos con más de 100,000 registros transaccionales (Volumen) de manera programática en tiempo real o por lotes de ingesta (Velocidad). Incorporamos datos estructurados y semiestructurados desde historiales de venta y reseñas de usuario (Variedad) filtrando irregularidades del modelo (Veracidad) para culminar en conclusiones accionables para el negocio (Valor).

### 3. Procesamiento y Análisis Distribuido
1. **RDD y Esquemas DataFrames**: Utilizamos la configuración `SparkSession` del ecosistema Spark (PySpark) en memoria distribuida para leer las fuentes. A pesar de los grandes volúmenes, logramos iteraciones rápidas al optimizar los procesos mediante almacenamiento `MEMORY_AND_DISK` (`persist`). 
2. **Spark SQL**: Aprovechamos interfaces de DataFrame y el motor embebido de queries SQL para producir métricas consolidadas cruciales para RetailMax. Evaluamos las tendencias de ventas por mes, descubrimos la dominancia del top 10 de ítems de compra y mapeamos el estado habitual de transacciones con la métrica de éxito. Las exportaciones se realizaron al estándar optimizado columner: formato Parquet.
3. **Optimización con Caché y Persistencia**: A lo largo de la transformación de las 5 lecciones, establecimos `persistencia` explícita en el Pipeline RDD/DataFrames para impedir que los CSVs se analizaran repetidas veces, lo que agota y recarga al Engine.

### 4. Machine Learning Escalable (Spark MLlib)
Para convertir los datos transaccionales en ventaja competitiva del marketing de RetailMax, ensamblamos un pipeline escalable de Feature Engineering. Agrupamos predictores normalizados de forma estandarizada (`VectorAssembler` y `StandardScaler`) e indexamos métricas textuales complejas. Sobre estas variables independientes diseñamos dos algoritmos de gran impacto:
*   **Regresión Logística (Modelo Supervisado)**: Permitió clasificar perfiles de clientes como satisfechos e insatisfechos basado en el historial transaccional como predictor, logrando así predicciones operativas en el backend a un bajo gasto de rendimiento en tiempo real. 
*   **Segmentación K-Means (Modelo No Supervisado)**: Para descubrir clústeres inexplorados y patrones de comportamiento de gasto masivo en las transacciones que no son trivialmente visibles a través de búsquedas SQL, optimizando los recursos del Marketing y evaluado mediante validaciones matemáticas intrínsecas del método de *Silhouette Strategy*.

### 5. Conclusiones y Recomendaciones para Marketing
Toda la orquestación distribuida, exportación, segmentación e inferencias concluyen en puntos invaluables para campañas orientadas al data-driven product marketing en RetailMax:
* Los flujos generaron datos relacionales depurados que la empresa ahora puede consumir analíticamente para comprender tendencias de satisfacción sin gastar dinero extra en reordenamiento.
* RetailMax cuenta con perfiles de cliente que consumen altos presupuestos concentrados en clústeres específicos, el análisis en parquet puede potenciar la personalización de campañas web y newsletters.
* Ciertos picos en los diagramas de series temporales revelan estacionalidad de demanda.

Las exportaciones están operativas para el front-end web e integrables en tiempo-completo en un Data Lake de nueva generación.
