# Proyecto creado por Luis Diaz (Agosto-2023) para Analizar datos y agruparlos por el metodo K-Means

# Importar bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Cargar los datos de transacciones de clientes (ejemplo de nombre de archivo, archivo CSV llamado 'datos_clientes.csv')
data = pd.read_csv('datos_clientes.csv')

# Preprocesamiento de datos
# Realiza la limpieza y transformación de datos según sea necesario

# Aplicar StandardScaler para estandarizar los datos
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Aplicar PCA para reducción de dimensionalidad
pca = PCA(n_components=2)  # Supongamos que elegimos 2 componentes principales
principal_components = pca.fit_transform(scaled_data)

# Aplicar K-Means para la segmentación de clientes
num_clusters = 3  # Supongamos que elegimos 3 clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=0)
clusters = kmeans.fit_predict(principal_components)

# Evaluar la segmentación usando Silhouette Score
silhouette_avg = silhouette_score(principal_components, clusters)

# Mostrar los resultados
data['Cluster'] = clusters
print(data.head())

# Guardar los resultados en un archivo CSV
data.to_csv('resultados_segmentacion.csv', index=False)