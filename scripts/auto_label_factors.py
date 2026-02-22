import os
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity

# Ensure neurosym source is in path
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

print("=========================================================")
print("🎯 TRIADIC ENGINE: AUTO-ETIQUETADO SEMÁNTICO DE PRIMOS")
print("=========================================================")
print("Usando los Vectores para descubrir el 'Nombre Humano' de cada Ladrillo Matemático.\n")

input_csv = "examples/data/wordnet_2k.csv"
if not os.path.exists(input_csv):
    print("Error: Ejecuta primero 'generate_real_data.py'")
    sys.exit(1)

df = pd.read_csv(input_csv)
conceptos = df["concept"].dropna().astype(str).tolist()

print("1. Cargando Cerebro de IA (all-MiniLM-L6-v2) y Vectorizando...")
encoder = ContinuousEncoder(model_name="all-MiniLM-L6-v2")
validator = DiscreteValidator()

# Vectorize all words
embeddings = encoder.encode(conceptos)

print("2. Mapeando a Factores Primos...")
mapper = DiscreteMapper(n_bits=10, seed=42)
prime_map = mapper.fit_transform(conceptos, embeddings)

# --- AUTO-ETIQUETADO BASADO EN CENTROIDES ---
# 1. Agrupar los índices de las palabras por factor primo
factor_indices = defaultdict(list)

for idx, word in enumerate(conceptos):
    primo = prime_map[word]
    factores = validator._prime_factors(primo)
    for f in factores:
        factor_indices[f].append(idx)

f_primes = sorted(list(factor_indices.keys()))

print("\n3. Calculando Centroides (Corazón Semántico) y Buscando Etiquetas...\n")
print("=========================================================")
print("🏷️ DICCIONARIO TRADUCIDO AUTOMÁTICAMENTE POR LA MÁQUINA")
print("=========================================================")

# Let's analyze the first few dimensions
for p in f_primes[:10]:
    indices = factor_indices[p]
    
    # Take all vectors of words that share this prime factor
    cluster_embeddings = embeddings[indices]
    
    # Calculate the centroid (average vector of the cluster)
    centroid = np.mean(cluster_embeddings, axis=0).reshape(1, -1)
    
    # Find the closest concept in the entire database to this centroid
    similarities = cosine_similarity(centroid, embeddings)[0]
    
    # Get top 3 closest words to act as the "Label"
    top_indices = np.argsort(similarities)[::-1][:3]
    top_labels = [conceptos[i] for i in top_indices]
    
    # Get some random examples of words that contain this factor
    import random
    random.seed(p)
    words_with_p = [conceptos[i] for i in indices]
    sample_words = random.sample(words_with_p, min(5, len(words_with_p)))
    
    print(f"🔸 Factor Primo [{p}]:")
    print(f"   ETIQUETA DESCUBIERTA POR EL CÓDIGO: '{top_labels[0].upper()}' (o {top_labels[1]}, {top_labels[2]})")
    print(f"   Aplica a {len(words_with_p)} palabras, ej: {', '.join(sample_words)}")
    print("-" * 70)

print("\n¡Prueba exitosa! El código miró la base de datos y le puso nombre humano a sus reglas matemáticas.")
