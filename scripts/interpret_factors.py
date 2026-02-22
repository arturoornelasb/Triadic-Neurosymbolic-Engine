import os
import sys
import pandas as pd
from collections import defaultdict

# Ensure neurosym source is in path
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

print("=========================================================")
print("🧠 TRIADIC ENGINE: DECODIFICADOR DE FACTORES PRIMOS")
print("=========================================================")
print("Vamos a analizar qué significa cada 'Ladrillo Primo' en el cerebro de la IA.\n")

input_csv = "examples/data/wordnet_2k.csv"
if not os.path.exists(input_csv):
    print("Error: Ejecuta primero 'generate_real_data.py'")
    sys.exit(1)

df = pd.read_csv(input_csv)
conceptos = df["concept"].dropna().astype(str).tolist()

# Usaremos un solo modelo para entender su cerebro interno
print("1. Cargando Modelo de IA (all-MiniLM-L6-v2)...")
encoder = ContinuousEncoder(model_name="all-MiniLM-L6-v2")
validator = DiscreteValidator()

print(f"2. Vectorizando {len(conceptos)} conceptos...")
embeddings = encoder.encode(conceptos)

print("3. Asignando Factores Primos (LSH k=10)...")
mapper = DiscreteMapper(n_bits=10, seed=42)
prime_map = mapper.fit_transform(conceptos, embeddings)

# --- INGENIERÍA INVERSA ---
# Agruparemos las palabras basándonos en si poseen o no un factor primo específico.
factor_groups = defaultdict(list)

print("\n4. Escaneando la red para hacer Ingeniería Inversa de los Significados...\n")
for word, primo in prime_map.items():
    factores = validator._prime_factors(primo)
    for f in factores:
        factor_groups[f].append(word)

# Imprimimos los resultados para los primeros primos
f_primes = sorted(list(factor_groups.keys()))

print("=========================================================")
print("🧬 DICCIONARIO SEMÁNTICO DE NÚMEROS PRIMOS (INTERPRETACIÓN)")
print("=========================================================")
print("Si las palabras comparten un primo, comparten un rasgo semántico oculto.\n")

for p in f_primes[:8]:  # Solo mostramos los 8 primeros planos hiperdimensionales
    words_with_p = factor_groups[p]
    # Mostramos máximo 15 palabras al azar para ver el patrón
    import random
    random.seed(p) # Deterministic sample
    sample_words = random.sample(words_with_p, min(15, len(words_with_p)))
    
    print(f"Factor Primo [{p}] está presente en {len(words_with_p)} palabras.")
    print(f"   -> Ejemplos: {', '.join(sample_words)}")
    print("-" * 60)

print("\nConclusión: Al agrupar las palabras por factor primo, podemos deducir (o usar otro LLM para etiquetar) qué categoría humana real representa cada número primo.")
