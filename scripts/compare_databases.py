import time
import os
import sys

# Asegurar que el Engine está en el path (root dir)
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

print("=========================================================")
print("🤖 TRIADIC ENGINE: COMPARADOR DE BASES DE DATOS (Sesgos)")
print("=========================================================")
print("Vamos a simular dos 'Cerebros' de IA diferentes.")
print("BD A: Modelo moderno y general ('all-MiniLM-L6-v2')")
print("BD B: Modelo especializado/diferente ('paraphrase-MiniLM-L3-v2')\n")

# Para simular dos bases de datos con pesos/sesgos distintos, cargaremos dos modelos de embeddings diferentes.
print("1. Cargando BD A (Modelo General)...")
encoder_A = ContinuousEncoder(model_name='all-MiniLM-L6-v2')

print("2. Cargando BD B (Modelo Alternativo/Sesgado)...")
encoder_B = ContinuousEncoder(model_name='paraphrase-MiniLM-L3-v2') 

validator = DiscreteValidator()

conceptos = [
    "Nurse", "Doctor", "Engineer", "Teacher", 
    "CEO", "Assistant", "Leader", "Follower",
    "Man", "Woman"
]

print(f"\n3. Vectorizando conceptos en ambas BDs...")
embeddings_A = encoder_A.encode(conceptos)
embeddings_B = encoder_B.encode(conceptos)

print("\n4. Proyectando a Espacio Primo (LSH k=8)...")
# Usamos la MISMA semilla para que los LSH hyperplanes partan del mismo azar, 
# así cualquier diferencia en el primo final se debe PURAMENTE a los sesgos del modelo.
mapper_A = DiscreteMapper(n_bits=8, seed=42)
mapper_B = DiscreteMapper(n_bits=8, seed=42)

prime_map_A = mapper_A.fit_transform(conceptos, embeddings_A)
prime_map_B = mapper_B.fit_transform(conceptos, embeddings_B)

print("\n=========================================================")
print("🔎 ANÁLISIS DE BRECHAS (GAP ANALYSIS) ENTRE BD_A Y BD_B")
print("=========================================================")

for word in conceptos:
    p_A = prime_map_A[word]
    p_B = prime_map_B[word]
    
    if p_A == p_B:
        print(f"✅ CONSENSO TOTAL en '{word}': Ambas BDs estructuran este concepto de forma idéntica.")
    else:
        gap = validator.explain_gap(p_A, p_B)
        shared_factors = validator._prime_factors(gap['shared'])
        only_A_factors = validator._prime_factors(gap['only_in_a'])
        only_B_factors = validator._prime_factors(gap['only_in_b'])
        
        print(f"⚠️ DISCREPANCIA SEMÁNTICA en '{word}':")
        print(f"   -> Consenso (Comparten): {shared_factors}")
        print(f"   -> BD_A ve a '{word}' añadiendo los factores: {only_A_factors}")
        print(f"   -> BD_B ve a '{word}' añadiendo los factores: {only_B_factors}")
        print("-" * 50)

print("\nConclusión: Tu algoritmo extrae las diferencias matemáticas exactas entre cómo dos redes neuronales distintas perciben el mismo concepto humano.")
