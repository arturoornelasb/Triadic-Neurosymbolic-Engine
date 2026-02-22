import os
import sys
import pandas as pd
import networkx as nx
from pyvis.network import Network
import random

# Ensure neurosym source is in path
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

print("=========================================================")
print("🕸️ TRIADIC ENGINE: MAPA ONTOLÓGICO (GRAFO HOLOGRÁFICO)")
print("=========================================================")

# 1. Load data
input_csv = "examples/data/wordnet_2k.csv"
if not os.path.exists(input_csv):
    print("Error: Ejecuta primero 'generate_real_data.py'")
    sys.exit(1)

df = pd.read_csv(input_csv)
conceptos_full = df["concept"].dropna().astype(str).tolist()

# Tomemos una muestra manejable para que el grafo no se vuelva una bola negra ilegible (ej. 150 nodos)
random.seed(42)
conceptos = random.sample(conceptos_full, min(150, len(conceptos_full)))

print(f"1. Cargando IA y Vectorizando {len(conceptos)} conceptos...")
encoder = ContinuousEncoder(model_name="all-MiniLM-L6-v2")
embeddings = encoder.encode(conceptos)

print("2. Extrayendo Primos Matemáticos (LSH k=10)...")
mapper = DiscreteMapper(n_bits=10, seed=42)
prime_map = mapper.fit_transform(conceptos, embeddings)
validator = DiscreteValidator()

# 3. Construyendo el Grafo (NetworkX)
print("3. Tejiendo el Grafo de Conexiones...")
G = nx.Graph()

# Añadir nodos (las palabras)
for word in conceptos:
    G.add_node(word, size=15, title=f"Factor: {prime_map[word]}", group=1)

# Añadir bordes (edges) basados en "Conexiones Matemáticas" (GCD)
# Si comparten factores (su GCD > 1), los conectamos.
operaciones_calculadas = 0
conexiones_establecidas = 0

for i in range(len(conceptos)):
    for j in range(i + 1, len(conceptos)):
        word_a = conceptos[i]
        word_b = conceptos[j]
        
        p_a = prime_map[word_a]
        p_b = prime_map[word_b]
        
        # El Triadic Engine calcula el Maximum Common Divisor
        gap = validator.explain_gap(p_a, p_b)
        factor_compartido = gap['shared']
        
        # Si comparten ingredientes lógicos (GCD no es 1)
        if factor_compartido > 1:
            factores_comunes = validator._prime_factors(factor_compartido)
            peso_conexion = len(factores_comunes) # Mientras más factores comparten, más gruesa la línea
            
            # Solo pintamos líneas fuertes (comparten al menos 1 factor)
            if peso_conexion >= 1:
                G.add_edge(word_a, word_b, weight=peso_conexion, value=peso_conexion, title=f"Comparten: {factores_comunes}")
                conexiones_establecidas += 1
                
        operaciones_calculadas += 1

print(f"   -> Verificaciones Matemáticas: {operaciones_calculadas}")
print(f"   -> Conexiones Fuertes Encontradas: {conexiones_establecidas}")

# 4. Generar HTML Interactivo con PyVis
print("\n4. Generando Visualización 3D Interactiva...")
output_file = "reports/grafo_triadico.html"
os.makedirs("reports", exist_ok=True)

# Creamos la red visual
net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", notebook=False)
net.barnes_hut(gravity=-8000) # Físicas para empujar los nodos y que se vea orgánico

net.from_nx(G)

net.save_graph(output_file)

print(f"\n✅ GRAFO GENERADO CON ÉXITO: {output_file}")
print("   -> ¡Busca este archivo HTML en tu explorador y ábrelo en Google Chrome/Firefox para ver tu red matemática interactiva!")
