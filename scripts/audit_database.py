import time
import os
import sys

# Asegurar que el Engine está en el path (root dir)
ENGINE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ENGINE_PATH)

from src.neurosym.encoder import ContinuousEncoder, DiscreteMapper
from src.neurosym.triadic import DiscreteValidator

print("=========================================================")
print("🤖 TRIADIC ENGINE: AUDITOR DE BASES DE DATOS (WordNet)")
print("=========================================================")

print("\n1. Cargando Modelo de Lenguaje (Encoder)...")
encoder = ContinuousEncoder()
validator = DiscreteValidator()

# Nuestra pequeña Base de Datos Industrial (Conceptos a auditar)
conceptos_db = [
    "King", "Man", "Queen", "Woman", 
    "Doctor", "Nurse", "Car", "Vehicle", 
    "Dog", "Animal", "Apple", "Fruit"
]

print(f"\n2. Vectorizando la Base de Datos ({len(conceptos_db)} conceptos)...")
embeddings = encoder.encode(conceptos_db)

print("\n3. Mapeando Vectores (Decimales) a Factores Primos (Enteros)...")
# Usamos K=8 para una resolución moderada
mapper = DiscreteMapper(n_bits=8, seed=42)
prime_map = mapper.fit_transform(conceptos_db, embeddings)

for word, primo in prime_map.items():
    factores = validator._prime_factors(primo)
    print(f"   - {word}: {primo} (Factores: {factores})")


print("\n=========================================================")
print("🔎 INICIANDO AUDITORÍA LÓGICA DE CONEXIONES")
print("=========================================================")
print("El sistema dividirá los conceptos. Si hay residuo, alertará de la inconsistencia.")

def auditar_conexion(concepto_general, concepto_especifico):
    p_gen = prime_map[concepto_general]
    p_esp = prime_map[concepto_especifico]
    
    # Matemáticamente: Si Especifico (EJ. Perro) ES UN General (Ej. Animal)
    # Entonces Especifico DEBE ser divisible por General sin residuo.
    # Es decir: p_esp % p_gen == 0
    
    if p_esp % p_gen == 0:
        print(f"✅ LÓGICA PERFECTA: '{concepto_especifico}' ({p_esp}) subsume a '{concepto_general}' ({p_gen}).")
    else:
        # Hay inconsistencia. Averiguamos qué le falta.
        gap = validator.explain_gap(p_esp, p_gen)
        print(f"❌ INCONSISTENCIA: La BD dice que '{concepto_especifico}' es un tipo de '{concepto_general}' pero matemáticamente fallan.")
        print(f"   -> Comparten los factores: {validator._prime_factors(gap['shared'])}")
        print(f"   -> A '{concepto_especifico}' LE FALTAN los factores: {validator._prime_factors(gap['only_in_b'])} para ser un verdadero '{concepto_general}'.")

print("\n--- Test 1: Jerarquías ---")
auditar_conexion("Vehicle", "Car")
auditar_conexion("Animal", "Dog")
auditar_conexion("Fruit", "Apple")

print("\n--- Test 2: Inconsistencias (Analogías Rotas) ---")
# Auditar analogía famosa: Es (Rey / Hombre) proporcional a (Reina / Mujer)?
# Rey * Mujer = X
# Queen * Hombre = Y
k = prime_map["King"]
m = prime_map["Man"]
q = prime_map["Queen"]
w = prime_map["Woman"]

target_queen = validator.analogy_prediction(source_a=m, source_b=k, target_a=w)
print(f"Analogía: Man es a King como Woman es a... ?")
if target_queen.is_valid and target_queen.output_value == q:
    print(f"✅ Analogía Perfecta en la BD. Output esperado y real coinciden en: {q}")
else:
    print(f"❌ FALLA LÓGICA EN LA BBDD DE LA IA!")
    print(f"   -> El sistema intentó deducir 'Queen' matemáticamente: (King * Woman) / Man")
    print(f"   -> {target_queen.trace}")
    print(f"   -> Por lo tanto, la conexión semántica en la red neuronal tiene un 'glitch'. No es exacta.")

print("\n=========================================================")
print("Auditoría Finalizada.")
