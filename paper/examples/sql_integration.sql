-- =============================================
-- NEUROSYMBOLIC PRIME ENCODING → SQL
-- Archivo: examples/sql_integration.sql
-- =============================================

CREATE TABLE semantic_concepts (
    concept_name     TEXT PRIMARY KEY,
    prime_encoding   BIGINT NOT NULL   -- o NUMERIC si usas k>20
);

-- Datos reales del paper (k=8, seed=42)
INSERT INTO semantic_concepts (concept_name, prime_encoding) VALUES
    ('King',    3230),   -- 2×5×17×19
    ('Queen',   1615),   --   5×17×19
    ('Man',     85085),  -- 5×7×11×13×17
    ('Woman',   1105),   -- 5×13×17
    ('Dog',     19019),  -- 7×11×13×19
    ('Cat',     6783),   -- 3×7×17×19
    ('Love',    7735),   -- 5×7×13×17
    ('Hate',    1105);   -- 5×13×17

-- =============================================
-- OPERACIONES DEL PAPER DIRECTO EN SQL
-- =============================================

-- 1. Logical Subsumption
-- "¿Qué conceptos son más específicos que Queen?" (King % Queen == 0)
SELECT 
    c1.concept_name AS more_specific,
    c2.concept_name AS subsumed
FROM semantic_concepts c1
CROSS JOIN semantic_concepts c2
WHERE c1.prime_encoding % c2.prime_encoding = 0
  AND c1.concept_name != c2.concept_name;

-- 2. Algebraic Composition (LCM)
-- "Crea el concepto King + Dog"
WITH compose AS (
    SELECT 
        'King+Dog' AS new_concept,
        ABS(3230::numeric * 19019 / gcd(3230, 19019)) AS lcm_value
)
SELECT 
    new_concept,
    lcm_value,
    lcm_value % 3230  AS subsumes_king,   -- debe ser 0
    lcm_value % 19019 AS subsumes_dog;    -- debe ser 0

-- 3. Abductive Gap Analysis (el más bonito)
SELECT 
    'King' AS A,
    'Man' AS B,
    gcd(3230, 85085) AS shared_gcd,
    3230 / gcd(3230, 85085) AS unique_to_king,
    85085 / gcd(3230, 85085) AS unique_to_man;

-- Resultado esperado de la última query:
-- shared_gcd     = 85     → {5,17}
-- unique_to_king = 38     → {2,19}
-- unique_to_man  = 1001   → {7,11,13}
