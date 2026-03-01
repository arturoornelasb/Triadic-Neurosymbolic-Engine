"""
SQLite persistence layer for the Triadic Neurosymbolic Engine.
Stores prime indexes so they survive server restarts.
"""
import sqlite3
import os
import csv
import io
from typing import Dict, List, Optional
from datetime import datetime, timezone


class PrimeIndexDB:
    """
    Persistent storage for discrete prime indexes using SQLite.
    
    Default location: ~/.neurosym/index.db
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_dir = os.path.join(os.path.expanduser("~"), ".neurosym")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "index.db")
        
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS concepts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    prime_factor INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    lsh_bits INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(text, model, lsh_bits, seed)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    concept_a TEXT NOT NULL,
                    concept_b TEXT NOT NULL,
                    model_a TEXT NOT NULL,
                    model_b TEXT NOT NULL,
                    dist_model_a TEXT,
                    dist_model_b TEXT,
                    chain TEXT,
                    created_at TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_concepts_model 
                ON concepts(model, lsh_bits, seed)
            """)
    
    def save_index(self, prime_map: Dict[str, int], model: str, lsh_bits: int, seed: int):
        """Save a prime map to the database. Upserts on conflict."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for text, prime_factor in prime_map.items():
                conn.execute("""
                    INSERT INTO concepts (text, prime_factor, model, lsh_bits, seed, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(text, model, lsh_bits, seed) 
                    DO UPDATE SET prime_factor = excluded.prime_factor, created_at = excluded.created_at
                """, (text, prime_factor, model, lsh_bits, seed, now))
    
    def load_index(self, model: str, lsh_bits: int, seed: int) -> Dict[str, int]:
        """Load a previously saved prime map from the database."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT text, prime_factor FROM concepts WHERE model = ? AND lsh_bits = ? AND seed = ?",
                (model, lsh_bits, seed)
            ).fetchall()
        return {text: prime for text, prime in rows}
    
    def save_audit(self, results: List[dict], model_a: str, model_b: str):
        """Save audit discrepancy results."""
        now = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            for r in results:
                conn.execute("""
                    INSERT INTO audit_results (concept_a, concept_b, model_a, model_b, dist_model_a, dist_model_b, chain, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    r.get("concept_a", ""), r.get("concept_b", ""),
                    model_a, model_b,
                    str(r.get("distance_model_a", "")), str(r.get("distance_model_b", "")),
                    r.get("chain", ""), now
                ))
    
    def list_indexes(self) -> List[dict]:
        """List all stored indexes with their metadata."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute("""
                SELECT model, lsh_bits, seed, COUNT(*) as concept_count, MAX(created_at) as last_updated
                FROM concepts
                GROUP BY model, lsh_bits, seed
                ORDER BY last_updated DESC
            """).fetchall()
        return [
            {"model": r[0], "lsh_bits": r[1], "seed": r[2], "concept_count": r[3], "last_updated": r[4]}
            for r in rows
        ]
    
    def export_csv(self, model: str = None) -> str:
        """Export concepts to CSV string."""
        with sqlite3.connect(self.db_path) as conn:
            if model:
                rows = conn.execute(
                    "SELECT text, prime_factor, model, lsh_bits, seed, created_at FROM concepts WHERE model = ?",
                    (model,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT text, prime_factor, model, lsh_bits, seed, created_at FROM concepts"
                ).fetchall()
        
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["text", "prime_factor", "model", "lsh_bits", "seed", "created_at"])
        writer.writerows(rows)
        return output.getvalue()
    
    def delete_index(self, model: str, lsh_bits: int, seed: int) -> int:
        """Delete all concepts for a given named index. Returns number of rows deleted."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "DELETE FROM concepts WHERE model = ? AND lsh_bits = ? AND seed = ?",
                (model, lsh_bits, seed)
            )
        return cursor.rowcount

    def concept_count(self) -> int:
        """Return total number of stored concepts."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM concepts").fetchone()
        return row[0] if row else 0
