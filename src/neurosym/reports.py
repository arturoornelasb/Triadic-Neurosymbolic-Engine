"""
Report Generator for the Triadic Neurosymbolic Engine.

Generates exportable audit reports in HTML, JSON, and CSV formats.
No external dependencies — uses Python stdlib only.
"""
import json
import csv
import io
import html as html_mod
from datetime import datetime, timezone
from typing import Dict, List


def _engine_version() -> str:
    try:
        from importlib.metadata import version
        return version("triadic-engine")
    except Exception:
        return "0.4.0"


class ReportGenerator:
    """
    Generates structured audit reports from Triadic Engine results.
    Supports HTML (standalone, embeddable), JSON, and CSV output.
    """
    
    def __init__(self, title: str = "Triadic Neurosymbolic Audit Report"):
        self.title = title
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.sections: List[dict] = []
    
    def add_encoding_section(self, prime_map: Dict[str, int], model: str, lsh_bits: int, factorize_fn=None):
        """Add the prime encoding results to the report."""
        rows = []
        for concept, prime in sorted(prime_map.items(), key=lambda x: x[1]):
            factors = factorize_fn(prime) if factorize_fn else []
            rows.append({
                "concept": concept,
                "prime_factor": prime,
                "decomposition": factors,
            })
        
        self.sections.append({
            "type": "encoding",
            "title": "Prime Factor Encoding",
            "model": model,
            "lsh_bits": lsh_bits,
            "total_concepts": len(prime_map),
            "unique_primes": len(set(prime_map.values())),
            "rows": rows,
        })
    
    def add_audit_section(self, discrepancies: List[dict], model_a: str, model_b: str, 
                          total_pairs: int, total_concepts: int):
        """Add the model audit discrepancy results to the report."""
        self.sections.append({
            "type": "audit",
            "title": f"Model Comparison: {model_a} vs {model_b}",
            "model_a": model_a,
            "model_b": model_b,
            "total_concepts": total_concepts,
            "total_pairs": total_pairs,
            "discrepancies_found": len(discrepancies),
            "discrepancy_rate": f"{len(discrepancies) / max(1, total_pairs) * 100:.1f}%",
            "rows": discrepancies,
        })
    
    def add_graph_section(self, edges: list, node_count: int):
        """Add graph statistics to the report."""
        self.sections.append({
            "type": "graph",
            "title": "Semantic Graph Statistics",
            "total_nodes": node_count,
            "total_edges": len(edges),
            "avg_edge_weight": sum(e[2] for e in edges) / max(1, len(edges)) if edges else 0,
            "top_connections": [
                {"concept_a": a, "concept_b": b, "weight": w, "shared_primes": p}
                for a, b, w, p in sorted(edges, key=lambda x: -x[2])[:20]
            ],
        })
    
    # === Export Methods ===
    
    def to_json(self, indent: int = 2) -> str:
        """Export the full report as a JSON string."""
        report = {
            "title": self.title,
            "generated_at": self.timestamp,
            "engine": f"Triadic Neurosymbolic Engine v{_engine_version()}",
            "sections": self.sections,
        }
        return json.dumps(report, indent=indent, default=str)
    
    def to_csv(self) -> str:
        """Export encoding and audit rows as CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        
        for section in self.sections:
            if section["type"] == "encoding":
                writer.writerow(["--- Encoding Results ---"])
                writer.writerow(["Concept", "Prime Factor", "Decomposition"])
                for row in section["rows"]:
                    writer.writerow([row["concept"], row["prime_factor"], str(row["decomposition"])])
                writer.writerow([])
            
            elif section["type"] == "audit":
                writer.writerow([f"--- Audit: {section['model_a']} vs {section['model_b']} ---"])
                writer.writerow(["Concept A", "Concept B", f"Dist {section['model_a']}", f"Dist {section['model_b']}", "Chain"])
                for row in section["rows"]:
                    writer.writerow([
                        row.get("concept_a", ""),
                        row.get("concept_b", ""),
                        row.get("distance_model_a", ""),
                        row.get("distance_model_b", ""),
                        row.get("chain", ""),
                    ])
                writer.writerow([])
        
        return output.getvalue()
    
    def to_html(self) -> str:
        """Export a standalone HTML report with embedded dark-theme styling."""
        html_sections = []
        
        for section in self.sections:
            if section["type"] == "encoding":
                rows_html = ""
                for row in section["rows"]:
                    factors_str = ", ".join(str(f) for f in row["decomposition"])
                    safe_concept = html_mod.escape(str(row['concept']))
                    rows_html += f"""
                    <tr>
                        <td>{safe_concept}</td>
                        <td class="mono">{row['prime_factor']}</td>
                        <td class="mono">[{factors_str}]</td>
                    </tr>"""
                
                html_sections.append(f"""
                <div class="section">
                    <h2>🧬 {section['title']}</h2>
                    <div class="stats">
                        <div class="stat"><span class="stat-value">{section['total_concepts']}</span><span class="stat-label">Concepts</span></div>
                        <div class="stat"><span class="stat-value">{section['unique_primes']}</span><span class="stat-label">Unique Clusters</span></div>
                        <div class="stat"><span class="stat-value">{section['model']}</span><span class="stat-label">Model</span></div>
                        <div class="stat"><span class="stat-value">{section['lsh_bits']}</span><span class="stat-label">LSH Bits</span></div>
                    </div>
                    <table>
                        <thead><tr><th>Concept</th><th>Prime Factor</th><th>Decomposition</th></tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>""")
            
            elif section["type"] == "audit":
                rows_html = ""
                for row in section["rows"]:
                    safe_a = html_mod.escape(str(row.get('concept_a', '')))
                    safe_b = html_mod.escape(str(row.get('concept_b', '')))
                    safe_chain = html_mod.escape(str(row.get('chain', '')))
                    rows_html += f"""
                    <tr>
                        <td>{safe_a}</td>
                        <td>{safe_b}</td>
                        <td class="mono">{html_mod.escape(str(row.get('distance_model_a', '')))}</td>
                        <td class="mono">{html_mod.escape(str(row.get('distance_model_b', '')))}</td>
                        <td>{safe_chain}</td>
                    </tr>"""
                
                badge_class = "badge-ok" if section['discrepancies_found'] == 0 else "badge-warn"
                html_sections.append(f"""
                <div class="section">
                    <h2>🤖 {section['title']}</h2>
                    <div class="stats">
                        <div class="stat"><span class="stat-value">{section['total_concepts']}</span><span class="stat-label">Concepts</span></div>
                        <div class="stat"><span class="stat-value">{section['total_pairs']}</span><span class="stat-label">Pairs Tested</span></div>
                        <div class="stat"><span class="stat-value {badge_class}">{section['discrepancies_found']}</span><span class="stat-label">Discrepancies</span></div>
                        <div class="stat"><span class="stat-value">{section['discrepancy_rate']}</span><span class="stat-label">Bias Rate</span></div>
                    </div>
                    <table>
                        <thead><tr><th>Concept A</th><th>Concept B</th><th>Dist Model A</th><th>Dist Model B</th><th>Chain</th></tr></thead>
                        <tbody>{rows_html}</tbody>
                    </table>
                </div>""")
            
            elif section["type"] == "graph":
                conn_html = ""
                for c in section.get("top_connections", []):
                    safe_a = html_mod.escape(str(c['concept_a']))
                    safe_b = html_mod.escape(str(c['concept_b']))
                    conn_html += f"""
                    <tr>
                        <td>{safe_a}</td>
                        <td>{safe_b}</td>
                        <td class="mono">{c['weight']}</td>
                        <td class="mono">{c['shared_primes']}</td>
                    </tr>"""
                
                html_sections.append(f"""
                <div class="section">
                    <h2>🌐 {section['title']}</h2>
                    <div class="stats">
                        <div class="stat"><span class="stat-value">{section['total_nodes']}</span><span class="stat-label">Nodes</span></div>
                        <div class="stat"><span class="stat-value">{section['total_edges']}</span><span class="stat-label">Edges</span></div>
                        <div class="stat"><span class="stat-value">{section['avg_edge_weight']:.1f}</span><span class="stat-label">Avg Weight</span></div>
                    </div>
                    <h3>Top 20 Strongest Connections</h3>
                    <table>
                        <thead><tr><th>Concept A</th><th>Concept B</th><th>Weight</th><th>Shared Primes</th></tr></thead>
                        <tbody>{conn_html}</tbody>
                    </table>
                </div>""")
        
        sections_joined = "\n".join(html_sections)
        
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ 
            font-family: 'Segoe UI', -apple-system, system-ui, sans-serif; 
            background: #0d1117; color: #c9d1d9; 
            padding: 40px; line-height: 1.6;
        }}
        h1 {{ 
            color: #e94560; font-size: 2em; margin-bottom: 8px;
            background: linear-gradient(135deg, #e94560, #0f3460);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ color: #8b949e; margin-bottom: 32px; }}
        .section {{ 
            background: #161b22; border: 1px solid #30363d; 
            border-radius: 12px; padding: 24px; margin-bottom: 24px;
        }}
        h2 {{ color: #58a6ff; margin-bottom: 16px; }}
        h3 {{ color: #8b949e; margin: 16px 0 8px; }}
        .stats {{ 
            display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap;
        }}
        .stat {{ 
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            border: 1px solid #0f3460; border-radius: 8px; 
            padding: 12px 20px; text-align: center; min-width: 120px;
        }}
        .stat-value {{ display: block; font-size: 1.5em; font-weight: 700; color: #e94560; }}
        .stat-label {{ display: block; font-size: 0.8em; color: #8b949e; margin-top: 4px; }}
        table {{ 
            width: 100%; border-collapse: collapse; margin-top: 12px;
        }}
        th {{ 
            background: #21262d; color: #58a6ff; padding: 10px 14px; 
            text-align: left; font-weight: 600; border-bottom: 2px solid #30363d;
        }}
        td {{ 
            padding: 8px 14px; border-bottom: 1px solid #21262d; 
        }}
        tr:hover {{ background: #1c2128; }}
        .mono {{ font-family: 'Fira Code', 'Consolas', monospace; font-size: 0.9em; }}
        .badge-ok {{ color: #3fb950; }}
        .badge-warn {{ color: #f85149; }}
        .footer {{ 
            text-align: center; color: #484f58; margin-top: 40px; 
            padding-top: 20px; border-top: 1px solid #21262d;
        }}
    </style>
</head>
<body>
    <h1>⚛️ {self.title}</h1>
    <p class="subtitle">Generated {self.timestamp} · Triadic Neurosymbolic Engine v{_engine_version()}</p>
    {sections_joined}
    <div class="footer">
        <p>© 2026 J. Arturo Ornelas Brand · Triadic Neurosymbolic Engine</p>
        <p>Deterministic Algebraic Framework for AI Interpretability</p>
    </div>
</body>
</html>"""
    
    def save(self, path: str, format: str = "html"):
        """Save the report to a file."""
        if format == "html":
            content = self.to_html()
        elif format == "json":
            content = self.to_json()
        elif format == "csv":
            content = self.to_csv()
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'html', 'json', or 'csv'.")
        
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
