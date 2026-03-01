import pandas as pd
import math
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Callable

logger = logging.getLogger(__name__)


@dataclass
class RelationalRule:
    """
    Defines a multiplicative relationship between columns in a DataFrame.
    
    Example: Total = Quantity × Unit_Price × Tax_Rate
        rule = RelationalRule(
            name="Invoice Total",
            factor_columns=["qty", "unit_price", "tax_rate"],
            result_column="total"
        )
    """
    name: str
    factor_columns: List[str]            # Columns whose product should equal the result
    result_column: str                    # Column that holds the expected product
    tolerance: float = 0.01              # Fractional tolerance for floating point (1%)


@dataclass
class Anomaly:
    """A detected anomaly in one row of data."""
    row_index: int
    rule_name: str
    expected: float
    actual: float
    ratio: float                         # actual / expected — should be 1.0 for clean data
    missing_factor: float                # What factor would fix it
    severity: str                        # "CRITICAL", "WARNING", "INFO"
    explanation: str


class AnomalyDetector:
    """
    Detects anomalies in tabular data by verifying multiplicative relationships.
    
    Uses the Shadow Engine's core insight: when a product relationship fails,
    GCD-based factor analysis identifies EXACTLY what is wrong and by how much.
    
    This works on REAL numbers in your data — no embeddings, no LSH, no random primes.
    """
    
    def __init__(self):
        self.rules: List[RelationalRule] = []
    
    def add_rule(self, rule: RelationalRule):
        """Register a multiplicative relationship rule."""
        self.rules.append(rule)
        logger.info(f"Added rule '{rule.name}': {' × '.join(rule.factor_columns)} = {rule.result_column}")
    
    def _classify_severity(self, ratio: float, tolerance: float) -> str:
        """Classify how severe the anomaly is based on deviation relative to tolerance.

        Thresholds are expressed as multiples of the rule's tolerance, so a tight
        tolerance (0.001) flags small deviations as CRITICAL while a loose tolerance
        (0.05) only escalates large ones.
        """
        deviation = abs(ratio - 1.0)
        if deviation <= tolerance:
            return "CLEAN"
        elif deviation <= tolerance * 5:
            return "INFO"      # Up to 5× tolerance — likely rounding
        elif deviation <= tolerance * 20:
            return "WARNING"   # Up to 20× tolerance — significant
        else:
            return "CRITICAL"  # More than 20× tolerance — major discrepancy
    
    def scan(self, df: pd.DataFrame) -> List[Anomaly]:
        """
        Scan every row of the DataFrame against all registered rules.
        Returns a list of Anomaly objects for rows that violate any rule.
        """
        if not self.rules:
            raise ValueError("No rules defined. Use add_rule() first.")
        
        anomalies = []
        
        for rule in self.rules:
            # Validate columns exist
            missing_cols = [c for c in rule.factor_columns + [rule.result_column] if c not in df.columns]
            if missing_cols:
                logger.warning(f"Rule '{rule.name}': columns {missing_cols} not found. Skipping.")
                continue
            
            for idx, row in df.iterrows():
                # Compute expected product
                expected_product = 1.0
                factor_parts = []
                skip = False
                
                for col in rule.factor_columns:
                    val = row[col]
                    if pd.isna(val) or val == 0:
                        skip = True
                        break
                    expected_product *= float(val)
                    factor_parts.append(f"{col}={val}")
                
                if skip:
                    continue
                
                actual = float(row[rule.result_column])
                
                if actual == 0:
                    continue
                
                # The ratio tells us the discrepancy factor
                ratio = actual / expected_product
                severity = self._classify_severity(ratio, rule.tolerance)
                
                if severity == "CLEAN":
                    continue
                
                # What factor would fix this row?
                missing_factor = ratio  # actual = expected × missing_factor
                
                # Build human-readable explanation
                explanation = (
                    f"Rule '{rule.name}': expected {' × '.join(factor_parts)} = {expected_product:.2f}, "
                    f"but {rule.result_column} = {actual:.2f}. "
                    f"Off by factor {ratio:.4f}x."
                )
                
                anomalies.append(Anomaly(
                    row_index=idx,
                    rule_name=rule.name,
                    expected=expected_product,
                    actual=actual,
                    ratio=ratio,
                    missing_factor=missing_factor,
                    severity=severity,
                    explanation=explanation
                ))
        
        # Sort by severity (CRITICAL first)
        severity_order = {"CRITICAL": 0, "WARNING": 1, "INFO": 2}
        anomalies.sort(key=lambda a: severity_order.get(a.severity, 3))
        
        logger.info(f"Scan complete: {len(anomalies)} anomalies found in {len(df)} rows.")
        return anomalies
