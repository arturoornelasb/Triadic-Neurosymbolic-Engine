import math
from fractions import Fraction
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """
    Result of a discrete algebraic validation.
    """
    is_valid: bool
    output_value: float
    simplicity_k: float
    is_hypothetical: bool = False
    missing_factor: int = None
    trace: str = ""

class DiscreteValidator:
    """
    Abductive Algebraic Resolver for Relational Data.
    Merges Triadic Relational Framework and Shadow Engine logic.
    Project continuous latent spaces into discrete integer factors for rigorous validation.
    """
    def __init__(self):
        pass

    @staticmethod
    def _compute_simplicity(rule_a: int, rule_b: int) -> float:
        """
        Computes simplicity K constant based on the balancing rules.
        K = 1 / (a * b)
        """
        return float(Fraction(1, rule_a * rule_b))

    @staticmethod
    def discover_missing_factor(numerator: int, denominator: int) -> int:
        """
        Abductive Discovery: Identifies the missing integer factor that prevents
        the equation from resolving to a clean integer state (Topological Obstruction).
        """
        common = math.gcd(numerator, denominator)
        missing_link = denominator // common
        return missing_link

    def validate_relationship(self, source: int, transform1: int, transform2: int, 
                              rule_a: int = 1, rule_b: int = 1) -> ValidationResult:
        """
        Validates if the relationship: 
        (rule_a * transform1 * transform2) / (rule_b * source) resolves cleanly.
        
        Args:
            source (C1): The base concept/entity (e.g. integer projection of a word)
            transform1 (C2): The first relational modifier
            transform2 (C3): The second relational modifier
            rule_a, rule_b: The invariant ratio (default 1:1)
            
        Returns:
            ValidationResult containing the expected target (C4) or the missing abductive factor.
        """
        # 1. GCD Normalization (Projection to Terminal Object)
        gcd_in = math.gcd(source, math.gcd(transform1, transform2))
        c1_p = source // gcd_in
        c2_p = transform1 // gcd_in
        c3_p = transform2 // gcd_in
        
        # 2. Relational Path Construction
        numerator = rule_a * c2_p * c3_p
        denominator = rule_b * c1_p
        
        trace = f"({rule_a} * {c2_p} * {c3_p}) / ({rule_b} * {c1_p})"
        simplicity = self._compute_simplicity(rule_a, rule_b)
        
        # 3. Validation and Abductive Discovery
        if numerator % denominator != 0:
            logger.info("Topological obstruction detected. Initiating Abductive Discovery.")
            missing = self.discover_missing_factor(numerator, denominator)
            
            # Hypothetical resolution
            c4_hypothetical_p = (numerator * missing) // denominator
            c4_final = c4_hypothetical_p * gcd_in
            
            return ValidationResult(
                is_valid=False,
                output_value=c4_final,
                simplicity_k=simplicity,
                is_hypothetical=True,
                missing_factor=missing,
                trace=trace + f" -> Fails. Requires factor: {missing}"
            )
            
        # 4. Success Path
        c4_p = numerator // denominator
        c4 = c4_p * gcd_in
        
        return ValidationResult(
            is_valid=True,
            output_value=c4,
            simplicity_k=simplicity,
            is_hypothetical=False,
            trace=trace + f" -> Success: {c4}"
        )

    def analogy_prediction(self, source_a: int, source_b: int, target_a: int) -> ValidationResult:
        """
        Resolves semantic analogies (A:B :: C:D)
        Returns the integer prediction for D.
        Formula: D = (C * B) / A
        """
        return self.validate_relationship(
            source=source_a, 
            transform1=source_b, 
            transform2=target_a,
            rule_a=1, 
            rule_b=1
        )

    # --- Logical Verification API (Unique to Prime Representation) ---

    @staticmethod
    def _prime_factors(n: int) -> list[int]:
        """Returns the unique prime factors of n."""
        if n <= 1:
            return []
        factors = []
        d = 2
        temp = n
        while d * d <= temp:
            if temp % d == 0:
                factors.append(d)
                while temp % d == 0:
                    temp //= d
            d += 1
        if temp > 1:
            factors.append(temp)
        return factors

    @staticmethod
    def subsumes(a: int, b: int) -> bool:
        """
        Logical Subsumption: Does concept A contain ALL semantic features of B?
        
        In prime space: A subsumes B iff B divides A (A % B == 0).
        Example: King(2*3*5) subsumes Male(3) → True (King contains 'male' feature)
        
        This operation is IMPOSSIBLE with Hamming distance over bitstrings.
        Hamming only tells you "how many bits differ", not containment.
        """
        return a % b == 0

    @staticmethod
    def compose(*concepts: int) -> int:
        """
        Algebraic Composition: Create a new concept by combining features.
        
        In prime space: the union of all semantic features = LCM of the integers.
        This preserves uniqueness: compose(Royal, Male) contains all features of both.
        
        Example: compose(Royal=2*5, Male=3*7) → 2*3*5*7 = 210
        
        This operation is IMPOSSIBLE with vector cosine similarity.
        """
        result = concepts[0]
        for c in concepts[1:]:
            result = (result * c) // math.gcd(result, c)  # LCM
        return result

    @staticmethod
    def explain_gap(a: int, b: int) -> dict:
        """
        Abductive Discovery: Explains exactly WHY two concepts differ.
        
        Returns:
            - shared: GCD(a, b) — the common semantic backbone
            - only_in_a: features present in A but missing from B
            - only_in_b: features present in B but missing from A
        
        Example: explain_gap(King=30, Queen=10)
            → shared=10, only_in_king=3 (the 'male' factor), only_in_queen=1
        
        This deterministic decomposition is IMPOSSIBLE with continuous vectors.
        Cosine similarity only says "0.87 similar" — not WHICH features differ.
        """
        shared = math.gcd(a, b)
        only_in_a = a // shared
        only_in_b = b // shared
        return {
            "shared": shared,
            "only_in_a": only_in_a,
            "only_in_b": only_in_b,
            "a_contains_b": (a % b == 0),
            "b_contains_a": (b % a == 0),
        }

