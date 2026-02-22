# Prime Factorization as a Neurosymbolic Bridge: Projecting Continuous Embeddings into Discrete Algebraic Space for Deterministic Verification

**Arturo Ornelas**

---

## Abstract

We propose a method for bridging continuous neural embeddings and discrete symbolic reasoning by projecting dense vector representations into composite prime integers via Locality Sensitive Hashing (LSH). Each LSH hyperplane is assigned a unique prime number; a concept's discrete representation is the product of all primes corresponding to its active hyperplanes. This composite prime encoding enables three algebraic operations impossible under standard vector similarity metrics: (1) **logical subsumption** via divisibility testing, (2) **concept composition** via least common multiple, and (3) **abductive gap analysis** via GCD factorization. We benchmark the approach on a 20,000-word WordNet vocabulary and demonstrate 12× faster validation throughput compared to cosine similarity, while providing deterministic rather than probabilistic outputs. We discuss limitations including sensitivity to LSH hyperparameters and the distinction between hash-bucket coincidence and genuine semantic containment.

**Keywords:** Neurosymbolic AI, Prime Factorization, Locality Sensitive Hashing, Discrete Reasoning, Algebraic Verification

---

## 1. Introduction

Neural networks encode semantic concepts as dense vectors in $\mathbb{R}^n$, enabling powerful approximate reasoning through distance metrics such as cosine similarity. However, continuous representations inherently produce probabilistic outputs: two vectors are "87% similar," but the system cannot determine whether one concept logically *contains* another, nor can it explain *which specific features* differ between two concepts.

Symbolic AI systems, by contrast, operate on discrete structures that support formal logical operations—subsumption, composition, and contradiction detection—but lack the ability to handle the noise and ambiguity inherent in natural language. The neurosymbolic integration problem seeks to bridge these two paradigms (Garcez et al., 2019; Sarker et al., 2021).

We introduce a lightweight bridging mechanism that projects continuous embeddings into composite prime integers. The key insight is that **prime factorization endows integers with a lattice structure** where:

- **Divisibility** corresponds to logical subsumption
- **Least Common Multiple (LCM)** corresponds to feature union (composition)
- **Greatest Common Divisor (GCD)** decomposes shared versus unique features

These operations are exact, deterministic, and computationally inexpensive—requiring only integer arithmetic—while the underlying semantic signal is derived from pre-trained neural embeddings.

### 1.1 Contributions

1. A **composite prime encoding** scheme that maps dense vectors to integers via LSH, where each prime factor represents a semantic feature dimension.
2. Three **algebraic verification operations** (subsumption, composition, gap analysis) that are provably impossible under standard vector similarity.
3. An empirical evaluation on WordNet ($N = 20{,}000$) demonstrating computational efficiency gains with honest analysis of semantic accuracy tradeoffs.
4. A complete open-source implementation with interactive demonstration.

---

## 2. Related Work

**Neurosymbolic AI.** Recent work has focused on integrating neural and symbolic reasoning. DeepProbLog (Manhaeve et al., 2018) embeds neural predicates into probabilistic logic programs. Logic Tensor Networks (Badreddine et al., 2022) ground first-order logic in differentiable tensor operations. Our approach differs by operating *post-hoc* on frozen embeddings, requiring no joint training.

**Locality Sensitive Hashing.** LSH (Indyk & Motwani, 1998) maps similar vectors to identical hash codes with high probability via random hyperplane projections (Charikar, 2002). Standard LSH produces binary codes compared via Hamming distance. We extend this by mapping each hyperplane to a prime number, producing composite integers with richer algebraic structure than bitstrings.

**Integer Representations in NLP.** Word2Vec arithmetic (Mikolov et al., 2013) famously demonstrated $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$, but this operates in continuous space with approximate equality. Our framework reframes such relationships as exact integer divisibility tests.

**Knowledge Graph Embedding.** Methods like TransE (Bordes et al., 2013) and RotatE (Sun et al., 2019) embed relations as arithmetic operations on vectors. Our contribution is complementary: we project *any* pre-trained embedding into a discrete space where algebraic verification becomes exact.

---

## 3. Method

### 3.1 Continuous Encoding

We use a pre-trained sentence transformer $f: \mathcal{S} \rightarrow \mathbb{R}^d$ (default: `all-MiniLM-L6-v2`, $d = 384$) to encode natural language concepts into dense vectors. Any embedding model can be substituted.

### 3.2 Composite Prime Projection

Given a set of $k$ random hyperplanes $\{h_1, \ldots, h_k\}$ sampled from $\mathcal{N}(0, I_d)$, and a mapping $\pi: \{1, \ldots, k\} \rightarrow \mathbb{P}$ assigning the $i$-th prime number $p_i$ to each hyperplane, we define the composite prime encoding:

$$
\Phi(x) = \prod_{i=1}^{k} p_i^{\, \mathbb{1}[h_i \cdot f(x) > 0]}
$$

where $\mathbb{1}[\cdot]$ is the indicator function. That is, a concept's integer representation is the product of all primes whose corresponding hyperplanes yield a positive projection.

**Properties:**
- Two concepts share a prime factor $p_i$ if and only if they fall on the same side of hyperplane $h_i$.
- Semantically similar concepts (by the original embedding distance) are likely to share more active hyperplanes and thus more prime factors (by LSH's locality-preserving guarantee).
- The representation is deterministic given fixed hyperplanes.

### 3.3 Algebraic Verification Operations

The composite prime representation enables three operations that are impossible under standard cosine similarity or Hamming distance on bitstrings:

**Operation 1: Logical Subsumption**

$$
\text{subsumes}(A, B) \iff \Phi(A) \bmod \Phi(B) = 0
$$

This tests whether concept $A$ contains *all* semantic features (active hyperplanes) of concept $B$. Cosine similarity returns a scalar (e.g., 0.87) with no directional containment information. Hamming distance counts differing bits but cannot distinguish containment from partial overlap.

**Operation 2: Algebraic Composition**

$$
\text{compose}(A, B) = \text{lcm}(\Phi(A), \Phi(B))
$$

This creates a new integer whose prime factorization is the union of both inputs' features. The composed concept is guaranteed to subsume both inputs:

$$
\text{compose}(A, B) \bmod \Phi(A) = 0 \quad \wedge \quad \text{compose}(A, B) \bmod \Phi(B) = 0
$$

No analogous operation exists for continuous vectors because vector addition does not guarantee that the result "contains" the addends in any formal sense.

**Operation 3: Abductive Gap Analysis**

$$
\text{shared}(A, B) = \gcd(\Phi(A), \Phi(B))
$$
$$
\text{unique\_to\_A}(A, B) = \Phi(A) \, / \, \gcd(\Phi(A), \Phi(B))
$$
$$
\text{unique\_to\_B}(A, B) = \Phi(B) \, / \, \gcd(\Phi(A), \Phi(B))
$$

This decomposes the relationship between two concepts into three components: how they are alike (shared backbone), what $A$ has that $B$ lacks, and vice versa. Each component is itself a product of primes that can be factored to identify specific semantic dimensions. Cosine similarity can only report a single scalar distance.

### 3.4 Resolution–Collision Tradeoff

The parameter $k$ (number of hyperplanes/bits) controls a fundamental tradeoff:

- **Small $k$ ($\leq 4$):** High collision rate. Many semantically distinct concepts receive the same prime encoding, enabling spurious subsumption.
- **Large $k$ ($\geq 16$):** Near-unique encodings. Very few concepts share any prime factors, making GCD trivially 1 and subsumption trivially false.
- **Moderate $k$ (6–10):** A useful operating regime where semantically related concepts share some—but not all—prime factors.

We investigate this tradeoff empirically in Section 4.

---

## 4. Experimental Evaluation

### 4.1 Setup

We encode 20,000 words from WordNet using `all-MiniLM-L6-v2` and project them through our composite prime mapper with $k \in \{3, 6, 8, 12, 16\}$ using 10 random seeds each.

### 4.2 Computational Efficiency

We compare the wall-clock time for 50,000 pairwise relationship validations:

| Method | Time (s) | Operation | Output Type |
|:---|:---|:---|:---|
| Cosine Similarity | 11.21 | Dense vector dot product | Probabilistic ($\in [0,1]$) |
| Composite Prime (ours) | 0.93 | Integer GCD + modulo | Deterministic ($\in \mathbb{Z}$) |

The discrete representation achieves a **12× speedup** because integer arithmetic (GCD, modulo) is computationally cheaper than floating-point dot products, and the integers fit in CPU registers without matrix operations.

### 4.3 Subsumption Analysis

Using $k = 8$, we test all word pairs in a curated subset of 200 words from WordNet's hypernym hierarchy and measure how often *subsumes*$(A, B)$ aligns with WordNet's known is-a relationships.

We observe that subsumption in prime space captures LSH bucket containment, which is correlated with but not equivalent to genuine semantic containment. The precision depends heavily on $k$ and the random seed, as discussed in Section 5.

### 4.4 Composition Verification

We verify the theoretical guarantee that $\text{compose}(A, B)$ subsumes both $A$ and $B$ across all 20,000 word pairs tested. This property holds by construction (since LCM always divides evenly by its inputs) and is confirmed empirically with zero exceptions.

---

## 5. Limitations and Honest Assessment

We identify several important limitations that must be addressed transparently:

### 5.1 Hash Coincidence ≠ Semantic Containment

The most critical limitation is that **subsumption in prime space reflects LSH bucket containment, not genuine semantic subsumption**. When *subsumes*(King, Queen) returns True, it means every hyperplane that Queen activates is also activated by King. This is a property of random projections, not of the semantic relationship between royalty and gender.

Changing the random seed can reverse the subsumption direction entirely. This limits the reliability of subsumption for downstream applications without careful hyperparameter selection.

### 5.2 Sensitivity to $k$

The useful operating regime ($k = 6$–$10$) is narrow. There is no principled method for selecting $k$ without domain-specific validation, analogous to the challenge of selecting the number of hash tables in multi-probe LSH.

### 5.3 Integer Overflow

For large $k$, composite primes grow exponentially (product of $k/2$ primes on average). At $k = 32$, typical values exceed $10^{15}$, which remains within 64-bit integer range but approaches practical limits for $k > 50$. Arbitrary-precision arithmetic is available but degrades performance.

### 5.4 Loss of Continuous Nuance

The projection from $\mathbb{R}^{384}$ to $\mathbb{Z}$ is inherently lossy. Concepts that differ by subtle continuous gradients (e.g., "happy" vs. "elated") may receive identical prime encodings, erasing distinctions that the original embedding preserves.

---

## 6. Discussion

### 6.1 What Prime Factorization Adds Beyond Bitstrings

A natural question is whether the prime factorization layer provides genuine value over standard LSH bitstrings with Hamming distance. We argue that the value is specifically in the **algebraic operations**:

| Operation | Bitstring + Hamming | Primes + GCD/Modulo |
|:---|:---|:---|
| Similarity | ✓ (XOR + popcount) | ✓ (GCD ratio) |
| Subsumption | ✗ | ✓ ($A \bmod B = 0$) |
| Composition | ✗ | ✓ (LCM) |
| Gap decomposition | ✗ | ✓ (GCD + quotients) |

For pure nearest-neighbor search, bitstrings are faster and sufficient. The prime representation is justified only when downstream tasks require logical verification, composition, or explanatory decomposition.

### 6.2 Potential Applications

Based on our analysis, the most promising applications are:

1. **Post-hoc LLM output verification:** Given a knowledge base encoded as prime integers, verify whether an LLM's output is logically consistent with known facts via divisibility testing.
2. **Ontology debugging:** Detect inconsistencies in knowledge graph hierarchies by checking whether is-a relationships satisfy subsumption in prime space.
3. **Explainable similarity:** Provide human-interpretable explanations of why two concepts are similar or different, decomposed into specific feature dimensions.

### 6.3 Connection to Formal Concept Analysis

Our prime lattice structure bears resemblance to the concept lattice in Formal Concept Analysis (Ganter & Wille, 1999), where concepts are ordered by extent/intent inclusion. The mapping from continuous embeddings to a formal concept lattice via prime factorization may be a fruitful direction for future work.

---

## 7. Conclusion

We have presented a lightweight mechanism for projecting continuous neural embeddings into composite prime integers, enabling exact algebraic operations—subsumption, composition, and abductive gap analysis—that are impossible under standard vector similarity metrics. The approach achieves a 12× computational speedup over cosine similarity with deterministic guarantees.

We are transparent about the method's primary limitation: the semantic validity of the algebraic operations depends on the quality of the underlying LSH projection, which is sensitive to hyperparameter selection. The prime factorization layer adds algebraic structure but does not add semantic information beyond what LSH provides.

We believe this work contributes a useful tool to the neurosymbolic AI toolkit—not as a replacement for continuous methods, but as a complementary verification layer that trades continuous nuance for discrete certitude.

**Code availability:** The complete implementation is available at [repository URL].

---

## References

- Badreddine, S., et al. (2022). Logic Tensor Networks. *Artificial Intelligence*, 303.
- Bordes, A., et al. (2013). Translating Embeddings for Modeling Multi-relational Data. *NeurIPS*.
- Charikar, M. S. (2002). Similarity Estimation Techniques from Rounding Algorithms. *STOC*.
- Garcez, A. d., et al. (2019). Neural-Symbolic Computing: An Effective Methodology for Principled Integration of Machine Learning and Reasoning. *JAIR*.
- Ganter, B. & Wille, R. (1999). *Formal Concept Analysis: Mathematical Foundations*. Springer.
- Indyk, P. & Motwani, R. (1998). Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality. *STOC*.
- Manhaeve, R., et al. (2018). DeepProbLog: Neural Probabilistic Logic Programming. *NeurIPS*.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations in Vector Space. *ICLR Workshop*.
- Sarker, M. K., et al. (2021). Neuro-Symbolic Artificial Intelligence: The State of the Art. *IOS Press*.
- Sun, Z., et al. (2019). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. *ICLR*.
