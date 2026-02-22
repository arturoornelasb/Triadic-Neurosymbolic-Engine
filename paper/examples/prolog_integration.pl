% =============================================
% NEUROSYMBOLIC PRIME ENCODING → Prolog
% Archivo: examples/prolog_integration.pl
% =============================================

:- use_module(library(clpfd)).   % para big integers si es necesario

% ==================== HECHOS ====================
concept('King',    3230).
concept('Queen',   1615).
concept('Man',     85085).
concept('Woman',   1105).
concept('Dog',     19019).
concept('Cat',     6783).
concept('Love',    7735).
concept('Hate',    1105).

% ==================== PREDICADOS ====================

% Operation 1: Logical Subsumption
subsumes(A, B) :-
    concept(A, PhiA),
    concept(B, PhiB),
    PhiA mod PhiB =:= 0.

% Operation 2: Composition (LCM)
compose(A, B, LCM) :-
    concept(A, PhiA),
    concept(B, PhiB),
    G is gcd(PhiA, PhiB),
    LCM is abs(PhiA * PhiB) / G.

% Operation 3: Gap Analysis
gap_analysis(A, B, Shared, UniqueA, UniqueB) :-
    concept(A, PhiA),
    concept(B, PhiB),
    Shared is gcd(PhiA, PhiB),
    UniqueA is PhiA / Shared,
    UniqueB is PhiB / Shared.

% ==================== QUERIES DE EJEMPLO ====================
/*
?- subsumes('King', 'Queen').          % true
?- subsumes('King', 'Man').            % false

?- compose('King', 'Dog', LCM), 
   concept('King', K), LCM mod K =:= 0.   % true

?- gap_analysis('King', 'Man', S, UA, UB).
% S = 85, UA = 38, UB = 1001
*/
