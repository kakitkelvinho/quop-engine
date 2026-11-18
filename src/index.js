// @ts-check
// Public entry point. Re-exports only, with no logic here.
//
// Low-level modules are exported as namespaces so names can repeat across
// them without prefixes:  complex.add(a, b)  vs  matrix.add(A, B).
// The quantum layer (v0.3+) will be exported flat, QuTiP-style.

export * as complex from "./complex.js";
export * as matrix from "./matrix.js";
