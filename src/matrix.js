// @ts-check
/**
 * Dense complex matrices.
 *
 * Storage: `{ rows, cols, data }` where `data` is a Float64Array of length
 * `2·rows·cols`, interleaved [re, im] pairs, row-major:
 *
 *     entry (i, j)  →  k = 2·(i·cols + j),  re = data[k],  im = data[k + 1]
 *
 * Kets are `cols = 1`, bras are `rows = 1`.
 *
 * Rules:
 *  - Pure: never mutate inputs; always return a fresh matrix with its own
 *    Float64Array (never a view onto an input's buffer).
 *  - Shape mismatches throw RangeError naming both shapes, e.g.
 *    "matmul: A is 2×3, B is 2×3 (A.cols ≠ B.rows)".
 *  - Kernels loop over `data` directly; do not call complex.js per element.
 *
 * @module matrix
 */

/** @typedef {import("./complex.js").Complex} Complex */

/**
 * A dense complex matrix.
 * @typedef {object} Matrix
 * @property {number} rows
 * @property {number} cols
 * @property {Float64Array} data  length 2·rows·cols, interleaved, row-major
 */

/**
 * All-zero matrix (like `np.zeros((rows, cols), dtype=complex)`).
 * @param {number} rows  non-negative integer
 * @param {number} cols  non-negative integer
 * @returns {Matrix}
 * @throws {RangeError} if rows or cols is not a non-negative integer
 */
export function zeros(rows, cols) {
  throw new Error("not implemented");
}

/**
 * n×n identity matrix (like `np.eye(n)` / QuTiP `qeye(n)`).
 * @param {number} n  non-negative integer
 * @returns {Matrix}
 * @throws {RangeError} if n is not a non-negative integer
 */
export function identity(n) {
  throw new Error("not implemented");
}

/**
 * Elementwise sum A + B.
 * @param {Matrix} A
 * @param {Matrix} B  same shape as A
 * @returns {Matrix}
 * @throws {RangeError} if shapes differ
 */
export function add(A, B) {
  throw new Error("not implemented");
}

/**
 * Elementwise difference A − B.
 * @param {Matrix} A
 * @param {Matrix} B  same shape as A
 * @returns {Matrix}
 * @throws {RangeError} if shapes differ
 */
export function sub(A, B) {
  throw new Error("not implemented");
}

/**
 * Scalar multiple z·A.
 * @param {Matrix} A
 * @param {Complex} z
 * @returns {Matrix}
 */
export function scale(A, z) {
  throw new Error("not implemented");
}

/**
 * Matrix product A·B (like `np.matmul` / `A @ B`).
 * (A·B)ᵢⱼ = Σₖ Aᵢₖ Bₖⱼ
 *
 * Performance hint: loop order i → k → j reads B along its rows, which is
 * cache-friendly for row-major storage.
 *
 * @param {Matrix} A  m×n
 * @param {Matrix} B  n×p
 * @returns {Matrix}  m×p
 * @throws {RangeError} if A.cols ≠ B.rows
 */
export function matmul(A, B) {
  throw new Error("not implemented");
}

/**
 * Conjugate transpose A† (QuTiP `.dag()`).
 * (A†)ᵢⱼ = (Aⱼᵢ)*
 * @param {Matrix} A  m×n
 * @returns {Matrix}  n×m
 */
export function dagger(A) {
  throw new Error("not implemented");
}

/**
 * Kronecker (tensor) product A ⊗ B (like `np.kron`).
 * For A (m×n) and B (p×q) the result is (m·p)×(n·q) with
 * (A ⊗ B)[i·p + k, j·q + l] = Aᵢⱼ · Bₖₗ
 * @param {Matrix} A  m×n
 * @param {Matrix} B  p×q
 * @returns {Matrix}  (m·p)×(n·q)
 */
export function kron(A, B) {
  throw new Error("not implemented");
}

/**
 * Trace Tr A = Σᵢ Aᵢᵢ.
 * @param {Matrix} A  square
 * @returns {Complex}
 * @throws {RangeError} if A is not square
 */
export function trace(A) {
  throw new Error("not implemented");
}

/**
 * Commutator [A, B] = A·B − B·A.
 * @param {Matrix} A  n×n
 * @param {Matrix} B  n×n
 * @returns {Matrix}  n×n
 * @throws {RangeError} if A and B are not square matrices of the same size
 */
export function commutator(A, B) {
  throw new Error("not implemented");
}
