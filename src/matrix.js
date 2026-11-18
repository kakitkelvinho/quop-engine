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
 * Check non negative integer
 * @param {number} n
 * @param {string} fn
 */
function checkNonNegInt(n, fn) {
  if (!Number.isInteger(n) || n < 0) {
    throw new RangeError(
      `${fn}: number provided must be integer and non-negative! Got ${n} instead.`,
    );
  }
}

/**
 * Check same shape
 * @param {Matrix} A
 * @param {Matrix} B
 * @param {string} fn
 */
function checkSameShape(A, B, fn) {
  if (A.rows !== B.rows || A.cols !== B.cols) {
    throw new RangeError(
      `${fn}: A and B are not the same size! A: ${A.rows}x${A.cols}, B: ${B.rows}x${B.cols}.`,
    );
  }
}

/**
 * All-zero matrix (like `np.zeros((rows, cols), dtype=complex)`).
 * @param {number} rows  non-negative integer
 * @param {number} cols  non-negative integer
 * @returns {Matrix}
 * @throws {RangeError} if rows or cols is not a non-negative integer
 */
export function zeros(rows, cols) {
  checkNonNegInt(rows, "zeros");
  checkNonNegInt(cols, "zeros");
  const data = new Float64Array(2 * rows * cols);
  return { rows, cols, data };
}

/**
 * n×n identity matrix (like `np.eye(n)` / QuTiP `qeye(n)`).
 * @param {number} n  non-negative integer
 * @returns {Matrix}
 * @throws {RangeError} if n is not a non-negative integer
 */
export function identity(n) {
  checkNonNegInt(n, "identity");
  const eye = zeros(n, n);
  for (let i = 0; i < n; i++) {
    eye.data[2 * (i * n + i)] = 1;
  }
  return eye;
}

/**
 * Elementwise sum A + B.
 * @param {Matrix} A
 * @param {Matrix} B  same shape as A
 * @returns {Matrix}
 * @throws {RangeError} if shapes differ
 */
export function add(A, B) {
  checkSameShape(A, B, "add");
  const length = A.data.length;
  const added = zeros(A.rows, A.cols);
  for (let i = 0; i < length; i++) {
    added.data[i] = A.data[i] + B.data[i];
  }
  return added;
}

/**
 * Elementwise difference A − B.
 * @param {Matrix} A
 * @param {Matrix} B  same shape as A
 * @returns {Matrix}
 * @throws {RangeError} if shapes differ
 */
export function sub(A, B) {
  // smart solution: return add(A, scale(B, [-1, 0]));
  checkSameShape(A, B, "sub");
  const length = A.data.length;
  const added = zeros(A.rows, A.cols);
  for (let i = 0; i < length; i++) {
    added.data[i] = A.data[i] - B.data[i];
  }
  return added;
}

/**
 * Scalar multiple z·A.
 * @param {Matrix} A
 * @param {Complex} z
 * @returns {Matrix}
 */
export function scale(A, z) {
  const scaled = zeros(A.rows, A.cols);
  const [zre, zim] = z;
  for (let i = 0; i < A.data.length; i += 2) {
    const re = A.data[i];
    const im = A.data[i + 1];
    scaled.data[i] = zre * re - zim * im;
    scaled.data[i + 1] = zre * im + zim * re;
  }
  return scaled;
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
  if (A.cols !== B.rows) {
    throw new RangeError(
      `matmul: Shape not compatible! Given sizes: A: ${A.rows}x${A.cols}; B:${B.rows}x${B.cols}.`,
    );
  }
  const multiplied = zeros(A.rows, B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let k = 0; k < A.cols; k++) {
      const are = A.data[2 * (i * A.cols + k)];
      const aim = A.data[2 * (i * A.cols + k) + 1];
      for (let j = 0; j < B.cols; j++) {
        const bre = B.data[2 * (k * B.cols + j)];
        const bim = B.data[2 * (k * B.cols + j) + 1];
        multiplied.data[2 * (i * B.cols + j)] += are * bre - aim * bim;
        multiplied.data[2 * (i * B.cols + j) + 1] += are * bim + bre * aim;
      }
    }
  }
  return multiplied;
}

/**
 * Conjugate transpose A† (QuTiP `.dag()`).
 * (A†)ᵢⱼ = (Aⱼᵢ)*
 * @param {Matrix} A  m×n
 * @returns {Matrix}  n×m
 */
export function dagger(A) {
  const daggered = zeros(A.cols, A.rows);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      daggered.data[2 * (j * A.rows + i)] = A.data[2 * (i * A.cols + j)];
      daggered.data[2 * (j * A.rows + i) + 1] =
        -A.data[2 * (i * A.cols + j) + 1];
    }
  }
  return daggered;
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
  const kroned = zeros(A.rows * B.rows, A.cols * B.cols);
  for (let i = 0; i < A.rows; i++) {
    for (let j = 0; j < A.cols; j++) {
      const Aidx = 2 * (i * A.cols + j);
      const subblock = scale(B, [A.data[Aidx], A.data[Aidx + 1]]);
      for (let kr = 0; kr < B.rows; kr++) {
        const start = 2 * kr * B.cols;
        kroned.data.set(
          subblock.data.subarray(start, start + 2 * B.cols),
          2 * ((i * B.rows + kr) * kroned.cols + j * B.cols),
        );
      }
    }
  }
  return kroned;
}

/**
 * Trace Tr A = Σᵢ Aᵢᵢ.
 * @param {Matrix} A  square
 * @returns {Complex}
 * @throws {RangeError} if A is not square
 */
export function trace(A) {
  if (A.rows !== A.cols) {
    throw new RangeError(`trace: Matrix is not square ${A.rows}x${A.cols}`);
  }
  let re = 0,
    im = 0;
  for (let i = 0; i < A.rows; i++) {
    re += A.data[2 * (i * A.rows + i)];
    im += A.data[2 * (i * A.rows + i) + 1];
  }
  return [re, im];
}

/**
 * Commutator [A, B] = A·B − B·A.
 * @param {Matrix} A  n×n
 * @param {Matrix} B  n×n
 * @returns {Matrix}  n×n
 * @throws {RangeError} if A and B are not square matrices of the same size
 */
export function commutator(A, B) {
  if (A.rows !== A.cols || B.rows !== B.cols || A.rows !== B.rows) {
    throw new RangeError(
      `commutator: Matrices are not the same shape: A:${A.rows}x${A.cols} and B:${B.rows}x${B.cols}`,
    );
  }
  return sub(matmul(A, B), matmul(B, A));
}
