// @ts-check
// Shared test utilities. Not part of the public API. `npm test` only runs
// test/**/*.test.js, so this file is never run as a test itself.
//
// These helpers deliberately do NOT use src/ code, so a bug in the code
// under test can't hide itself by also breaking the helper.

import assert from "node:assert/strict";

/** @typedef {import("../src/complex.js").Complex} Complex */
/** @typedef {import("../src/matrix.js").Matrix} Matrix */

export const TOL = 1e-12;

/**
 * Build a matrix from nested rows. Entries may be real numbers or [re, im].
 *   mat([[0, [0, -1]], [[0, 1], 0]])  // σy
 * @param {(number | Complex)[][]} entries
 * @returns {Matrix}
 */
export function mat(entries) {
  const rows = entries.length;
  const cols = rows === 0 ? 0 : entries[0].length;
  const data = new Float64Array(2 * rows * cols);
  for (let i = 0; i < rows; i++) {
    if (entries[i].length !== cols) throw new Error("mat: ragged rows");
    for (let j = 0; j < cols; j++) {
      const z = entries[i][j];
      const [re, im] = typeof z === "number" ? [z, 0] : z;
      data[2 * (i * cols + j)] = re;
      data[2 * (i * cols + j) + 1] = im;
    }
  }
  return { rows, cols, data };
}

// Fresh copies on every call, so one test can never corrupt another's input.
export const I2 = () => mat([[1, 0], [0, 1]]);
export const SX = () => mat([[0, 1], [1, 0]]);
export const SY = () => mat([[0, [0, -1]], [[0, 1], 0]]);
export const SZ = () => mat([[1, 0], [0, -1]]);

/** A generic non-Hermitian, non-normal complex 2×2 for algebraic identities. */
export const M = () => mat([[[1, 2], [3, -1]], [[0, -2], [4, 0.5]]]);
/** Another one, which does not commute with M. */
export const N = () => mat([[[-1, 1], [2, 0]], [[0.5, 3], [0, -1]]]);

/**
 * @param {Complex} actual
 * @param {Complex} expected
 * @param {number} [tol]
 */
export function assertComplexClose(actual, expected, tol = TOL) {
  assert.ok(Array.isArray(actual), `expected [re, im] array, got ${actual}`);
  assert.equal(actual.length, 2, "complex must have length 2");
  const ok =
    Math.abs(actual[0] - expected[0]) <= tol &&
    Math.abs(actual[1] - expected[1]) <= tol;
  assert.ok(ok, `got [${actual}], expected [${expected}]`);
}

/**
 * Compare shape, storage type and every entry within tolerance.
 * @param {Matrix} actual
 * @param {Matrix} expected
 * @param {number} [tol]
 */
export function assertMatClose(actual, expected, tol = TOL) {
  assert.equal(actual.rows, expected.rows, "rows");
  assert.equal(actual.cols, expected.cols, "cols");
  assert.ok(actual.data instanceof Float64Array, "data must be a Float64Array");
  assert.equal(actual.data.length, 2 * actual.rows * actual.cols, "data length");
  for (let k = 0; k < expected.data.length; k += 2) {
    const e = k / 2;
    const i = Math.floor(e / expected.cols);
    const j = e % expected.cols;
    const ok =
      Math.abs(actual.data[k] - expected.data[k]) <= tol &&
      Math.abs(actual.data[k + 1] - expected.data[k + 1]) <= tol;
    assert.ok(
      ok,
      `entry (${i},${j}): got [${actual.data[k]}, ${actual.data[k + 1]}], ` +
        `expected [${expected.data[k]}, ${expected.data[k + 1]}]`,
    );
  }
}

/**
 * Deep copy for "inputs were not mutated" checks.
 * @param {Matrix} A
 * @returns {Matrix}
 */
export function snapshot(A) {
  return { rows: A.rows, cols: A.cols, data: Float64Array.from(A.data) };
}
