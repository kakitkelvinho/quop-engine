// @ts-check
/**
 * Complex scalars.
 *
 * A complex number is a plain 2-element array `[re, im]`.
 * Every function is pure: it never mutates its arguments and always returns
 * a new array.
 *
 * These are convenience functions for scalar code. Matrix kernels in
 * matrix.js should NOT call them per element: each call allocates a new
 * array, which is slow in hot loops. Kernels index Float64Array data directly.
 *
 * @module complex
 */

/**
 * A complex number `re + i·im`.
 * @typedef {[number, number]} Complex
 */

/**
 * Sum of two complex numbers.
 * @param {Complex} a
 * @param {Complex} b
 * @returns {Complex} a + b
 */
export function add([re1, im1], [re2, im2]) {
  return [re1 + re2, im1 + im2];
}

/**
 * Product of two complex numbers.
 * (a + ib)(c + id) = (ac − bd) + i(ad + bc)
 * @param {Complex} a
 * @param {Complex} b
 * @returns {Complex} a · b
 */
export function mul([re1, im1], [re2, im2]) {
  return [re1 * re2 - im1 * im2, re1 * im2 + im1 * re2];
}

/**
 * Complex conjugate.
 * @param {Complex} a
 * @returns {Complex} a*
 */
export function conj([re, im]) {
  return [re, -im];
}

/**
 * Squared modulus |a|² = re² + im².
 * Squared on purpose: it avoids a sqrt and is what Born-rule
 * probabilities need.
 * @param {Complex} a
 * @returns {number} |a|²
 */
export function abs2([re, im]) {
  return re * re + im * im;
}

/**
 * Unit phasor e^{iθ} = cos θ + i sin θ.
 * @param {number} theta  phase in radians
 * @returns {Complex} e^{iθ}
 */
export function expi(theta) {
  return [Math.cos(theta), Math.sin(theta)];
}
