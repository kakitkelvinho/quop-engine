// @ts-check
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import * as complex from "../src/complex.js";
import { assertComplexClose } from "./helpers.js";

const { add, mul, conj, abs2, expi } = complex;

describe("complex.add", () => {
  test("adds componentwise", () => {
    assertComplexClose(add([1, 2], [3, -5]), [4, -3]);
  });

  test("is commutative", () => {
    assertComplexClose(add([1.5, -2], [0.25, 7]), add([0.25, 7], [1.5, -2]));
  });

  test("zero is the identity", () => {
    assertComplexClose(add([3, 4], [0, 0]), [3, 4]);
  });
});

describe("complex.mul", () => {
  test("i · i = −1", () => {
    assertComplexClose(mul([0, 1], [0, 1]), [-1, 0]);
  });

  test("(1 + 2i)(3 + 4i) = −5 + 10i", () => {
    assertComplexClose(mul([1, 2], [3, 4]), [-5, 10]);
  });

  test("is commutative", () => {
    assertComplexClose(mul([1.5, -2], [0.25, 7]), mul([0.25, 7], [1.5, -2]));
  });

  test("one is the identity", () => {
    assertComplexClose(mul([3, 4], [1, 0]), [3, 4]);
  });

  test("is distributive", () => {
    /** @type {[number, number]} */ const a = [4, 2];
    /** @type {[number, number]} */ const b = [3, 2];
    /** @type {[number, number]} */ const c = [2, 1];
    assertComplexClose(mul(a, add(b, c)), add(mul(a, b), mul(a, c)));
  });
});

describe("complex.conj", () => {
  test("flips the sign of the imaginary part", () => {
    assertComplexClose(conj([3, 4]), [3, -4]);
  });

  test("is an involution: (z*)* = z", () => {
    assertComplexClose(conj(conj([-1.5, 2.5])), [-1.5, 2.5]);
  });

  test("z · z* = |z|² (real)", () => {
    const z = /** @type {[number, number]} */ ([3, -4]);
    assertComplexClose(mul(z, conj(z)), [25, 0]);
  });
});

describe("complex.abs2", () => {
  test("|3 + 4i|² = 25", () => {
    assert.equal(abs2([3, 4]), 25);
  });

  test("returns a plain number", () => {
    assert.equal(typeof abs2([1, 1]), "number");
  });

  test("is multiplicative: |ab|² = |a|²|b|²", () => {
    /** @type {[number, number]} */ const a = [1, 2];
    /** @type {[number, number]} */ const b = [-3, 0.5];
    assert.ok(Math.abs(abs2(mul(a, b)) - abs2(a) * abs2(b)) < 1e-12);
  });
});

describe("complex.expi (analytic: Euler's formula)", () => {
  test("e^{i·0} = 1", () => {
    assertComplexClose(expi(0), [1, 0]);
  });

  test("e^{iπ/2} = i", () => {
    assertComplexClose(expi(Math.PI / 2), [0, 1]);
  });

  test("e^{iπ} = −1", () => {
    assertComplexClose(expi(Math.PI), [-1, 0]);
  });

  test("|e^{iθ}|² = 1 for all θ", () => {
    for (const theta of [-7, -1, 0.3, 2, 100]) {
      assert.ok(Math.abs(abs2(expi(theta)) - 1) < 1e-12, `θ = ${theta}`);
    }
  });

  test("e^{ia} · e^{ib} = e^{i(a+b)}", () => {
    assertComplexClose(mul(expi(0.7), expi(-2.1)), expi(0.7 - 2.1));
  });
});

describe("purity", () => {
  test("no function mutates its inputs", () => {
    /** @type {[number, number]} */ const a = [1, 2];
    /** @type {[number, number]} */ const b = [3, 4];
    add(a, b);
    mul(a, b);
    conj(a);
    abs2(a);
    assert.deepEqual(a, [1, 2]);
    assert.deepEqual(b, [3, 4]);
  });

  test("returns a new array, never an input", () => {
    /** @type {[number, number]} */ const a = [1, 0];
    /** @type {[number, number]} */ const zero = [0, 0];
    assert.notEqual(add(a, zero), a);
    assert.notEqual(mul(a, a), a);
    assert.notEqual(conj([1, 0]), a);
  });
});
