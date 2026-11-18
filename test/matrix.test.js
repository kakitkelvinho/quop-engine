// @ts-check
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import * as matrix from "../src/matrix.js";
import {
  mat, I2, SX, SY, SZ, M, N,
  assertMatClose, assertComplexClose, snapshot,
} from "./helpers.js";

const {
  zeros, identity, add, sub, scale, matmul, dagger, kron, trace, commutator,
} = matrix;

/** Expected error for shape problems: RangeError naming the function. */
const shapeError = (/** @type {string} */ fn) => ({
  name: "RangeError",
  message: new RegExp(fn),
});

// ---------------------------------------------------------------------------
describe("matrix.zeros", () => {
  test("has the requested shape and all-zero interleaved data", () => {
    const Z = zeros(2, 3);
    assert.equal(Z.rows, 2);
    assert.equal(Z.cols, 3);
    assert.ok(Z.data instanceof Float64Array);
    assert.equal(Z.data.length, 12);
    assert.ok(Z.data.every((x) => x === 0));
  });

  test("allows empty matrices (NumPy convention)", () => {
    const Z = zeros(0, 3);
    assert.equal(Z.rows, 0);
    assert.equal(Z.data.length, 0);
  });

  test("throws RangeError on invalid sizes", () => {
    assert.throws(() => zeros(-1, 2), { name: "RangeError" });
    assert.throws(() => zeros(2, 1.5), { name: "RangeError" });
    assert.throws(() => zeros(NaN, 2), { name: "RangeError" });
  });
});

// ---------------------------------------------------------------------------
describe("matrix.identity", () => {
  test("identity(2) is diag(1, 1)", () => {
    assertMatClose(identity(2), I2());
  });

  test("identity(3) has ones on the diagonal only", () => {
    assertMatClose(identity(3), mat([[1, 0, 0], [0, 1, 0], [0, 0, 1]]));
  });

  test("returns a fresh buffer on every call (no shared cache)", () => {
    assert.notEqual(identity(2).data, identity(2).data);
  });

  test("throws RangeError on invalid size", () => {
    assert.throws(() => identity(-2), { name: "RangeError" });
    assert.throws(() => identity(2.5), { name: "RangeError" });
  });
});

// ---------------------------------------------------------------------------
describe("matrix.add / matrix.sub", () => {
  test("σx + σz", () => {
    assertMatClose(add(SX(), SZ()), mat([[1, 1], [1, -1]]));
  });

  test("complex entries add componentwise", () => {
    assertMatClose(
      add(M(), N()),
      mat([[[0, 3], [5, -1]], [[0.5, 1], [4, -0.5]]]),
    );
  });

  test("A − A = 0", () => {
    assertMatClose(sub(M(), M()), zeros(2, 2));
  });

  test("(A + B) − B = A", () => {
    assertMatClose(sub(add(M(), N()), N()), M());
  });

  test("throw RangeError on shape mismatch", () => {
    assert.throws(() => add(zeros(2, 2), zeros(2, 3)), shapeError("add"));
    assert.throws(() => sub(zeros(2, 2), zeros(3, 2)), shapeError("sub"));
  });
});

// ---------------------------------------------------------------------------
describe("matrix.scale", () => {
  test("i · σx", () => {
    assertMatClose(scale(SX(), [0, 1]), mat([[0, [0, 1]], [[0, 1], 0]]));
  });

  test("complex scalar times complex matrix", () => {
    // (1 + i) · M
    assertMatClose(
      scale(M(), [1, 1]),
      mat([[[-1, 3], [4, 2]], [[2, -2], [3.5, 4.5]]]),
    );
  });

  test("scaling by 0 gives the zero matrix", () => {
    assertMatClose(scale(M(), [0, 0]), zeros(2, 2));
  });
});

// ---------------------------------------------------------------------------
describe("matrix.matmul (analytic: Pauli algebra)", () => {
  test("σx² = σy² = σz² = I", () => {
    assertMatClose(matmul(SX(), SX()), I2());
    assertMatClose(matmul(SY(), SY()), I2());
    assertMatClose(matmul(SZ(), SZ()), I2());
  });

  test("σx σy = i σz", () => {
    assertMatClose(matmul(SX(), SY()), mat([[[0, 1], 0], [0, [0, -1]]]));
  });

  test("identity is neutral on both sides", () => {
    assertMatClose(matmul(identity(2), M()), M());
    assertMatClose(matmul(M(), identity(2)), M());
  });

  test("complex 2×2 product", () => {
    // M·N computed by hand
    assertMatClose(
      matmul(M(), N()),
      mat([[[1.5, 7.5], [1, 1]], [[2.5, 14.25], [0.5, -8]]]),
    );
  });

  test("non-square: (2×3)(3×1) = 2×1", () => {
    const A = mat([[1, 2, 3], [4, 5, 6]]);
    const x = mat([[1], [[0, 1]], [-1]]);
    assertMatClose(matmul(A, x), mat([[[-2, 2]], [[-2, 5]]]));
  });

  test("bra · ket is a 1×1 inner product: ⟨0|0⟩ = 1", () => {
    assertMatClose(matmul(mat([[1, 0]]), mat([[1], [0]])), mat([[1]]));
  });

  test("throws RangeError naming both shapes when A.cols ≠ B.rows", () => {
    assert.throws(() => matmul(zeros(2, 3), zeros(2, 3)), {
      name: "RangeError",
      message: /matmul.*2\s*[×x]\s*3/,
    });
  });
});

// ---------------------------------------------------------------------------
describe("matrix.dagger", () => {
  test("Pauli matrices are Hermitian: σ† = σ", () => {
    assertMatClose(dagger(SX()), SX());
    assertMatClose(dagger(SY()), SY());
    assertMatClose(dagger(SZ()), SZ());
  });

  test("conjugates and transposes a general matrix", () => {
    assertMatClose(dagger(M()), mat([[[1, -2], [0, 2]], [[3, 1], [4, -0.5]]]));
  });

  test("ket† is a bra: shape 2×1 → 1×2, entries conjugated", () => {
    const ket = mat([[[1, 1]], [[0, -2]]]);
    assertMatClose(dagger(ket), mat([[[1, -1], [0, 2]]]));
  });

  test("(A†)† = A", () => {
    assertMatClose(dagger(dagger(M())), M());
  });

  test("(AB)† = B†A†", () => {
    assertMatClose(dagger(matmul(M(), N())), matmul(dagger(N()), dagger(M())));
  });
});

// ---------------------------------------------------------------------------
describe("matrix.kron", () => {
  test("shape is (m·p)×(n·q)", () => {
    const K = kron(zeros(2, 3), zeros(4, 5));
    assert.equal(K.rows, 8);
    assert.equal(K.cols, 15);
  });

  test("|0⟩ ⊗ |1⟩ = |01⟩", () => {
    const ket0 = mat([[1], [0]]);
    const ket1 = mat([[0], [1]]);
    assertMatClose(kron(ket0, ket1), mat([[0], [1], [0], [0]]));
  });

  test("σz ⊗ I acts on the first qubit (ordering convention)", () => {
    assertMatClose(
      kron(SZ(), I2()),
      mat([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
      ]),
    );
  });

  test("σx ⊗ σy with complex entries", () => {
    const i = /** @type {[number, number]} */ ([0, 1]);
    const mi = /** @type {[number, number]} */ ([0, -1]);
    assertMatClose(
      kron(SX(), SY()),
      mat([
        [0, 0, 0, mi],
        [0, 0, i, 0],
        [0, mi, 0, 0],
        [i, 0, 0, 0],
      ]),
    );
  });

  test("mixed-product property: (A⊗B)(C⊗D) = (AC)⊗(BD)", () => {
    assertMatClose(
      matmul(kron(M(), SY()), kron(N(), SX())),
      kron(matmul(M(), N()), matmul(SY(), SX())),
    );
  });

  test("non-square operands: (2×1) ⊗ (1×3) = 2×3", () => {
    const col = mat([[1], [[0, 1]]]);
    const row = mat([[1, 2, 3]]);
    assertMatClose(
      kron(col, row),
      mat([[1, 2, 3], [[0, 1], [0, 2], [0, 3]]]),
    );
  });
});

// ---------------------------------------------------------------------------
describe("matrix.trace", () => {
  test("Tr I_n = n", () => {
    assertComplexClose(trace(identity(3)), [3, 0]);
  });

  test("Pauli matrices are traceless", () => {
    assertComplexClose(trace(SX()), [0, 0]);
    assertComplexClose(trace(SY()), [0, 0]);
    assertComplexClose(trace(SZ()), [0, 0]);
  });

  test("complex trace", () => {
    assertComplexClose(trace(M()), [5, 2.5]);
  });

  test("cyclic: Tr(AB) = Tr(BA)", () => {
    assertComplexClose(trace(matmul(M(), N())), trace(matmul(N(), M())));
  });

  test("Tr(A ⊗ B) = Tr A · Tr B", () => {
    // Tr M = 5 + 2.5i, Tr I2 = 2
    assertComplexClose(trace(kron(M(), I2())), [10, 5]);
  });

  test("throws RangeError on non-square input", () => {
    assert.throws(() => trace(zeros(2, 3)), shapeError("trace"));
  });
});

// ---------------------------------------------------------------------------
describe("matrix.commutator (analytic: su(2) algebra)", () => {
  /** @param {import("../src/matrix.js").Matrix} S */
  const times2i = (S) => scale(S, [0, 2]);

  test("[σx, σy] = 2i σz", () => {
    assertMatClose(commutator(SX(), SY()), times2i(SZ()));
  });

  test("[σy, σz] = 2i σx", () => {
    assertMatClose(commutator(SY(), SZ()), times2i(SX()));
  });

  test("[σz, σx] = 2i σy", () => {
    assertMatClose(commutator(SZ(), SX()), times2i(SY()));
  });

  test("antisymmetric: [A, B] = −[B, A]", () => {
    assertMatClose(commutator(M(), N()), scale(commutator(N(), M()), [-1, 0]));
  });

  test("[A, A] = 0 and [A, I] = 0", () => {
    assertMatClose(commutator(M(), M()), zeros(2, 2));
    assertMatClose(commutator(M(), identity(2)), zeros(2, 2));
  });

  test("commutators are traceless", () => {
    assertComplexClose(trace(commutator(M(), N())), [0, 0]);
  });

  test("throws RangeError on non-square or mismatched inputs", () => {
    assert.throws(() => commutator(zeros(2, 3), zeros(3, 2)), shapeError("commutator"));
    assert.throws(() => commutator(zeros(2, 2), zeros(3, 3)), shapeError("commutator"));
  });
});

// ---------------------------------------------------------------------------
describe("purity", () => {
  test("no function mutates its inputs", () => {
    const A = M();
    const B = N();
    const a0 = snapshot(A);
    const b0 = snapshot(B);
    add(A, B);
    sub(A, B);
    scale(A, [2, 1]);
    matmul(A, B);
    dagger(A);
    kron(A, B);
    trace(A);
    commutator(A, B);
    assertMatClose(A, a0, 0);
    assertMatClose(B, b0, 0);
  });

  test("results never share a buffer with an input", () => {
    const A = M();
    const Z = zeros(2, 2);
    const I = identity(2);
    assert.notEqual(add(A, Z).data, A.data);
    assert.notEqual(sub(A, Z).data, A.data);
    assert.notEqual(scale(A, [1, 0]).data, A.data);
    assert.notEqual(matmul(A, I).data, A.data);
    assert.notEqual(matmul(I, A).data, A.data);
    const one = mat([[[2, 0]]]);
    assert.notEqual(dagger(one).data, one.data);
    assert.notEqual(kron(one, A).data, A.data);
  });
});
