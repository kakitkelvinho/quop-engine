# QUOP Engine

Pure JavaScript library to run quantum optics calculations.

## Installation

`"quop-engine": "github:kakitkelvinho/quop-engine#v0.1.0"`

## Definitions

- Complex numbers are defined as arrays: `[re, im]`
- Matrices are objects: `{rows, cols, data}`
  - Stored in interleaved, row-major convention
  - `data` is a `Float64Array` of length `2*rows*cols`
- Pure functions: all functions return new data

## License

MIT
