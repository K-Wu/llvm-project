// RUN: mlir-opt %s --linalg-generalize-named-ops \
// RUN:             --sparsification="enable-gpu-libgen" --gpu-to-llvm='use-opaque-pointers=1'  | FileCheck %s

#CSR = #sparse_tensor.encoding<{ lvlTypes = [ "dense", "compressed" ] }>
func.func @matmul(%A: memref<128x128xf16>, %B: memref<128x128xf16>, %C_in: memref<128x128xf16>) -> memref<128x128xf16> {
  %token0 = gpu.wait async
  %C_out, %token1  = gpu.customop async [%token0]  %A, %B, %C_in : memref<128x128xf16> and memref<128x128xf16> and memref<128x128xf16> into memref<128x128xf16>
  return %C_out: memref<128x128xf16>
}
