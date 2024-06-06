handshake.func @test_divui(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.divui %arg0, %arg1 : index
  return %0, %arg2 : index, none
}