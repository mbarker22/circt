handshake.func @test_divsi(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.divsi %arg0, %arg1 : index
  return %0, %arg2 : index, none
}