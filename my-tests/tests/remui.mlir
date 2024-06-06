handshake.func @test_remui(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.remui %arg0, %arg1 : index
  return %0, %arg2 : index, none
}