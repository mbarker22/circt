handshake.func @test_xori(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.xori %arg0, %arg1 : index
  return %0, %arg2 : index, none
}