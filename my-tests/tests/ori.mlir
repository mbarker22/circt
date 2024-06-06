handshake.func @test_ori(%arg0: index, %arg1: index, %arg2: none, ...) -> (index, none) {
  %0 = arith.ori %arg0, %arg1 : index
  return %0, %arg2 : index, none
}