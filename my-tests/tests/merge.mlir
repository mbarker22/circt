handshake.func @test_merge(%arg0: index, %arg1: index, %arg2: index) -> (index) {
  %0 = handshake.merge %arg0, %arg1, %arg2 : index
  return %0 : index
}