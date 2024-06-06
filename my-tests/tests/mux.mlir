handshake.func @test_mux(%arg0: index, %arg1: index, %arg2: index) -> (index) {
  %0 = handshake.mux %arg0 [%arg1, %arg2] : index, index
  return %0 : index
}


