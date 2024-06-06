handshake.func @test_cond_br(%arg0: i1, %arg1: index) -> (index, index) {
  %0:2 = handshake.cond_br %arg0, %arg1 : index
  return %0#0, %0#1 : index, index
}