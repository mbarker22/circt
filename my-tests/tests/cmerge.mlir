handshake.func @test_cmerge(%arg0: index, %arg1: index, %arg2: index) -> (index, index) {
  %0:2 = handshake.control_merge %arg0, %arg1, %arg2 : index, index
  return %0#0, %0#1 : index, index
}