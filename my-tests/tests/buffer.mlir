handshake.func @test_buffer(%arg0: index, %arg1: none, ...) -> (index, none) {
  %0 = handshake.buffer [3] seq %arg0 : index
  return %0, %arg1 : index, none
}