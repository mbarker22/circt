handshake.func @test_fork(%arg0: index, %arg1: none, ...) -> (index, index, none) {
  %0:2 = handshake.fork [2] %arg0 : index
  return %0#0, %0#1, %arg1 : index, index, none
}
