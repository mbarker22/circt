handshake.func @test_constant(%arg0: none, ...) -> (index) {
  %1 = handshake.constant %arg0 {value = 42 : index} : index
  return %1 : index
}
