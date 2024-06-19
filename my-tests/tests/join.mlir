handshake.func @test_join(%arg0: index, %arg1: index, %arg2: none, ...) -> (none, none) {
  %0 = join %arg0, %arg1 : index, index
  return %0, %arg2 : none, none
}