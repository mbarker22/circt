handshake.func @test_join(%arg0: none, %arg1: none, %arg2: none, ...) -> (none, none) {
  %0 = join %arg0, %arg1 : none, none
  return %0, %arg2 : none, none
}