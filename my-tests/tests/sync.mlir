handshake.func @test_sync(%arg0: none, %arg1: i32) -> (none, i32) {
  %res:2 = sync %arg0, %arg1 : none, i32
  return %res#0, %res#1 : none, i32
}
