handshake.func @test_pack_unpack(%arg0: i64, %arg1: i32, %ctrl: none, ...) -> (i64, i32, none) {
  %0 = handshake.pack %arg0, %arg1 : tuple<i64, i32>
  %1, %2 = handshake.unpack %0 : tuple<i64, i32>
  return %1, %2, %ctrl : i64, i32, none
}