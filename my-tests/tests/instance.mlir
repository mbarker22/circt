handshake.func @foo(%in : i32) -> (i32) {
    handshake.return %in : i32
}

handshake.func @test_instance(%in : i32) -> (i32) {
    %out = handshake.instance @foo(%in) : (i32) -> (i32)
    handshake.return %out : i32
}