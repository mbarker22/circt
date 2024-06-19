handshake.func @test_gcd(%arg0: i32, %arg1: i32) -> (i32) {
  %aIn = handshake.buffer [1] seq %arg0 : i32
  %bIn = handshake.buffer [1] seq %arg1 : i32
  %c = buffer [1] seq %e1 {initValues = [1]} : i1
  %c1, %c2 = handshake.fork [2] %c : i1

  // first muxes
  %a = handshake.mux %c1 [%aFBb, %aIn] : i1, i32
  %a1, %a2 = handshake.fork [2] %a : i32
  %b = handshake.mux %c2 [%bFBb, %bIn] : i1, i32
  %b1, %b2 = handshake.fork [2] %b : i32

  // equality test and first cond branch
  %e = arith.cmpi eq, %a1, %b1 : i32
  %e1, %e2, %e3 = handshake.fork [3] %e : i1
  %result, %a3 = handshake.cond_br %e2, %a2 : i32
  %a4, %a5 = handshake.fork [2] %a3 : i32
  %discard, %b3 = handshake.cond_br %e3, %b2 : i32
  handshake.sink %discard : i32
  %b4, %b5 = handshake.fork [2] %b3 : i32

  // less-thank test and second cond branch
  %lt = arith.cmpi slt, %a4, %b4 : i32
  %lt1, %lt2, %lt3, %lt4 = handshake.fork [4] %lt : i1
  %a6, %a7 = handshake.cond_br %lt1, %a5 : i32
  %a8, %a9 = handshake.fork [2] %a6 : i32
  %b6, %b7 = handshake.cond_br %lt2, %b5 : i32
  %b8, %b9 = handshake.fork [2] %b7 : i32

  // subtractors
  %a10 = arith.subi %a7, %b9 : i32
  %b10 = arith.subi %b6, %a9 : i32

  // final muxes
  %aFB = handshake.mux %lt3 [%a10, %a8] : i1, i32
  %bFB = handshake.mux %lt4 [%b8, %b10] : i1, i32

  // feedback buffers
  %aFBb = buffer [1] seq %aFB : i32
  %bFBb = buffer [1] seq %bFB : i32

  return %result : i32
}