; RUN: firtool %s | FileCheck %s

FIRRTL version 3.3.0
circuit test :%[[{
  "class":"sifive.enterprise.firrtl.FullAsyncResetAnnotation",
    "target":"~test|test>reset"
  }]]
  module test :
    input clock : Clock
    input reset : AsyncReset
    input in : { foo : UInt<8>, bar : UInt<8>}
    output out : { foo : UInt<8>, bar : UInt<8>}

    wire reg1_w : { foo : UInt<8>, bar : UInt<8>}
    invalidate reg1_w.bar
    invalidate reg1_w.foo
    ; CHECK: reg1_foo <= 8'hC;
    ; CHECK: reg1_bar <= 8'h0;
    connect reg1_w.foo, UInt<8>(0hc)
    invalidate reg1_w.bar
    ; CHECK: reg1_foo = 8'hC;
    ; CHECK: reg1_bar = 8'h0;
    regreset reg1 : { foo : UInt<8>, bar : UInt<8>}, clock, reset, reg1_w
    wire reg2 : { foo : UInt<8>, bar : UInt<8>}
    connect reg1, in
    connect reg2, reg1
    connect out, reg2
