handshake.func @test_memory(%arg0: index, %arg1: index, %v: i32, %argCtrl: none) -> none {
  %ldData, %stCtrl, %ldCtrl = handshake.memory[ld=1, st=1](%storeData, %storeAddr, %loadAddr) {id = 0 : i32, lsq = false} :  memref<10xi32>, (i32, index, index) -> (i32, none, none)
  %fCtrl:2 = fork [2] %argCtrl : none
  %loadData, %loadAddr = load [%arg0] %ldData, %fCtrl#0 : index, i32
  %storeData, %storeAddr = store [%arg1] %v, %fCtrl#1 : index, i32
  sink %loadData : i32
  %finCtrl = join %stCtrl, %ldCtrl : none, none
  return %finCtrl : none
}