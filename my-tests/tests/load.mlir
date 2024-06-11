handshake.func @test_load(%arg0: index, %argCtrl: none)  {
  %ldData, %ldCtrl = handshake.memory[ld=1, st=0](%loadAddr) {id = 0 : i32, lsq = false} : memref<10xi32>, (index) -> (i32, none)
  %loadData, %loadAddr = load [%arg0] %ldData, %argCtrl : index, i32
  return 
}