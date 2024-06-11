handshake.func @test_store(%arg0: index, %v : i32, %argCtrl: none) {
  %stCtrl = handshake.memory[ld=0, st=1](%storeData, %storeAddr) {id = 0 : i32, lsq = false} : memref<10xi32>, (i32, index) -> (none)
  %storeData, %storeAddr = store [%arg0] %v, %argCtrl : index, i32
  return
}