handshake.func @top(%arg0: index, %arg1: index, %arg2: index, %arg3: none, ...) -> (index, none) {
  %0 = handshake.mux %arg0 [%arg1, %arg2] : index, index
  return %0, %arg3 : index, none
}


