//===- HandshakeToHW.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares passes which together will lower the Handshake dialect to
// CIRCT RTL dialects.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_CONVERSION_HANDSHAKETOHW_H
#define CIRCT_CONVERSION_HANDSHAKETOHW_H

#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/Builders.h"
#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace circt {

namespace esi {
class ChannelType;
} // namespace esi

std::unique_ptr<mlir::Pass> createHandshakeToHWPass();

namespace handshake {

// Attribute name for the name of a predeclaration of the to-be-lowered
// hw.module from a handshake function.
static constexpr const char *kPredeclarationAttr = "handshake.module_name";

// Converts 't' into a valid HW type. This is strictly used for converting
// 'index' types into a fixed-width type.
Type toValidType(Type t);

// Wraps a type into an ESI ChannelType type. The inner type is converted to
// ensure comprehensability with the RTL dialects.
esi::ChannelType esiWrapper(mlir::Type t);

// Returns the hw::ModulePortInfo that corresponds to the given handshake
// operation and its in- and output types.
hw::ModulePortInfo getPortInfoForOpTypes(mlir::Operation *op, TypeRange inputs,
                                         TypeRange outputs);

 
  // Helper class for partitioning module and adding wake signals
  class WakeSignalHelper {
  public:
    WakeSignalHelper(hw::HWModuleLike &module) : module(module) {
      findTraverseOrder();
    }

    struct netValue {
      bool operator!=(const netValue &rhs) const {
	return x ? (rhs.x ? false : true) : (rhs.x ? true : (value != rhs.value));
      }
      
      bool x;
      APInt value;
    };
         
    void simulateOps(llvm::DenseMap<Value, netValue> &stateMap, llvm::DenseMap<Value, netValue> &prevStateMap);
    
    bool findIdleState(llvm::DenseMap<Value, netValue> &idleState);

    llvm::SetVector<Value> getOutReady();
    llvm::SetVector<Value> getInData();
    llvm::SetVector<Value> getInValid();
    llvm::SetVector<Value> getInReady();
    llvm::SetVector<Value> getOutValid();

    void getDataOps(llvm::SetVector<Operation*> &data_ops, llvm::SmallVector<Value> &input_args, llvm::SetVector<Value> &output_args);

    void getStateTransitionInputs(llvm::SmallVector<Value> &inputs, llvm::DenseMap<Value, WakeSignalHelper::netValue> &state, llvm::SmallVector<std::string> &bit_vectors);

  private:
    void findTraverseOrder();
    void findHandshakeIO();
    
    hw::HWModuleLike module;
    llvm::SmallVector<Operation*> traverseOrder;
    llvm::SetVector<Value> out_ready;
    llvm::SetVector<Value> out_valid;
    llvm::SetVector<Value> out_data;
    llvm::SetVector<Value> in_ready;
    llvm::SetVector<Value> in_valid;
    llvm::SetVector<Value> in_data;
  };
    
} // namespace handshake
} // namespace circt

#endif // CIRCT_CONVERSION_HANDSHAKETOHW_H
