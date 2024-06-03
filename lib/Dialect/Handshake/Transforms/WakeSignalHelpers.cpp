// Contains helper functions to add sleepable domains to handshake ops during lowering

#include "PassDetails.h"
#include "circt/Conversion/HandshakeToHW.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iostream>

using namespace circt;
using namespace handshake;
using namespace mlir;

namespace circt {
namespace handshake {

  llvm::SetVector<Value> WakeSignalHelper::getOutReady() {
    if (out_ready.empty())
      findHandshakeIO();
    return out_ready;
  }
  
  llvm::SetVector<Value> WakeSignalHelper::getInData() {
    if (in_data.empty())
      findHandshakeIO();
    return in_data;
  }
  
  llvm::SetVector<Value> WakeSignalHelper::getInValid() {
    if (in_valid.empty())
      findHandshakeIO();
    return in_valid;
  }

  llvm::SetVector<Value> WakeSignalHelper::getInReady() {
    if (in_ready.empty())
      findHandshakeIO();
    return in_ready;
  }
  
  llvm::SetVector<Value> WakeSignalHelper::getOutValid() {
    if (out_valid.empty())
      findHandshakeIO();
    return out_valid;
  }
  
  void WakeSignalHelper::findHandshakeIO() {
    for (auto i : module.getBodyBlock()->getArguments()) {
      if (isa<esi::ChannelType>(i.getType())) {
	auto unwrapOp = i.getUsers().begin();
	in_ready.insert(unwrapOp->getOperand(1));
	in_data.insert(unwrapOp->getResult(0));
	in_valid.insert(unwrapOp->getResult(1));
      } 
    }
    for (auto i : module.getBodyBlock()->getTerminator()->getOperands()) {
      if (isa<esi::ChannelType>(i.getType())) {
	auto wrapOp = i.getDefiningOp();
	out_data.insert(wrapOp->getOperand(0));
	out_valid.insert(wrapOp->getOperand(1));
	out_ready.insert(wrapOp->getResult(1));
      }
    }
  }
  
  bool WakeSignalHelper::findIdleState(llvm::DenseMap<Value, netValue> &idleState) {
    llvm::SmallVector<llvm::DenseMap<Value, netValue>> history;
    while (true) {
      llvm::DenseMap<Value, netValue> stateMap;
      // simulate with no backpressure or valid input to find idle state
      for (auto i : getOutReady())
	stateMap[i] = {false, APInt(1, 1)};
      for (auto i : getInData())
	stateMap[i] = {true, APInt(1, 1)};
      for (auto i : getInValid())
	stateMap[i] = {false, APInt(1, 0)};
      
      DenseMap<Value, netValue> prevStateMap;
      if (!history.empty()) {
	prevStateMap = history.back();
      }
      simulateOps(stateMap, prevStateMap); 

      if (history.size() > 0) {
	if (history.back() == stateMap) {
	  // self edge - idle state found
	  for (auto [key, val] : stateMap) {
	    if (isa<seq::CompRegOp>(key.getDefiningOp()) && !val.x) {
	      //std::cerr << "add reg to idle\n";
	      idleState[key] = val;
	    }
	  }
	  return true;
	} else if (std::find(history.begin(), history.end(), stateMap) != history.end()) {
	  // cycle - no idle state
	  return false;
	}
      }
      history.push_back(stateMap);
    }	
  }

  void WakeSignalHelper::getStateTransitionInputs(llvm::SmallVector<Value> &inputs, llvm::DenseMap<Value, WakeSignalHelper::netValue> &state, llvm::SmallVector<std::string> &transition_bit_vectors) {
    llvm::SmallVector<Value> stack;
    llvm::SetVector<Value> visited;
    for (auto [key, val] : state) {
      if (!val.x) {
	// don't worry about don't cares
	stack.push_back(key.getDefiningOp()->getOperand(0));
	stack.push_back(key.getDefiningOp()->getOperand(3));
      }
    }

    while (!stack.empty()) {
      auto net = stack.back();
      stack.pop_back();
      visited.insert(net);
		  
      if (isa<BlockArgument>(net)) {
	inputs.push_back(net);
      } else {
	auto op = net.getDefiningOp();
	if (isa<esi::WrapValidReadyOp>(op) || isa<esi::UnwrapValidReadyOp>(op)) {
	  // handshake inputs
	  for (auto result : op->getResults()) {
	    if (result == net) {
	      inputs.push_back(result);
	    }
	  }
	} else if (isa<seq::CompRegOp>(op)) {
	  stack.push_back(op->getOperand(0));
	  stack.push_back(op->getOperand(3));
	} else {
	  for (auto operand : op->getOperands()) {
	    if (!visited.contains(operand)) {
	      stack.push_back(operand);
	    }
	  }
	}
      }
    }

    // all combinations of input bits
    llvm::SmallVector<std::string> bit_vectors;
    size_t num_bits = 0;
    for (auto i : inputs) {
      num_bits += i.getType().getIntOrFloatBitWidth();
    }
    bit_vectors.push_back("0");
    bit_vectors.push_back("1");
    for (size_t i = 0; i < num_bits-1; i++) {
      llvm::SmallVector<std::string> update;
      while (!bit_vectors.empty()) {
	auto back = bit_vectors.back();
	bit_vectors.pop_back();
	auto zero = back + "0";
	update.push_back(zero);
	back = back + "1";
	update.push_back(back);
      }
      bit_vectors.append(update);
    }

    // simulate to find inputs that transition out of idle state
    llvm::DenseMap<Value, WakeSignalHelper::netValue> prevStateMap;
    for (auto [key, value] : state) {
      prevStateMap[key.getDefiningOp()->getOperand(0)] = value;
    }
    for (auto input_bits : bit_vectors) {
      llvm::DenseMap<Value, WakeSignalHelper::netValue> stateMap;
      size_t bit_start = 0;
      for (auto i : inputs) {
	auto num_bits = i.getType().getIntOrFloatBitWidth();
	std::string val = input_bits.substr(bit_start, num_bits);
	stateMap[i] = {false, APInt(num_bits, StringRef(val), 2)};
	bit_start += num_bits;
      }
      simulateOps(stateMap, prevStateMap);
      for (auto [key, val] : state) {
	if (stateMap[key.getDefiningOp()->getOperand(0)] != val) {
	  // transition from idle state
	  transition_bit_vectors.push_back(input_bits);
	  break;
	}
      }
    }
  }

  void WakeSignalHelper::getDataOps(llvm::SetVector<Operation*> &data_ops, llvm::SmallVector<Value> &input_args, llvm::SetVector<Value> &output_args) {
    llvm::SmallVector<Value> trace;
    llvm::SetVector<Value> ctrl;
    // trace backwards from in_ready and out_valid signals to find ctrl nets
    for (auto i : getInReady()) {
      trace.push_back(i);
    }
    for (auto i : getOutValid()) {
      trace.push_back(i);
    }
    while (!trace.empty()) {
      auto net = trace.back();
      trace.pop_back();
      if (!ctrl.contains(net)) {
	ctrl.insert(net);
	if (!isa<BlockArgument>(net) && !isa<esi::WrapValidReadyOp>(net.getDefiningOp()) && !isa<esi::UnwrapValidReadyOp>(net.getDefiningOp())) {
	  for (auto i : net.getDefiningOp()->getOperands()) {
	    trace.push_back(i);
	  }
	}
      }
    }
    // data ops are not wrap/unwrap and have an operand that isn't in control
    for (auto &op : module.getBodyBlock()->getOperations()) {
      if (!isa<esi::WrapValidReadyOp>(op) && !isa<esi::UnwrapValidReadyOp>(op) && !isa<hw::OutputOp>(op)) {
	for (auto i : op.getOperands()) {
	  if (!ctrl.contains(i)) {
	    data_ops.insert(&op);
	    break;
	  }
	}
      }
    }
    
    // inputs to sleepable are operands that aren't also sleepable
    // outputs from sleepable are results that aren't also sleepable
    for (auto &op : data_ops) {
      for (auto i : op->getOperands()) {
	if (!data_ops.contains(i.getDefiningOp())) {
	  // operand not produced by sleepable op
	  input_args.push_back(i);
	}
      }
      for (auto res : op->getResults()) {
	for (auto use : res.getUsers()) {
	  if (!data_ops.contains(use)) {
	    output_args.insert(res);
	  }
	}
      }
    }
  }

  void WakeSignalHelper::findTraverseOrder() {
    llvm::SetVector<Operation*> ordered;
    llvm::SmallVector<Operation*> stack;
    for (auto &op : module.getBodyBlock()->getOperations()) {
      if (isa<esi::WrapValidReadyOp>(op) || isa<esi::UnwrapValidReadyOp>(op)) {
	ordered.insert(&op);
      }
      if (!ordered.contains(&op)) {
	stack.push_back(&op);
      }
      while (!stack.empty()) {
	auto peek = stack.back();
	bool pop = true;
	if (isa<esi::WrapValidReadyOp>(peek) || isa<esi::UnwrapValidReadyOp>(peek)) {
	  stack.pop_back();
	} else {
	  if (!isa<seq::CompRegOp>(peek)) {
	    for (auto operand : peek->getOperands()) {
	      if (!isa<BlockArgument>(operand) && !ordered.contains(operand.getDefiningOp())) {
		stack.push_back(operand.getDefiningOp());
		pop = false;
	      }
	    }
	  } else {
	    if (!ordered.contains(peek->getOperand(3).getDefiningOp())) {
	      stack.push_back(peek->getOperand(3).getDefiningOp());
	      pop = false;
	    }
	  }
	  if (pop) {
	    traverseOrder.push_back(peek);
	    stack.pop_back();
	    ordered.insert(peek);
	  }
	}
      }
    }
  }
  
  void WakeSignalHelper::simulateOps(llvm::DenseMap<Value, netValue> &stateMap, llvm::DenseMap<Value, netValue> &prevStateMap) {
    // TODO: more than 2 inputs
    for (auto &op : traverseOrder) {
      if (isa<hw::ConstantOp>(op)) {
	auto attr = op->getAttrOfType<mlir::IntegerAttr>("value");
	stateMap[op->getResult(0)] = {false, attr.getValue()};
      } else if (isa<seq::CompRegOp>(op)) {
	if (prevStateMap.empty()) {
	  stateMap[op->getResult(0)] = stateMap[op->getOperand(3)]; // initial value
	} else {
	  stateMap[op->getResult(0)] = prevStateMap[op->getOperand(0)]; // prev cycle value
	} 
      } else if (isa<comb::XorOp>(op)) {
	auto res = stateMap[op->getOperand(0)].value ^ stateMap[op->getOperand(1)].value;
	stateMap[op->getResult(0)] = {false, res};
      } else if (isa<comb::AndOp>(op)) {
	auto res = stateMap[op->getOperand(0)].value & stateMap[op->getOperand(1)].value;
	stateMap[op->getResult(0)] = {false, res};
      } else if (isa<comb::OrOp>(op)) {
	auto res = stateMap[op->getOperand(0)].value | stateMap[op->getOperand(1)].value;
	stateMap[op->getResult(0)] = {false, res};
      } 
    }
  }

 
  
}
}
