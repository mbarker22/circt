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

  size_t getBitWidth(Value v) {
    size_t ret = 0;
    if (isa<hw::StructType>(v.getType())) {
      auto structType = cast<hw::StructType>(v.getType());
      llvm::SmallVector<Type> innerTypes;
      structType.getInnerTypes(innerTypes);
      for (auto t : innerTypes) {
	ret += t.getIntOrFloatBitWidth();
      }
    } else {
      ret += v.getType().getIntOrFloatBitWidth();
    }
    return ret;
  }

  static std::string valueName(Operation *scopeOp, Value v) {
    std::string s;
    llvm::raw_string_ostream os(s);
    // CAVEAT: Since commit 27df7158fe MLIR prefers verifiers to print errors for
    // operations in generic form, and the printer by default runs a verification.
    // `valueName` is used in some of these verifiers where preferably the generic
    // operand form should be used instead.
    AsmState asmState(scopeOp, OpPrintingFlags().assumeVerified());
    v.printAsOperand(os, asmState);
    return s;
  }

  static std::string netValueStr(WakeSignalHelper::netValue val) {
    std::string s;
    llvm::raw_string_ostream os(s);
    if (val.x) {
      s = "x";
    } else {
      val.value.print(os, false);
    }    
    return s;
  }
  
  bool WakeSignalHelper::findIdleState(llvm::DenseMap<Value, netValue> &idleState) {
    llvm::SmallVector<llvm::DenseMap<Value, netValue>> history;
    int timeout = 0;
    while (true) {
      llvm::DenseMap<Value, netValue> stateMap;
      // simulate with no backpressure or valid input to find idle state
      for (auto i : getOutReady())
	stateMap[i] = {false, APInt(1, 1)};
      for (auto i : getInData())
	stateMap[i] = {true, APInt(getBitWidth(i), 0)};
      for (auto i : getInValid())
	stateMap[i] = {false, APInt(1, 0)};
      
      DenseMap<Value, netValue> prevStateMap;
      if (!history.empty()) {
	prevStateMap = history.back();
      }
      simulateOps(stateMap, prevStateMap);
      // std::cerr << "simulation cycle:\n";
      // for (auto reg : module.getBodyBlock()->getOps<seq::CompRegOp>()) {
      // 	reg->print(llvm::errs());
      // 	std::cerr << " -> " << netValueStr(stateMap[reg]) << "\n";
      // }
      
      if (history.size() > 0) {
	if (history.back() == stateMap) {
	  // self edge - idle state found
	  //std::cerr << "idle state:\n";
	  //int num_bits = 0;
	  for (auto [key, val] : stateMap) {
	    if (isa<seq::CompRegOp>(key.getDefiningOp())) { // && !val.x) {
	      //num_bits += getBitWidth(key);
	      idleState[key] = val;
	      //std::cerr << valueName(key.getDefiningOp()->getParentOp(), key) << " -> " << netValueStr(val) << "\n";
	    }
	  }
	  //if (num_bits > 10) {
	    //std::cerr << "too many bits\n";
	    //return false;
	  //} else {
	  return true;
	  // }
	} else if (std::find(history.begin(), history.end(), stateMap) != history.end()) {
	  // cycle - no idle state
	  return false;
	}
      }
      history.push_back(stateMap);
      timeout++;
      if (timeout > 10) {
	std::cerr << "Find idle state: simulation timeout\n";
	return false;
      }
    }	
  }

  void WakeSignalHelper::getStateTransitionInputs(llvm::SmallVector<Value> &inputs, llvm::DenseMap<Value, WakeSignalHelper::netValue> &state, llvm::SmallVector<std::string> &transition_bit_vectors) {
    llvm::SmallVector<Value> stack;
    llvm::SetVector<Value> visited;
    // find inputs that drive state registers
    //std::cerr << "starting vals:\n";
    for (auto [key, val] : state) {
      if (!val.x) {
	// don't worry about don't cares
	stack.push_back(key.getDefiningOp()->getOperand(0));
	stack.push_back(key.getDefiningOp()->getOperand(3));
	//std::cerr << valueName(key.getDefiningOp()->getParentOp(), key.getDefiningOp()->getOperand(0)) << "\n" << valueName(key.getDefiningOp()->getParentOp(), key.getDefiningOp()->getOperand(0)) << "\n";
      }
    }

    while (!stack.empty()) {
      auto net = stack.back();
      // std::cerr << "net: " << valueName(net.getDefiningOp()->getParentOp(), net) << "\n";
      // net.getDefiningOp()->print(llvm::errs());
      // std::cerr << "\n";
      stack.pop_back();
      visited.insert(net);
		  
      if (isa<BlockArgument>(net)) {
	if (!getInData().contains(net) && std::find(inputs.begin(), inputs.end(), net) == inputs.end()) {
	  inputs.push_back(net);
	}
      } else {
	auto op = net.getDefiningOp();
	if (isa<esi::WrapValidReadyOp>(op) || isa<esi::UnwrapValidReadyOp>(op)) {
	  // handshake inputs
	  for (auto result : op->getResults()) {
	    if (result == net) {
	      if (!getInData().contains(net) && std::find(inputs.begin(), inputs.end(), net) == inputs.end()) {
		inputs.push_back(result);
	      }
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
    // std::cerr << "transition inputs\n";
    // for (auto i : inputs) {
    //   std::cerr << valueName(i.getDefiningOp()->getParentOp(), i) << " " << i.getType().getIntOrFloatBitWidth() << "\n";
    // }

    // all combinations of those input bits
    llvm::SmallVector<std::string> bit_vectors;
    size_t num_bits = 0;
    for (auto i : inputs) {
      //num_bits += i.getType().getIntOrFloatBitWidth();
      //if (!getInData().contains(i)) {
      num_bits += getBitWidth(i);
      //}
    }
    //std::cerr << "total input bits: " << num_bits << "\n";
    
    assert(num_bits < 10 && "Too many input bits to simulate");
    
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

    //std::cerr << "bit vectors\n";

    // find all other inputs
    llvm::SmallVector<Value> xInputs;
    for (auto i : getInValid()) {
      if (std::find(inputs.begin(), inputs.end(), i) == inputs.end()) {
	xInputs.push_back(i);
      }
    }
    for (auto i : getInData()) {
      if (std::find(inputs.begin(), inputs.end(), i) == inputs.end()) {
	xInputs.push_back(i);
      } else {
	assert (true && "ERROR: DATA WIRE IN TRANSITION INPUTS\n");
      }
    }
    for (auto i : getOutReady()) {
      if (std::find(inputs.begin(), inputs.end(), i) == inputs.end()) {
	xInputs.push_back(i);
      }
    }

    //std::cerr << "don't care inputs\n";
    
    // simulate to find inputs that transition out of idle state
    llvm::DenseMap<Value, WakeSignalHelper::netValue> prevStateMap;
    for (auto [key, value] : state) {
      prevStateMap[key.getDefiningOp()->getOperand(0)] = value;
    }
    for (auto input_bits : bit_vectors) {
      //std::cerr << "simulate " << input_bits << "\n";
      llvm::DenseMap<Value, WakeSignalHelper::netValue> stateMap;
      size_t bit_start = 0;
      for (auto i : inputs) {
	//auto num_bits = i.getType().getIntOrFloatBitWidth();
	auto num_bits = getBitWidth(i);
	std::string val = input_bits.substr(bit_start, num_bits);
	stateMap[i] = {false, APInt(num_bits, StringRef(val), 2)};
	bit_start += num_bits;
      }
      //std::cerr << "set x inputs:\n";
      for (auto i : xInputs) {
	//std::cerr << valueName(i.getDefiningOp()->getParentOp(), i) << "\n";
	stateMap[i] = {true, APInt(getBitWidth(i), 0)};
      }
      simulateOps(stateMap, prevStateMap);
      for (auto [key, val] : state) {
	//key.getDefiningOp()->print(llvm::errs());
	//std::cerr << "\n" << valueName(key.getDefiningOp()->getParentOp(), key) << " = " << netValueStr(val) << " " << (val.x ? ("") : (std::to_string(val.value.getBitWidth()))) << "\n";
	//std::cerr << valueName(key.getDefiningOp()->getParentOp(), key.getDefiningOp()->getOperand(0)) << " = " << netValueStr(stateMap[key.getDefiningOp()->getOperand(0)]) << " " << (stateMap[key.getDefiningOp()->getOperand(0)].x ? ("") : (std::to_string(stateMap[key.getDefiningOp()->getOperand(0)].value.getBitWidth()))) << "\n------------------\n";
	if (!val.x && stateMap[key.getDefiningOp()->getOperand(0)] != val) {
	  // transition from idle state
	  transition_bit_vectors.push_back(input_bits);
	  //std::cerr << "transition: " << input_bits << "\n";
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
	//}
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
    // TODO: don't cares
    for (auto &op : traverseOrder) {
      // op->print(llvm::errs());
      // std::cerr << "\n";
      TypeSwitch<Operation*>(op)
	.Case([&](hw::ConstantOp op) {
	  //auto valAttr = op->getAttrOfType<mlir::IntegerAttr>("value");
	  //auto value = op.getValue();
	  stateMap[op->getResult(0)] = {false, op.getValue()};
	})
	.Case([&](seq::CompRegOp op) {
	  if (prevStateMap.empty()) {
	    stateMap[op->getResult(0)] = stateMap[op->getOperand(3)]; // initial value
	  } else {
	    stateMap[op->getResult(0)] = prevStateMap[op->getOperand(0)]; // prev cycle value
	  }
	})
	.Case([&](comb::XorOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; // ^ stateMap[op->getOperand(1)].value;
	  bool resX = stateMap[op->getOperand(0)].x; // | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	      if (stateMap[*operand].x) {
		resX = true;
		break;
	      }
	      resVal = resVal ^ stateMap[*operand].value;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::AndOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; 
	  bool resX = stateMap[op->getOperand(0)].x;
	  for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	    if (resX) {
	      if (!stateMap[*operand].x && stateMap[*operand].value.isZero()) {
		resX = false;
		resVal = stateMap[*operand].value;
	      }
	    } else if (stateMap[*operand].x) {
	      if (resVal.isOne()) {
		resX = true;
	      }
	    } else {
	      resVal = stateMap[op->getOperand(0)].value & stateMap[op->getOperand(1)].value;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::OrOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; 
	  bool resX = stateMap[op->getOperand(0)].x;
	  for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	    if (resX) {
	      if (!stateMap[*operand].x && stateMap[*operand].value.isOne()) {
		resX = false;
		resVal = stateMap[*operand].value;
	      }
	    } else if (stateMap[*operand].x) {
	      if (resVal.isZero()) {
		resX = true;
	      }
	    } else {
	      resVal = stateMap[op->getOperand(0)].value | stateMap[op->getOperand(1)].value;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::MuxOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x;
	  if (!resX) {
	    if (stateMap[op->getOperand(0)].value == 1) {
	      resVal = stateMap[op->getOperand(1)].value;
	      resX = stateMap[op->getOperand(1)].x;
	    } else {
	      resVal = stateMap[op->getOperand(2)].value;
	      resX = stateMap[op->getOperand(2)].x;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ICmpOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    switch(op.getPredicate()) { 
	    case comb::ICmpPredicate::eq:
	      resVal = (stateMap[op->getOperand(0)].value.eq(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::ne:
	      resVal = (stateMap[op->getOperand(0)].value.ne(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::slt:
	      resVal = (stateMap[op->getOperand(0)].value.slt(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::ult:
	      resVal = (stateMap[op->getOperand(0)].value.ult(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::sle:
	      resVal = (stateMap[op->getOperand(0)].value.sle(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::ule:
	      resVal = (stateMap[op->getOperand(0)].value.ule(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::sgt:
	      resVal = (stateMap[op->getOperand(0)].value.sgt(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::ugt:
	      resVal = (stateMap[op->getOperand(0)].value.ugt(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::sge:
	      resVal = (stateMap[op->getOperand(0)].value.sge(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    case comb::ICmpPredicate::uge:
	      resVal = (stateMap[op->getOperand(0)].value.uge(stateMap[op->getOperand(1)].value)) ? (APInt(1, 1)) : (APInt(1, 0));
	      break;
	    default:
	      std::cerr << "FIX: invalid icmp op\n";
	      break;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ExtractOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x;
	  if (!resX) {
	    auto extractFrom = stateMap[op->getOperand(0)].value;
	    auto numBits = op->getResult(0).getType().getIntOrFloatBitWidth();
	    auto lowBit = op.getLowBit();
	    resVal = extractFrom.extractBits(numBits, lowBit);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ConcatOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; 
	  bool resX = stateMap[op->getOperand(0)].x;
	  if (!resX) {
	    for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	      if (stateMap[*operand].x) {
		resX = true;
		break;
	      }
	      resVal = resVal.concat(stateMap[*operand].value);
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ShlOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.shl(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ShrUOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.lshr(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ShrSOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.ashr(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ModSOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.srem(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::ModUOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.urem(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::DivUOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.udiv(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::DivSOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value.sdiv(stateMap[op->getOperand(1)].value);
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::SubOp op) {
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x | stateMap[op->getOperand(1)].x;
	  if (!resX) {
	    resVal = stateMap[op->getOperand(0)].value - stateMap[op->getOperand(1)].value;
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::AddOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; 
	  bool resX = stateMap[op->getOperand(0)].x; 
	  if (!resX) {
	    for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	      if (stateMap[*operand].x) {
		resX = true;
		break;
	      }
	      resVal = resVal + stateMap[*operand].value;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](comb::MulOp op) {
	  APInt resVal = stateMap[op->getOperand(0)].value; 
	  bool resX = stateMap[op->getOperand(0)].x; 
	  if (!resX) {
	    for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	      if (stateMap[*operand].x) {
		resX = true;
		break;
	      }
	      resVal = resVal * stateMap[*operand].value;
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](hw::StructCreateOp op) {
	  //std::cerr << "StructCreateOp: ";
	  //op.print(llvm::errs());
	  //std::cerr << "\n"; 
	  APInt resVal = stateMap[op->getOperand(0)].value;
	  bool resX = stateMap[op->getOperand(0)].x;
	  // if (resX) {
	  //   std::cerr << "struct: X\n";
	  // } else {
	  //   std::cerr << "struct: ";
	  //   resVal.print(llvm::errs(), false);
	  //   std::cerr << "\n";
	  // }
	  if (!resX) {
	    for (auto operand = op->getOperands().begin()+1; operand < op->getOperands().end(); operand++) {
	      if (stateMap[*operand].x) {
		resX = true;
		//std::cerr << " struct: X";
		break;
	      }
	      resVal = resVal.concat(stateMap[*operand].value);
	      // std::cerr << "append: ";
	      // stateMap[*operand].value.print(llvm::errs(), false);
	      // std::cerr << " struct: ";
	      // resVal.print(llvm::errs(), false);
	      // std::cerr << "\n";
	    }
	  }
	  stateMap[op->getResult(0)] = {resX, resVal};
	})
	.Case([&](hw::StructExplodeOp op) {
	  // std::cerr << "StructExplodeOp: ";
	  //op.print(llvm::errs());
	  //std::cerr << "\n";
	  APInt resVal;
	  bool resX = stateMap[op->getOperand(0)].x;
	  auto extractFrom = stateMap[op->getOperand(0)].value;
	  // std::cerr << "input: ";
	  // if (resX)
	  //   std::cerr << "X\n";
	  // else {
	  //   extractFrom.print(llvm::errs(), false);
	  //   std::cerr << " " << extractFrom.getBitWidth() << "\n";
	  // }
	  int lowBit = 0;
	  for (auto res : op->getResults()) {
	    //std::cerr << "output: ";
	    if (resX) {
	      //std::cerr << "X\n";
	      stateMap[res] = {resX, resVal};
	    } else {
	      auto numBits = res.getType().getIntOrFloatBitWidth();
	      //std::cerr << "extract " << numBits << " from " << lowBit << "\n";
	      resVal = extractFrom.extractBits(numBits, lowBit);
	      stateMap[res] = {resX, resVal};
	      //resVal.print(llvm::errs(), false);
	      //std::cerr << "\n";
	      lowBit += numBits;
	    }
	  }
	})
	.Default([&](Operation* op) {
	  if (!isa<hw::OutputOp>(op)) {
	    std::cerr << "FIX: unknown op: " << op->getName().getStringRef().str() << "\n";
	  }
	});
   
    }
  }

 
  
}
}
