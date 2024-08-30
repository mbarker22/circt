//===- MergeUnitRate.cpp - lock functions pass ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the merge unit rate pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Handshake/HandshakeOps.h"
#include "circt/Dialect/Handshake/HandshakePasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace handshake;
using namespace mlir;

static void getUnitRateOps(handshake::FuncOp &funcOp, llvm::SetVector<Operation*> &unit_ops) {
  funcOp->walk([&](Operation* op) {
    if (isa<arith::ArithDialect, comb::CombDialect>(op->getDialect()) && isa<handshake::FuncOp>(op->getParentOp())) {
      unit_ops.insert(op);
    }
  });
}

static void findUnitRateGroups(llvm::SetVector<Operation*> &unit_ops, llvm::SmallVector<llvm::SetVector<Operation*>> &groups) {
    int set_id = 0;
    llvm::SetVector<Operation*> visited;
    for (auto op : unit_ops) {
      //llvm::outs() << "unit op: " << op->getName().getStringRef().str() << "\n";
      if (!visited.contains(op)) {
	//llvm::outs() << "  not visited\n";
	llvm::SetVector<Operation*> current_set;
	llvm::SmallVector<Operation*> trace;
	trace.push_back(op);
	visited.insert(op);
	while (!trace.empty()) {
	  auto trace_op = trace.back();
	  //llvm::outs() << "  pop: " << trace_op->getName().getStringRef().str() << "\n";
	  trace.pop_back();
	  current_set.insert(trace_op);
	  //op_to_set_map[trace_op] = set_id;
	  for (auto arg : trace_op->getOperands()) {
	    //llvm::outs() << "    arg: " << arg.getDefiningOp()->getName().getStringRef().str() << "\n";
	    if (unit_ops.contains(arg.getDefiningOp()) && !visited.contains(arg.getDefiningOp())) {
	      //llvm::outs() << "      trace\n";
	      trace.push_back(arg.getDefiningOp());
	      visited.insert(arg.getDefiningOp());
	    }
	  }
	  for (auto res : trace_op->getResults()) {
	    for (auto use : res.getUsers()) {
	      //llvm::outs() << "    use: " << use->getName().getStringRef().str() << "\n";
	      if (unit_ops.contains(use) && !visited.contains(use)) {
		//llvm::outs() << "      trace\n";
		trace.push_back(use);
		visited.insert(use);
	      }
	    }
	  }
	}
	groups.push_back(current_set);
	//set_to_op_map[set_id] = current_set;
	set_id++;
      }
    }
  }

static void findGroupIO(llvm::SetVector<Operation*> &group, llvm::SmallVector<Value> &input_args, llvm::SetVector<Value> &output_args) {
  // inputs to group are operands that aren't also in group
  // outputs from group are results that aren't also in group
  for (auto &op : group) {
    for (auto i : op->getOperands()) {
      if (!group.contains(i.getDefiningOp())) {
	input_args.push_back(i);
      }
    }
    for (auto res : op->getResults()) {
      for (auto use : res.getUsers()) {
	if (!group.contains(use)) {
	  output_args.insert(res);
	}
      }
    }
  }
}

LogicalResult handshake::mergeUnitRate(handshake::FuncOp &op, OpBuilder &builder) {
  // find all unit rate ops with funcOp as parent
  llvm::SetVector<Operation*> unit_ops;
  getUnitRateOps(op, unit_ops);
  // llvm::outs() << "unit rate ops: \n";
  // for (auto i : unit_ops) {
  //   llvm::outs() << "  " << i->getName().getStringRef().str() << "\n";
  // }

  // find groups of unit rate ops that can be merged
  llvm::SmallVector<llvm::SetVector<Operation*>> groups;
  findUnitRateGroups(unit_ops, groups);

  // for each group, create a unitrate op, clone ops into it, update uses
  for (auto group : groups) {
    if (group.size() > 1) {
      // llvm::outs() << "group: \n";
      // for (auto op : group) {
      //   llvm::outs() << "  " << op->getName().getStringRef().str() << "\n";
      // }
      llvm::SmallVector<Value> input_args;
      llvm::SetVector<Value> output_args;
      findGroupIO(group, input_args, output_args);
      // llvm::outs() << "inputs:\n";
      // for (auto i : input_args) {
      //   llvm::outs() << i << "\n";
      // }
      // llvm::outs() << "\noutputs:\n";
      // for (auto i : output_args) {
      //   llvm::outs() << i << "\n";
      // }
      // llvm::outs() << "\n";

      auto inputs = ArrayRef(input_args);
      auto outputs = output_args.getArrayRef();
      builder.setInsertionPoint(group.front());
      auto unitRateOp = builder.create<handshake::UnitRateOp>(group.front()->getLoc(), TypeRange(outputs), ValueRange(inputs));
    
      IRMapping valueMap;
      for (auto [idx, arg] : llvm::enumerate(input_args)) {
	valueMap.map(arg, unitRateOp.getBodyBlock()->getArguments()[idx]);
      }
      llvm::DenseMap<Value, Value> output_map;
      for (auto [idx, arg] : llvm::enumerate(output_args)) {
	output_map[arg] = unitRateOp->getResult(idx);
      }

      llvm::DenseMap<Operation*, llvm::SmallVector<int>> oooArgs;
      llvm::DenseMap<Value, Value> return_map;
      builder.setInsertionPointToStart(unitRateOp.getBodyBlock());
      for (auto op : group) {
	auto newOp = builder.cloneWithoutRegions(*op, valueMap);
	llvm::SmallVector<int> args;
	for (auto [idx, arg] : llvm::enumerate(op->getOperands())) {
	  if (!valueMap.contains(arg)) {
	    args.push_back(idx);
	  }
	}
	if (!args.empty())
	  oooArgs[newOp] = args;
	for (auto [idx, res] : llvm::enumerate(op->getResults())) {
	  if (output_args.contains(res)) {
	    return_map[res] = newOp->getResult(idx);
	  }
	}
      }

      llvm::SmallVector<Value> returnOperands;
      for (auto output : output_args) {
	returnOperands.push_back(return_map[output]);
      }
      auto oldReturnOp = unitRateOp.getBodyBlock()->getTerminator();
      builder.setInsertionPoint(oldReturnOp);
      builder.create<handshake::UnitRateReturnOp>(group.front()->getLoc(), returnOperands);
      oldReturnOp->erase();
    
      for (auto [key, val] : oooArgs) {
	for (auto idx : val) {
	  key->setOperand(idx, valueMap.lookup(key->getOperand(idx)));
	}
      }

      for (auto res : output_args) {
	res.replaceAllUsesWith(output_map[res]);
      }

      for (auto op : group) {
	op->dropAllUses();
	op->erase();
      }
    }
  }
  
  return success();
}

namespace {

struct HandshakeMergeUnitRatePass
    : public HandshakeMergeUnitRateBase<HandshakeMergeUnitRatePass> {
  void runOnOperation() override {
    handshake::FuncOp op = getOperation();
    if (op.isExternal())
      return;

    OpBuilder builder(op);
    if (failed(mergeUnitRate(op, builder)))
      signalPassFailure();
  };
};
} // namespace

std::unique_ptr<mlir::Pass>
circt::handshake::createHandshakeMergeUnitRatePass() {
  return std::make_unique<HandshakeMergeUnitRatePass>();
}
