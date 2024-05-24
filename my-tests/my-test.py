#!/usr/bin/python3

# expects an mlir input file with a single handshake.func op named top

import sys
import subprocess
import re
import random
import os

assert len(sys.argv) == 2
test = sys.argv[1]
print("testing %s"%(test))

# run lowering passes
def lower(wake_signals):
    cmd = "circt-opt %s.mlir -lower-handshake-to-hw=\"add-wake-signals=%d\""%(test, wake_signals)
    cmd += " --lower-esi-ports --lower-esi-to-hw"
    cmd += " --lower-seq-to-sv"
    cmd += " -export-verilog -o %s.mlir-out"%(test)
    cmd += " > top.sv"
    print("  lowering: " + cmd)
    
    result = subprocess.run(cmd, shell=True)
    if (result.returncode == 0):
        print("  lowering successful")
    else:
        print("  lowering failed")
        exit()

# run Verilator simulation
def run(inputs, wake_signals):
    cmd = "circt-rtl-sim.py top.sv my-driver.cpp --no-default-driver"
    cmd += " --simargs=\"%s %s\""%(test+wake_signals, inputs)
    print("  simulating: " + cmd)
    
    result = subprocess.run(cmd, shell=True, capture_output=True)
    if (result.returncode == 0):
        print("  sumulation successful")
    else:
        print("  simulation failed")
        print(result.stdout)
        print(result.stderr)
        exit()

# get func op io
inputFile = open("%s.mlir"%(test), "r")
for line in inputFile.readlines():
    if re.findall(r'handshake.func', line):
        inputs = re.findall(r'%(.*?): index', line)
        outputs = re.findall(r'index', line.split("->")[1])
        for i in range(len(outputs)):
            outputs[i] = "out%d"%(i)
inputFile.close()

# generate verilator driver file
templateFile = open("driver-template.cpp", "r")
template = templateFile.readlines();
templateFile.close()

insertPoint = [idx for idx, line in enumerate(template) if re.search('SETUP', line)][0]
for idx, i in enumerate(inputs):
    template.insert(insertPoint, "std::vector<int> %s_offered;\nstd::vector<int> %s;\ngetInputVector(std::string(argv[%d]), %s_offered);\n"%(i, i, idx+2, i))
for o in outputs:
    template.insert(insertPoint, "std::vector<int> %s;\n"%(o))

insertPoint = [idx for idx, line in enumerate(template) if re.search('INPUTS', line)][0]
for i in inputs:
    template.insert(insertPoint, "tb->%s = %s_offered.back();\ntb->%s_valid = (%s_offered.size() > 0) ? (myrand() & 0x1) : 0x0;\n"%(i, i, i, i))
for o in outputs:
    template.insert(insertPoint, "tb->%s_ready = myrand() & 0x1;\n"%(o))

insertPoint = [idx for idx, line in enumerate(template) if re.search('HANDSHAKE', line)][0]
for i in inputs:
    template.insert(insertPoint, "acceptInput(tb->%s_ready, tb->%s_valid, tb->%s, %s_offered, %s);\ntrace(traceFile, tb->%s_ready, tb->%s_valid, tb->%s);\n"%(i, i, i, i, i, i, i, i))
for o in outputs:
    template.insert(insertPoint, "recordOutput(tb->%s_ready, tb->%s_valid, tb->%s, out0);\ntrace(traceFile, tb->%s_ready, tb->%s_valid, tb->%s);\n"%(o, o, o, o, o, o))

insertPoint = [idx for idx, line in enumerate(template) if re.search('RESULTS', line)][0]
for i in inputs:
    template.insert(insertPoint, "printResult(outFile, %s);\n"%(i))
for o in outputs:
    template.insert(insertPoint, "printResult(outFile, %s);\n"%(o))

# write driver file
driverFile = open("my-driver.cpp", "w")
template_string = "".join(x for x in template)
driverFile.write(template_string)
driverFile.close()

# random input strings
input_arg = ""
for i in inputs:
    input_vals = [str(random.randint(0,10)) for i in range(4)]
    input_arg += ",".join(input_vals) + " "

# lower to sv and run simulation
for mode in [0, 1]:
    print("power gated" if mode else "standard")
    lower(mode)
    run(input_arg, str(mode))

# compare results
result = subprocess.run("diff top.sv.d/%s0.out top.sv.d/%s1.out"%(test, test), shell=True, capture_output=True)
cleanup = ["my-driver.cpp", "%s.mlir-out"%(test), "top.sv"]
if (len(result.stdout) == 0):
    print("%s PASS"%(test))
    for f in cleanup:
        if (os.path.isfile(f)):
            os.remove(f)
else:
    print("%s FAIL"%(test))
