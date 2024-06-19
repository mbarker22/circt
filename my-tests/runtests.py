#!/usr/bin/python3

# expects an mlir input file with a single handshake.func op named test_basename

import sys
import subprocess
import re
import random
import os
import shutil

if len(sys.argv) > 1 :
    tests = [sys.argv[1]]
else :
    tests_dir = os.path.join(os.getcwd(), "tests")
    tests = [os.path.join(tests_dir, f) for f in os.listdir(tests_dir) if f.split('.')[1] == "mlir"]
print("running: ", tests)

# run lowering passes
def lower(path, basename, wake_signals):
    cmd = "circt-opt %s "%(path)
    cmd += "--handshake-materialize-forks-sinks" # handshake lowering requires every SSA value to have exactly one use
    cmd += " -lower-handshake-to-hw=\"add-wake-signals=%d\""%(wake_signals) # lower handshake and add wake signals
    cmd += " --lower-esi-ports --lower-esi-to-hw" # lower esi channels
    cmd += " -lower-comb -canonicalize" # lower combinational and run canonicalizeation
    cmd += " --lower-seq-hlmem --lower-seq-to-sv" # lower sequential (clock, reg, mem) to sv
    cmd += " -hw-cleanup -canonicalize" # cleanup ir and canonicalize again 
    cmd += " -hw-legalize-modules -prettify-verilog" # last steps before exporing verilog
    cmd += " -export-verilog -o %s.mlir-out"%(basename)
    cmd += " > %s.sv"%(basename)
    print("  lowering: " + cmd)
    
    result = subprocess.run(cmd, shell=True)
    if (result.returncode == 0):
        print("  lowering successful")
        return True
    else:
        print("  lowering failed")
        return False

# run Verilator simulation
def run(basename, inputs, wake_signals):
    cmd = "circt-rtl-sim.py %s.sv %s-driver.cpp --no-default-driver --top test_%s"%(basename, basename, basename)
    cmd += " --simargs=\"%s %s\""%(basename+("-wake" if wake_signals else "-std"), inputs)
    print("  simulating: " + cmd)
    
    result = subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
    if (result.returncode == 0):
        print("  sumulation successful")
        return True
    else:
        print("  simulation failed")
        print(result.stdout)
        print(result.stderr)
        return False

def setup(path, basename):
# get func op io
    inputFile = open("%s"%(path), "r")
    for line in inputFile.readlines():
        if re.findall(r'handshake.func', line):
            inputs = [i for i in re.findall(r'%(\w+?) *: (\w+)', line)]
            if "->" in line:
                outputs = [("out%s"%(idx), i) for [idx, i] in enumerate(re.findall(r'(\w+)', line.split("->")[1]))]             
            else :
                outputs = []
            
    inputFile.close()

    # generate verilator driver file
    templateFile = open("driver-template.cpp", "r")
    template = templateFile.readlines();
    templateFile.close()

    insertPoint = [idx for idx, line in enumerate(template) if re.search('SETUP', line)][0]
    for idx, i in enumerate(inputs):
        template.insert(insertPoint, "std::vector<int> %s_offered;\nbool %s_accepted = false;\nstd::vector<int> %s;\ngetInputVector(std::string(argv[%d]), %s_offered);\n"%(i[0], i[0], i[0], idx+2, i[0]))
    for o in outputs:
        template.insert(insertPoint, "std::vector<int> %s;\n"%(o[0]))

    insertPoint = [idx for idx, line in enumerate(template) if re.search('HANDSHAKE', line)][0]
    for i in inputs:
        template.insert(insertPoint, "if (%s_accepted) {\ntb->%s_valid = 0x0;\n%s_accepted = false;\n}\n"%(i[0], i[0], i[0]))
   
        
    insertPoint = [idx for idx, line in enumerate(template) if re.search('INPUTS', line)][0]
    for i in inputs:
        if i[1] != "none":
            template.insert(insertPoint, "tb->%s = (tb->%s_valid == 1) ? %s_offered.back() : 0;\n"%(i[0], i[0], i[0]))
        template.insert(insertPoint, "tb->%s_valid = (tb->%s_valid == 0) ? ((%s_offered.size() > 0) ? (myrand() & 0x1) : 0x0) : (tb->%s_valid = 0x1);\n"%(i[0], i[0], i[0], i[0]))
    for o in outputs:
        template.insert(insertPoint, "tb->%s_ready = myrand() & 0x1;\n"%(o[0]))

    insertPoint = [idx for idx, line in enumerate(template) if re.search('TRACE', line)][0]
    for i in inputs:
        if i[1] != "none":
            template.insert(insertPoint, "%s_accepted = acceptInput(tb->%s_ready, tb->%s_valid, tb->%s, %s_offered, %s);\n"%(i[0], i[0], i[0], i[0], i[0], i[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, tb->%s);\n"%(i[0], i[0], i[0]))
        else:
            template.insert(insertPoint, "%s_accepted = acceptInput(tb->%s_ready, tb->%s_valid, 0x0, %s_offered, %s);\n"%(i[0], i[0], i[0], i[0], i[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, 0x0);\n"%(i[0], i[0]))
    for o in outputs:
        if o[1] != "none":
            template.insert(insertPoint, "recordOutput(tb->%s_ready, tb->%s_valid, tb->%s, %s);\n"%(o[0], o[0], o[0], o[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, tb->%s);\n"%(o[0], o[0], o[0]))
        else:
            template.insert(insertPoint, "recordOutput(tb->%s_ready, tb->%s_valid, 0x0, %s);\n"%(o[0], o[0], o[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, 0x0);\n"%(o[0], o[0]))

    insertPoint = [idx for idx, line in enumerate(template) if re.search('RESULTS', line)][0]
    for i in inputs:
        template.insert(insertPoint, "printResult(outFile, %s);\n"%(i[0]))
    for o in outputs:
        template.insert(insertPoint, "printResult(outFile, %s);\n"%(o[0]))

    inputs_exist = "";
    for i in inputs:
        inputs_exist += "!%s_offered.empty() && "%(i[0])
    inputs_exist = inputs_exist[:-3]
    if len(inputs) > 0:
        inputs_exist = "(" + inputs_exist
        inputs_exist += ") || "
    
    no_inputs = ""
    for i in inputs:
        no_inputs += "%s_offered.empty() || "%(i[0])
    no_inputs = no_inputs[:-3]
    if len(inputs) > 0:
        no_inputs = "if (" + no_inputs + ") delay--;"
    else:
        no_inputs += "delay--;"
    
    # write driver file
    driverFile = open("%s-driver.cpp"%(basename), "w")
    template_string = "".join(x for x in template)
    template_string = template_string.replace("NAME", "test_%s"%(basename))
    template_string = template_string.replace("EXIST", inputs_exist)
    template_string = template_string.replace("DELAY", no_inputs)
    driverFile.write(template_string)
    driverFile.close()

    # random input strings
    input_arg = ""
    for i in inputs:
        if i[1] == "none":
            input_vals = ['n' for j in range(4)]
        else:
            bits = 64 if i[1] == "index" else int(i[1].split("i")[1])
            input_vals = [str(random.randint(0,pow(2, min(bits-1, 8)))) for j in range(4)]
        input_arg += ",".join(input_vals) + " "
    return input_arg

def cleanup(basename):
    files = ["%s-driver.cpp"%(basename), "%s.mlir-out"%(basename), "%s.sv"%(basename), "%s.sv.d"%(basename)]
    for f in files:
        if (os.path.isfile(f)):
            os.remove(f)
        elif (os.path.exists(f)):
            shutil.rmtree(f)

for test in tests:
    basename = os.path.basename(test).split('.')[0]
    print("testing %s %s"%(test, basename))

    #cleanup(basename)
    #continue

    input_arg = setup(test, basename)

    # lower to sv and run simulation
    run_success = [False, False]
    for mode in [0, 1]:
        print("power gated" if mode else "standard")
        if lower(test, basename, mode):
            run_success[mode] = run(basename, input_arg, mode)

    # compare results
    if run_success[0] and run_success[1]:
        result = subprocess.run("diff %s.sv.d/%s-std.out %s.sv.d/%s-wake.out"%(basename, basename, basename, basename), shell=True, capture_output=True)
        if (len(result.stdout) == 0):
            print("%s PASS"%(basename))
            cleanup(basename)
        else:
            print("%s FAIL"%(basename))
    else:
        print("%s FAIL"%(basename))
    print("---------")
