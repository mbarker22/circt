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

def setup(path, basename, mode, inputs, outputs):
    
    if mode == 1:
        awakeSignals = getAwakeSignals(basename)
        
    # generate verilator driver file
    templateFile = open("driver-template.cpp", "r")
    template = templateFile.readlines();
    templateFile.close()

    insertPoint = [idx for idx, line in enumerate(template) if re.search('SETUP', line)][0]
    for idx, i in enumerate(inputs):
        template.insert(insertPoint, "std::vector<int> %s_offered;\nbool %s_accepted = false;\nstd::vector<int> %s;\ngetInputVector(std::string(argv[%d]), %s_offered);\n"%(i[0], i[0], i[0], idx+2, i[0]))
        template.insert(insertPoint, "dataFile << \"%s: \" << argv[%d] << std::endl;\n"%(i[0], idx+2))
    for o in outputs:
        template.insert(insertPoint, "std::vector<int> %s;\n"%(o[0]))
    if mode == 1:
        for a in awakeSignals:
            template.insert(insertPoint, "int %s_awake = 0;\n"%(a[0]))
            template.insert(insertPoint, "int %s_wake = 0;\n"%(a[0]))
            template.insert(insertPoint, "int %s_awake_prev = 1;\n"%(a[0]))
            template.insert(insertPoint, "int %s_wakeup = 0;\n"%(a[0]))

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
            template.insert(insertPoint, "if (recordOutput(tb->%s_ready, tb->%s_valid, tb->%s, %s)) last_cycle = main_time/4;\n"%(o[0], o[0], o[0], o[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, tb->%s);\n"%(o[0], o[0], o[0]))
        else:
            template.insert(insertPoint, "if (recordOutput(tb->%s_ready, tb->%s_valid, 0x0, %s)) last_cycle = main_time/4;\n"%(o[0], o[0], o[0]))
            template.insert(insertPoint, "trace(traceFile, tb->%s_ready, tb->%s_valid, 0x0);\n"%(o[0], o[0]))
    if mode == 1:
        for a in awakeSignals:
            template.insert(insertPoint, "%s_awake_prev = tb->rootp->%s;\n"%(a[0], a[1]))
            template.insert(insertPoint, "if (tb->rootp->%s == 1) %s_wake++;\n"%(a[2], a[0]))
            template.insert(insertPoint, "if (tb->rootp->%s == 1) {\n %s_awake++;\n if (%s_awake_prev == 0) %s_wakeup++;\n}\n"%(a[1], a[0], a[0], a[0]))

    insertPoint = [idx for idx, line in enumerate(template) if re.search('RESULTS', line)][0]
    for i in inputs:
        template.insert(insertPoint, "printResult(outFile, %s);\n"%(i[0]))
        template.insert(insertPoint, "outFile << \"%s: \";\n"%(i[0]))
    for o in outputs:
        template.insert(insertPoint, "printResult(outFile, %s);\n"%(o[0]))
        template.insert(insertPoint, "outFile << \"%s: \";\n"%(o[0]))
    template.insert(insertPoint, "outFile << std::hex;\n")
    if mode == 1:
        for a in awakeSignals:
            template.insert(insertPoint, "dataFile << \"%s_awake: \" << %s_awake << std::endl;\n"%(a[0], a[0]))
            template.insert(insertPoint, "dataFile << \"%s_wake: \" << %s_wake << std::endl;\n"%(a[0], a[0]))
            template.insert(insertPoint, "dataFile << \"%s_wakeup: \" << %s_wakeup << std::endl;\n"%(a[0], a[0]))
            
    template.insert(insertPoint, "dataFile << \"simulation cycles: \" << (main_time/4) << std::endl;")
    template.insert(insertPoint, "dataFile << \"last output: \" << last_cycle << std::endl;")

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

def cleanup(basename):
    files = ["%s-driver.cpp"%(basename), "%s.mlir-out"%(basename), "%s.sv"%(basename), "%s.sv.d"%(basename)]
    for f in files:
        if (os.path.isfile(f)):
            os.remove(f)
        elif (os.path.exists(f)):
            shutil.rmtree(f)

def getInputs(path, basename, num_inputs):
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

    # random input strings
    input_arg = ""
    for i in inputs:
        if i[1] == "none":
            input_vals = ['n' for j in range(num_inputs)]
        else:
            bits = 64 if i[1] == "index" else int(i[1].split("i")[1])
            input_vals = [str(random.randint(0,pow(2, min(bits-1, 8)))) for j in range(num_inputs)]
        input_arg += ",".join(input_vals) + " "
    return inputs, outputs, input_arg

def getAwakeSignals(basename):
    print("get metrics")
    svFile = open("%s.sv"%(basename), "r")
    svString = svFile.read()
    sleepModules = re.findall(r'wire *_(\w+)_sleep_awake;', svString)
    awakeSignals = []
    for module in sleepModules:
        awakeSignals += [("%s"%(i), "test_%s__DOT__%s__DOT___%s_sleep_awake"%(basename, i, module), "test_%s__DOT__%s__DOT__%s_sleep__DOT__wake"%(basename, i, module)) for i in re.findall(r'%s +(\w+) +\('%(module), svString)];
    svFile.close()
    svString = svString.replace('wake,', 'wake /* verilator public_flat */,')
    with open("%s.sv"%(basename), "w") as file:
        file.write(svString)
    return awakeSignals

for test in tests:
    basename = os.path.basename(test).split('.')[0]
    print("testing %s %s"%(test, basename))

    #cleanup(basename)
    #continue

    inputs, outputs, input_arg = getInputs(test, basename, 2)
    
    # lower to sv and run simulation
    run_success = [False, False]
    for mode in [0, 1]:
        print("power gated" if mode else "standard")
        if lower(test, basename, mode):
            setup(test, basename, mode, inputs, outputs)
            run_success[mode] = run(basename, input_arg, mode)

    # compare results
    if run_success[0] and run_success[1]:
        result = subprocess.run("diff %s.sv.d/%s-std.out %s.sv.d/%s-wake.out"%(basename, basename, basename, basename), shell=True, capture_output=True)
        if (len(result.stdout) == 0):
            print("%s PASS"%(basename))
            # generated_dir = os.path.join(os.getcwd(), "%s.sv.d"%(basename))
            # os.replace("%s/%s-std.data"%(generated_dir, basename), "simulation_data/%s-std.data"%(basename))
            # os.replace("%s/%s-wake.data"%(generated_dir, basename), "simulation_data/%s-wake.data"%(basename))
            #cleanup(basename)
        else:
            print("%s FAIL"%(basename))
    else:
        print("%s FAIL"%(basename))
    print("---------")
