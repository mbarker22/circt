#!/usr/bin/python3

import sys
import subprocess
import os
import shutil

assert len(sys.argv) == 2
test = sys.argv[1]

def run(basename, inputs, wake_signals, config):
    cmd = "circt-rtl-sim.py generated_files/%s.sv generated_files/%s-driver.cpp --no-default-driver --top test_%s"%(basename+("-wake" if wake_signals else "-std"), basename+("-wake" if wake_signals else "-std"), basename)
    cmd += " --simargs=\"%s-%s %s\""%(basename+("-wake" if wake_signals else "-std"), config, inputs)
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

def cleanup(basename):
    files = ["%s.mlir-out"%(basename), "%s-std.sv.d"%(basename), "%s-wake.sv.d"%(basename)]
    for f in files:
        if (os.path.isfile(f)):
            os.remove(f)
        elif (os.path.exists(f)):
            shutil.rmtree(f)

basename = os.path.basename(test).split('.')[0]
print("running %s %s"%(test, basename))

inputFile = open("data/%s-inputs.txt"%(basename), "r")
input_strings = inputFile.readlines();
inputFile.close()

for s in input_strings[1:]:
    print(s[:-1])
    inputs = s[:-1].split(",")
    config = inputs[0]
    input_arg = " ".join(inputs[1:])
    run_success = [False, False]
    for mode in [0, 1]:
        run_success[mode] = run(basename, input_arg, delay, mode, config)

    if run_success[0] and run_success[1]:
        result = subprocess.run("diff %s-std.sv.d/%s-std.out %s-wake.sv.d/%s-wake.out"%(basename, basename, basename, basename), shell=True, capture_output=True)
        if (len(result.stdout) == 0):
            print("%s PASS"%(basename))
            current_dir = os.getcwd()
            os.replace("%s/%s-std.sv.d/%s-std-%s.data"%(current_dir, basename, basename, config), "output_files/%s-std-%s.data"%(basename, config))
            os.replace("%s/%s-wake.sv.d/%s-wake-%s.data"%(current_dir, basename, basename, config), "output_files/%s-wake-%s.data"%(basename, config))
            cleanup(basename)
        else:
            print("%s FAIL"%(basename))
    else:
        print("%s FAIL"%(basename))
    print("---------")
