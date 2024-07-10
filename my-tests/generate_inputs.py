#!/usr/bin/python3

import sys
import os
import re
import random

assert len(sys.argv) == 3
test = sys.argv[1]
num_configs = int(sys.argv[2])

def getInputs(path):
    # get func op io
    inputFile = open("%s"%(path), "r")
    for line in inputFile.readlines():
        if re.findall(r'handshake.func', line):
            inputs = [i for i in re.findall(r'%(\w+?) *: (\w+)', line)]
           
    inputFile.close()
    return inputs

basename = os.path.basename(test).split('.')[0]
inputFile = open("data/%s-inputs.txt"%(basename), "w")
inputs = getInputs(test)

header = "config,"
for i in inputs:
    header += "%s,"%(i[0])
header = header[:-1]
header += "\n"
inputFile.write(header)
   
for config in range(num_configs):
    config_string = "%d,"%(config)
    for i in inputs:
        if i[1] == "none":
            config_string += "n,"
        else:
            bits = 64 if i[1] == "index" else int(i[1].split("i")[1])
            config_string += "%s,"%(str(random.randint(0,pow(2, min(bits-1, 8)))))
    config_string = config_string[:-1]
    config_string += "\n"
    inputFile.write(config_string)
    
inputFile.close()
