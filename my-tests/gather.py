#!/usr/bin/python3

import sys
import os

assert len(sys.argv) == 2
test = sys.argv[1]

basename = os.path.basename(test).split('.')[0]
print("gathering %s %s"%(test, basename))

inputFile = open("data/%s-inputs.txt"%(basename), "r")
input_strings = inputFile.readlines();
inputFile.close()

with open("output_files/%s-wake-0.data"%(basename)) as file:
    instances = ["%s_wake,%s_awake,%s_wakeup"%(i.split(':')[0], i.split(':')[0], i.split(':')[0]) for i in file.readlines()[2:]]

header = input_strings[0][:-1] + ",std_sim_end,std_sim_cycles,wake_sim_end,wake_sim_cycles," + ",".join(instances)

csvFile = open("data/%s.csv"%(basename), "w")
csvFile.write(header+"\n")

for s in input_strings[1:]:
    config = s.split(',')[0]
    config_string = s[:-1] + ","
    #csvFile.write(config_string+"\n")
    for mode in [0, 1]:
        with open("output_files/%s-%s-%s.data"%(basename, ("wake" if mode else "std"), config)) as file:
            data = [line.split(': ')[1][:-1].split(' ') for line in file.readlines()]
            config_string += ",".join([x for row in data for x in row])
        config_string += ","
    csvFile.write(config_string[:-1]+"\n")
csvFile.close()
