import sys
import json

f = open(sys.argv[1], "r")  # input ends with .ipynb
j = json.load(f)
of = open(sys.argv[2], "w")  # output ends with .py
if j["nbformat"] >= 4:
    for i, cell in enumerate(j["cells"]):
        of.write("#cell " + str(i) + "\n")
        for line in cell["source"]:
            of.write(line)
        of.write("\n\n")
else:
    for i, cell in enumerate(j["worksheets"][0]["cells"]):
        of.write("#cell " + str(i) + "\n")
        for line in cell["input"]:
            of.write(line)
        of.write("\n\n")

of.close()
