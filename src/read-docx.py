import docx
import sys
import os
import csv


def process(path):
    record = 0  # skip the first few lines of every document
    doc = docx.Document(path)
    label = os.path.basename(path).replace(".docx", "")
    # extract the label from the path
    text = ""

    for i in doc.paragraphs:
        line = i.text  # read doc line by line, including empty lines
        if line.startswith(("=",)):
            if record == 1:
                # previously recording. stop recording when = is encountered for the first time. flush the text buffer
                row = [text, label]
                # write a row to the csv file
                writer.writerow(row)
                text = ""
            record = 0
        elif line.startswith(("Date",)):
            record = 1

        if record:
            text = text + line


# the input is paths to all docx files.
# usage example:time python src/read-docx.py data/qemu*/*.docx
# open the file in the write mode. assume that the path exists.
with open("data/data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    header = ["text", "label"]
    writer.writerow(header)
    for path in sys.argv[1:]:
        process(path)
