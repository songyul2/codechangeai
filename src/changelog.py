# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See LICENSE for more details.
#
# Copyright: Red Hat Inc. 2021
# Author: Songyu Liu <sonliu@redhat.com>

import docx
import sys
import os
import csv


def read_docx(path, writer):

    record = 0
    # skip the first few lines of every document. record is a flag variable. when flag == 1, we record a line.
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
        if record:
            text = text + line
        elif line.startswith(("Date",)):
            record = 1


def read_changelog(path, writer):
    record = 0
    # skip the first few lines of every document. record is a flag variable. when flag == 1, we record a line.
    doc = open(path, encoding="latin-1")
    label = os.path.basename(path)
    # extract the label from the path
    text = ""

    for line in doc:

        if line.startswith(("=", "commit")):
            if record == 1:
                # previously recording. stop recording when = is encountered for the first time. flush the text buffer
                row = [text, label]
                # write a row to the csv file
                writer.writerow(row)
                text = ""
            record = 0
        if record:
            text = text + line
        elif line.startswith(("Date",)):
            record = 1  # start recording the next line


def write_changelog(path, class_preds):
    """
    path is the path to a changelog. preds is a list of predictions.
    these predictions will be added to the commit line.
    the output will be in a different file under the same dir as the path.
    """
    path_output = path + "-preds"
    doc = open(path, encoding="latin-1")
    text = ""

    for line in doc:

        if line.startswith(("commit")):
            if len(class_preds) > 0:
                text = text + str(class_preds[0])
                class_preds.pop(0)  # remove the 0th element. the argument is modified!
            else:
                break

        text = text + line

    with open(path_output, "w") as f:
        f.write(text)


def write_csv(paths):
    """
    input is a list of paths to all changelog/docx files. a csv that combines them will be written.
    """
    # open the file in the write mode. assume that the path exists. if newline='' is not specified, newlines embedded inside quoted fields will not be interpreted correctly,
    with open("data/data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["text", "label"]
        writer.writerow(header)
        for path in paths:
            if "docx" in path:
                read_docx(path, writer)
            else:
                read_changelog(path, writer)


# usage example:time python src/read-docx.py data/qemu*/*.docx
if __name__ == "__main__":
    paths = sys.argv[1:]  # slicing returns a list of paths
    write_csv(paths)
