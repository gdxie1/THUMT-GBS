#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os
import json
import codecs
import operator
import cPickle
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Path of input file")

    # input files
    parser.add_argument("--source", type=str, default= None, required=False,
                        help="Path of source corpus")
    parser.add_argument("--target", type=str, default= None, required=False,
                        help="Path of target corpus")
    return parser.parse_args()

def main(args):
    source_sen = None
    with codecs.open(args.source, "r",  encoding='utf8') as input_file:
        source_sen = [line.strip().split() for line in input_file]  # if line.strip()

    target_sen = None
    with codecs.open(args.target, "r", encoding='utf8') as input_file:
        target_sen = [line.strip().split() for line in input_file]

    with open(args.source+'.len', "w") as len_file:
        for line in source_sen:
            len_file.write('%d\n' % len(line))

    with codecs.open(args.target+'.len', "w", encoding='utf8') as len_file:
        for i, line in enumerate(target_sen):
            len_file.write('%d\n' % len(line))
    # with codecs.open(args.target+'.len1', "w", encoding='utf8') as len_file:
    #     for i, line in enumerate(target_sen):
    #         print(i)
    #         # if i == 1141397:
    #         #     a = 10
    #         len_file.write('%i %d %s\n' % (i, len(line), ' '.join(line)))
    # with codecs.open(args.target+'1', "w", encoding='utf8') as new_file:
    #     for i, line in enumerate(target_sen):
    #         new_file.write('%s\n' % (' '.join(line)))

if __name__ == "__main__":
    main(parse_args())