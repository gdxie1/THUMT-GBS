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
                        help="Path of source corpus length")
    parser.add_argument("--target", type=str, default= None, required=False,
                        help="Path of target corpus length")
    parser.add_argument("--rev_align", type=str, default=None, required=False,
                        help="Path of reverse alignment")
    return parser.parse_args()

def main(args):
    # 读取src tgt的句长
    src_lens = None
    with open(args.source + '.len', "r") as len_file:
        src_lens = [int(line) for line in len_file]

    tgt_lens = None
    with open(args.target + '.len', "r") as len_file:
        tgt_lens = [int(line) for line in len_file]

    align_pos = []
    with open(args.rev_align, "r") as align_file:
        # i = 1
        for i, line in enumerate(align_file):
            # print(i)
            tgt_align_list = line.split()
            sen_align_src = []
            sen_align_tgt = []
            tgt_pos_set = set()
            for word_align in tgt_align_list:
                src_pos, tgt_pos = word_align.split('-')
                sen_align_src.append(src_pos)
                sen_align_tgt.append(tgt_pos)
                tgt_pos_set.add(int(tgt_pos))
            # aligning missed tgt word to eos
            for xpos in range(tgt_lens[i] + 1):  # +3 to add eos--eos
                if xpos not in tgt_pos_set:
                    sen_align_src.append(str(src_lens[i]))  # eos positon
                    sen_align_tgt.append(str(xpos))
            align_pos_mask = ['1'] * len(sen_align_tgt)  # 产生mask以便GPU处理
            align_pos.append([' '.join(sen_align_tgt), ' '.join(sen_align_src), ' '.join(align_pos_mask)])

    mask_file = open(args.target+'.align_mask', "w")
    src_align_pos_file = open(args.source + '.align_pos', "w")
    tgt_align_pos_file = open(args.target + '.align_pos', "w")
    for tgt_align_line, src_align_line, align_mask_line in align_pos:
        tgt_align_pos_file.write(tgt_align_line)
        tgt_align_pos_file.write('\n')
        src_align_pos_file.write(src_align_line)
        src_align_pos_file.write('\n')
        mask_file.write(align_mask_line)
        mask_file.write('\n')
    mask_file.close()
    src_align_pos_file.close()
    tgt_align_pos_file.close()

if __name__ == "__main__":
    main(parse_args())