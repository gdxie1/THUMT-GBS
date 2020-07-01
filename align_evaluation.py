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
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--align_file", type=str, required=True,
                        help="alignment file")
    parser.add_argument("--fast_align_file", type=str, required=True,
                        help="alignment file")
    parser.add_argument("--head", type=int, required=True, choices=range(0, 8),
                        help="head")
    parser.add_argument("--layer", type=int, required=True, choices=range(0, 8),
                        help="layer")



    return parser.parse_args()

def main(args):
    source_sen = None
    with open(args.source, "r") as input_file:
        source_sen = [line.strip().split() for line in input_file]  # if line.strip()

    target_sen = None
    with open(args.target, "r") as input_file:
        target_sen = [line.strip().split() for line in input_file]

    # read file
    restored_alignment = []
    with open(args.align_file, "r") as align_file:
        line = align_file.readline()
        count = int(line)
        alignment = cPickle.load(align_file)
        while 1:
            restored_alignment.append(alignment)
            # count = int(align_file.readline())
            # alignment = cPickle.load(align_file)
            try:
                count = int(align_file.readline())
                alignment = cPickle.load(align_file)
                #print(count, len(alignment))
            except ValueError as ex:
                #print(ex)
                break
    # read the fast-align file and convert to tgt-src format
    fast_align_matrixes=[]
    with open(args.fast_align_file, "r") as align_file:
        for i, line in enumerate(align_file):
            align_list = line.strip().split()
            sen_align_src = []
            sen_align_tgt = []
            tgt_pos_set = set()
            for word_align in align_list:
                src_pos, tgt_pos = word_align.split('-')
                sen_align_src.append(int(src_pos))
                sen_align_tgt.append(int(tgt_pos))
                tgt_pos_set.add(int(tgt_pos))
            tgt_src_align_pos = zip(sen_align_tgt, sen_align_src)
            # get the src and tgt's length
            tgt_len = len(target_sen[i])
            src_len = len(source_sen[i])
            align_matrix = np.zeros([tgt_len, src_len], dtype=np.float32)
            for tgt, src in tgt_src_align_pos:
                align_matrix[tgt, src] = 1
            fast_align_matrixes.append(align_matrix)

    decoded_align_matrixs = []

    for i, alignment in enumerate(restored_alignment):  # each sentence's alignment
        tgt_align_matrixs = [] # np.zeros([0, alignment[0]["layer_0"].shape[2]])
        for word_align in alignment:  # 每一句的每个target词对的align
            layer_head_align = word_align["layer_%d" % args.layer][0, args.head]  # extract from 1 8 SL
            tgt_align_matrixs.append(layer_head_align)
        #print('src len %d | align src len %d' % (len(source_sen[i]), np.shape(sen_align_matrixs[0])[0]))

        # get the src and tgt's length
        tgt_len = len(target_sen[i])
        src_len = len(source_sen[i])
        sen_align_matrix = np.concatenate([tgt_align_matrixs], 0)
        sen_align_matrix = sen_align_matrix[:, :src_len]  # remove
        decoded_align_matrixs.append(sen_align_matrix)
        # print(np.shape(fast_align_matrixes[i])[1], np.shape(sen_align_matrix)[1])
        # if np.shape(fast_align_matrixes[i]) != np.shape(sen_align_matrix):
        #     a=1
    # 因为当初decode设计的有裁剪，encoder结果的源语言句可能比实际的短，导致 enc dec attention矩阵也短
    for i, fast_align_matrix in enumerate(fast_align_matrixes):
        fast_align_matrixes[i] = fast_align_matrix[:, :np.shape(decoded_align_matrixs[i])[1]]

    square_means = []
    cross_entropy = []
    loss_mode = 'square-mean'
    for fast_align_matrix, sen_align_matrix in zip(fast_align_matrixes, decoded_align_matrixs):
        #if loss_mode == square_means:
        _err = np.square(np.subtract(fast_align_matrix, sen_align_matrix))
        _err_sum = np.sum(_err, 1)  # sum on src
        square_means.append(_err_sum.mean())  # average on Target
        #else:
        # _err = -np.multiply(fast_align_matrix, np.log(sen_align_matrix))
        # _err_sum = np.sum(_err, 1)  # sum on src
        # cross_entropy.append(_err_sum.mean())  # average on Target
        # for i in range(np.shape(sen_align_matrix)[0]):
        #     for j in range(np.shape(sen_align_matrix)[1]):
        #         if sen_align_matrix[i, j] == 0.0:
        #             print(i, j)
        #
        #square_means.append(np.square(np.subtract(fast_align_matrix, sen_align_matrix)).mean())
    average_means = np.mean(np.asarray(square_means), dtype=np.float32)
    #average_entropy = np.mean(np.asarray(cross_entropy), dtype=np.float32)
    #print('square_mean: %d  cross_entropy: %d' % (average_means, average_entropy))

    print(average_means)

if __name__ == "__main__":
    main(parse_args())
