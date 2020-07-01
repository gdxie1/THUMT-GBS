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
    # parser.add_argument("--output", type=str, required=True,
    #                     help="Path of output file")
    # parser.add_argument("--grid_file", type=str, required=True,
    #                     help="alignment file")
    # parser.add_argument("--time_file", type=str, required=True,
    #                     help="alignment file")
    parser.add_argument('--constraints', type=str, default=None, required=False,
                        help='(Optional) json file containing one (possibly empty) list of constraints per input line')
    parser.add_argument("--weight_threshold", type=float, default=0.3,
                        help="the threshold to start a constraint")
    parser.add_argument("--encdec_att_layer", type=int, default=5, choices=range(0, 6),
                        help="the layer to perform weight evaluation for constraint")
    parser.add_argument("--heads", type=int, nargs="+", required=True, choices=range(0, 8),
                        help="Path of trained models")
    parser.add_argument("--head_model", type=str, default="union",
                        help="how to use the weights between heads: union|average")
    parser.add_argument("--layer_model", type=int, default=0,
                        help="how to use the weights between layers")
    return parser.parse_args()

def main(args):
    source_sen = None
    with codecs.open(args.source, "r",  encoding='utf8') as input_file:
        source_sen = [line.strip().split() for line in input_file]  # if line.strip()

    # for src in source_sen:
    #     src.split()
    # source_sen = [line.split(line.strip()) for line in input_file if line.strip()]

    target_sen = None
    with codecs.open(args.target, "r", encoding='utf8') as input_file:
        target_sen = [line.strip().split() for line in input_file]

    # read file
    #restored_grid = None
    with open(args.target+'.time_hyps', "r") as grid_file:
        sen_decode_time = cPickle.load(grid_file)
        restored_grid = cPickle.load(grid_file)

    constraints = None
    if args.constraints is not None:
        constraints = json.loads(codecs.open(args.constraints, encoding='utf8').read())

    file_output_analysis = codecs.open(args.target+".analysis.txt", 'w', encoding='utf8')
    print("id | src_length | cons_num | time | hyps | step_hyps |grid_length | grid_height")
    file_output_analysis.write("id | src_length | cons_num | time | hyps | step_hyps | grid_length | grid_height\n")
    for i, sen in enumerate(source_sen):
        src = source_sen[i]
        src_len = len(src)
        grid = restored_grid[i]
        top_row = max(k[1] for k in grid.keys())
        last_col = max(k[0] for k in grid.keys())
        hyps_num = sum(grid.values()[1:])
        step_hyps_num = float(hyps_num)/(last_col) + 0.5  #四舍五入

        print("%4d | %4d | %2d | %10f | %5d | %4d | %4d | %4d" %
              (i, src_len, len(constraints[i]), sen_decode_time[i], hyps_num, step_hyps_num, last_col+1, top_row+1))
        file_output_analysis.write("%4d | %4d | %2d | %10f | %5d | %4d | %4d | %4d\n" %
              (i, src_len, len(constraints[i]), sen_decode_time[i], hyps_num, step_hyps_num, last_col+1, top_row+1))
    file_output_analysis.close()
    #file_output = open(args.output, "w")
    file_output = codecs.open(args.target+'.grid.txt', 'w', encoding='utf8')
    for i, sen in enumerate(target_sen):
        # if i ==25:
        #     print(i)
        # sp = ' '* (max_len+1)
        # print cons and source
        # first print cons x
        cons_src_list = []
        for word in source_sen[i]:
            cons_src_list.append(word)

        file_output.write("\n----------------------\n")
        if len(constraints[i]) != 0:  # 在cons前后加上[]
            sen_cons_tgt = ""
            for cons in constraints[i]:
                cons_tgt = " ".join(cons["tgt"])
                sen_cons_tgt = sen_cons_tgt + '|' + cons_tgt
                start = cons["src_pos"][0]
                end = cons["src_pos"][1]-1
                cons_src_list[start] = '[' + cons_src_list[start]
                cons_src_list[end] = cons_src_list[end] +']'
                # for pos in range(cons["src_pos"][0],cons["src_pos"][1]):
                #     cons_src_list[pos] = '[' + cons_src_list[pos][1:]
            file_output.write(sen_cons_tgt)
            file_output.write('\n')


        cons_src = " ".join(cons_src_list)

        file_output.write("%d   %s\n" % (i, cons_src))

        # 开始打印grid
        grid = restored_grid[i]
        top_row = max(k[1] for k in grid.keys())
        last_col = max(k[0] for k in grid.keys())
        for r in reversed(range(top_row+1)):
            file_output.write('  	')  # 每一行进行缩进
            for c in range(1, last_col+1):
                if (c, r) in grid:
                    file_output.write('%2d	' % grid[(c, r)])
                else:
                    file_output.write('  	')
            file_output.write('\n')

        #  开始的打印target
        # tgt_out_line = sen[:]
        # sens_max_index = max_index_list[i]
        # if len(sen) == 0:
        #     continue
        #
        # lens = [len(word) for word in sen]
        # max_len = max(lens)
        #
        # for j, word in enumerate(tgt_out_line):
        #     try:
        #         max_index, _ = sens_max_index[j]
        #         tgt_out_line[j] = " "*(max_len-len(word)) + word + ":" + source_sen[i][max_index]
        #     except IndexError as ex:
        #         file_output.write(ex)
        line = ' '.join(sen)
        file_output.write(line)

    file_output.close()

        #
        #     count = 0
        #     for output, score, ratio in zip(restored_outputs, restored_scores, restored_ratio):
        #         decoded = []
        #         for idx in output:
        #             if idx == params.mapping["target"][params.eos]:
        #                 break
        #             decoded.append(vocab[idx])
        #         decoded = " ".join(decoded)
        #
        #         if not args.verbose:
        #             outfile.write("%s\n" % decoded)
        #         else:
        #             pattern = "%d ||| %s ||| %s ||| %f ||| %f ||| %d\n"
        #             source = restored_inputs[count]
        #             cons = restored_constraints[count]
        #             cons_token_num = 0
        #             for cons_item in cons:
        #                 cons_token_num+=cons_item["tgt_len"]
        #             values = (count, source, decoded, score, ratios[0], cons_token_num)
        #             outfile.write(pattern % values)
        #         count += 1
        #
        # with open(args.output+".alignment", "w") as outfile:
        #     count = 0
        #     for alignment in restored_grid:
        #         outfile.write("%d\n" % count)
        #         cPickle.dump(alignment, outfile)
        #         count += 1
if __name__ == "__main__":
    main(parse_args())