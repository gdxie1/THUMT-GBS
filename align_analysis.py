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
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    return parser.parse_args()

def main(args):
    source_sen = None
    with open(args.source, "r") as input_file:
        source_sen = [line.strip().split() for line in input_file]  # if line.strip()

    for word in source_sen[0]:
        print(word)
    # read file
#    restored_alignment = []
#    with open(args.align_file, "r") as align_file:
#        line = align_file.readline()
#        count = int(line)
#        alignment = cPickle.load(align_file)
#        while 1:
#            restored_alignment.append(alignment)
#            # count = int(align_file.readline())
#            # alignment = cPickle.load(align_file)
#            try:
#                count = int(align_file.readline())
#                alignment = cPickle.load(align_file)
#                #print(count, len(alignment))
#            except ValueError as ex:
#                #print(ex)
#                break
#    max_index_list = []
#    for i, alignment in enumerate(restored_alignment):
#        sens_max_index = []
#        for word_align in alignment:  # 每一句的每个target词对的align
#            layer_align = np.zeros([0, word_align["layer_0"].shape[2]])
#            for layerid in range(0, 6):
#                one_layer_align = word_align["layer_%d" % layerid]
#                one_layer_align = np.squeeze(one_layer_align, 0)  # 1 8 Sl remove the 1
#                one_layer_align = np.mean(one_layer_align, 0, keepdims=True)  # average on head
#                layer_align = np.concatenate([layer_align, one_layer_align], 0)
#            layer_align = np.mean(layer_align, 0)
#            layer_align = layer_align[:-1]  # remove eos
#            # sen = source_sen[i]
#            # # if sen[-1] == '。' and layer_align.shape > 2:
#            #     layer_align = layer_align[:-1]  # remove 。
#
#            max_index = np.argmax(layer_align)
#            max_v = layer_align[max_index]
#            sens_max_index.append((max_index, max_v))
#
#
#            # layer_align = word_align["layer_%d" % args.encdec_att_layer]
#            # layer_align = np.squeeze(layer_align, 0)  # 1 8 Sl remove the 1
#            # layer_align = np.mean(layer_align, 0)  # average on head
#            # layer_align = layer_align[:-1]  # remove eos
#            # max_index = np.argmax(layer_align)
#            # max_v = layer_align[max_index]
#            # sens_max_index.append((max_index, max_v))
#        max_index_list.append(sens_max_index)
#    constraints = None
#    if args.constraints is not None:
#        constraints = json.loads(codecs.open(args.constraints, encoding='utf8').read())
#
#
    file_output = open(args.output, "w")
    for i, sen in enumerate(target_sen):
        # if i ==25:
        #     print(i)
        # sp = ' '* (max_len+1)
        # print cons and source
        # first print cons x
        cons_src_list = []
        for word in source_sen[i]:
            cons_src_list.append(word)

        # for word in source_sen[i]:  # 空格无法与显示的汉字对齐
        #     cons_src_list.append(' '*len(word))
        if len(constraints[i]) != 0:
            sen_cons_tgt = ""
            for cons in constraints[i]:
                cons_tgt = " ".join(cons["tgt"])
                sen_cons_tgt = sen_cons_tgt + cons_tgt
                start = cons["src_pos"][0]
                end = cons["src_pos"][1]-1
                cons_src_list[start] = '[' + cons_src_list[start]
                cons_src_list[end] = cons_src_list[end] +']'
                # for pos in range(cons["src_pos"][0],cons["src_pos"][1]):
                #     cons_src_list[pos] = '[' + cons_src_list[pos][1:]
            print(sen_cons_tgt)

        cons_src = " ".join(cons_src_list)
        print("----------------------")
        print("%d   %s" % (i, cons_src))
        #  开始的打印target
        tgt_out_line = sen[:]
        sens_max_index = max_index_list[i]
        if len(sen) == 0:
            continue

        lens = [len(word) for word in sen]
        max_len = max(lens)

        for j, word in enumerate(tgt_out_line):
            try:
                max_index, _ = sens_max_index[j]
                tgt_out_line[j] = " "*(max_len-len(word)) + word + ":" + source_sen[i][max_index]
            except IndexError as ex:
                print(ex)
        # for cons in constraints[i]:
        #     src_start = cons["src_pos"][0]
        #     src_end = cons["src_pos"][1]
            #
            # start = cons["tgt_pos"][0]
            # end = cons["tgt_pos"][1]
            # for pos in range(start, end):
            #     tgt_out_line[pos] = tgt_out_line[pos]+'*'
            # tgt_out_line[start] = tgt_out_line[start] +"|   "+" ".join(cons["src"])
        for line in tgt_out_line:
            print(line)

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
        #     for alignment in restored_alignment:
        #         outfile.write("%d\n" % count)
        #         cPickle.dump(alignment, outfile)
        #         count += 1
if __name__ == "__main__":
    main(parse_args())



        # for i, sen in enumerate(target_sen):
        #     print(i)
        #     lens = [len(word) for word in sen]
        #     max_len = max(lens)
        #     sp = ' '* (max_len+1)
        #     # print cons and source
        #     # first print cons x
        #     cons_src_list=[]
        #     for word in source_sen[i]:  # 空格无法与显示的汉字对齐
        #         cons_src_list.append(word)
        #
        #     # for word in source_sen[i]:  # 空格无法与显示的汉字对齐
        #     #     cons_src_list.append(' '*len(word))
        #     if len(constraints[i]) == 0:
        #         continue
        #     for cons in constraints[i]:
        #         start = cons["src_pos"][0]
        #         end = cons["src_pos"][1]-1
        #         cons_src_list[start] = '[' + cons_src_list[start]
        #         cons_src_list[end] = cons_src_list[end] +']'
        #         # for pos in range(cons["src_pos"][0],cons["src_pos"][1]):
        #         #     cons_src_list[pos] = '[' + cons_src_list[pos][1:]
        #     cons_src = sp + " ".join(cons_src_list)
        #     print(cons_src)
        #
        #     # src = " ".join(source_sen[i])
        #     # print(src)
        #     tgt_out_line = sen[:]
        #     for cons in constraints[i]:
        #         src_start = cons["src_pos"][0]
        #         src_end = cons["src_pos"][1]
        #
        #         start = cons["tgt_pos"][0]
        #         end = cons["tgt_pos"][1]
        #         for pos in range(start, end):
        #             tgt_out_line[pos] = tgt_out_line[pos]+'*'
        #         tgt_out_line[start] = tgt_out_line[start] +"    |   "+" ".join(cons["src"])
        #     for line in tgt_out_line:
        #         print(line)
