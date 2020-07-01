# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import operator

import numpy as np
import tensorflow as tf
import cPickle
import json
import codecs



def sort_and_zip_files(names):
    inputs = []
    input_lens = []
    files = [tf.gfile.GFile(name) for name in names]

    count = 0

    for lines in zip(*files):
        lines = [line.strip() for line in lines]
        input_lens.append((count, len(lines[0].split())))
        inputs.append(lines)
        count += 1

    # Close files
    for fd in files:
        fd.close()

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=True)
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])

    return [list(x) for x in zip(*sorted_inputs)]


def get_evaluation_input(inputs, params):
    with tf.device("/cpu:0"):
        # Create datasets
        datasets = []

        for data in inputs:
            dataset = tf.data.Dataset.from_tensor_slices(data)
            # Split string
            dataset = dataset.map(lambda x: tf.string_split([x]).values,
                                  num_parallel_calls=params.num_threads)
            # Append <eos>
            dataset = dataset.map(
                lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
                num_parallel_calls=params.num_threads
            )
            datasets.append(dataset)

        dataset = tf.data.Dataset.zip(tuple(datasets))

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda *x: {
                "source": x[0],
                "source_length": tf.shape(x[0])[0],
                "references": x[1:]
            },
            num_parallel_calls=params.num_threads
        )

        dataset = dataset.padded_batch(
            params.eval_batch_size,
            {
                "source": [tf.Dimension(None)],
                "source_length": [],
                "references": (tf.Dimension(None),) * (len(inputs) - 1)
            },
            {
                "source": params.pad,
                "source_length": 0,
                "references": (params.pad,) * (len(inputs) - 1)
            }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Covert source symbols to ids
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])

    return features



def batch_examples(example, batch_size, max_length, mantissa_bits,
                   shard_multiplier=1, length_multiplier=1, constant=False,
                   num_threads=4, drop_long_sequences=True):
    """ Batch examples

    :param example: A dictionary of <feature name, Tensor>.
    :param batch_size: The number of tokens or sentences in a batch
    :param max_length: The maximum length of a example to keep
    :param mantissa_bits: An integer
    :param shard_multiplier: an integer increasing the batch_size to suit
        splitting across data shards.
    :param length_multiplier: an integer multiplier that is used to
        increase the batch sizes and sequence length tolerance.
    :param constant: Whether to use constant batch size
    :param num_threads: Number of threads
    :param drop_long_sequences: Whether to drop long sequences

    :returns: A dictionary of batched examples
    """

    with tf.name_scope("batch_examples"):
        max_length = max_length or batch_size
        min_length = 8
        mantissa_bits = mantissa_bits

        # Compute boundaries
        x = min_length
        boundaries = []

        while x < max_length:
            boundaries.append(x)
            x += 2 ** max(0, int(math.log(x, 2)) - mantissa_bits)

        # Whether the batch size is constant
        if not constant:
            batch_sizes = [max(1, batch_size // length)
                           for length in boundaries + [max_length]]
            batch_sizes = [b * shard_multiplier for b in batch_sizes]
            bucket_capacities = [2 * b for b in batch_sizes]
        else:
            batch_sizes = batch_size * shard_multiplier
            bucket_capacities = [2 * n for n in boundaries + [max_length]]

        max_length *= length_multiplier
        boundaries = [boundary * length_multiplier for boundary in boundaries]
        max_length = max_length if drop_long_sequences else 10 ** 9

        # The queue to bucket on will be chosen based on maximum length
        max_example_length = 0
        for v in example.values():
            if v.shape.ndims > 0:
                seq_length = tf.shape(v)[0]
                max_example_length = tf.maximum(max_example_length, seq_length)

        (_, outputs) = tf.contrib.training.bucket_by_sequence_length(
            max_example_length,
            example,
            batch_sizes,
            [b + 1 for b in boundaries],
            num_threads=num_threads,
            capacity=2,  # Number of full batches to store, we don't need many.
            bucket_capacities=bucket_capacities,
            dynamic_pad=True,
            keep_input=(max_example_length <= max_length)
        )

    return outputs

def get_training_input_with_alignment(filenames, params):
    """ Get input for training stage

    :param filenames: A list contains [source_filename, target_filename]
    :param params: Hyper-parameters

    :returns: A dictionary of pair <Key, Tensor>
    """
    with tf.device("/cpu:0"):
        src_dataset = tf.data.TextLineDataset(filenames[0])
        tgt_dataset = tf.data.TextLineDataset(filenames[1])
        src_alignment = tf.data.TextLineDataset(filenames[0]+'.align_pos')
        tgt_alignment = tf.data.TextLineDataset(filenames[1]+'.align_pos')
        align_mask = tf.data.TextLineDataset(filenames[1]+'.align_mask')
        #对齐的数据，以tgt为核心，tgt的词序列 0 1 2 3 4 ....每个对应一个src的词编号

        dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset, tgt_alignment, src_alignment, align_mask))
        dataset = dataset.shuffle(params.buffer_size)
        dataset = dataset.repeat()

        # Split string
        dataset = dataset.map(
            lambda src, tgt, tgt_align, src_align, align_mask: (
                tf.string_split([src]).values,
                tf.string_split([tgt]).values,
                tf.string_split([tgt_align]).values,
                tf.string_split([src_align]).values,
                tf.string_split([align_mask]).values  # mask
            ),
            num_parallel_calls=params.num_threads
        )

        # Append <eos> symbol
        dataset = dataset.map(
            lambda src, tgt, align_tgt, align_src, align_mask: (
                tf.concat([src, [tf.constant(params.eos)]], axis=0),
                tf.concat([tgt, [tf.constant(params.eos)]], axis=0),
                tf.string_to_number(align_tgt, tf.int32),
                tf.string_to_number(align_src, tf.int32),
                tf.string_to_number(align_mask, tf.int32),
            ),
            num_parallel_calls=params.num_threads
        )

        # Convert to dictionary
        dataset = dataset.map(
            lambda src, tgt, align_tgt, align_src, align_mask: {
                "source": src,
                "target": tgt,
                "align_tgt": align_tgt,
                "align_src": align_src,
                "align_mask": align_mask,
                "align_length": tf.shape(align_tgt),
                "source_length": tf.shape(src),
                "target_length": tf.shape(tgt)
            },
            num_parallel_calls=params.num_threads
        )

        # Create iterator
        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        # Create lookup table
        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        # String to index lookup
        features["source"] = src_table.lookup(features["source"])
        features["target"] = tgt_table.lookup(features["target"])

        # Batching
        shard_multiplier = len(params.device_list) * params.update_cycle
        features = batch_examples(features, params.batch_size,
                                  params.max_length, params.mantissa_bits,
                                  shard_multiplier=shard_multiplier,
                                  length_multiplier=params.length_multiplier,
                                  constant=params.constant_batch_size,
                                  num_threads=params.num_threads)

        # Convert to int32
        features["source"] = tf.to_int32(features["source"])
        features["target"] = tf.to_int32(features["target"])
        features["align_tgt"] = tf.to_int32(features["align_tgt"])
        features["align_src"] = tf.to_int32(features["align_src"])
        features["align_mask"] = tf.to_int32(features["align_mask"])
        features["align_length"] = tf.to_int32(features["align_length"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["target_length"] = tf.to_int32(features["target_length"])
        features["source_length"] = tf.squeeze(features["source_length"], 1)
        features["target_length"] = tf.squeeze(features["target_length"], 1)

        return features


def get_input_with_src_constraints(inputs, input_constraints, params):
    with tf.device("/cpu:0"):

        # [  cons should be something like this. The useful info is tgt and src_pos
        #     {
        #         "src": [
        #             "25.@@"
        #         ],
        #         "tgt": [
        #             "25.@@",
        #             "27.@@",
        #             "27.@@"
        #         ],
        #         "src_pos": [
        #             41,
        #             42
        #         ],
        #         "tgt_pos": [
        #             54,
        #             57
        #         ]
        #     }
        # ]
    
        # 当batch为多个的时候，几个句子之间的cons的数量需要对齐，所有的cons-item 需要pad对齐
        # 以下为所有句子的cons完全对齐 后面的 tf.data.Dataset.from_tensor_slices 需要对齐
        # 尝试将此改为 在 Dataset 中 batch 方法
        max_cons_num_of_sen=0  # each sentence's max cons phrase number
        max_item_tgt_len=0   # each cons tgt phrase max length
        if len(input_constraints):
            # json always read and convert to unicode, so here convert from unicode to str
            # each source's constraints
            for constraints_of_sen in input_constraints:
                if len(constraints_of_sen) > max_cons_num_of_sen:
                    max_cons_num_of_sen=len(constraints_of_sen)
                #cons_of_sen = []
                for cons_item in constraints_of_sen:
                    cons_item_tgt_len = len(cons_item["tgt"])
                    if cons_item_tgt_len > max_item_tgt_len:
                        max_item_tgt_len = cons_item_tgt_len
                    tgt = [word.encode('utf-8') for word in cons_item["tgt"]]
                    cons_item["tgt"] = tgt
                    # extend the src_pos according parameter
                    cons_item["src_pos"][0] = cons_item["src_pos"][0]-params.pos_extend
                    cons_item["src_pos"][1] = cons_item["src_pos"][1]+params.pos_extend

            cons_len = [] # each sentence's real cons number
            #pad to align
            for constraints_of_sen in input_constraints:  # each source's constraints
                constraints_of_sen_num = len(constraints_of_sen)
                if constraints_of_sen_num<max_cons_num_of_sen:
                    constraints_of_sen.extend([{"src": [], "tgt": [], "src_pos": [0,0], "tgt_pos": [0,0]} 
                                               for i in range(max_cons_num_of_sen-constraints_of_sen_num)])

                cons_item_tgt_len_list=[]
                for cons_item in constraints_of_sen:
                    cons_item_tgt_len = len(cons_item["tgt"])
                    cons_item["tgt_len"] = cons_item_tgt_len  # 保存此数据，在translator检查原始cons的时候用
                    cons_item_tgt_len_list.append(cons_item_tgt_len)

                    if cons_item_tgt_len <= max_item_tgt_len:
                        #the last position of each constrints  asigned an eos
                        cons_item["tgt"].extend([params.pad]*(max_item_tgt_len-cons_item_tgt_len))
                        cons_item["tgt"].append(params.eos)
                cons_len.append(cons_item_tgt_len_list)

        else:
            #create empty constraint list to match the calculating
            input_constraints = [[{"src": [], "tgt": [], "src_pos": [0,0], "tgt_pos": [0,0]}]]*len(inputs)
            cons_len = [[0]]*len(inputs)

        #extract src_pos, tgt sperately. because these infomation is useful
        cons_src_pos= []
        cons_tgt = []

        for constraints_of_sen in input_constraints:
            src_pos = []
            tgts = []
            for cons_item in constraints_of_sen:
                src_pos.append(cons_item["src_pos"])
                tgts.append(cons_item["tgt"])
            cons_src_pos.append(src_pos)
            cons_tgt.append(tgts)

                    #assert max_item_tgt_len > 0
        assert len(inputs) == len(input_constraints)

        #convert to tensor dataset
        dataset_cons_tgt = tf.data.Dataset.from_tensor_slices(
            tf.constant(cons_tgt, dtype=tf.string))

        dataset_cons_src_pos = tf.data.Dataset.from_tensor_slices(
            tf.constant(cons_src_pos, dtype=tf.int32))

        dataset_cons_len = tf.data.Dataset.from_tensor_slices(
            tf.constant(cons_len, dtype=tf.int32))

        #convert to tensor dataset
        dataset = tf.data.Dataset.from_tensor_slices(
            tf.constant(inputs)
        )
        # Split string
        dataset = dataset.map(lambda x: tf.string_split([x]).values,
                              num_parallel_calls=params.num_threads)

        # Append <eos>
        dataset = dataset.map(
            lambda x: tf.concat([x, [tf.constant(params.eos)]], axis=0),
            num_parallel_calls=params.num_threads
        )

        # Convert tuple to dictionary
        dataset = dataset.map(
            lambda x: {"source": x, "source_length": tf.shape(x)[0]},
            num_parallel_calls=params.num_threads
        )


        dataset = dataset.zip((dataset, dataset_cons_src_pos, dataset_cons_tgt, dataset_cons_len))
        dataset = dataset.map(
            lambda src, _cons_src_pos, _cons_tgt, _cons_len: {
                'source': src['source'],
                'source_length': src['source_length'],
                "constraints_src_pos": _cons_src_pos,
                'constraints': _cons_tgt,
                'constraints_len': _cons_len
            }
        )
        dataset = dataset.padded_batch(
            params.decode_batch_size * len(params.device_list),
            {"source": [tf.Dimension(None)],
             "source_length": [],
             "constraints_src_pos":[tf.Dimension(None), 2],
             "constraints": [tf.Dimension(None), tf.Dimension(None)],
             "constraints_len": [tf.Dimension(None)]
             },
            # 这里需要补齐
            {"source": params.pad,
             "source_length": 0,
             "constraints_src_pos": 0,
             "constraints": params.pad,
             "constraints_len": 0
             }
        )

        iterator = dataset.make_one_shot_iterator()
        features = iterator.get_next()

        src_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["source"]),
            default_value=params.mapping["source"][params.unk]
        )
        tgt_table = tf.contrib.lookup.index_table_from_tensor(
            tf.constant(params.vocabulary["target"]),
            default_value=params.mapping["target"][params.unk]
        )

        features["source"] = src_table.lookup(features["source"])
        features["constraints"] =tgt_table.lookup(features["constraints"])

        features["source"] = tf.to_int32(features["source"])
        features["source_length"] = tf.to_int32(features["source_length"])
        features["constraints_src_pos"] = tf.to_int32(features["constraints_src_pos"])
        features["constraints"] = tf.to_int32(features["constraints"])
        features["constraints_len"] = tf.to_int32(features["constraints_len"])

        return features

def sort_input_src_cons(filename, cons_filename, reverse=True):
    # Read file
    with tf.gfile.Open(filename) as fd:
        inputs = [line.strip() for line in fd if line.strip()]

    # for i , line in enumerate(inputs):
    #     if(len(line) > 350):
    #         inputs[i] = line[0:350]  # cut to be shorter  这里是人为加的代码，导致一个大bug

    input_lens = [
        (i, len(line.strip().split())) for i, line in enumerate(inputs)
    ]

    sorted_input_lens = sorted(input_lens, key=operator.itemgetter(1),
                               reverse=reverse)


    sorted_keys = {}
    sorted_inputs = []

    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i # index 原来句子的索引，i 现在句子的索引

    # Read input constraints
    sorted_constraints = []
    if cons_filename is not None:
        constraints = json.loads(codecs.open(cons_filename, encoding='utf8').read())
        for i, (index, _) in enumerate(sorted_input_lens):
            sorted_constraints.append(constraints[index])

        assert (len(sorted_inputs) == len(sorted_constraints))


        # # sorted_constraints = [constraints[sorted_keys[i]] for i in range(input_len)]
        # cons = None
        # for i in range(input_len):
        #     j = sorted_keys[i]
        #     cons = constraints[j]
        #     sorted_constraints.append(cons)
        #     print('%d-%d:%s', (i, j, sorted_inputs[i]))
        #     print(cons)



    return sorted_keys, sorted_inputs, sorted_constraints
