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
import tensorflow as tf
import src_cons_inference as cons_inference
import thumt.data.vocab as vocabulary
import src_cons_transformer as transformer
# import thumt.utils.inference as inference
import thumt.utils.parallel as parallel
import thumt.utils.sampling as sampling
import src_cons_dataset as src_cons_dataset

from constrained_decoding import create_constrained_decoder
from constrained_decoding.translation_model.thumt_tm import ThumtTranslationModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    # parser.add_argument("--input", type=str, required=True,
    #                     help="Path of input file")

    # input files
    parser.add_argument("--input", type=str, required=True,
                        help="Path of source and target corpus")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    parser.add_argument('--constraints', type=str, default=None, required=False,
                        help='(Optional) json file containing one (possibly empty) list of constraints per input line')

    parser.add_argument("--pos_extend", type=int, default=0,
                        help="the value to extend pos of cons. pos0- and pos1+")
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
    parser.add_argument("--hgbs", action="store_true",
                        help="Enable hgbs method")
    parser.add_argument("--decode_hyp_num", type=int, default=100,
                        help="number of hyps that each gpu can decode at same time. Some gpu my OMM if this number too big")
    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        device_list=[0],
        num_threads=1,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=32,
        # sampling
        generate_samples=False,
        num_samples=1,
        min_length_ratio=0.0,
        max_length_ratio=1.5,
        min_sample_length=0,
        max_sample_length=0,
        sample_batch_size=32
    )
    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    if model_name.startswith("experimental_"):
        model_name = model_name[13:]

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    params.add_hparam("constraints", args.constraints)
    params.add_hparam("pos_extend", args.pos_extend)
    params.add_hparam("weight_threshold", args.weight_threshold)
    params.add_hparam("encdec_att_layer", args.encdec_att_layer)
    params.add_hparam("heads", args.heads)
    params.add_hparam("head_model", args.head_model)
    params.add_hparam("hgbs", args.hgbs)
    params.add_hparam("decode_hyp_num", args.decode_hyp_num)
    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break
    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat
                n = num_shards

    new_predictions = [prediction[:n] for prediction in predictions]  #这样确保留下有效的GPU OP没有错误
    return new_predictions, feed_dict

def main(args):
    tf.logging.set_verbosity(tf.logging.INFO)
    # Load configs
    model_cls_list = [transformer.Transformer for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]


    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor
            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_rerank_inference_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Read input file
        sorted_keys, sorted_inputs, sorted_constraints = \
            src_cons_dataset.sort_input_src_cons(args.input, args.constraints)

        # Build input queue
        features = src_cons_dataset.get_input_with_src_constraints(sorted_inputs, sorted_constraints, params)

        print(sorted_keys)

        #Create placeholder
        placeholders = []
        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i),
                "constraints_src_pos": tf.placeholder(tf.int32, [None, None, None], "constraints_src_pos_%d" % i),
                "constraints": tf.placeholder(tf.int32, [None, None, None], "constraints_%d" % i),
                "constraints_len": tf.placeholder(tf.int32, [None, None], "constraints_len_%d" % i)
            })
        encoding_fn = model_fns[0][0]

        encoder_op = parallel.data_parallelism(
            params.device_list, lambda f: encoding_fn(f, params),
            placeholders)

        # Create assign ops
        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        results = []

        # Create session
        with tf.Session(config=session_config(params)) as sess:

            # Restore variables
            sess.run(assign_op)
            sess.run(tf.tables_initializer())
            decoder_input_list = []
            encoder_output_list = []
            while True:
                try:
                    feats = sess.run(features)
                    encoder_op, feed_dict = shard_features(feats, placeholders,
                                                   encoder_op)
                    #print("encoding %d" % i)
                    encoder_state = sess.run(encoder_op, feed_dict=feed_dict)

                    for j in range(len(feats["source"])):
                        decoder_input_item = {
                            "source": [feats["source"][j]],
                            "source_length": [feats["source_length"][j]],
                            "constraints_src_pos": feats["constraints_src_pos"][j],
                            "constraints": feats["constraints"][j],
                            "constraints_len": feats["constraints_len"][j],
                        }
                        decoder_input_list.append(decoder_input_item)
                    # 不能简单的用GPU数量来循环,要用实际的输出来循环，因为有时候会空出GPU，比如最后一句或几句，无法凑够给1个GPU
                    for i in range(len(encoder_state[0])):
                        state_len = len(encoder_state[0][i])
                        for j in range(state_len):
                            encoder_output_item ={
                                "encoder": encoder_state[0][i][j:j+1],
                                "encoder_weight": encoder_state[1][i][j:j+1]
                            }
                            encoder_output_list.append(encoder_output_item)
                            # if  np.shape(encoder_output_item['encoder'])[1] != decoder_input_list[i]["source_length"]
                    #for input, encoder_output in zip(decoder_input_list, encoder_output_list):

                    message = "Finish encoding sentences: %d" % len(decoder_input_list)
                    tf.logging.log(tf.logging.INFO, message)
                except tf.errors.OutOfRangeError:
                    break

        # vocab = params.vocabulary["source"]
        # for decoder_input, encoder_output in zip(decoder_input_list, encoder_output_list):
        #     #print(decoder_input["source_length"][0], np.shape(encoder_output['encoder'])[1])
        #     sen = []
        #     for idx in decoder_input["source"][0]:
        #         if idx == params.mapping["source"][params.eos]:
        #             break
        #         sen.append(vocab[idx])
        #     s1 = " ".join(sen)
        #     print(s1)


        # print(encoder_result.shape)
        # for i in range(encoder_result.shape[0]):
        #     print('[')
        #     for j in range(encoder_result.shape[1]):
        #         print('[')
        #         for k in range(encoder_result.shape[2]):
        #             print("%f" % encoder_result[i][j][k])
        #         print(']')
        #     print(']')

        with open(args.output, "w") as outfile:
            cPickle.dump(sorted_keys, outfile)
            cPickle.dump(decoder_input_list, outfile)
            cPickle.dump(encoder_output_list, outfile)
            # count = 0
            # for input, encoder_output in zip(decoder_input_list, encoder_output_list):
            #     outfile.write("%d\n" % count)
            #     cPickle.dump(input, outfile)
            #     cPickle.dump(encoder_output, outfile)
            #     count += 1

if __name__ == "__main__":
    main(parse_args())
