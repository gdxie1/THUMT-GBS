#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import os
import cPickle
import numpy as np
import tensorflow as tf
import thumt.data.vocab as vocabulary
import src_cons_transformer as transformer
import thumt.utils.parallel as parallel

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

    control_symbols = [params.pad, params.bos, params.eos, params.unk,
                       '!', '\"', '&', '\\', ',', '(', ')', '*',
                       ',', '-', '.', '...', '/', ':', ';', '?',
                       '{', '|', '}', '~', '、', '。','—', '……',
                       '‘', '’ ', '“', '”', '〈', '〉', '《', '》',
                       '！', '（', '）', '，', '．', '：', '；']
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
        with open(args.input, "r") as encoded_file:
            sorted_keys = cPickle.load(encoded_file)
            decoder_input_list = cPickle.load(encoded_file)
            encoder_output_list = cPickle.load(encoded_file)

        state_placeholders = []
        for i in range(len(params.device_list)):
            decode_state = {
                "encoder": tf.placeholder(tf.float32, [None, None, params.hidden_size],
                                          "encoder_%d" % i),
                #"encoder_weight": we doesn't need encoder weight
                "source": tf.placeholder(tf.int32, [None, None], "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None], "source_length_%d" % i),
                # [bos_id, ...] => [..., 0]
                "target": tf.placeholder(tf.int32, [None, None], "target_%d" % i),
                #"target_length": tf.placeholder(tf.int32, [None, ], "target_length_%d" % i)
            }
            #需要这些值，以进行增量式解码
            for j in range(params.num_decoder_layers):
                decode_state["decoder_layer_%d_key" % j] = tf.placeholder(tf.float32, [None, None, params.hidden_size],
                                                   "decoder_layer_%d_key_%d" % (j,i))
                decode_state["decoder_layer_%d_value" % j] = tf.placeholder(tf.float32, [None, None, params.hidden_size],
                                        "decoder_layer_%d_value_%d" % (j,i))  # layer and GPU
                # we only need the return value of this
                # decode_state["decoder_layer_%d_att_weight" % j] = tf.placeholder(tf.float32, [None, None, None, None],
                #                              # N Head T S  inference的时候,T总是为1,表示1步
                #                              "decoder_layer_%d_att_weight" % j),
            state_placeholders.append(decode_state)


        def decoding_fn(s):
            _decoding_fn = model_fns[0][1]
            #split s to state and feature, and 转换为嵌套的结构，以满足transformer模型
            state = {
                "encoder": s["encoder"],
                "decoder": {
                    "layer_%d" % j: {
                        "key": s["decoder_layer_%d_key" % j],
                        "value": s["decoder_layer_%d_value" % j],
                    } for j in range(params.num_decoder_layers)
                }
            }
            inputs = s["target"]
            #inputs = tf.Print(inputs, [inputs], "before target", 100, 10000)
            feature = {
                "source": s["source"],
                "source_length": s["source_length"],
                # [bos_id, ...] => [..., 0]
                # "target": tf.pad(inputs[:,1:], [[0, 0], [0, 1]])
                #"target": tf.pad(inputs, [[0, 0], [0, 1]]),  # 前面没有bos_id，因此直接补上0，这是为了和decode_graph中的补bos相配合
                "target": inputs,
                "target_length": tf.fill([tf.shape(inputs)[0]], tf.shape(inputs)[1])
            }
            #feature["target"] = tf.Print(feature["target"], [feature["target"]], "target", 100,10000)
            ret = _decoding_fn(feature, state, params)
            return ret

        decoder_op = parallel.data_parallelism(
            params.device_list, lambda s: decoding_fn(s),
            state_placeholders)

        #batch = tf.shape(encoder_output)[0]

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
        sen_decode_time = []
        grid_hyps = []  #存放每个句子中每个grid中的hyps，以便后期分析和统计
        # Create session
        with tf.Session(config=session_config(params)) as sess:
            # from tensorflow.python import debug as tf_debug
            # sess = tf_debug.LocalCLIDebugWrapperSession(sess,ui_type='curses')#readline

            # Restore variables
            sess.run(assign_op)
            #startpoint=320
            for i, (decode_input, encoder_output) in enumerate(zip(decoder_input_list, encoder_output_list)):
                # if i < startpoint:
                #     continue

                # if i == startpoint:
                #     break
                # print(input["source"])
                # print(input["constraints"])
                #################
                # create constraint translation related model
                # build ensembled TM
                thumt_tm = ThumtTranslationModel(sess, decoder_op, encoder_output, state_placeholders,
                                                 decode_input, params)

                # Build GBS search
                cons_decoder = create_constrained_decoder(thumt_tm)
                ##################
                max_length = decode_input["source_length"][0] + params.decode_length
                beam_size = params.beam_size
                # top_beams = params.top_beams
                top_beams = 1
                start_time = time.time()
                best_output, search_grid = decode(encoder_output, sess, decoder_op, state_placeholders, params,
                                     cons_decoder,
                                     thumt_tm, decode_input, top_beams,
                                     max_hyp_len=max_length,
                                     beam_size=beam_size,
                                     return_alignments=True,
                                     length_norm=False
                                     )
                sen_decode_time.append(time.time() - start_time)
                hyps_num = {k:len(search_grid[k]) for k in search_grid.keys()}
                grid_hyps.append(hyps_num)

                # output_beams = [search_grid[k] for k in search_grid.keys() if k[1] == top_row]
                # output_hyps = [h for beam in output_beams for h in beam]

                # constraints=input_constraints,
                # return_alignments=return_alignments,
                # length_norm=length_norm)
                results.append(best_output)

                message = "Finished decoding sentences index: %d" % (i)
                tf.logging.log(tf.logging.INFO, message)

        # Convert to plain text
        vocab = params.vocabulary["target"]
        outputs = []
        scores = []
        mask_ratio = []
        best_alignment = []

        for result in results:
            sub_result = zip(*result[0])
            outputs.extend(sub_result[0])
            scores.extend(sub_result[1])
            best_alignment.extend(result[1])

            # for sub_result in result:  # 每次解码结果可能有多个bestscore
            #     outputs.append(sub_result[0][0][1:])  # seqs
            #     scores.append(sub_result[0][1])  # score
            #     mask_ratio.append(0)
            #     best_alignment.extend(sub_result[1])
        new_outputs = []
        for s in outputs:
            new_outputs.append(s[1:])
        outputs = new_outputs

        for s, score in zip(outputs, scores):
            s1 = []
            for idx in s:
                if idx == params.mapping["target"][params.eos]:
                    break
                s1.append(vocab[idx])
            s1 = " ".join(s1)
            #print("%s" % s1)
            print("%f   %s" % (score, s1))


        restored_inputs = []
        restored_outputs = []
        restored_scores = []
        restored_constraints = []
        restored_alignment = []
        restored_sen_decode_time = []
        restored_grid_hyps = []
        for index in range(len(sorted_keys)):
            restored_outputs.append(outputs[sorted_keys[index]])
            restored_scores.append(scores[sorted_keys[index]])
            #restored_constraints.append(sorted_constraints[sorted_keys[index]])
            restored_alignment.append(best_alignment[sorted_keys[index]])
            restored_sen_decode_time.append(sen_decode_time[sorted_keys[index]])
            restored_grid_hyps.append(grid_hyps[sorted_keys[index]])

        # restored_outputs = outputs
        # restored_scores = scores
        # restored_alignment = best_alignment
        # restored_sen_decode_time = sen_decode_time
        # restored_grid_hyps = grid_hyps

        # Write to file
        with open(args.output, "w") as outfile:
            count = 0
            for output, score, de_time in zip(restored_outputs, restored_scores, restored_sen_decode_time):
                decoded = []
                for idx in output:
                    if idx == params.mapping["target"][params.eos]:
                        break
                    decoded.append(vocab[idx])
                decoded = " ".join(decoded)

                if not args.verbose:
                    outfile.write("%s\n" % decoded)
                else:
                    pattern = "%d |%s |%f |%f \n"
                    # cons = restored_constraints[count]
                    # cons_token_num = 0
                    # for cons_item in cons:
                    #     cons_token_num += cons_item["tgt_len"]
                    values = (count, decoded, score, de_time)
                    outfile.write(pattern % values)
                count += 1

        with open(args.output+".alignment", "w") as outfile:
            count = 0
            for alignment in restored_alignment:
                outfile.write("%d\n" % count)
                cPickle.dump(alignment, outfile)
                count += 1
        #  保存解码时间和grid中的hyps，以便进行分析
        with open(args.output + ".time_hyps", "w") as outfile:
            cPickle.dump(restored_sen_decode_time, outfile)
            cPickle.dump(restored_grid_hyps, outfile)
        with open(args.output + ".time", "w") as outfile:
            time_sen = np.asarray(restored_sen_decode_time)
            ave = np.average(time_sen)
            outfile.write("average time:%f\n" % ave)
            cPickle.dump(restored_sen_decode_time, outfile)


def decode(encoder_output, sess, decoder_op, decoder_placeholder, params,
           cons_decoder,translation_model, inputs, n_best, max_hyp_len, beam_size=5,
           constraints=None,
           mert_nbest=False,
           return_alignments=False,
           length_norm=True):

    input_constraints = []
    input_cons_src_pos = []
    cons_lens = inputs["constraints_len"]
    for i, cons in enumerate(inputs["constraints"]):
        cons_len = cons_lens[i]
        cons = cons[0:cons_len]
        if cons_len:
            input_constraints.append(cons)
            input_cons_src_pos.append(inputs["constraints_src_pos"])

    # if constraints is not None:
    #     input_constraints = translation_model.map_constraints(constraints)
    inputs = [inputs]
    start_hyp = translation_model.start_hypothesis(inputs, input_constraints)

    # Note: the length_factor is used with the length of the first model input of the ensemble
    if params.hgbs:
        search_grid = cons_decoder.search_hgbs(inputs, encoder_output, sess, decoder_op, decoder_placeholder, params,start_hyp=start_hyp,
                                     constraints=input_constraints,
                                     max_hyp_len=max_hyp_len,
                                     beam_size=beam_size)
    else:
        search_grid = cons_decoder.search(inputs, encoder_output, sess, decoder_op, decoder_placeholder, params,
                                               start_hyp=start_hyp,
                                               constraints=input_constraints,
                                               max_hyp_len=max_hyp_len,
                                               beam_size=beam_size)

    # best_output, best_alignments = decoder.best_n(search_grid, translation_model.eos_id,
    #                                               n_best=n_best,
    #                                               return_model_scores=mert_nbest,
    #                                               return_alignments=return_alignments,
    #                                               length_normalization=length_norm)

    ret1 = cons_decoder.best_n(search_grid, translation_model.eos_id,
                               n_best=n_best,
                               return_model_scores=mert_nbest,
                               return_alignments=return_alignments,
                               length_normalization=length_norm,
                               prefer_eos=True)

    return ret1, search_grid  # 返回 grid以统计效率
    # best_output = ret1[0][1:]
    # #return best_output
    # if return_alignments:
    #      return best_output, best_alignments
    #  else:
    #      return best_output

if __name__ == "__main__":
    main(parse_args())
