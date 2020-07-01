# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import tensorflow as tf

from collections import namedtuple
from tensorflow.python.util import nest


class BeamSearchState(namedtuple("BeamSearchState",
                                 ("inputs", "state", "finish"))):
    pass


def _get_inference_fn(model_fns, features):
    def inference_fn(inputs, state):
        local_features = {
            "source": features["source"],
            "source_length": features["source_length"],
            #"constraints": features["constraints"],
            # [bos_id, ...] => [..., 0]
            "target": tf.pad(inputs[:, 1:], [[0, 0], [0, 1]]),
            "target_length": tf.fill([tf.shape(inputs)[0]],
                                     tf.shape(inputs)[1])
        }

        outputs = []
        next_state = []

        for (model_fn, model_state) in zip(model_fns, state):
            if model_state:
                output, new_state = model_fn(local_features, model_state)
                outputs.append(output)
                next_state.append(new_state)
            else:
                output = model_fn(local_features)
                outputs.append(output)
                next_state.append({})

        # Ensemble
        log_prob = tf.add_n(outputs) / float(len(outputs))

        return log_prob, next_state

    return inference_fn


def _infer_shape(x):
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.shape.dims is None:
        return tf.shape(x)

    static_shape = x.shape.as_list()
    dynamic_shape = tf.shape(x)

    ret = []
    for i in range(len(static_shape)):
        dim = static_shape[i]
        if dim is None:
            dim = dynamic_shape[i]
        ret.append(dim)

    return ret


def _infer_shape_invariants(tensor):
    shape = tensor.shape.as_list()
    for i in range(1, len(shape) - 1):
        shape[i] = None
    return tf.TensorShape(shape)


def _merge_first_two_dims(tensor):
    shape = _infer_shape(tensor)
    shape[0] *= shape[1]
    shape.pop(1)
    return tf.reshape(tensor, shape)


def _merge_first_three_dims(tensor):
    shape = _infer_shape(tensor)
    shape[0] *= shape[1] * shape[2]
    shape.pop(2)
    shape.pop(1)
    return tf.reshape(tensor, shape)


def _merge_beam_cons(tensor):
    shape = _infer_shape(tensor)
    shape[2] *= shape[3]
    shape.pop(3)
    return tf.reshape(tensor, shape)


def _split_first_two_dims(tensor, dim_0, dim_1):
    shape = _infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _split_first_three_dims(tensor, dim_0, dim_1, dim_2):
    shape = _infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + [dim_2] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _split_first_four_dims(tensor, dim_0, dim_1, dim_2, dim_3):
    shape = _infer_shape(tensor)
    new_shape = [dim_0] + [dim_1] + [dim_2] + [dim_3] + shape[1:]
    return tf.reshape(tensor, new_shape)


def _tile_to_beam_size(tensor, beam_size):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = beam_size

    return tf.tile(tensor, tile_dims)


def _tile_to_constraints_size(tensor, cons_size):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=1)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[1] = cons_size

    return tf.tile(tensor, tile_dims)


def _tile_to_constraints_num(tensor, cons_num):
    """Tiles a given tensor by beam_size. """
    tensor = tf.expand_dims(tensor, axis=3)
    tile_dims = [1] * tensor.shape.ndims
    tile_dims[3] = cons_num

    return tf.tile(tensor, tile_dims)


def _gather_2d(params, indices, name=None):
    """ Gather the 2nd dimension given indices
    :param params: A tensor with shape [batch_size, M, ...]
    :param indices: A tensor with shape [batch_size, N]  每一行的N个值代表在 params中该行的位置，存的不是slice，所以需要转换为slice再提取
    :return: A tensor with shape [batch_size, N, ...]
    """
    batch_size = tf.shape(params)[0]
    range_size = tf.shape(indices)[1]
    batch_pos = tf.range(batch_size * range_size) // range_size  # get 00001111222233334444 ...
    batch_pos = tf.reshape(batch_pos, [batch_size, range_size])  # get 0000 1111 2222 3333 4444 ....
    indices = tf.stack([batch_pos, indices], axis=-1)  # add the specific position to a new dimension to indicate the ..
    output = tf.gather_nd(params, indices, name=name)

    return output


def _gather_3d(params, indices, name=None):
    """ Gather the 3nd dimension given indices
    :param params: A tensor with shape [batch_size, G, M, ...]
    :param indices: A tensor with shape [batch_size, G, N]
    :return: A tensor with shape [batch_size, G,  N, ...]
    """
    batch_size = tf.shape(params)[0]
    grid_size = tf.shape(params)[1]

    # range_size = tf.shape(indices)[2]

    params = _merge_first_two_dims(params)
    indices = _merge_first_two_dims(indices)

    output = _gather_2d(params, indices, name)
    output = _split_first_two_dims(output, batch_size, grid_size)

    #
    # #batch_size * grid_size * range_size 先算出有多少个位置  除grid_size * range_size得到处于那一层
    # batch_pos = tf.range(batch_size * grid_size * range_size) // (grid_size * range_size)
    # grid_pos = tf.range(batch_size * grid_size * range_size) %  (grid_size * range_size)
    # grid_pos = grid_pos // range_size
    # batch_pos = tf.reshape(batch_pos, [batch_size, grid_size, range_size])
    # grid_pos = tf.reshape(grid_pos, [batch_size, grid_size, range_size])
    #
    #
    # batch_grid_pos = tf.stack([batch_pos, grid_pos], axis=-1)
    #
    # indices = tf.expand_dims(indices, -1)
    # indices = tf.concat([batch_grid_pos,indices], axis=-1)
    #
    # output = tf.gather_nd(params, indices, name=name)

    return output


def _beam_search_step(time, func, state, batch_size, beam_size, alpha,
                      pad_id, eos_id, cons_ids, cons_len, features, params, r):
    cons_num_per_source = tf.shape(cons_len)[1]
    # cons_all_words_num+1 will be the grid number
    cons_all_words_num = tf.reduce_max(tf.reduce_sum(cons_len, 1))  # cons_len:[[1,3,5],[2,0,4]...]

    # Compute log probabilities
    # shape: batch x (cons_all_words_num+1) x beam x ...
    seqs, log_probs = state.inputs[:2]
    # merge first three dims because of cons_num
    # batch*(cons_all_words_num+1)*beam x ...
    flat_seqs = _merge_first_three_dims(seqs)

    # flat it to prepare produce next word and probabilities of the seqs
    flat_state = nest.map_structure(lambda x: _merge_first_three_dims(x),
                                    state.state)
    # log_prob, {"encoder": encoder_output, "decoder": decoder_state}
    step_log_probs, next_state = func(flat_seqs, flat_state)


    grid_height =cons_all_words_num + 1

    #############################################
    # N*G*B x Num_head x Source Length
    step_encdec_att_weight = next_state[0]["decoder"]["layer_%d_encdec_output" % params.encdec_att_layer]["weight"]
    # only select certain heads
    step_encdec_att_weight = tf.gather(step_encdec_att_weight, params.heads, axis=1)  # 选择部分 head进行实验

    if params.head_model == 'average':
        # 将所有的heads的weight合并为一行，然后求平均。后面的代码继续使用，后面的代码以为head是一个
        step_encdec_att_weight = tf.reduce_mean(step_encdec_att_weight, 1, keep_dims=True)  # sum all heads weight

    step_encdec_att_weight = step_encdec_att_weight[:, :, :-1]  # remove the weight of eos
    step_encdec_att_weight_sum = tf.reduce_sum(step_encdec_att_weight, -1, keep_dims=True)
    step_encdec_att_weight = tf.div(step_encdec_att_weight, step_encdec_att_weight_sum)  # recalculate the distribution

    weight_shape = tf.shape(step_encdec_att_weight)
    padded_source_len = weight_shape[2]
    # will be N G B 1 Head source-Length
    step_encdec_att_weight = tf.reshape(step_encdec_att_weight, [batch_size, grid_height, beam_size, 1, weight_shape[1], weight_shape[2]])


    # extend to N G B C H L 为每个N*G*B中的每个c准备一份
    step_encdec_att_weight = tf.tile(step_encdec_att_weight, [1, 1, 1, cons_num_per_source, 1, 1])

    src_len_range = tf.range(padded_source_len, dtype=tf.int32)

    # will be N G B C H padded_source_Len
    src_len_range = tf.tile(
        tf.reshape(src_len_range,[1, 1, 1, 1, 1, padded_source_len]),
        [batch_size, grid_height, beam_size, cons_num_per_source, weight_shape[1], 1])  # weight_shape[1] is head number

    # 计算attention weight 在每个约束上的值之和
    # 可以通过调整src_pos0 1的位置，实现位置的扩展或偏移
    src_pos0 = features["constraints_src_pos"][:, :, 0]  # N C  # features["constraints_src_pos"] is N C 2
    src_pos0 = tf.reshape(src_pos0, [batch_size, 1, 1, cons_num_per_source, 1, 1])
    # will be N G B C H 1   每个N中的cons的src端的起始位置给每个G B H赋值一份
    src_pos0 = tf.tile(src_pos0, [1, grid_height, beam_size, 1, weight_shape[1], 1])

    src_pos1 = features["constraints_src_pos"][:, :, 1]  # N C
    src_pos1 = tf.reshape(src_pos1, [batch_size, 1, 1, cons_num_per_source, 1, 1])
    src_pos1 = tf.tile(src_pos1, [1, grid_height, beam_size, 1, weight_shape[1], 1])

    src_cons_mask0 = tf.greater_equal(src_len_range, src_pos0)
    src_cons_mask1 = tf.less(src_len_range, src_pos1)

    #src_cons_mask1 = tf.Print(src_cons_mask1, [features["constraints_src_pos"], src_cons_mask0, src_cons_mask1], "src_cons_mask0, src_cons_mask1", 1000, 10000)
    # will be N G B C H L
    src_cons_mask = tf.logical_and(src_cons_mask0, src_cons_mask1)
                                # N G B C H L                             N G B C H L
    masked_weight = tf.multiply(step_encdec_att_weight, tf.to_float(src_cons_mask))  #

    #masked_weight = tf.Print(masked_weight, [features["constraints_src_pos"], tf.shape(step_encdec_att_weight), step_encdec_att_weight, masked_weight], "step_encdec_att_weight, masked_weight ", 1000, 10000)

    cons_span_weight_sum = tf.reduce_sum(masked_weight, -1)  # 在L维求和；will be N G B C H(each C's sum of weight on each head)

    cons_span_weight_sum_mask = tf.greater_equal(cons_span_weight_sum, params.weight_threshold)  # N G B C H每句话中，每个C的source端attention weight得分是否大于给定值
    cons_span_weight_sum_mask = tf.reduce_sum(tf.to_int32(cons_span_weight_sum_mask), -1)  # add on head dimension; will be N G B C
    cons_span_weight_sum_mask = tf.cast(cons_span_weight_sum_mask, tf.bool)

    # cons_span_weight_sum_mask = tf.Print(cons_span_weight_sum_mask,
    #                                      [cons_span_weight_sum_mask], "cons_span_weight_sum_mask", 1000, 10000)

    # split first three dims because of cons_num
    # split to batch x (cons_all_words_num+1) x beam x vocab_size
    # as each beam-item radiates vacab_size posibile symbols
    step_log_probs = _split_first_three_dims(step_log_probs, batch_size,
                                             cons_all_words_num + 1, beam_size)
    next_state = nest.map_structure(
        lambda x: _split_first_three_dims(x, batch_size, cons_all_words_num + 1, beam_size), next_state)

    # expand_dims axis=3 because of cons_num
    # log_probs is N G B  step_log_probs is N*G*B*V
    curr_log_probs = tf.expand_dims(log_probs, 3) + step_log_probs

    # Apply length penalty
    length_penalty = tf.pow((5.0 + tf.to_float(time + 1)) / 6.0, alpha)
    curr_scores = curr_log_probs / length_penalty
    vocab_size = curr_scores.shape[-1].value or tf.shape(curr_scores)[-1]

    ######## Start GBS
    # before select top-k, we perform constrained decoding
    # batch x c+1 x beam x cons_num
    cons_active_pos = state.inputs[-1]

    # induce open flags
    # batch x 1 x 1 x cons_num ; as each grid each beam-item face the same constraints
    cons_len_expand = tf.expand_dims(cons_len, 1)  #每一条约束的长度
    cons_len_expand = tf.expand_dims(cons_len_expand, 2)

    open_flags = tf.logical_or(
        tf.equal(cons_active_pos, 0),  #
        tf.greater_equal(
            cons_active_pos,  # N*G*B*C
            cons_len_expand))   #
    # will be batch x G x beam ; when position flags of all n-cons of a beam-item are 1 in the open_flag
    open_flags = tf.reduce_all(open_flags, axis=-1)  # logical AND; open_flags is for a beam-item

    # generate mode: only allow open hyps
    # set closed hyp scores to -inf
    flat_curr_scores = _merge_first_three_dims(curr_scores)  # (N*G*B)*V

    gen_curr_scores = _split_first_three_dims(
        tf.where(
            # another possible way is do not flat open_flag but till the open_flag to V so can direct operate on curr_scores
            tf.reshape(open_flags, [-1]),  # reshape to (N*G*B)
            flat_curr_scores,
            tf.float32.min * tf.ones_like(flat_curr_scores)),
        batch_size, cons_all_words_num + 1, beam_size)

    # start mode: only allow open hyps and first ids in constraints
    # continue mode: only allow closed hyps and non-first ids in constraints

    # first according to open_flag set unvalid cons_active_pos to eos_id
    # valid: (open and cons_pos=0) or (close and cons_pos>0)
    tiled_open_flags = tf.tile(
        tf.expand_dims(open_flags, -1),
        [1, 1, 1, cons_num_per_source])


    start_cons_mask = tf.logical_and(  # find pos is 0 among all 0 or 0/eos
        tiled_open_flags,
        tf.equal(cons_active_pos, 0))

    # 利用attention再次决定是否启动start, 对后续的代码逻辑无影响
    # 计算 mask的比率
    ###########
    start_cons_mask_ft = tf.to_float(start_cons_mask)  # N G B C
    all_start = tf.reduce_sum(start_cons_mask_ft, [1, 2, 3])  # sum of all start
    ###########

    start_cons_mask_plus_src_cons = tf.logical_and(start_cons_mask, cons_span_weight_sum_mask)

    ###########
    start_cons_mask_plus_src_cons_int = tf.to_float(start_cons_mask_plus_src_cons)  # N G B C
    filtered_all_start = tf.reduce_sum(start_cons_mask_plus_src_cons_int, [1, 2, 3])  # sum of all start
    ###########
    # mask率是保留下来的start和原来要开始是start之比，越低越好
    # 防止除0 ； 约束为0个的，本质不需要mask，设置mask率为1，
    all_start_0flag = tf.equal(all_start, 0)
    all_start_N0_flag = tf.greater(all_start, 0)
    all_start = tf.add(all_start, tf.to_float(all_start_0flag))
    all_one = tf.ones(tf.shape(all_start))

    ratio = tf.div(filtered_all_start, all_start)
    ratio = tf.where(all_start_N0_flag, ratio, all_one)  # 初始 all start 为0的地方，直接赋值为1，其他用计算出来的ratio
    r = tf.concat([r, tf.reshape(ratio, [batch_size, 1])], 1)

    ###########
    # 约束长度为0的，虽然每次都要在grid最下层start，而 mask却作用到所有1，导致mask率为0,这里要恢复mask率为1
    # 在统计的时候，此类句子是否要计入？
    # start_cons_mask_plus_src_cons = tf.Print(start_cons_mask_plus_src_cons, [cons_all_words_num, ratio, all_start,
    #                                                                          tf.shape(start_cons_mask_plus_src_cons),
    #                                                                          tf.shape(start_cons_mask),
    #                                                                          start_cons_mask_plus_src_cons,
    #                                                                          start_cons_mask],
    #                                          "cons_all_words_num ratio, shape(start_cons_mask_plus_src_cons), shape(start_cons_mask)", 10000, 10000)

    # N x G x B x C
    # maybe include already-processed constraints
    valid_cons = tf.logical_or(
        start_cons_mask_plus_src_cons,
        tf.logical_and(  # find pos is no 0/eos
            tf.logical_not(tiled_open_flags),
            tf.logical_and(
                tf.greater(cons_active_pos, 0),
                tf.less(cons_active_pos, cons_len_expand))))

    cons_item_ending_pos = tf.expand_dims(cons_len, 1)
    cons_item_ending_pos = tf.expand_dims(cons_item_ending_pos, 1)
    # only keep valid pos, invalid pos directly point to eos
    # to eradicate its effects; con_valid_pos is a temporary variable
    # con_valid_pos will be batch x Grid size x beam x cons_num
    cons_valid_pos = tf.where(
        valid_cons, cons_active_pos,
        tf.zeros_like(cons_active_pos) + cons_item_ending_pos)

    # then gather cons ids for each beam items
    # shape: batch x (cons_all_words_num+1) x beam x cons_num
    # first extend to N*G*B*C*CL

    # cons_ids is N*C*CL
    # cons_ids_extended will be batch x G x beam x cons_num x cons_len;
    cons_ids_extended = tf.reshape(cons_ids, [batch_size, 1, 1, cons_num_per_source, -1])
    # to distribute C*CL to each grid and each beam-item
    cons_ids_extended = tf.tile(cons_ids_extended, [1, cons_all_words_num + 1, beam_size, 1, 1])  # now is N G B C CL
    # merge first 4 dimsension
    cons_ids_extended = _merge_first_three_dims(cons_ids_extended)
    cons_ids_extended = _merge_first_two_dims(cons_ids_extended)  # will be N*G*B*C CL

    # extend to N*G*B*C*1
    cons_valid_pos_extended = tf.expand_dims(cons_valid_pos, -1)
    cons_valid_pos_extended = _merge_first_three_dims(cons_valid_pos_extended)
    cons_valid_pos_extended = _merge_first_two_dims(cons_valid_pos_extended)  # will be N*G*B*C 1

    # batch x c+1 x beam x cons_num x 1
    # N x G x B X C x 1
    active_cons_ids = _split_first_four_dims(
        _gather_2d(
            cons_ids_extended,
            cons_valid_pos_extended
        ),
        batch_size, cons_all_words_num + 1, beam_size, cons_num_per_source)
    # remove the last demension as it's 1; become N*G*B*C
    active_cons_ids = tf.reshape(active_cons_ids, [batch_size, cons_all_words_num + 1, beam_size, cons_num_per_source])

    # create one-hot mask, shape: batch x (cons_all_words_num+1) x beam x cons_num x vocab_size
    # =1 means valid cons ids
    active_cons_masks = tf.one_hot(active_cons_ids, vocab_size, dtype=tf.int32)

    # all </s> (eos_id) and <pad> (pad_id) are invalid as well
    not_end_mask = tf.cast(
        tf.logical_and(
            tf.not_equal(active_cons_ids,
                         tf.zeros_like(cons_valid_pos) + eos_id),
            tf.not_equal(active_cons_ids,
                         tf.zeros_like(cons_valid_pos) + pad_id)),
        dtype=tf.int32)

    # shape: batch x (cons_all_words_num+1) x beam x vocab_size
    # TODO not work when same beginning words in constraints
    # active_cons_masks = tf.reduce_sum(
    #     active_cons_masks * tf.expand_dims(not_end_mask, -1),
    #     axis=-2)

    # active_cons_masks is N x G x B x C x V(V is one-hot,means to select one word form each cons );
    # not_end_mask is N*G*B*C
    # this operate will eradicate the effects of EOS
    active_cons_masks = active_cons_masks * \
                        tf.expand_dims(not_end_mask, -1)

    # set scores with non-cons ids to -inf
    # N x G x B x C x V
    # to assign each cons all the normal probabilities of V
    curr_scores_cons = tf.tile(
        tf.expand_dims(curr_scores, 3),  # curr_scores  N x G x B x V
        [1, 1, 1, cons_num_per_source, 1])  # N x G x B x C x V
    # only keep those probabilities that is valid among each V
    cons_curr_scores = tf.where(
        tf.greater(active_cons_masks, 0),
        curr_scores_cons,
        tf.float32.min * tf.ones_like(curr_scores_cons))
    # will be N G B*C V; this will be concatenated to grid_N+1's G mode scores
    cons_curr_scores = tf.reshape(
        cons_curr_scores,
        [batch_size, cons_all_words_num + 1,
         beam_size * cons_num_per_source, -1])

    # before concat, shift cons column by 1 then padding
    # because one more constraint word is added
    def _shift_and_pad(tensor, pad_value):
        # tensor shape: batch x (cons_all_words_num+1) x beam x vocab_size
        return tf.concat(
            [tf.ones_like(tensor[:, :1]) * pad_value, tensor[:, :-1]],
            axis=1)

    cons_curr_scores = _shift_and_pad(
        cons_curr_scores, tf.float32.min)

    # shape: batch x (cons_all_words_num+1) x (C+1)*beam x vocab_size
    curr_scores = tf.concat(
        [gen_curr_scores,
         cons_curr_scores],
        axis=-2)

    def _tile_cons_reshape(tensor):
        tensor = _tile_to_constraints_num(
            tensor, cons_num_per_source)
        return _merge_beam_cons(tensor)

    # do the same to seqs
    # at beginning , each beam-item has a seq, now tile the seq to cons_num, this is corresponding to each cons
    seqs_cons = _tile_cons_reshape(seqs)  # after this, seqs_cons should be N G B*C L
    seqs_shifted = _shift_and_pad(seqs_cons, eos_id)
    seqs = tf.concat(
        [seqs,
         seqs_shifted],
        axis=-2)  # after this, the seqes should be N G B*(1+C) L(seqs' length)

    def concat_shift_tile_and_pad(x):
        x_grid0 = x[:, :1]  # keep the lowest value as pad
        x_shifted = tf.concat([x_grid0, x[:, :-1]], 1)  # concat on batch-size dimension
        x_shifted = _tile_cons_reshape(x_shifted)
        return tf.concat([x, x_shifted], 2)

    # change the next_state from N G B to N G B 1+C so that next step can choice arbitrarily
    next_state = nest.map_structure(concat_shift_tile_and_pad,
                                    next_state)
    ######## end GBS

    # Select top-k candidates
    # *2 because of GBS
    # [batch_size, cons_all_words_num+1, beam_size * vocab_size * 2]
    curr_scores = tf.reshape(curr_scores,
                             [batch_size, cons_all_words_num + 1,
                              beam_size * vocab_size * (1 + cons_num_per_source)])
    # [batch_size, cons_all_words_num+1, 2 * beam_size]
    top_scores, top_indices = tf.nn.top_k(curr_scores,
                                          k=2 * beam_size)  # top_indices is  N G 2B ; find 2B sub-seqs for each G
    # Shape: [batch_size, cons_all_words_num+1, 2 * beam_size]
    beam_indices = top_indices // vocab_size  # N G 2B
    symbol_indices = top_indices % vocab_size  # N G 2B

    # Expand sequences
    # [batch_size, cons_all_words_num+1, 2 * beam_size, time]
    # reshape to [batch_size*(cons_all_words_num+1), 2*beam_size, ...]
    # then call _gather_2d, then reshape back
    candidate_seqs = _gather_2d(
        _merge_first_two_dims(seqs),  # before merge seqs is  N G B*(1+C) L(seqs' length)
        _merge_first_two_dims(beam_indices))  # before merge is N G 2B

    candidate_seqs = _split_first_two_dims(
        candidate_seqs, batch_size, cons_all_words_num + 1)

    candidate_seqs = tf.concat([candidate_seqs,
                                tf.expand_dims(symbol_indices, 3)], 3)

    # Expand sequences
    # Suppress finished sequences  flag is N G 2B  representing each Grid's potential symbol
    flags = tf.equal(symbol_indices, eos_id)
    # [batch, G, 2 * beam_size]
    alive_scores = top_scores + tf.to_float(flags) * tf.float32.min
    # [batch, G, beam_size] ; to select top k from alive_scores--[batch, G, 2*beam_size]
    alive_scores, alive_indices = tf.nn.top_k(alive_scores,
                                              beam_size)  # alive-indices is indices in batch, G, 2*beam_size

    # N x G x B           ;N G 2B          N G B
    alive_symbols = _gather_3d(symbol_indices, alive_indices)
    alive_indices = _gather_3d(beam_indices, alive_indices)  # find the actual indices in N G B*(1+C); alive_indices will be N G B
    alive_seqs = _gather_3d(seqs, alive_indices)  # find the seqs from the N G B*(1+C) L

    # [batch_size, grid, beam_size, time + 1]
    alive_seqs = tf.concat([alive_seqs, tf.expand_dims(alive_symbols, -1)], -1)
    alive_state = nest.map_structure(lambda x: _gather_3d(x, alive_indices),
                                     next_state)

    alive_log_probs = alive_scores * length_penalty

    # update cons_active_pos
    # in generate mode, cons_active_pos keep the same
    # in start and continue mode, the selected cons_active_pos + 1
    # we need to shift cons_active_pos at first
    cons_active_pos_shiftpad = _shift_and_pad(
        cons_active_pos, tf.reduce_max(cons_len))  # con_active_pos is N G B C
    cons_active_pos_shiftpad = _tile_cons_reshape(
        # after this, cons_active_pos_shiftpad is N G B*C C  # ensure each of B*C extended beam-item have cons pos information
        cons_active_pos_shiftpad)

    # we need to first select alive cons pos
    # N x G x (C+1)B x C
    cons_active_pos_concat = tf.concat(
        [cons_active_pos, cons_active_pos_shiftpad],  # cons_active_pos只是起到占位的作用，由于处于前beam-size个候选分支，肯定不能被选上
        axis=2)
    # 挑选出与B个优选分支相对应的cons_active_pos
    cons_active_pos_shiftpad = _gather_3d(
        cons_active_pos_concat,
        alive_indices)  # cons_active_pos_shiftpad will be N G B C ; last two dimension is each beam's original cons pos



    # extend to N G B C 1
    cons_active_pos_shiftpad_extended = tf.expand_dims(cons_active_pos_shiftpad, -1)
    cons_active_pos_shiftpad_extended = _merge_first_three_dims(cons_active_pos_shiftpad_extended)
    cons_active_pos_shiftpad_extended = _merge_first_two_dims(cons_active_pos_shiftpad_extended)  # will be N*G*B*C 1

    # 优选的每个beam-item对应的所有cons 的active symbols' ids
    active_cons_ids = _split_first_four_dims(
        _gather_2d(
            cons_ids_extended,  # is  N*G*B*C CL
            cons_active_pos_shiftpad_extended  # is N*G*B*C 1
        ),
        batch_size, cons_all_words_num + 1, beam_size, cons_num_per_source)  # N G B C 1 each Cons's current ids

    # remove last demension as it's 1; selected each sub-beam-item's potential ids
    # N G B C  优选的每个beam-item对应的所有cons 的active symbols' ids
    active_cons_ids = tf.reshape(
        active_cons_ids,
        [batch_size, cons_all_words_num + 1,
         beam_size, cons_num_per_source])

    # N x G x B x C; alive_indices is N G B(here each value representing the order in (c+1)*B)
    # 前面的操作，已经选择出了 优选的B个分支对应的C个 cons的activate pos ids

    # 已经为每个Grid选择了B个新的符号alive_symbols ，这B个新符号，代表了新的B个beam-item
    # 这B个新的符号，来自于C个的cons的各自activate位置或者Gen模式中的symbols，
    # 这里判断，新选择的B个符号(alive_symbols)，到底与那个cons中的activate pos ids相等
    # 当两个cons的第一个词汇相同，且新选择了这两个词汇为新分支，此时是什么情况？ alive_symbols [a a] active_cons_ids [[a a c] [a a c]]
    # 通过 valid_cons 来mask
    cons_selected = tf.logical_and(
        tf.expand_dims(tf.greater_equal(alive_indices, beam_size), -1),  # order greater than B is cons activate symbols
        tf.equal(tf.expand_dims(alive_symbols, -1),  # now alive_symbols is N x G x B ; the selected B symbols will compare with each next activate symbols in all C cons
                 active_cons_ids))  # now active_cons_ids is N G B C(number of next position symbols is C)


    # N G B C 1
    def _shift(x):
        x_grid0 = x[:, :1]  # keep the lowest value as pad
        x_shifted = tf.concat([x_grid0, x[:, :-1]], 1)  # concat on grid dimension
        return x_shifted

    # N x G x B x C x 1  ; valid_cons 代表每个beam-item对应的C个约束的 valid 状态; valid 本身就是N G B C
    valid_cons_shifted = tf.expand_dims(
        tf.cast(_shift(valid_cons), tf.int32), -1)
    # N x G x B x C x C; BxC个分支，每个C个约束的valid_cons位
    valid_cons_full = tf.tile(
        tf.reshape(
            tf.one_hot(tf.range(cons_num_per_source), cons_num_per_source, dtype=tf.int32),
            [1, 1, 1, cons_num_per_source, cons_num_per_source]),
        [batch_size, cons_all_words_num + 1, beam_size, 1, 1])
    # N x G x B x C x C
    # [1 0 1]'  multiply
    # [1 0 0
    #  0 1 0
    #  0 0 1
    #  ]  相乘的结果是，
    valid_cons_shifted = tf.multiply(
        valid_cons_shifted,
        valid_cons_full)

    # N x G x BC x C
    valid_cons_shifted = tf.cast(
        _merge_beam_cons(valid_cons_shifted), tf.bool)
    # valid_cons = _merge_beam_cons(
    #     tf.expand_dims(valid_cons, -1))
    # will be N G B(优选出来的B个) C(one-hot having been masked)
    valid_cons = _gather_3d(
        tf.concat([valid_cons, valid_cons_shifted], axis=2),
        alive_indices)
    # will be N G B C
    cons_selected = tf.logical_and(
        cons_selected, valid_cons)

    # assert: only at most one cons word selected
    assert_op = tf.Assert(
        tf.reduce_all(tf.less_equal(
            tf.reduce_sum(tf.cast(cons_selected, tf.int32), axis=-1),
            1)),
        [cons_selected])

    with tf.control_dependencies([assert_op]):
        # 调整pos
        # cons_active_pos_shiftpad is N G B C
        new_cons_active_pos = tf.minimum(  # 个数不一样，顶不一样，限制batchsize=1走通
            cons_active_pos_shiftpad + tf.cast(cons_selected, dtype=tf.int32),
            cons_len_expand)

    # 对于没有选中的pos，保持pos不变 ﻿
    # cons_selected 这个标志是用来调整位置的。只要有一个约束中的词汇被采纳了， 位置调整了，则整个beam-item的pos要保留调整后的新位置，而不能按每个约束单独操作

    # cons_selected_flag= tf.reduce_any(cons_selected,-1, True)
    # cons_selected_flag = tf.tile(cons_selected_flag,[1,1,1,cons_num_per_source])
    #
    # #对于仅仅G模式的（没有接受任何新约束的），要保留其原来约束的pos，而不是下一个grid来的pos的位置
    # new_cons_active_pos= tf.where(cons_selected_flag, new_cons_active_pos, cons_active_pos)

    # 上面的where确保了其最下grid的pos均为0
    # change the lowest grid's pos to 0 to ensure
    # new_cons_active_pos = tf.concat([tf.zeros_like(new_cons_active_pos[:,0:1]), new_cons_active_pos[:,1:]], 1)

    # end update cons_active_pos

    # Select finished sequences
    prev_fin_flags, prev_fin_seqs, prev_fin_scores = state.finish

    # only use seqs which contain all constraints
    candidate_seqs = candidate_seqs[:, -1]
    flags = flags[:, -1]

    # some flag is not real end flag as the seqs containing no any constraints
    flags_mask = tf.greater_equal(time, cons_all_words_num)

    flags = tf.where(flags_mask, flags, tf.zeros_like(flags))

    top_scores = top_scores[:, -1]

    # [batch, 2 * beam_size]
    step_fin_scores = top_scores + (1.0 - tf.to_float(flags)) * tf.float32.min
    # [batch, 3 * beam_size]
    fin_flags = tf.concat([prev_fin_flags, flags], axis=1)
    fin_scores = tf.concat([prev_fin_scores, step_fin_scores], axis=1)

    # [batch, beam_size]
    fin_scores, fin_indices = tf.nn.top_k(fin_scores, beam_size)
    fin_flags = _gather_2d(fin_flags, fin_indices)

    pad_seqs = tf.fill([batch_size, beam_size, 1],
                       tf.constant(pad_id, tf.int32))
    prev_fin_seqs = tf.concat([prev_fin_seqs, pad_seqs], axis=2)
    fin_seqs = tf.concat([prev_fin_seqs, candidate_seqs], axis=1)
    fin_seqs = _gather_2d(fin_seqs, fin_indices)

    # add new constraint-related tensors to state.inputs
    new_state = BeamSearchState(
        inputs=(alive_seqs, alive_log_probs, alive_scores, new_cons_active_pos),
        state=alive_state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    return time + 1, new_state, r


def beam_search(func, state, batch_size, beam_size, max_length, alpha,
                pad_id, bos_id, eos_id, constraints, constraints_len, features, params):
    # TODO: replace None with real tensors or nums
    # the number of constrained words
    # 限制的数量，不是所有限制的总词汇
    cons_shape = tf.shape(constraints)

    cons_num = cons_shape[1]  # None
    # todo should get the max number
    cons_all_words_num = tf.reduce_max(tf.reduce_sum(constraints_len, -1))
    # constraint lex ids
    # shape: batch x cons_num x Length
    # the last column is eos_id
    # batch中每一句话的每一个限制item的所有词的ID，后面以eos_id补齐
    cons_ids = constraints
    # shape: batch x cons_num
    # 每一句话的每一个限制的长度
    cons_len = constraints_len

    # cons 是二维数组   第几个词

    # store cons pos for current step
    # shape: batch x cons_num+1 x beam_size x cons_num
    # 每个beam_item里面要保存下一步可以走每一条constraint的哪个字符
    # initialize to 0
    cons_active_pos = tf.fill([batch_size, cons_all_words_num + 1, beam_size, cons_num], 0)  # None
    # Open表示，可以随便生成  close表示，只能在constraints的范围内生成
    # open/close flag, open=1, close=0
    # shape: batch x cons_num+1 x beam
    # open_flags = None

    # init_seqs shape: batch x cons_num x beam x 1
    init_seqs = tf.fill([batch_size, cons_all_words_num + 1, beam_size, 1], bos_id)
    # init_log_probs shape: batch x cons_num x beam
    # [:, :, 0] = 0, others -inf

    init_log_probs00 = tf.constant([[[0.] + [tf.float32.min] * (
            beam_size - 1)]])  # init_log_probs = tf.constant([[0.] + [tf.float32.min] * (beam_size - 1)])
    init_log_probs00 = tf.tile(init_log_probs00, [batch_size, 1, 1])
    # extend to constraints
    init_log_probs = tf.constant([[tf.float32.min] * beam_size])
    init_log_probs = tf.expand_dims(init_log_probs, 0)
    init_log_probs = tf.tile(init_log_probs, [batch_size, cons_all_words_num, 1])
    init_log_probs = tf.concat([init_log_probs00, init_log_probs], 1)

    init_scores = tf.zeros_like(init_log_probs)
    # keep unchanged
    fin_seqs = tf.zeros([batch_size, beam_size, 1], tf.int32)
    fin_scores = tf.fill([batch_size, beam_size], tf.float32.min)
    fin_flags = tf.zeros([batch_size, beam_size], tf.bool)

    # extend the state to batch*cons+1*beams
    state = nest.map_structure(lambda x: _tile_to_constraints_size(x, cons_all_words_num + 1),
                               state)

    # cons_ids, cons_masks, open_flags as inputs
    state = BeamSearchState(
        inputs=(init_seqs, init_log_probs, init_scores, cons_active_pos),
        state=state,
        finish=(fin_flags, fin_seqs, fin_scores),
    )

    max_step = tf.reduce_max(max_length)

    def _is_finished(t, s, r):
        log_probs = s.inputs[1]
        finished_flags = s.finish[0]
        finished_scores = s.finish[2]
        max_lp = tf.pow(((5.0 + tf.to_float(max_step)) / 6.0), alpha)
        # across last 2 cons_num
        # TODO: check if cause erros when no constaints
        best_alive_score = log_probs[:, -2:] / max_lp
        # take the best
        best_alive_score = tf.reduce_max(best_alive_score, axis=1)

        worst_finished_score = tf.reduce_min(
            finished_scores * tf.to_float(finished_flags), axis=1)

        add_mask = 1.0 - tf.to_float(tf.reduce_any(finished_flags, 1))
        worst_finished_score += tf.float32.min * add_mask
        worst_finished_score = tf.expand_dims(worst_finished_score, -1)

        #worst_finished_score = tf.Print(worst_finished_score, [worst_finished_score], "worst_finished_score")

        bound_is_met = tf.reduce_all(tf.greater(worst_finished_score,
                                                best_alive_score))

        cond = tf.logical_and(tf.less(t, max_step),
                              tf.logical_not(bound_is_met))

        return cond

    def _loop_fn(t, s, r):
        outs = _beam_search_step(t, func, s, batch_size, beam_size, alpha,
                                 pad_id, eos_id, cons_ids, cons_len, features, params, r)  # directly add real features to inner function
        return outs

    time = tf.constant(0, name="time")
    mask_ratio = tf.zeros([batch_size, 0])

    # TODO: add shapes for cons_active_pos
    shape_invariants = BeamSearchState(
        inputs=(tf.TensorShape([None, None, None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None, None, None])),
        state=nest.map_structure(_infer_shape_invariants, state.state),
        finish=(tf.TensorShape([None, None]),
                tf.TensorShape([None, None, None]),
                tf.TensorShape([None, None]))
    )
    outputs = tf.while_loop(_is_finished, _loop_fn, [time, state, mask_ratio],
                            shape_invariants=[tf.TensorShape([]),
                                              shape_invariants, tf.TensorShape([None, None])],
                            parallel_iterations=1,
                            back_prop=False)

    final_state = outputs[1]
    alive_seqs = final_state.inputs[0]
    alive_scores = final_state.inputs[2]
    final_flags = final_state.finish[0]
    final_seqs = final_state.finish[1]
    final_scores = final_state.finish[2]

    alive_seqs.set_shape([None, None, beam_size, None])
    final_seqs.set_shape([None, beam_size, None])
    alive_seqs = alive_seqs[:, -1]
    final_seqs = tf.where(tf.reduce_any(final_flags, 1), final_seqs,
                          alive_seqs)
    alive_scores = alive_scores[:, -1]

    final_scores = tf.where(tf.reduce_any(final_flags, 1), final_scores,
                            alive_scores)
    #alive_state = final_state.state
    ratio = outputs[2]
    average = tf.reduce_mean(ratio,1)
    ratio = tf.concat([tf.expand_dims(average,-1), ratio],1)
    return final_seqs, final_scores, ratio  #, alive_state

def create_inference_graph(model_fns, features, params):
    if not isinstance(model_fns, (list, tuple)):
        raise ValueError("mode_fns must be a list or tuple")

    features = copy.copy(features)

    decode_length = params.decode_length
    beam_size = params.beam_size
    top_beams = params.top_beams
    alpha = params.decode_alpha

    # Compute initial state if necessary
    states = []
    funcs = []


    for model_fn in model_fns:
        if callable(model_fn):
            # For non-incremental decoding
            states.append({})
            funcs.append(model_fn)
        else:
            # For incremental decoding where model_fn is a tuple:
            # (encoding_fn, decoding_fn)
            #保留 encoder_weight以便输出
            encoder_states, encoder_weights = model_fn[0](features)
            states.append(encoder_states)
            funcs.append(model_fn[1])

    #states[0]["encoder"] = tf.Print(states[0]["encoder"], [tf.shape(states[0]["encoder"])], "states[0]-encoder", 100, 1000)

    batch_size = tf.shape(features["source"])[0]
    pad_id = params.mapping["target"][params.pad]
    bos_id = params.mapping["target"][params.bos]
    eos_id = params.mapping["target"][params.eos]

    # Expand the inputs in to the beam size
    # [batch, length] => [batch, beam_size, length]
    #  这里要考虑constraints的长度
    cons_all_words_num = tf.reduce_max(tf.reduce_sum(features['constraints_len'], -1))

    features["source"] = tf.expand_dims(features["source"], 1)
    features["source"] = tf.tile(features["source"], [1, beam_size * (cons_all_words_num + 1), 1])
    shape = tf.shape(features["source"])

    # [batch, beam_size, length] => [batch * beam_size, length]
    features["source"] = tf.reshape(features["source"],
                                    [shape[0] * shape[1], shape[2]])

    # For source sequence length
    features["source_length"] = tf.expand_dims(features["source_length"], 1)
    features["source_length"] = tf.tile(features["source_length"],
                                        [1, beam_size * (cons_all_words_num + 1)])

    shape = tf.shape(features["source_length"])

    max_length = features["source_length"] + decode_length

    # [batch, beam_size, length] => [batch * beam_size, length]
    features["source_length"] = tf.reshape(features["source_length"],
                                           [shape[0] * shape[1]])
    decoding_fn = _get_inference_fn(funcs, features)
    states = nest.map_structure(lambda x: _tile_to_beam_size(x, beam_size),
                                states)

    seqs, scores, ratio = beam_search(decoding_fn, states, batch_size, beam_size,
                               max_length, alpha, pad_id, bos_id, eos_id,
                               features["constraints"], features["constraints_len"], features, params)  #features参数有重复，但为了尽量保持一致

    return seqs[:, :top_beams, 1:], scores[:, :top_beams], ratio #, encoder_weights, alive_state
