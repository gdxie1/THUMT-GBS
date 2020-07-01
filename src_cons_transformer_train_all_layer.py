# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import tensorflow as tf
import thumt.interface as interface
import thumt.layers as layers


def _layer_process(x, mode):
    if not mode or mode == "none":
        return x
    elif mode == "layer_norm":
        return layers.nn.layer_norm(x)
    else:
        raise ValueError("Unknown mode %s" % mode)


def _residual_fn(x, y, keep_prob=None):
    if keep_prob and keep_prob < 1.0:
        y = tf.nn.dropout(y, keep_prob)
    return x + y


def _ffn_layer(inputs, hidden_size, output_size, keep_prob=None,
              dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="ffn_layer", values=[inputs],
                           dtype=dtype):
        with tf.variable_scope("input_layer"):
            hidden = layers.nn.linear(inputs, hidden_size, True, True)
            hidden = tf.nn.relu(hidden)

        if keep_prob and keep_prob < 1.0:
            hidden = tf.nn.dropout(hidden, keep_prob)

        with tf.variable_scope("output_layer"):
            output = layers.nn.linear(hidden, output_size, True, True)

        return output


def transformer_encoder(inputs, bias, params, dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="encoder", dtype=dtype,
                           values=[inputs, bias]):
        #inputs = tf.Print(inputs, [dtype], 'dtype')
        x = inputs
        for layer in range(params.num_encoder_layers):
            with tf.variable_scope("layer_%d" % layer):
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout
                    )
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        return outputs

def transformer_decoder(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None

                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )
                    y = y["outputs"]

                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state

        return outputs


def transformer_decoder_output_cur_encdec_atten(inputs, memory, bias, mem_bias, params, state=None,
                        dtype=None, scope=None):
    """
        output current step's encdec attention result
    """

    with tf.variable_scope(scope, default_name="decoder", dtype=dtype,
                           values=[inputs, memory, bias, mem_bias]):
        x = inputs
        next_state = {}
        att_weight=[]
        for layer in range(params.num_decoder_layers):
            layer_name = "layer_%d" % layer
            with tf.variable_scope(layer_name):
                layer_state = state[layer_name] if state is not None else None
                with tf.variable_scope("self_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        None,
                        bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                        state=layer_state
                    )

                    if layer_state is not None:
                        next_state[layer_name] = y["state"]

                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

                with tf.variable_scope("encdec_attention"):
                    y = layers.attention.multihead_attention(
                        _layer_process(x, params.layer_preprocess),
                        memory,
                        mem_bias,
                        params.num_heads,
                        params.attention_key_channels or params.hidden_size,
                        params.attention_value_channels or params.hidden_size,
                        params.hidden_size,
                        1.0 - params.attention_dropout,
                    )

                    #combined_weight = tf.squeeze(y["weights"], 2)  # remove the 1 dimension and become 2 8 45
                    combined_weight = y["weights"]  # remove the 1 dimension and become 2 8 45
                    y = y["outputs"]
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)
                    att_weight.append(combined_weight)

                with tf.variable_scope("feed_forward"):
                    y = _ffn_layer(
                        _layer_process(x, params.layer_preprocess),
                        params.filter_size,
                        params.hidden_size,
                        1.0 - params.relu_dropout,
                    )
                    x = _residual_fn(x, y, 1.0 - params.residual_dropout)
                    x = _layer_process(x, params.layer_postprocess)

        outputs = _layer_process(x, params.layer_preprocess)

        if state is not None:
            return outputs, next_state, att_weight

        return outputs, next_state, att_weight

def encoding_graph(features, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    hidden_size = params.hidden_size

    src_seq = features["source"]
    #src_seq = tf.Print(src_seq, [features["align_pos"]], "align_pos", 10, 1000)

    src_len = features["source_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)

    svocab = params.vocabulary["source"]
    src_vocab_size = len(svocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        src_embedding = tf.get_variable("weights",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    else:
        src_embedding = tf.get_variable("source_embedding",
                                        [src_vocab_size, hidden_size],
                                        initializer=initializer)
    #src_embedding = tf.cast(src_embedding, dtype=tf.float64)
    bias = tf.get_variable("bias", [hidden_size])

    # id => embedding
    # src_seq: [batch, max_src_length]
    inputs = tf.gather(src_embedding, src_seq) * (hidden_size ** 0.5)
    inputs = inputs * tf.expand_dims(src_mask, -1)

    # Preparing encoder
    encoder_input = tf.nn.bias_add(inputs, bias)
    encoder_input = layers.attention.add_timing_signal(encoder_input)

    #encoder_input = add_timing_signal_float64(encoder_input)

    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        encoder_input = tf.nn.dropout(encoder_input, keep_prob)

    encoder_output = transformer_encoder(encoder_input, enc_attn_bias, params) #,  dtype=tf.float64

    # encoder_output = tf.cast(encoder_output, dtype=tf.float32)
    # encoder_output = tf.cast(encoder_output, dtype=tf.float32)
    #

    return encoder_output

def decoding_graph(features, state, mode, params):
    if mode != "train":
        params.residual_dropout = 0.0
        params.attention_dropout = 0.0
        params.relu_dropout = 0.0
        params.label_smoothing = 0.0

    tgt_seq = features["target"]
    src_len = features["source_length"]
    tgt_len = features["target_length"]
    src_mask = tf.sequence_mask(src_len,
                                maxlen=tf.shape(features["source"])[1],
                                dtype=tf.float32)
    tgt_mask = tf.sequence_mask(tgt_len,
                                maxlen=tf.shape(features["target"])[1],
                                dtype=tf.float32)

    hidden_size = params.hidden_size
    tvocab = params.vocabulary["target"]
    tgt_vocab_size = len(tvocab)
    initializer = tf.random_normal_initializer(0.0, params.hidden_size ** -0.5)

    if params.shared_source_target_embedding:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tgt_embedding = tf.get_variable("weights",
                                            [tgt_vocab_size, hidden_size],
                                            initializer=initializer)
    else:
        tgt_embedding = tf.get_variable("target_embedding",
                                        [tgt_vocab_size, hidden_size],
                                        initializer=initializer)

    if params.shared_embedding_and_softmax_weights:
        weights = tgt_embedding
    else:
        weights = tf.get_variable("softmax", [tgt_vocab_size, hidden_size],
                                  initializer=initializer)

    # id => embedding
    # tgt_seq: [batch, max_tgt_length]

    #tgt_seq = tf.Print(tgt_seq, [tgt_seq], "tgt_seq", 1000, 1000)

    targets = tf.gather(tgt_embedding, tgt_seq) * (hidden_size ** 0.5)

    targets = targets * tf.expand_dims(tgt_mask, -1)

    #targets = tf.Print(targets, [tf.shape(targets), tf.shape(tgt_mask)], "tf.shape(targets), tf.shape(tgt_mask)", 10, 1000)

    # Preparing encoder and decoder input
    enc_attn_bias = layers.attention.attention_bias(src_mask, "masking")
    dec_attn_bias = layers.attention.attention_bias(tf.shape(targets)[1],
                                                    "causal")

    #Shift left
    # if mode != "infer":
    #     decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    #     decoder_input = layers.attention.add_timing_signal(decoder_input)
    # else:  thumt 本身的 beamsearch需要以前的encoder结果，不能简单的裁剪，不像改造的chris的结果
    #     decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, -1:, :]
    #     actual_len = tf.shape(state["decoder"]["layer_0"]["key"])[1]
    #     not_bos_flag = tf.greater(actual_len, 0)
    #     decoder_input = decoder_input * tf.cast(not_bos_flag,dtype=tf.float32)
    #     decoder_input = add_onestep_timing_signal(decoder_input, actual_len+1)

    # Shift left
    decoder_input = tf.pad(targets, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    decoder_input = layers.attention.add_timing_signal(decoder_input)

    if params.residual_dropout:
        keep_prob = 1.0 - params.residual_dropout
        decoder_input = tf.nn.dropout(decoder_input, keep_prob)

    encoder_output = state["encoder"]

    if mode != "infer":
        # decoder_output = transformer_decoder(decoder_input, encoder_output,
        #                                      dec_attn_bias, enc_attn_bias,
        #                                      params)
        #decoder_input = tf.Print(decoder_input, [tf.shape(decoder_input)], "decoder_input", 10, 1000)
        decoder_outputs = transformer_decoder_output_cur_encdec_atten(decoder_input, encoder_output,
                                                                      dec_attn_bias, enc_attn_bias,
                                                                      params)

        decoder_output, decoder_state, decoder_weight = decoder_outputs
    else:
        #decoder_input = tf.Print(decoder_input, [decoder_input], "decoder_input before -1", 1000, 1000)

        decoder_input = decoder_input[:, -1:, :]
        #decoder_input = tf.Print(decoder_input, [decoder_input], "decoder_input after -1", 1000, 1000)

        dec_attn_bias = dec_attn_bias[:, :, -1:, :]
        decoder_outputs = transformer_decoder_output_cur_encdec_atten(decoder_input, encoder_output,
                                              dec_attn_bias, enc_attn_bias,
                                              params, state=state["decoder"])

        decoder_output, decoder_state, decoder_weight = decoder_outputs
        decoder_output = decoder_output[:, -1, :]
        logits = tf.matmul(decoder_output, weights, False, True)
        log_prob = tf.nn.log_softmax(logits)

        return log_prob, {"decoder": decoder_state, "encoder": encoder_output,}

    # [batch, length, channel] => [batch * length, vocab_size]
    decoder_output = tf.reshape(decoder_output, [-1, hidden_size])
    logits = tf.matmul(decoder_output, weights, False, True)
    labels = features["target"]

    # label smoothing
    ce = layers.nn.smoothed_softmax_cross_entropy_with_logits(
        logits=logits,
        labels=labels,
        smoothing=params.label_smoothing,
        normalize=True
    )
    ce = tf.reshape(ce, tf.shape(tgt_seq))

    if mode == "eval":
        return -tf.reduce_sum(ce * tgt_mask, axis=1)

    loss = tf.reduce_sum(ce * tgt_mask) / tf.reduce_sum(tgt_mask)
    #########################################
    # 开始计算 alignment的loss  要形成 tgt--->src的对齐矩阵
    # layer_weight = decoder_weight[5]  # layer5 128 8 32 34:128是batch_size，每一层都是针对batch进行操作，所以每一层的结果是一个batch的结果
    # att_matrix = layer_weight[:, 5]  # only keep the 5th head: 128 32 34

    ######################
    # 仅考虑第六层，如果考虑所有层，则注释掉
    decoder_weight = decoder_weight[5:6]
    #
    ###################
    layer_weight = tf.transpose(decoder_weight, [1, 0, 2, 3, 4])  #6layers Nbatch 8heads 32 34 -->
    att_matrix = tf.reduce_mean(layer_weight, [1, 2])  # will be N 32 34
    att_matrix_shape = tf.shape(att_matrix)

    # att_matrix 也需要Maks掉eos以后的概率
    src_mask = tf.sequence_mask(src_len, maxlen=tf.shape(features["source"])[1])  # 有词存在的位置是True, padded的位置是False: N SL
    src_mask = tf.tile(tf.expand_dims(src_mask, 2), [1, 1, att_matrix_shape[1]])  # 扩展到padded target长   N SL TL
    src_mask = tf.transpose(src_mask, [0, 2, 1])  # 变成 N T S，以便和tgt_mask相乘   # N TL SL

    tgt_mask = tf.sequence_mask(tgt_len, maxlen=tf.shape(features["target"])[1])  # 有词存在的位置是True, padded的位置是False: N TL
    tgt_mask = tf.tile(tf.expand_dims(tgt_mask,2), [1, 1, att_matrix_shape[2]])   # 扩展到padded source长

    att_matrix_mask = tf.logical_and(tgt_mask, src_mask)  # 生成 mask matrix
    # 最小平方差 和 交叉熵的算法不一样，挪到后面统一处理
    #att_matrix = tf.multiply(att_matrix, tf.cast(att_matrix_mask, dtype=tf.float32))  #屏蔽掉padded部分的值

    padded_align_len = tf.shape(features["align_tgt"])[1]  #对齐矩阵中的tgt元素数量(和src相同)
    batch_size = att_matrix_shape[0]

    # 消除 后面padded的0-0 将 0-0转换为eos-eos: 实际的对齐单元数量   padded的对齐单元数量: features["align_mask"] 也是经过对齐的
    padded_mask = tf.cast(tf.equal(features["align_mask"], 0), dtype=tf.int32)  # mask中后面的0就是padded mask：将1的位置转换为0， 0的位置转换为1：  N Padded_Len
    tgt_padded_mask = tf.multiply(padded_mask, tf.expand_dims(features["target_length"]-1, 1))  # target_length 已近包含eos， 长度减1得到eos位置索引， Padded的位置都填上eos位置索引
    src_padded_mask = tf.multiply(padded_mask, tf.expand_dims(features["source_length"]-1, 1))

    align_tgt = features["align_tgt"] + tgt_padded_mask  # 1 2 3 .xxx 0 0 0 ---> 1 2 3 .xxx 10 10 10  # 10是eos的索引
    align_src = features["align_src"] + src_padded_mask  # 0 2 4 .xxx 0 0 0 ---> 1 2 3 .xxx 11 11 11  # 11是eos的索引

    indices = tf.stack([align_tgt, align_src], 2)  # N 40 2
    indices = tf.reshape(indices, [-1, 2])
    batch_indices = tf.range(batch_size)  # 0 1 2 3 ...
    batch_indices = tf.expand_dims(batch_indices, -1)
    batch_indices = tf.tile(batch_indices, [1, padded_align_len])  # 0 0 0 ... 1 1 1 1 ... 2 2 2 2
    batch_indices = tf.reshape(batch_indices, [-1, 1])
    indices = tf.concat([batch_indices, indices], 1)   # [[0 x x] [0 x x ] ... [1 x x ] [1 x x ]....]
    indices = tf.cast(indices, dtype=tf.int64)
    indicator_shape = tf.cast(att_matrix_shape, dtype=tf.int64)

    # loss = tf.Print(loss, [tgt_padded_mask, src_padded_mask], "tgt_padded_mask, src_padded_mask", 100, 10000)
    # loss = tf.Print(loss, [tf.shape(features["target"]), tf.shape(features["source"])], "target source", 100, 10000)
    # loss = tf.Print(loss, [features["target_length"], features["source_length"]], "real target source len", 100, 10000)
    # loss = tf.Print(loss, [tf.shape(features["align_tgt"]), tf.shape(features["align_src"])], "align_tgt align_src", 100, 10000)
    # loss = tf.Print(loss, [features["align_tgt"], features["align_src"]], "",
    #                 100, 10000)
    # loss = tf.Print(loss, [att_matrix_shape], "att_matrix_shape", 100, 10000)
    # loss = tf.Print(loss, [tf.shape(indices), indices], "indices", 100, 10000)
    # loss = tf.Print(loss, [tf.shape(att_matrix), att_matrix], "layer_head_weight", 100, 10000)

    indicator = tf.sparse_to_dense(indices, indicator_shape, 1.0, validate_indices=False)
    indicator_src_sum = tf.reduce_sum(indicator, -1, keepdims=True)
    zero_sum_pos = tf.equal(indicator_src_sum, 0)  # 找到为sum为0的位置
    indicator_src_sum = indicator_src_sum + tf.cast(zero_sum_pos, dtype=tf.float32)  #sum为1 确保不除0
    indicator = tf.div(indicator, indicator_src_sum)  # 归一化
    align_loss_model = 'square-mean'
    if align_loss_model == 'square-mean':
        att_matrix = tf.multiply(att_matrix, tf.cast(att_matrix_mask, dtype=tf.float32))  # 屏蔽掉padded部分的值
        _err = tf.square(tf.subtract(att_matrix, indicator))  # N Tl Sl
        # square_err_sum = tf.reduce_sum(square_err, [1, 2])  # add each source and each target word
        # square_err_sum_av = tf.div(square_err_sum, features['target_length'])  # 每个句子内部平均，以实际的长度来除，以消除padded单元0的影响
        # align_loss = tf.reduce_mean(square_err_sum_av)  # batch内的句子平均
        # #align_loss = tf.losses.mean_squared_error(att_matrix, indicator)
    else:
        att_matrix_mask = tf.cast(tf.logical_not(att_matrix_mask), dtype=tf.float32)  # padded部分变为1, log以后就是0，不计入损失
        att_matrix = att_matrix + att_matrix_mask  # 屏蔽padded部分的值，使其变为1，防止Log0 出现， 且取log后就是0
        mini_val_mask = tf.equal(att_matrix, 0.0)*0.00000001  # 先进行上一步，再进行下一步
        att_matrix = att_matrix + mini_val_mask
        _err = -indicator * tf.log(att_matrix)

    err_sum = tf.reduce_sum(_err, [1, 2])  # add each source and each target word
    err_sum_av = tf.div(err_sum, tf.cast(features['target_length'], dtype=tf.float32))  # 每个句子内部平均，以实际的长度来除，以消除padded单元的影响
    align_loss = tf.reduce_mean(err_sum_av)  # batch内的句子平均
    # loss = tf.Print(loss, [tf.shape(indicator), indicator], "indicator", 100, 10000)
    # loss = tf.Print(loss, [indicator_src_sum], "indicator_src_sum", 100, 10000)
    # loss = tf.Print(loss, [zero_sum_pos], "zero_sum_pos", 100, 10000)

    #loss = tf.Print(loss, [loss, align_loss], "loss, align_loss", 100000, 10000)


    start_step = tf.constant(390000, dtype=tf.float32)
    end_step = tf.constant(550000, dtype=tf.float32)

    curr_step = tf.train.get_or_create_global_step()
    pass_step = tf.cast(curr_step, dtype=tf.float32) - start_step
    flag_start = tf.cast(tf.greater(pass_step, 0.0), dtype=tf.float32)  # 0 / 1  start_step之前不衰减，之后才衰减
    ratio = pass_step / (end_step - start_step) * flag_start

    decay_ratio = 1 - ratio

    flag_end = tf.greater(decay_ratio, 0)  # can't be negative

    decay_ratio = decay_ratio * tf.cast(flag_end, dtype=tf.float32)

    loss = tf.Print(loss, [decay_ratio, align_loss], 'decay_ratio, align_loss', 10000, 1000)
    return loss + decay_ratio*align_loss

    #return loss

def add_onestep_timing_signal(x, length, min_timescale=1.0, max_timescale=1.0e4, name=None):
    """
    This function adds a bunch of sinusoids of different frequencies to a
    Tensor. See paper: `Attention is all you need'

    :param x: A tensor with shape [batch, length, channels]
    :param length: the actual length of the current decoding target
    :param min_timescale: A floating point number
    :param max_timescale: A floating point number
    :param name: An optional string

    :returns: a Tensor the same shape as x.
    """

    with tf.name_scope(name, default_name="add_timing_signal", values=[x]):

        channels = tf.shape(x)[2]
        position = tf.to_float(tf.range(length-1, length))
        #position = tf.to_float(tf.range(length))
        num_timescales = channels // 2

        log_timescale_increment = (
                math.log(float(max_timescale) / float(min_timescale)) /
                (tf.to_float(num_timescales) - 1)
        )
        inv_timescales = min_timescale * tf.exp(
            tf.to_float(tf.range(num_timescales)) * -log_timescale_increment
        )

        scaled_time = (tf.expand_dims(position, 1) *
                       tf.expand_dims(inv_timescales, 0))
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
        #signal = tf.reshape(signal, [1, length, channels])
        signal = tf.reshape(signal, [1, 1, channels])
        #x = tf.Print(x, [length, x, signal], "length, x, signal")
        return x + signal[-1]



def model_graph(features, mode, params):


    encoder_output = encoding_graph(features, mode, params)

    state = {
        "encoder": encoder_output
    }
    output = decoding_graph(features, state, mode, params)

    return output


class Transformer(interface.NMTModel):

    def __init__(self, params, scope="transformer"):
        super(Transformer, self).__init__(params=params, scope=scope)

    def get_training_func(self, initializer):
        def training_fn(features, params=None, reuse=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope, initializer=initializer,
                                   reuse=reuse):
                loss = model_graph(features, "train", params)
                return loss

        return training_fn

    def get_evaluation_func(self):
        def evaluation_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                score = model_graph(features, "eval", params)

            return score

        return evaluation_fn
    def get_inference_func(self):
        def encoding_fn(features, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                encoder_output = encoding_graph(features, "infer", params)
                batch = tf.shape(encoder_output)[0]

                state = {
                    "encoder": encoder_output,
                    "decoder": {
                        "layer_%d" % i: {
                            "key": tf.zeros([batch, 0, params.hidden_size]),
                            "value": tf.zeros([batch, 0, params.hidden_size]),
                        }
                        for i in range(params.num_decoder_layers)
                    }
                }

            return state

        def decoding_fn(features, state, params=None):
            if params is None:
                params = copy.copy(self.parameters)
            else:
                params = copy.copy(params)

            with tf.variable_scope(self._scope):
                log_prob, new_state = decoding_graph(features, state, "infer",
                                                     params)

            return log_prob, new_state

        return encoding_fn, decoding_fn

    @staticmethod
    def get_name():
        return "transformer"

    @staticmethod
    def get_parameters():
        params = tf.contrib.training.HParams(
            pad="<pad>",
            bos="<eos>",
            eos="<eos>",
            unk="<unk>",
            append_eos=False,
            hidden_size=512,
            filter_size=2048,
            num_heads=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            attention_dropout=0.0,
            residual_dropout=0.1,
            relu_dropout=0.0,
            label_smoothing=0.1,
            attention_key_channels=0,
            attention_value_channels=0,
            multiply_embedding_mode="sqrt_depth",
            shared_embedding_and_softmax_weights=False,
            shared_source_target_embedding=False,
            # Override default parameters
            learning_rate_decay="linear_warmup_rsqrt_decay",
            initializer="uniform_unit_scaling",
            initializer_gain=1.0,
            learning_rate=1.0,
            layer_preprocess="none",
            layer_postprocess="layer_norm",
            batch_size=4096,
            constant_batch_size=False,
            adam_beta1=0.9,
            adam_beta2=0.98,
            adam_epsilon=1e-9,
            clip_grad_norm=0.0

        )

        return params
