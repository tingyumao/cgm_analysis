import mxnet as mx
import yaml


def cgm_net(in_seq, h_seq, num_hidden=32, sequence_length=32):

    init_c = mx.symbol.FullyConnected(data=h_seq, num_hidden=num_hidden, name='init_c_encoder')
    init_h = mx.symbol.FullyConnected(data=h_seq, num_hidden=num_hidden, name='init_h_encoder')

    lstm_cell = mx.rnn.LSTMCell(num_hidden=num_hidden)

    outputs, states = lstm_cell.unroll(length=sequence_length, inputs=in_seq,
                                       layout='NTC', begin_state=[init_h, init_c], merge_outputs=False)

    outputs = mx.sym.concat(*outputs, dim=1) # NTC

    outputs = outputs.reshape((-1, num_hidden))
    pred_seq = mx.symbol.FullyConnected(data=outputs, num_hidden=1, name='output_seq_decoder')
    pred_seq = pred_seq.reshape((-1, sequence_length, 1))

    return pred_seq  # mx.sym.Group([outputs] + states)


def cgm_net_train(config):

    num_hidden = config.get('NUM_HIDDEN', 32)
    sequence_length = config.get('SEQUENCE_LENGTH', 32)

    h_seq = mx.sym.Variable(name='hist_sequence')
    in_seq = mx.sym.Variable(name='input_sequence')
    gt_seq = mx.sym.Variable(name='gt_sequence')

    pred_seq = cgm_net(in_seq, h_seq, num_hidden=num_hidden, sequence_length=sequence_length)

    # loss
    mse = mx.sym.mean((pred_seq - gt_seq)**2, axis=1)
    loss_mse = mx.sym.MakeLoss(mse, normalization='batch', name='mean_squared_loss')

    return loss_mse


def cgm_net_test(config):

    num_hidden = config.get('NUM_HIDDEN', 32)
    sequence_length = config.get('SEQUENCE_LENGTH', 32)

    h_seq = mx.sym.Variable(name='hist_sequence')
    in_seq = mx.sym.Variable(name='input_sequence')

    pred_seq = cgm_net(in_seq, h_seq, num_hidden=num_hidden, sequence_length=sequence_length)

    return pred_seq # mx.sym.Group([h_seq, pred_seq])


if __name__ == '__main__':

    with open('../config/all_config.yaml') as f:
        config = yaml.load(f)

    sym = cgm_net_test(config['NET'])

    # viewer = mx.viz.plot_network(sym)
    # viewer.view()
    # save net architecture to json file
    # sym.save('../sample/net.json')

    # fake sequence data
    batch_size = config['TRAIN']['BATCH_SIZE']
    seq_len = config['NET']['SEQUENCE_LENGTH']
    fake_in_seq = mx.nd.ones([batch_size, seq_len, 5])
    fake_h_seq = mx.nd.ones([batch_size, seq_len, 1])

    # distributed computation
    ctx = [mx.cpu(i) for i in range(4)]

    data_shapes = [('input_sequence', (batch_size, seq_len, 5)),
                   ('hist_sequence', (batch_size, seq_len, 1))]

    label_shapes = []  # [('gt_sequence', (batch_size,seq_len, 1))]

    mod = mx.mod.Module(sym, data_names=[s[0] for s in data_shapes], label_names=[s[0] for s in label_shapes],
                        context=ctx)
    # initializer = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)

    mod.bind(data_shapes=data_shapes, for_training=False)

    mod.init_params()

    fake_data_batch = mx.io.DataBatch(data=[fake_in_seq, fake_h_seq], label=None,
                                      provide_data=data_shapes, provide_label=[])
    mod.forward(data_batch=fake_data_batch)

    test_out = mod.get_outputs()

    print(test_out[0].shape)