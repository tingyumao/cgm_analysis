import mxnet as mx
from src.net import cgm_net_train
from src.data_loader import SubjectDataLoader


def train_loop(config):

    # training settings
    batch_size = config['TRAIN']['BATCH_SIZE']
    seq_len = config['NET']['SEQUENCE_LENGTH']

    # prepare dataset
    train_data = SubjectDataLoader(config)  # mx.io.DataIter()
    val_data = SubjectDataLoader(config)  # mx.io.DataIter()

    # define network
    sym = cgm_net_train(config=config)

    # distributed computation
    ctx = [mx.cpu(i) for i in range(4)]

    # create training module
    data_shapes = [('input_sequence', (batch_size, seq_len, 5)),
                   ('hist_sequence', (batch_size, seq_len, 1))]

    label_shapes = [('gt_sequence', (batch_size,seq_len, 1))]

    mod = mx.mod.Module(sym, data_names=[s[0] for s in data_shapes], label_names=[s[0] for s in label_shapes],
                        context=ctx)
    initializer = mx.init.Xavier(factor_type="in", rnd_type='gaussian', magnitude=2)

    mod.bind(data_shapes=data_shapes, label_shapes=label_shapes, for_training=False)

    mod.init_params(initializer=initializer)

    eval_metric = mx.metric.Loss(name='bgv_mse', output_names=['mean_squared_loss'], label_names=['gt_sequences'])

    # optimizer, opt_params = _get_optimization(config, train_batch_num, batch_size)
    optimizer = "adam"
    lr_scheduler = mx.lr_scheduler.MultiFactorScheduler(step=[stepPerEpoch * x for x in steps], factor=0.1)
    optmizer_params = {'learning_rate': config['TRAIN']['LEARNING_RATE'],
                       'momentum': 0.9,
                       'wd': 0.0001,
                       'clip_gradient': 10.0,
                       'rescale_grad': 1.0 / config['TRAIN'][''],
                       'lr_scheduler': lr_scheduler}

    mod.fit(train_data=train_data,
            eval_data=val_data,
            eval_metric=eval_metric,
            optimizer=optimizer,
            optimizer_params=optmizer_params)




