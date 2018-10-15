import yaml


def net_config(config):
    net_cfg = dict()
    net_cfg['SEQUENCE_LENGTH'] = 32
    net_cfg['HIST_LENGTH'] = 16
    net_cfg['NUM_HIDDEN'] = 32
    config['NET'] = net_cfg


def train_config(config):
    train_cfg = dict()
    train_cfg['BATCH_SIZE'] = 16
    train_cfg['LEARNING_RATE'] = 1e-3
    train_cfg['L2_REG_WEIGHT'] = 0.1
    train_cfg['PRETRAIN_MODEL_PREFIX'] = './models/pretrained'
    train_cfg['EPOCHS'] = 20
    config['TRAIN'] = train_cfg


def test_config(config):

    test_cfg = dict()
    test_cfg['BATCH_SIZE'] = 1
    config['TEST'] = test_cfg


def data_config(config):
    data_cfg = dict()
    # data_cfg['subject1'] = {'begin_date': }
    # subject3: start_date = '2018-06-25 00:00:00', end_date = '2018-07-07 00:00:00'
    # subject4: start_date = '2018-06-28 00:00:00', end_date = '2018-07-07 00:00:00'
    # subject5: start_date = '2018-06-28 00:00:00', end_date = '2018-07-12 00:00:00'

    data_cfg['subject3'] = {'begin_date': '2018-06-25 00:00:00', 'end_date': '2018-07-07 00:00:00'}
    data_cfg['subject4'] = {'begin_date': '2018-06-28 00:00:00', 'end_date': '2018-07-07 00:00:00'}
    data_cfg['subject5'] = {'begin_date': '2018-06-28 00:00:00', 'end_date': '2018-07-12 00:00:00'}

    data_cfg['root_dir'] = '../data/type1'

    data_cfg['total_num_subject'] = 6

    config['DATA'] = data_cfg


if __name__ == '__main__':

    config = dict()
    net_config(config)
    train_config(config)
    test_config(config)
    data_config(config)

    with open('../config/all_config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)



