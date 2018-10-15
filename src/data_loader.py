import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)-s\t%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
import mxnet as mx
import numpy as np
import random
import yaml
from src.utils import *
from src.preprocess import *


class SubjectDataLoader(mx.io.DataIter):
    def __init__(self, config, **kwargs):
        super(SubjectDataLoader, self).__init__()

        self.data_config = config['DATA']
        self.root_dir = self.data_config['root_dir']
        self.total_num_subject = self.data_config['total_num_subject']
        self.subject = kwargs.get('subject', 'subject3')
        self.begin_date = self.data_config[self.subject]['begin_date']
        self.end_date = self.data_config[self.subject]['end_date']

        self.data_root_dir = self.data_config['root_dir']

        self.phase = kwargs.get('phase', 'TRAIN')
        assert self.phase in ['TRAIN', 'TEST'], print('Invalid mode {}'.format(self.phase))

        self.batch_size = config[self.phase]['BATCH_SIZE']
        self.seq_len = config['NET']['SEQUENCE_LENGTH']
        self.hist_len = config['NET']['HIST_LENGTH']

        self.in_feats = kwargs.get("input_features",
                                   ["Correction Bolus", "Meal Bolus", "gCarbs",
                                    "Carbohydrate", "hr", "steps", "sleep_code"])
        self.out_feat = ["bg"]

        self.num_in = len(self.in_feats)
        self.num_out = len(self.out_feat)

        self.seqs = self._load_cgm_data()
        self.data_shapes = [(self.batch_size, self.seq_len, self.num_in), (self.batch_size, self.hist_len, 1)]
        self.label_shapes = [(self.batch_size, self.seq_len, self.num_out)]

        self.total_size = len(self.seqs)
        self.batch_num = self.total_size // self.batch_size

        self.reset()

    def reset(self):
        self.current = 0
        random.shuffle(self.seqs)

    @property
    def provide_data(self):
        return self.data_shapes

    @property
    def provide_label(self):
        return self.label_shapes

    def next(self):
        if self.current + self.batch_size > self.total_size:
            raise StopIteration
        xs = [[] for _ in range(len(self.data_shapes))]
        ys = [[] for _ in range(len(self.label_shapes))]
        for batch_ind, i in enumerate(range(self.current, self.current + self.batch_size)):
            data = [self.seqs[i][0], self.seqs[i][2]]
            label = [self.seqs[i][1]]

            for j in range(len(data)):
                xs[j].append(data[j])
            for j in range(len(label)):
                ys[j].append(label[j])

        xs = [mx.ndarray.array(x) for x in xs]
        ys = [mx.ndarray.array(y) for y in ys]
        self.current += self.batch_size

        return mx.io.DataBatch(data=xs, label=ys)

    def _load_cgm_data(self):
        data_dir = self.root_dir
        num_subject = self.total_num_subject
        file_dict = parse_data_files(data_dir, num_subject)

        # read data
        pat = self.subject
        sub_files = file_dict[pat]
        all_data = read_one_subject_data(sub_files)

        start_date = self.begin_date
        end_date = self.end_date
        for k in all_data:
            all_data[k] = select_time_window(all_data[k], start_date, end_date)

        s = all_data['bg']['minutes'][:1].values[0]
        e = all_data['bg']['minutes'][-1:].values[0]

        k = self.num_in
        x = np.arange(s, e, 5.)
        y = np.zeros((x.shape[0], k))

        cutoff_frequency = 5.0
        sample_rate = 200

        for i, k in enumerate(self.in_feats):
            if k in ["Correction Bolus", "Meal Bolus", "gCarbs", "Carbohydrate"]:
                tmp = all_data[k]
                tmp = tmp[tmp[k] < 1000]
                if k == "Meal Bolus":
                    tmp = tmp[tmp[k] < 50]

                max_tmp = max(tmp[k])
                tmp = all_data[k]
                tmp[tmp[k] > max_tmp][k] = max_tmp

                cb = tmp[k].values
                cbt = tmp['minutes'].values
                tt = np.minimum(np.round((cbt - s) / 5).astype('int'), x.shape[0] - 1)
                tv = np.zeros_like(x)
                tv[tt] = cb
                y[:, i] = tv
            elif k == "steps":
                stept = all_data['steps']['minutes'].values
                step = all_data['steps']['steps'].values
                tv = np.interp(x, stept, step)
                tv = np.convolve(tv, np.ones((12,)) / 12., mode='same')
                tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
                y[:, i] = tv
            elif k == "sleep_code":
                def find_sleep_interval(all_data):
                    sleep_data = all_data['sleep_code']
                    sleep_time_points = sorted(sleep_data['minutes'])
                    sleep_interval = []
                    rest_limit = 60
                    s = sleep_time_points[0]
                    for i, _ in enumerate(sleep_time_points[1:-1]):
                        e = sleep_time_points[i]
                        if sleep_time_points[i + 1] - e > rest_limit:
                            sleep_interval += [[s, e]]
                            s = sleep_time_points[i + 1]
                    return sleep_interval

                sleep_interval = find_sleep_interval(all_data)
                tv = np.zeros_like(x)
                for ss, se in sleep_interval:
                    ss = max(ss, s)
                    se = min(se, e)
                    tv[int((ss - s) // 5):int((se - s) // 5)] = 1.
                y[:, i] = tv
            else:
                vt = all_data[k]['minutes'].values
                v = all_data[k][k].values
                tv = np.interp(x, vt, v)
                tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
                y[:, i] = tv

        in_seqs = y[:-1, :]

        # cgm change rate
        y = np.zeros((x.shape[0], self.num_out))
        for i, k in enumerate(self.out_feat):
            vt = all_data[k]['minutes'].values
            v = all_data[k][k].values
            tv = np.interp(x, vt, v)
            tv = butter_lowpass_filter(tv, cutoff_frequency, sample_rate / 2)
            y[:, i] = tv
        bgv = np.zeros((x.shape[0]-1, self.num_out))
        bgv[:, 0] = np.diff(y[:, 0])

        # normalization
        ynorm = np.zeros_like(in_seqs)
        for i in range(ynorm.shape[1]):
            ynorm[:, i] = (in_seqs[:, i] - in_seqs[:, i].min()) / (in_seqs[:, i].max() - in_seqs[:, i].min())
        in_seqs = ynorm

        ynorm = np.zeros_like(bgv)
        for i in range(ynorm.shape[1]):
            ynorm[:, i] = 0.01 * (bgv[:, i] - np.mean(bgv[:, i]))

        # Generate sequences
        state_size = self.hist_len
        time_steps = self.seq_len

        in_data = []
        out_data = []
        hist_data = []
        all_data = []

        t1 = state_size
        t2 = t1 + time_steps
        while t2 < bgv.shape[0]:
            in_data.append(in_seqs[t1:t2])
            out_data.append(bgv[t1:t2])
            hist_data.append(bgv[t1-state_size:t1])
            all_data.append([in_seqs[t1:t2], bgv[t1:t2], bgv[t1-state_size:t1]])
            t1 += 2
            t2 = t1 + time_steps

        in_data = np.array(in_data)
        out_data = np.array(out_data)
        hist_data = np.array(hist_data)

        num_train = int(len(all_data) * 0.9)
        if self.phase == "TRAIN":
            all_data = all_data[:num_train]
        else:
            all_data = all_data[num_train:]

        logging.info("Input sequence shape: {}".format(in_data.shape))
        logging.info("Output sequence shape: {}".format(out_data.shape))
        logging.info("History sequence shape: {}".format(hist_data.shape))
        logging.info("#Sample: {}".format(len(all_data)))

        return all_data

    def viz_sample(self):
        sample = random.choice(self.seqs)
        in_data, out_data, hist_data = sample
        f, axarr = plt.subplots(self.num_in+self.num_out, 1, figsize=(10, 3*(self.num_in+self.num_out)))
        for i in range(self.num_in):
            axarr[i].plot(in_data[:, i], label=self.in_feats[i])
            axarr[i].legend(loc='center left', bbox_to_anchor=(1., 0.5),
                            fancybox=True, shadow=True, ncol=5)
        axarr[-1].plot(out_data[:, 0], label=self.out_feat[0])
        axarr[-1].legend(loc='center left', bbox_to_anchor=(1., 0.5),
                         fancybox=True, shadow=True, ncol=5)
        plt.show()


if __name__ == '__main__':

    with open('../config/all_config.yaml') as f:
        config = yaml.load(f)

    train_data = SubjectDataLoader(config)
    train_data.viz_sample()







