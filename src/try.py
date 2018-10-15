import mxnet as mx


if __name__ == '__main__':

    data = mx.sym.Variable('data')
    fc1 = mx.sym.FullyConnected(data, name='fc1', num_hidden=128)
    act1 = mx.sym.Activation(fc1, name='relu1', act_type="relu")
    fc2 = mx.sym.FullyConnected(act1, name='fc2', num_hidden=10)
    out = mx.sym.SoftmaxOutput(fc2, name='softmax')
    mod = mx.mod.Module(out, context=mx.cpu(0))  # create a module by given a Symbol

    batch_size = 16
    num_in = 200
    mod.bind(data_shapes=[('data', (batch_size, num_in))], for_training=False)  # create memory by given input shapes
    mod.init_params()  # initial parameters with the default random initializer

    tmp_data_batch = mx.io.DataBatch(data=[mx.nd.ones((batch_size, num_in))], provide_data=[('data', (batch_size, num_in))])
    mod.forward(tmp_data_batch)

    tmp = mod.get_outputs()

    print(len(tmp), tmp[0].shape)



