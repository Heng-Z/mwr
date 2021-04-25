from IsoNet.models.unet import builder,builder_fullconv,builder_fullconv_old
from tensorflow.keras.layers import Input,Add,Activation
from tensorflow.keras.models import Model
from IsoNet.losses.losses import loss_mae,loss_mse
from IsoNet.losses.wedge_power import wedge_power_gain
from tensorflow.keras.optimizers import Adam
def Unet(filter_base=32,
        depth=2,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=False,
        dropout=0.0,
        pool=(2,2,2),residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004,
        test_shape=None):

    # model = builder.build_unet(filter_base,depth,convs_per_depth,
    #            kernel,
    #            batch_norm,
    #            dropout,
    #            pool)
    model = builder_fullconv.build_unet(filter_base,depth,convs_per_depth,
            kernel,
            batch_norm,
            dropout,
            pool)
    # import os
    # import sys
    # cwd = os.getcwd()
    # sys.path.insert(0,cwd)
    # import train_settings 
    # model = builder_fullconv_old.build_unet(train_settings)
    
    #***** Construct complete model from unet output
    if test_shape is None:
        inputs = Input((None,None,None,1))
    elif type(test_shape) is int:
        inputs = Input((test_shape,test_shape,test_shape,1))
    unet_out = model(inputs) 
    if residual:
        outputs = Add()([unet_out, inputs])
    else:
        outputs = unet_out
    outputs = Activation(activation=last_activation)(outputs)
    model = Model(inputs=inputs, outputs=outputs)
    optimizer = Adam(lr=lr)
    if loss == 'mae' or loss == 'mse':
        metrics = ('mse', 'mae')
        _metrics = [eval('loss_%s()' % m) for m in metrics]
        loss = eval('loss_%s()' % loss)
    elif loss == 'binary_crossentropy':
        _metrics = ['accuracy']
    model.compile(optimizer=optimizer, loss=loss, metrics=_metrics)
    return model

if __name__ == "__main__":
    keras_model = Unet(filter_base=64,
        depth=3,
        convs_per_depth=3,
        kernel=(3,3,3),
        batch_norm=True,
        dropout=0.5,
        pool=(2,2,2),residual = True,
        last_activation = 'linear',
        loss = 'mae',
        lr = 0.0004)
    print(keras_model.summary())
