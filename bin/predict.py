#!/usr/bin/env python3
import numpy as np
import os
import logging
from IsoNet.preprocessing.simulate import apply_wedge
from IsoNet.util.norm import normalize
from IsoNet.util.toTile import reform3D
import mrcfile
from IsoNet.util.image import *
from IsoNet.util.metadata import MetaData,Label,Item
from IsoNet.util.dict2attr import idx2list
def predict(args):

    if args.log_level == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    import tensorflow as tf

    import logging
    tf_logger = tf.get_logger()
    tf_logger.setLevel(logging.ERROR)

    logger = logging.getLogger('predict')
    args.gpuID = str(args.gpuID)
    ngpus = len(args.gpuID.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID 


    logger.debug('percentile:{}'.format(args.norm))

    logger.info('gpuID:{}'.format(args.gpuID))
    if ngpus >1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = tf.keras.models.load_model(args.model)
    else:
        model = tf.keras.models.load_model(args.model)

    logger.info("Loaded model from disk")

    if not os.path.isdir(args.output_dir):
            os.mkdir(args.output_dir)
        # write star percentile threshold
    md = MetaData()
    md.read(args.star_file)
    if not 'rlnCorrectedTomoName' in md.getLabels():    
        md.addLabels('rlnCorrectedTomoName')
        for it in md:
            md._setItemValue(it,Label('rlnCorrectedTomoName'),None)
    args.tomo_idx = idx2list(args.tomo_idx)
    for it in md:
        if args.tomo_idx is None or str(it.rlnIndex) in args.tomo_idx:
            if args.use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels():
                tomo_file = it.rlnDeconvTomoName
            else:
                tomo_file = it.rlnMicrographName
            tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]
            if os.path.isfile(tomo_file):
                tomo_out_name = '{}/{}_corrected.mrc'.format(args.output_dir,tomo_root_name)
                predict_one(args,tomo_file,model,output_file=tomo_out_name)
                md._setItemValue(it,Label('rlnCorrectedTomoName'),tomo_out_name)
        md.write(args.star_file)
    
def predict_one(args,one_tomo,model,output_file=None):
    #predict one tomogram in mrc format INPUT: mrc_file string OUTPUT: output_file(str) or <root_name>_corrected.mrc

    root_name = one_tomo.split('/')[-1].split('.')[0]

    if output_file is None:
        if os.path.isdir(args.output_file):
            output_file = args.output_file+'/'+root_name+'_corrected.mrc'
        else:
            output_file = root_name+'_corrected.mrc'

    print('predicting:{}'.format(root_name))

    with mrcfile.open(one_tomo) as mrcData:
        real_data = mrcData.data.astype(np.float32)*-1
    real_data = normalize(real_data,percentile=args.norm)
    data=np.expand_dims(real_data,axis=-1)
    reform_ins = reform3D(data)
    data = reform_ins.pad_and_crop_new(args.cube_size,args.crop_size)
    #to_predict_data_shape:(n,cropsize,cropsize,cropsize,1)
    #imposing wedge to every cubes
    #data=wedge_imposing(data)

    ngpus = len(args.gpuID.split(','))
    N = args.batch_size * ngpus * args.Ntile
    num_patches = data.shape[0]
    if num_patches%N == 0:
        append_number = 0
    else:
        append_number = N - num_patches%N
    data = np.append(data, data[0:append_number], axis = 0)

    outData = np.zeros(data.shape)
    #for count in range(data.shape[0]//N):
    #    outData[count*N:count*N+N]=model.predict(data, batch_size= args.batch_size,verbose=1)
    global_batch = data.shape[0]//args.Ntile
    for i in range(args.Ntile-1):
        outData[i*global_batch:(i+1)*global_batch] = model.predict(data[i*global_batch:(i+1)*global_batch], batch_size= args.batch_size,verbose=1)
    outData[(args.Ntile-1)*global_batch:] = model.predict(data[(args.Ntile-1)*global_batch:], batch_size= args.batch_size,verbose=1)
    # outData=model.predict(data, batch_size= args.batch_size,verbose=1)
    outData = outData[0:num_patches]

    outData=reform_ins.restore_from_cubes_new(outData.reshape(outData.shape[0:-1]), args.cube_size, args.crop_size)

    outData = normalize(outData,percentile=args.norm)
    with mrcfile.new(output_file, overwrite=True) as output_mrc:
        output_mrc.set_data(-outData)
    print('Done predicting')
    # predict(args.model,args.weight,args.mrc_file,args.output_file, cubesize=args.cubesize, cropsize=args.cropsize, batch_size=args.batch_size, gpuID=args.gpuID, if_percentile=if_percentile)