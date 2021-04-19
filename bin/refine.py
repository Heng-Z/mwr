import logging
from IsoNet.preprocessing.cubes import prepare_cubes
from IsoNet.preprocessing.img_processing import normalize
from IsoNet.preprocessing.prepare import prepare_first_iter,get_cubes_list
import glob
import mrcfile
import numpy as np
import glob
import os
import shutil

def run(args):
    #*******set fixed parameters*******
    args.reload_weight = True
    args.noise_mode = 1
    args.result_dir = 'results'
    args.continue_from = "training"
    args.predict_cropsize = args.crop_size
    args.predict_batch_size = args.batch_size
    args.noise_dir = None
    args.lr = 0.0004
    #args.subtomo_dir = args.result_dir + '/subtomo'
    #if args.log_level == "debug":
    #    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
    #else:
    #    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.INFO)
    logger = logging.getLogger('IsoNet.refine')
    # Specify GPU(s) to be used
    args.gpuID = str(args.gpuID)
    args.ngpus = len(args.gpuID.split(','))
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpuID
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

    if args.log_level == 'debug':
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    else:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    #import tensorflow related modules after setting environment
    from IsoNet.training.predict import predict
    from IsoNet.training.train import prepare_first_model, train_data

    if args.continue_iter == 0 or args.pretrained_model is None:
        args = prepare_first_iter(args)
        logger.info("Done preperation for the first iteration!")
        args.continue_iter = 1
        if args.pretrained_model is not None:
            args.init_model = args.pretrained_model
        else:
            args = prepare_first_model(args)
    else: #mush has pretrained model and continue_iter >0
        args.init_model = args.pretrained_model

    args.mrc_list = os.listdir(args.subtomo_dir)
    args.mrc_list = ['{}/{}'.format(args.subtomo_dir,i) for i in args.mrc_list]

    continue_from_training = not os.path.isfile('{}/model_iter{:0>2d}.h5'.format
    (args.result_dir,args.continue_iter))

    #************************************
    for num_iter in range(args.continue_iter,args.iterations):
        args.iter_count = num_iter
        logger.info("Start Iteration{}!".format(num_iter))
        if num_iter > args.continue_iter: # set the previous iteration's result model as the init_model 
            args.init_model = '{}/model_iter{:0>2d}.h5'.format(args.result_dir,args.iter_count-1)
        args.noise_factor = ((num_iter - args.noise_start_iter)//args.noise_pause)+1 if num_iter >= args.noise_start_iter else 0
        logging.info("noise_factor:{}".format(args.noise_factor))

        if continue_from_training:
            try:
                shutil.rmtree(args.data_dir)
            except OSError:
                pass
                # logging.debug("No previous data folder!")
            get_cubes_list(args)
            logger.info("Done preparing subtomograms!")
            logger.info("Start training!")
            history = train_data(args) #train based on init model and save new one as model_iter{num_iter}.h5
            # losses.append(history.history['loss'][-1])
            logger.info("Done training!")
            logger.info("Start predicting subtomograms!")
            predict(args)
            logger.info("Done predicting subtomograms!")
        
        else:
            logger.info("Model for iteration {} exists".format(args.continue_iter))
            logger.info("Start cube predicting!")
            predict(args)
            logger.info("Done cube predicting!")
            continue_from_training = True

        logger.info("Done Iteration{}!".format(num_iter))

if __name__ == "__main__":
    from IsoNet.util.dict2attr import Arg
    arg = {'input_dir': 'subtomo/', 'gpuID': '4,5,6,7', 
    'mask_dir': None, 'noise_dir': None, 'iterations': 50, 
    'data_dir': 'data', 'pretrained_model': './results/model_iter35.h5', 
    'log_level': 'debug', 'continue_training': False, 
    'continue_iter': 36, 'noise_mode': 1, 'noise_level': 0.05, 
    'noise_start_iter': 15, 'noise_pause': 5, 'cube_size': 64, 
    'crop_size': 96, 'ncube': 1, 'preprocessing_ncpus': 16, 
    'epochs': 10, 'batch_size': 8, 'steps_per_epoch': 150, 
    'drop_out': 0.3, 'convs_per_depth': 3, 'kernel': (3, 3, 3),
     'unet_depth': 3, 'filter_base': 32, 'batch_normalization': False, 
     'normalize_percentile': True}
    d_args = Arg(arg)
    print(mrcfile.__file__)
    run(d_args)
