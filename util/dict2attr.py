class Arg:
    def __init__(self,dictionary):
        import logging
        # Do not set global basic logging 
        #logging.basicConfig(level=logging.DEBUG,format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        #datefmt='%Y-%m-%d:%H:%M:%S')
        #logger = logging.getLogger('IsoNet.util.dict2attr')
        for k, v in dictionary.items():
            # if k not in param_list:
            #     logging.warning("{} not recognized!".format(k))
            if k == 'gpuID' and type(v) is tuple:
                v = ','.join([str(i) for i in v])
            setattr(self, k, v)
            # print(k,v)
            # param_list.append(k)
        
def check_args(args):
    train_params = ['self','train','normalize_percentile', 'batch_normalization', 'filter_base', 'unet_depth', 'kernel', 'convs_per_depth', 'drop_out', 'steps_per_epoch', 'batch_size', 'epochs', 'preprocessing_ncpus', 'ncube','filter_base', 'crop_size', 'cube_size', 'noise_pause', 'noise_start_iter', 'noise_level', 'continue_iter', 'continue_training', 'log_level', 'pretrained_model', 'data_dir', 'subtomo_dir', 'datas_are_subtomos', 'iterations', 'noise_dir', 'mask_dir', 'gpuID', 'input_dir']
    predict_params = ['self', 'predict','norm', 'batch_size', 'crop_size', 'cube_size', 'gpuID', 'model', 'output_file', 'mrc_file']
    mask_param = ['tomo_path','mask_name,side','percentile','threshold']
    noise_param = ['output_folder', 'number_volume', 'cubesize', 'minangle', 'maxangle','anglestep', 'start','ncpus', 'mode']
    param_list = train_params + predict_params + mask_param + noise_param
    for i in args[2:]:
        if i not in param_list:
            pass
            # logging.warning("{} not recognized!".format(i))

def idx2str(tomo_idx):
    if tomo_idx is not None:
            if type(tomo_idx) is tuple:
                tomo_idx = list(map(str,tomo_idx))
            elif type(tomo_idx) is int:
                tomo_idx = [str(tomo_idx)]
            else:
                tomo_idx = tomo_idx.split(',')

    return tomo_idx