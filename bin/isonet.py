#!/usr/bin/env python3
import fire
import logging
import os
from IsoNet.util.dict2attr import Arg,check_args
import sys
from fire import core
import time
from IsoNet.util.metadata import MetaData,Label,Item

class ISONET:
    """
    ISONET: Train on tomograms and Predict to restore missing-wedge\n
    please run one of the following commands:
    isonet.py deconv
    isonet.py make_mask
    isonet.py refine
    isonet.py predict
    """
    def refine(self,
        subtomo_star: str = None,
        gpuID: str = '0,1,2,3',
        iterations: int = 50,
        data_dir: str = "data",
        pretrained_model = None,
        log_level: str = "info",
        continue_iter: int = 0,

        # cube_size: int = 64,
        # crop_size: int = 96,
        # ncube: int = 1,
        preprocessing_ncpus: int = 16,

        epochs: int = 10,
        batch_size: int = 8,
        steps_per_epoch: int = 100,

        noise_level:  float= 0.05,
        noise_start_iter: int = 15,
        noise_pause: int = 5,

        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = 64,
        batch_normalization: bool = False,
        normalize_percentile: bool = True,
    ):
        """
        Extract subtomogram and train neural network to correct missing wedge on generated subtomos
        :param input_dir: (None) directory containing tomogram(s) from which subtomos are extracted; format: .mrc or .rec
        :param mask_dir: (None) folder containing mask files, Eash mask file corresponds to one tomogram file, usually basename-mask.mrc
        :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
        :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
        :param iterations: (50) Number of training iterations.
        :param data_dir: (data) Temperary folder to save the generated data used for training.
        :param log_level: (info) debug level
        :param continue_iter: (0) Which iteration you want to start from?

        ************************Subtomo extraction settings************************

        :param cube_size: (64) Size of training cubes, this size should be divisible by 2^unet_depth.
        :param crop_size: (96) Size of cubes to impose missing wedge. Should be same or larger than size of cubes. Recommend 1.5 times of cube size
        :param ncube: (1) Number of cubes generated for each tomogram. Because each sampled subtomogram rotates 16 times, the actual number of subtomograms for trainings is ncube*16.
        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************Training settings************************

        :param epochs: (10) Number of epoch for each iteraction.
        :param batch_size: (8) Size of the minibatch.
        :param steps_per_epoch: (100) Step per epoch. A good estimation of this value is tomograms * ncube * 16 / batch_size *0.9.")

        ************************Denoise settings************************

        :param noise_level: (0.05) Level of noise STD(added noise)/STD(data) to start with. Set zero to disable noise reduction.
        :param noise_start_iter: (15) Iteration that start to add trainning noise.
        :param noise_pause: (5) Iters trainning noise remain at one level. The noise_level in each iteraion is defined as (((num_iter - noise_start_iter)//noise_pause)+1)*noise_level

        ************************Network settings************************

        :param drop_out: (0.3) Drop out rate to reduce overfitting.
        :param convs_per_depth: (3) Number of convolution layer for each depth.
        :param kernel: (3,3,3) Kernel for convolution
        :param unet_depth: (3) Number of convolution layer for each depth.
        :param filter_base: (64) The base number of channels after convolution
        :param batch_normalization: (False) Sometimes batch normalization may induce artifacts for extreme pixels in the first several iterations. Those could be restored in further iterations.
        :param normalize_percentile: (True) Normalize the 5 percent and 95 percent pixel intensity to 0 and 1 respectively. If this is set to False, normalize the input to 0 mean and 1 standard dievation.

        Typical training strategy:
        1. Train tomo with no pretrained model
        2. Continue train with previous interupted model
        3. Continue train with pre-trained model
        """

        from IsoNet.bin.refine import run
        d = locals()
        d_args = Arg(d)
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)

        logger = logging.getLogger('IsoNet.bin.refine')
        # d_args.only_extract_subtomos = False
        run(d_args)

    def predict(self, mrc_file: str, output_file: str, model: str, gpuID: str = None, cube_size:int=48,crop_size:int=64, batch_size:int=8,norm: bool=True,log_level: str="debug",Ntile:int=1):
        """
        Predict tomograms using trained model including model.json and weight(xxx.h5)
        :param mrc_file: path to tomogram, format: .mrc or .rec
        :param output_file: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping strategy
        :param batch_size: The batch size of the cubes grouped into for network predicting
        :param norm: (True) if normalize the tomograms by percentile
        :param log_level: ("debug") level of message to be displayed
        :param Ntile: divide data into Ntile part and then predict. 
        :raises: AttributeError, KeyError
        """
        from IsoNet.bin.predict import predict
        d = locals()

        d_args = Arg(d)
        predict(d_args)

    def make_mask(self,star_file,
        mask_path: str = 'mask',
        side: int=8,
        percentile: int=None,
        threshold: int=None,
        use_deconv_tomo: bool=True,
        mask_type: str="statistical",

        tomo_idx: str=None):
        """
        generate a mask to constrain sampling area of the tomogram
        :param tomo_path: path to the tomogram or tomogram folder
        :param mask_path: path and name of the mask to save as
        :param side: (8) The size of the box from which the max-filter and std-filter are calculated. *side* is suggested to be set close to the size of interested particles
        :param percentile: (99) The approximate percentage, ranging from 0 to 100, of the area of meaningful content in tomograms. 
        :param threshold: (1) A factor of overall standard deviation and its default value is 1. This parameter only affect the std-mask. Make the threshold smaller (larger) when you want to enlarge (shrink) mask area. When you don't want to use the std-mask, set the value to 0.
        :param mask_type: 'statistical' or 'surface': Masks can be generated based on the statistics or just take the middle part of tomograms
        """
        from IsoNet.bin.make_mask import make_mask,make_mask_dir
        if not os.path.isdir(mask_path):
            os.mkdir(mask_path)
        # write star percentile threshold
        md = MetaData()
        md.read(star_file)
        if not 'rlnMaskPercentile' in md.getLabels():    
            md.addLabels('rlnMaskPercentile','rlnMaskThreshold','rlnMaskName')
            for it in md:
                md._setItemValue(it,Label('rlnMaskPercentile'),90)
                md._setItemValue(it,Label('rlnMaskThreshold'),0.85)
                md._setItemValue(it,Label('rlnMaskName'),None)

        if tomo_idx is not None:
            if type(tomo_idx) is tuple:
                tomo_idx = list(map(str,tomo_idx))
            elif type(tomo_idx) is int:
                tomo_idx = [str(tomo_idx)]
            else:
                tomo_idx = tomo_idx.split(',')
        for it in md:
            if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                if percentile is not None:
                    md._setItemValue(it,Label('rlnMaskPercentile'),percentile)
                if threshold is not None:
                    md._setItemValue(it,Label('rlnMaskThreshold'),threshold)
                if use_deconv_tomo and "rlnDeconvTomoName" in md.getLabels():
                    tomo_file = it.rlnDeconvTomoName
                else:
                    tomo_file = it.rlnMicrographName
                tomo_root_name = os.path.splitext(os.path.basename(tomo_file))[0]

                if os.path.isfile(tomo_file):
                    mask_out_name = '{}/{}_mask.mrc'.format(mask_path,tomo_root_name)
                    make_mask(tomo_file,
                            mask_out_name,
                            side=side,
                            percentile=it.rlnMaskPercentile,
                            threshold=it.rlnMaskThreshold,
                            mask_type=mask_type)
                
                md._setItemValue(it,Label('rlnMaskName'),mask_out_name)
        md.write(star_file)

    def check(self):
        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        print('IsoNet --version 0.9.9 installed')

    def generate_command(self, tomo_dir: str, mask_dir: str=None, ncpu: int=10, gpu_memory: int=10, ngpu: int=4, pixel_size: float=10, also_denoise: bool=True):
        """
        \nGenerate recommanded parameters for "isonet.py refine" for users\n
        Only print command, not run it.
        :param input_dir: (None) directory containing tomogram(s) from which subtomos are extracted; format: .mrc or .rec
        :param mask_dir: (None) folder containing mask files, Eash mask file corresponds to one tomogram file, usually basename-mask.mrc
        :param ncpu: (10) number of avaliable cpu cores
        :param ngpu: (4) number of avaliable gpu cards
        :param gpu_memory: (10) memory of each gpu
        :param pixel_size: (10) pixel size in anstroms
        :param: also_denoise: (True) Preform denoising after 15 iterations when set true
        """
        import mrcfile
        import numpy as np
        s="isonet.py refine --input_dir {} ".format(tomo_dir)
        if mask_dir is not None:
            s+="--mask_dir {} ".format(mask_dir)
            m=os.listdir(mask_dir)
            with mrcfile.open(mask_dir+"/"+m[0]) as mrcData:
                mask_data = mrcData.data
            # vsize=np.count_nonzero(mask_data)
        else:
            m=os.listdir(tomo_dir)
            with mrcfile.open(tomo_dir+"/"+m[0]) as mrcData:
                tomo_data = mrcData.data
            sh=tomo_data.shape
            mask_data = np.ones(sh)
        num_tomo = len(m)

        s+="--preprocessing_ncpus {} ".format(ncpu)
        s+="--gpuID "
        for i in range(ngpu-1):
            s+=str(i)
            s+=","
        s+=str(ngpu-1)
        s+=" "
        if pixel_size < 15.0:
            filter_base = 64
            s+="--filter_base 64 "
        else:
            filter_base = 32
            s+="--filter_base 32"
#        if ngpu < 6:
#            batch_size = 2 * ngpu
#            s+="--batch_size {} ".format(batch_size)
        # elif ngpu == 3:
        #     batch_size = 6
        #     s+="--batch_size 6 "
 #       else:
        batch_size = (int(ngpu/7.0)+1) * ngpu
        s+="--batch_size {} ".format(ngpu)
        if filter_base==64:
            cube_size = int((gpu_memory/(batch_size/ngpu)) ** (1/3.0) *40 /16)*16
        elif filter_base ==32:
            cube_size = int((gpu_memory*3/(batch_size/ngpu)) ** (1/3.0) *40 /16)*16

        if cube_size == 0:
            print("Please use larger memory GPU or use more GPUs")

        s+="--cube_size {} --crop_size {} ".format(cube_size, int(cube_size*1.5))

        # num_per_tomo = int(vsize/(cube_size**3) * 0.5)
        from IsoNet.preprocessing.cubes import mask_mesh_seeds
        num_per_tomo = len(mask_mesh_seeds(mask_data,cube_size,threshold=0.1))
        s+="--ncube {} ".format(num_per_tomo)

        num_particles = int(num_per_tomo * num_tomo * 16 * 0.9)
        s+="--epochs 10 --steps_per_epoch {} ".format(int(num_particles/batch_size*0.4))

        if also_denoise:
            s+="--iterations 40 --noise_level 0.05 --noise_start_iter 15 --noise_pause 3"
        else:
            s+="--iterations 15 --noise_level 0 --noise_start_iter 100"
        print(s)

    
    def deconv(self, star_file: str, 
        deconv_folder:str="deconv", 
        snrfalloff: float=None, 
        deconvstrength: float=None, 
        highpassnyquist: float=0.1,
        tile: tuple=(1,4,4),
        ncpu:int=4,
        tomo_idx: str=None):
        from IsoNet.util.deconvolution import deconv_one
        md = MetaData()
        md.read(star_file)
        if not 'rlnSnrFalloff' in md.getLabels():
            md.addLabels('rlnSnrFalloff','rlnDeconvStrength','rlnDeconvTomoName')
            for it in md:
                md._setItemValue(it,Label('rlnSnrFalloff'),1.0)
                md._setItemValue(it,Label('rlnDeconvStrength'),1.0)
                md._setItemValue(it,Label('rlnDeconvTomoName'),None)

        if not os.path.isdir(deconv_folder):
            os.mkdir(deconv_folder)

        if tomo_idx is not None:
            if type(tomo_idx) is tuple:
                tomo_idx = list(map(str,tomo_idx))
            elif type(tomo_idx) is int:
                tomo_idx = [str(tomo_idx)]
            else:
                tomo_idx = tomo_idx.split(',')

        for it in md:
            if tomo_idx is None or str(it.rlnIndex) in tomo_idx:
                if snrfalloff is not None:
                    md._setItemValue(it,Label('rlnSnrFalloff'), snrfalloff)
                if deconvstrength is not None:
                    md._setItemValue(it,Label('rlnDeconvStrength'),deconvstrength)
                if (it.rlnDeconvTomoName is None) or (it.rlnDeconvTomoName == "None"):
                    tomo_file = it.rlnMicrographName
                    base_name = os.path.basename(tomo_file)                                        
                    deconv_tomo_name = '{}/{}'.format(deconv_folder,base_name)
                else:
                    deconv_tomo_name = it.rlnDeconvTomoName
                deconv_one(it.rlnMicrographName,deconv_tomo_name,defocus=it.rlnDefocus/10000.0, pixel_size=it.rlnPixelSize,snrfalloff=it.rlnSnrFalloff, deconvstrength=it.rlnDeconvStrength,highpassnyquist=highpassnyquist,tile=tile,ncpu=ncpu)
                md._setItemValue(it,Label('rlnDeconvTomoName'),deconv_tomo_name)
        md.write(star_file)

    def prepare_star(self,folder_name,output_star='tomograms.star',pixel_size = 10.0, defocus = 0.0, number_subtomos = 100):
        """
        \nGenerate recommanded parameters for "isonet.py refine" for users\n
        if is phase plate, keep defocus 0.0 if defocus different change manually in the output tomogram.star
        Only print command, not run it.
        :param input_dir: (None) directory containing tomogram(s) from which subtomos are extracted; format: .mrc or .rec
        :param mask_dir: (None) folder containing mask files, Eash mask file corresponds to one tomogram file, usually basename-mask.mrc
        :param ncpu: (10) number of avaliable cpu cores
        :param ngpu: (4) number of avaliable gpu cards
        :param gpu_memory: (10) memory of each gpu
        :param pixel_size: (10) pixel size in anstroms
        :param: also_denoise: (True) Preform denoising after 15 iterations when set true
        """       
        md = MetaData()
        md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo')
        tomo_list = sorted(os.listdir(folder_name))
        for i,tomo in enumerate(tomo_list):
            it = Item()
            md.addItem(it)
            md._setItemValue(it,Label('rlnIndex'),str(i+1))
            md._setItemValue(it,Label('rlnMicrographName'),os.path.join(folder_name,tomo))
            md._setItemValue(it,Label('rlnPixelSize'),pixel_size)
            md._setItemValue(it,Label('rlnDefocus'),defocus)
            md._setItemValue(it,Label('rlnNumberSubtomo'),number_subtomos)
            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        md.write(output_star)

    def prepare_subtomo_star(self, folder_name, output_star='subtomo.star', cube_size = None):
        """
        \nGenerate recommanded parameters for "isonet.py refine" for users\n
        if is phase plate, keep defocus 0.0 if defocus different change manually in the output tomogram.star
        Only print command, not run it.
        :param input_dir: (None) directory containing tomogram(s) from which subtomos are extracted; format: .mrc or .rec
        :param mask_dir: (None) folder containing mask files, Eash mask file corresponds to one tomogram file, usually basename-mask.mrc
        :param ncpu: (10) number of avaliable cpu cores
        :param ngpu: (4) number of avaliable gpu cards
        :param gpu_memory: (10) memory of each gpu
        :param pixel_size: (10) pixel size in anstroms
        :param: also_denoise: (True) Preform denoising after 15 iterations when set true
        """       
        #TODO check folder valid
        if not os.path.isdir(folder_name):
            print("the folder does not exist")
        import mrcfile
        md = MetaData()
        md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize')
        subtomo_list = sorted(os.listdir(folder_name))
        for i,subtomo in enumerate(subtomo_list):
            subtomo_name = os.path.join(folder_name,subtomo)
            try: 
                with mrcfile.open(subtomo_name, mode='r') as s:
                    crop_size = s.header.nx
            except:
                print("Warning: Can not process the subtomogram: {}!".format(subtomo_name))
                continue
            if cube_size is not None:
                cube_size = int(cube_size)
                if cube_size >= crop_size:
                    cube_size = int(crop_size / 1.5 + 1)//16 * 16
                    print("Warning: Cube size should be smaller than the size of subtomogram volume! Using cube size {}!".format(cube_size))
            else:
                cube_size = int(crop_size / 1.5 + 1)//16 * 16
            it = Item()
            md.addItem(it)
            md._setItemValue(it,Label('rlnSubtomoIndex'),str(i+1))
            md._setItemValue(it,Label('rlnImageName'),subtomo_name)
            md._setItemValue(it,Label('rlnCubeSize'),cube_size)
            md._setItemValue(it,Label('rlnCropSize'),crop_size)

            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        md.write(output_star)

    def extract(self,
        star_file: str = None,
        use_deconv_tomo: bool = True,
        subtomo_dir: str = "subtomo",
        subtomo_star: str = "subtomo.star",
        cube_size: int = 64,
        log_level: str="info"
        ):

        """
        \nExtract subtomograms\n
        :param input_dir: (None) directory containing tomogram(s) from which subtomos are extracted; format: .mrc or .rec
        :param mask_dir: (None) folder containing mask files, Eash mask file corresponds to one tomogram file, usually basename-mask.mrc
        :param crop_size: (96) Size of cubes to impose missing wedge. Should be same or larger than size of cubes. Recommend 1.5 times of cube size
        :param ncube: (1) Number of cubes generated for each tomogram. Because each sampled subtomogram rotates 16 times, the actual number of subtomograms for trainings is ncube*16.
        """

        d = locals()
        d_args = Arg(d)
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
        logger = logging.getLogger('IsoNet.extract')

        if  os.path.isdir(subtomo_dir):
            logger.warning("subtomo directory exists, the current directory will be overwriten")
            import shutil
            shutil.rmtree(subtomo_dir)
        os.mkdir(subtomo_dir)

        from IsoNet.preprocessing.prepare import extract_subtomos
        d_args.crop_size = int(int(cube_size) * 1.5)
        d_args.subtomo_dir = subtomo_dir
        extract_subtomos(d_args)

        

    def gui(self):
        import IsoNet.gui.Isonet_app as app
        app.main()

def Display(lines, out):
    text = "\n".join(lines) + "\n"
    out.write(text)

def pool_process(p_func,chunks_list,ncpu):
    from multiprocessing import Pool
    with Pool(ncpu,maxtasksperchild=1000) as p:
        # results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
        results = list(p.map(p_func,chunks_list))
    # return results
    
if __name__ == "__main__":
    core.Display = Display
    fire.Fire(ISONET)
