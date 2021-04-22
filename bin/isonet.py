#!/usr/bin/env python3
import fire
import logging
import os
from IsoNet.util.dict2attr import Arg,check_args,idx2list
import sys
from fire import core
import time
from IsoNet.util.metadata import MetaData,Label,Item

class ISONET:
    """
    ISONET: Train on tomograms and Predict to restore missing-wedge\n
    for detail discription, run one of the following commands:

    isonet.py prepare_star -h
    isonet.py prepare_subtomo_star -h
    isonet.py deconv -h
    isonet.py make_mask -h
    isonet.py extract -h
    isonet.py refine -h
    isonet.py predict -h

    """
    
    def prepare_star(self,folder_name, output_star='tomograms.star',pixel_size = 10.0, defocus = 0.0, number_subtomos = 100):
        """
        \nThis command generates a tomograms.star file from a folder containing only tomogram files (.mrc or .rec).\n
        isonet.py prepare_star folder_name [--output_star] [--pixel_size] [--defocus] [--number_subtomos]
        :param folder_name: (None) directory containing tomogram(s). Usually 1-5 tomograms are sufficient.
        :param output_star: (tomograms.star) star file similar to that from "relion". You can modify this file manually or with gui.
        :param pixel_size: (10) pixel size in anstroms. Usually you want to bin your tomograms to about 10A pixel size. 
        Too large or too small pixel sizes are not recommanded, since the target resolution on Z-axis should be about 30A.
        :param defocus: (0.0) defocus in Angstrom. Only need for ctf deconvolution For phase plate data, you can leave defocus 0. 
        If you have multiple tomograms with different defocus, please modify them in star file or with gui.
        :param number_subtomos: (100) Number of subtomograms to be extracted in later processes. 
        If you have multiple tomograms with different sizes or masks, you can modify them in star file or with gui, in later steps.

        """       
        md = MetaData()
        md.addLabels('rlnIndex','rlnMicrographName','rlnPixelSize','rlnDefocus','rlnNumberSubtomo')
        tomo_list = sorted(os.listdir(folder_name))
        for i,tomo in enumerate(tomo_list):
            #TODO check the folder contains only tomograms.
            it = Item()
            md.addItem(it)
            md._setItemValue(it,Label('rlnIndex'),str(i+1))
            md._setItemValue(it,Label('rlnMicrographName'),os.path.join(folder_name,tomo))
            md._setItemValue(it,Label('rlnPixelSize'),pixel_size)
            md._setItemValue(it,Label('rlnDefocus'),defocus)
            md._setItemValue(it,Label('rlnNumberSubtomo'),number_subtomos)
            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        md.write(output_star)

    def prepare_subtomo_star(self, folder_name, output_star='subtomo.star', pixel_size: float=10.0,cube_size = None):
        """
        \nThis command generates a subtomo.star file from a folder containing only subtomogram files (.mrc).
        This command is usually not necessary because "isonet.py extract" will generate this subtomo.star for you.\n
        isonet.py prepare_subtomo_star folder_name [--output_star] [--cube_size] 
        :param folder_name: (None) directory containing subtomogram(s).
        :param output_star: (subtomo.star) star file for subtomograms, will be used as input in refinement.
        :param cube_size: (None) This is the size of the cubic volumes used for training. This values should smaller than the size of subtomogram. 
        And the cube_size should be divisible by 8, eg. 32, 64. If this value isn't set, cube_size is automatically determined as int(subtomo_size / 1.5 + 1)//16 * 16
        """       
        #TODO check folder valid, logging
        if not os.path.isdir(folder_name):
            print("the folder does not exist")
        import mrcfile
        md = MetaData()
        md.addLabels('rlnSubtomoIndex','rlnImageName','rlnCubeSize','rlnCropSize','rlnPixelSize')
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
            md._setItemValue(it,Label('rlnPixelSize'),pixel_size)

            # f.write(str(i+1)+' ' + os.path.join(folder_name,tomo) + '\n')
        md.write(output_star)

    def deconv(self, star_file: str, 
        deconv_folder:str="deconv", 
        snrfalloff: float=None, 
        deconvstrength: float=None, 
        highpassnyquist: float=0.1,
        tile: tuple=(1,4,4),
        overlap_rate = 0.25,
        ncpu:int=4,
        tomo_idx: str=None):
        """
        \nCTF deconvolution for the tomograms.\n
        isonet.py deconv star_file [--deconv_folder] [--snrfalloff] [--deconvstrength] [--highpassnyquist] [--tile] [--overlap_rate] [--ncpu] [--tomo_idx]
        No need for phase plate data. IsoNet generated tomogram can hardly reach resolution beyond ctf first zero. However, this step is recommanded because it enhances low resolution information for a better visulization. 
        :param star_file: (None) Star file for tomograms.
        :param snrfalloff: (None) SNR fall rate with the frequency. High values means losing more high frequency. 
        If this value is not set, the program will look for the parameter in the star file. 
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param deconvstrength: (None) Strength of the deconvolution.  
        If this value is not set, the program will look for the parameter in the star file. 
        If this value is not set and not found in star file, the default value 1.0 will be used.
        :param highpassnyquist: (0.1) Keep this default value.
        :param tile: (1,4,4) The program crop the tomogram in multiple tiles (z,y,x) for multiprocessing and assembly them into one. e.g. (1,2,2)
        :param overlap_rate: (None) The overlapping rate for adjecent tiles.
        :param ncpu: (4) Number of cpus to use. 
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16  
        """    
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
        
        tomo_idx = idx2list(tomo_idx)
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

    def make_mask(self,star_file, 
                mask_folder: str = 'mask', 
                patch_size: int=4, 
                percentile: int=30,
                threshold: float=1.0,
                use_deconv_tomo:bool=True,
                z_crop:float=None,
                tomo_idx=None):
        """
        \ngenerate a mask to include sample area and exclude "empty" area of the tomogram. The masks do not need to be precise. In general, the number of subtomograms (a value in star file) should be lesser if you masked out larger area. \n
        isonet.py make_mask star_file [--mask_folder] [--patch_size] [--percentile] [--threshold] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: path to the tomogram or tomogram folder
        :param mask_folder: path and name of the mask to save as
        :param patch_size: (4) The size of the box from which the max-filter and std-filter are calculated. 
        It is suggested to be set close to the size of interested particles. 
        :param percentile: (30) The approximate percentage, ranging from 0 to 100, of the area of meaningful content in tomograms. 
        If this value is not set, the program will look for the parameter in the star file. 
        If this value is not set and not found in star file, the default value 99 will be used.
        :param threshold: (1.0) A factor of overall standard deviation and its default value is 1. 
        This parameter only affect the std-mask. 
        Make the threshold smaller (larger) when you want to enlarge (shrink) mask area. 
        When you don't want to use the std-mask, set the value to 0.
        If this value is not set, the program will look for the parameter in the star file. 
        If this value is not set and not found in star file, the default value 1.0 will be used.      
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead. 
        :param z_crop: If exclude the top and bottum regions of tomograms along z axis. For example, "--z_crop 0.2" will mask out the top 20% and bottum 20% region along z axis. 
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16   
        """
        #TODO the meaning of the parameter is not intuitive.
        from IsoNet.bin.make_mask import make_mask
        if not os.path.isdir(mask_folder):
            os.mkdir(mask_folder)
        # write star percentile threshold
        md = MetaData()
        md.read(star_file)
        if not 'rlnMaskPercentile' in md.getLabels():    
            md.addLabels('rlnMaskPercentile','rlnMaskThreshold','rlnMaskName')
            for it in md:
                md._setItemValue(it,Label('rlnMaskPercentile'),90)
                md._setItemValue(it,Label('rlnMaskThreshold'),0.85)
                md._setItemValue(it,Label('rlnMaskName'),None)

        tomo_idx = idx2list(tomo_idx)
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
                    mask_out_name = '{}/{}_mask.mrc'.format(mask_folder,tomo_root_name)
                    make_mask(tomo_file,
                            mask_out_name,
                            side=patch_size,
                            percentile=it.rlnMaskPercentile,
                            threshold=it.rlnMaskThreshold,
                            surface = z_crop)
                
                md._setItemValue(it,Label('rlnMaskName'),mask_out_name)
            md.write(star_file)

    def extract(self,
        star_file: str,
        use_deconv_tomo: bool = True,
        subtomo_folder: str = "subtomo",
        subtomo_star: str = "subtomo.star",
        cube_size: int = 64,
        log_level: str="info"
        ):

        """
        \nExtract subtomograms\n
        isonet.py extract star_file [--subtomo_folder] [--subtomo_star] [--cube_size] [--use_deconv_tomo] [--tomo_idx]
        :param star_file: tomogram star file
        :param subtomo_folder: (subtomo) folder for output subtomograms.
        :param subtomo_star: (subtomo.star) star file for output subtomograms.
        :param cube_size: (64) Size of cubes for training, should be divisible by 8, eg. 32, 64.
        :param log_level: ("info") level of the output, either "info" or "debug"
        The actual sizes of extracted subtomograms are 1.5 times of this value.
        :param use_deconv_tomo: (True) If CTF deconvolved tomogram is found in tomogram.star, use that tomogram instead. 
        """

        d = locals()
        d_args = Arg(d)
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
        logger = logging.getLogger('IsoNet.extract')

        if  os.path.isdir(subtomo_folder):
            logger.warning("subtomo directory exists, the current directory will be overwriten")
            import shutil
            shutil.rmtree(subtomo_folder)
        os.mkdir(subtomo_folder)

        from IsoNet.preprocessing.prepare import extract_subtomos
        d_args.crop_size = int(int(cube_size) * 1.5)
        d_args.subtomo_dir = subtomo_folder
        extract_subtomos(d_args)

    def refine(self,
        subtomo_star: str = None,
        gpuID: str = '0,1,2,3',
        iterations: int = 50,
        data_folder: str = "data",
        pretrained_model = None,
        log_level: str = "info",
        continue_iter: int = 0,

        preprocessing_ncpus: int = 16,

        epochs: int = 10,
        batch_size: int = None,
        steps_per_epoch: int = 100,

        noise_level:  float= 0.05,
        noise_start_iter: int = 15,
        noise_pause: int = 5,

        drop_out: float = 0.3,
        convs_per_depth: int = 3,
        kernel: tuple = (3,3,3),
        pool: tuple = None,
        unet_depth: int = 3,
        filter_base: int = None,
        batch_normalization: bool = False,
        normalize_percentile: bool = True,
    ):
        """
        \ntrain neural network to correct missing wedge\n
        isonet.py refine subtomo_star [--iterations] [--gpuID] [--preprocessing_ncpus] [--batch_size] [--steps_per_epoch] [--noise_start_iter] [--noise_level] [--noise_pause] ...
        :param subtomo_star: (None) star file containing subtomogram(s).
        :param gpuID: (0,1,2,3) The ID of gpu to be used during the training. e.g 0,1,2,3.
        :param pretrained_model: (None) A trained neural network model in ".h5" format to start with.
        :param iterations: (50) Number of training iterations.
        :param data_folder: (data) Temperary folder to save the generated data used for training.
        :param log_level: (info) debug level
        :param continue_iter: (0) Which iteration you want to start from?

        ************************preparation settings************************

        :param preprocessing_ncpus: (16) Number of cpu for preprocessing.

        ************************Training settings************************

        :param epochs: (10) Number of epoch for each iteraction.
        :param batch_size: (None) Size of the minibatch.If None, batch_size will be the max(2 * number_of_gpu,4). batch_size should be divisible by the number of gpu.
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
        #TODO: Test rotation list 16/20
        from IsoNet.bin.refine import run
        d = locals()
        d_args = Arg(d)
        d_args.data_dir = data_folder
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)

        logger = logging.getLogger('IsoNet.bin.refine')
        # d_args.only_extract_subtomos = False
        run(d_args)

    def predict(self, star_file: str, model: str, output_dir: str='./corrected_tomos', gpuID: str = None, cube_size:int=48,
    crop_size:int=64,use_deconv_tomo=True, batch_size:int=8,norm: bool=True,log_level: str="info",Ntile:int=1,tomo_idx=None):
        """
        \nPredict tomograms using trained model including model.json and weight(xxx.h5)\n
        isonet.py predict star_file model [--gpuID] [--output_dir] [--cube_size] [--crop_size] [--batch_size] [--tomo_idx] [--ntile]
        :param star_file: star for tomogram
        :param output_file: file_name of output predicted tomograms
        :param model: path to trained network model .h5
        :param gpuID: (0,1,2,3) The gpuID to used during the training. e.g 0,1,2,3.
        :param cube_size: (64) The tomogram is divided into cubes to predict due to the memory limitation of GPUs.
        :param crop_size: (96) The side-length of cubes cropping from tomogram in an overlapping strategy
        :param batch_size: The batch size of the cubes grouped into for network predicting
        :param norm: (True) if normalize the tomograms by percentile
        :param log_level: ("debug") level of message to be displayed
        :param Ntile: divide data into Ntile part and then predict. 
        :param tomo_idx: (None) If this value is set, process only the tomograms listed in this index. e.g. 1,2,4 or 5-10,15,16  
        :raises: AttributeError, KeyError
        """
        d = locals()
        d_args = Arg(d)
        from IsoNet.bin.predict import predict
        if d_args.log_level == "debug":
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',datefmt="%H:%M:%S",level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s, %(levelname)-8s %(message)s',datefmt="%m-%d %H:%M:%S",level=logging.INFO)
        predict(d_args)
  
    def check(self):
        from IsoNet.bin.predict import predict
        from IsoNet.bin.refine import run
        print('IsoNet --version 0.1 installed')
       
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
