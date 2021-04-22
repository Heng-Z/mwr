def generate_command(star_file: str, mask_dir: str=None, ncpu: int=10, gpu_memory: int=10, ngpu: int=4, pixel_size: float=10, also_denoise: bool=True):
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

        return 