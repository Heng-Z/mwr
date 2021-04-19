import numpy as np
import cupy as cp
from multiprocessing import Pool
from functools import partial
def tom_ctf1d(pixelsize, voltage, cs, defocus, amplitude, phaseshift, bfactor, length=2048):

    ny = 1 / pixelsize


    lambda1 = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda1 * 2


    points = np.arange(0,length)
    points = points.astype(np.float)
    points = points/(2 * length)*ny

    k2 = points**2;
    term1 = lambda1**3 * cs * k2**2

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)


    return (pcurve + acurve)*bfactor

def wiener1d(angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass)

    snr = np.exp(-data * snrfalloff * 100 / angpix) * (10^(3 * deconvstrength)) * highpass
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0);
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr)
    return ctf, wiener

def tom_deconv_tomo(inp, angpix, defocus, snrfalloff, deconvstrength, highpassnyquist, phaseflipped, phaseshift):
    #inp is a list containing ndarray and gpuID
    vol = inp[0]
    gpuID = inp[1]
    data = np.arange(0,1+1/2047.,1/2047.)
    highpass = np.minimum(np.ones(data.shape[0]), data/highpassnyquist) * np.pi
    highpass = 1-np.cos(highpass);
    eps = 1e-10
    snr = np.exp(-data * snrfalloff * 100 / angpix) * np.power(10.0,(3.0 * deconvstrength)) * highpass + eps 
    #snr[0] = -1
    ctf = tom_ctf1d(angpix*1e-10, 300e3, 2.7e-3, -defocus*1e-6, 0.07, phaseshift / 180 * np.pi, 0)
    if phaseflipped:
        ctf = abs(ctf)

    wiener = ctf/(ctf*ctf+1/snr);

    denom = ctf*ctf+1/snr
    #np.savetxt('den.txt',denom)
    #np.savetxt('snr.txt',snr)
    #np.savetxt('hipass.txt',highpass)
    #np.savetxt('ctf.txt',ctf)
    #np.savetxt('wiener.txt',wiener, fmt='%f')

    s1 = - int(np.shape(vol)[1] / 2)
    f1 = s1 + np.shape(vol)[1] - 1
    m1 = np.arange(s1,f1+1)     

    s2 = - int(np.shape(vol)[0] / 2)
    f2 = s2 + np.shape(vol)[0] - 1
    m2 = np.arange(s2,f2+1)     

    s3 = - int(np.shape(vol)[2] / 2)
    f3 = s3 + np.shape(vol)[2] - 1
    m3 = np.arange(s3,f3+1)     

#s3 = -floor(size(vol,3)/2);
#f3 = s3 + size(vol,3) - 1;
    x, y, z = np.meshgrid(m1,m2,m3);
    print(np.shape(x))
    x = x.astype(np.float32) / np.abs(s1);
    y = y.astype(np.float32) / np.abs(s2);
    z = z.astype(np.float32) / np.maximum(1, np.abs(s3));
    #z = z.astype(float) / np.abs(s3);
    r = np.sqrt(x**2+y**2+z**2)
    del x,y,z
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)

    #x = 0:1/2047:1;
    print(data.shape, wiener.shape, r.shape)
    ramp = np.interp(r, data,wiener).astype(np.float32);
    del r
    #ramp = np.interp(data,wiener,r);
    with cp.cuda.Device(gpuID):
        deconv = cp.real(cp.fft.ifftn(cp.fft.fftn(cp.asarray(vol)) * cp.asarray(ramp)))
        out = cp.asnumpy(deconv).astype(np.float32)
    #return real(ifftn(fftn(single(vol)).*ramp));
    return out

class Chunks:
    def __init__(self,num=2,overlap=0.25):
        self.overlap = overlap
        #num can be either int or tuple
        if type(num) is int:
            self.num = (num,num,num)
        else:
            self.num = num

    def get_chunks(self,vol):
        #side*(1-overlap)*(num-1)+side = sp + side*overlap -> side *(1-overlap) * num = side
        
        cube_size = np.round(np.array(vol.shape)/((1-self.overlap)*np.array(self.num))).astype(np.int16)
        overlap_len = np.round(cube_size*self.overlap).astype(np.int16)
        overlap_len = overlap_len + overlap_len %2
        eff_len = cube_size-overlap_len
        padded_vol = np.pad(vol,pad_width=[(ol//2,ol//2) for ol in overlap_len],mode='symmetric')
        sp = padded_vol.shape
        chunks_list = []
        slice1 = [(i*eff_len[0],i*eff_len[0]+cube_size[0]) for i in range(self.num[0]-1)]
        slice1.append(((self.num[0]-1)*eff_len[0],sp[0]))
        slice2 = [(i*eff_len[1],i*eff_len[1]+cube_size[1]) for i in range(self.num[1]-1)]
        slice2.append(((self.num[1]-1)*eff_len[1],sp[1]))
        slice3 = [(i*eff_len[2],i*eff_len[2]+cube_size[2]) for i in range(self.num[2]-1)]
        slice3.append(((self.num[2]-1)*eff_len[2],sp[2]))
        # print(slice1)
        # print(slice2)
        # print(slice3)
        for i in slice1:
            for j in slice2:
                for k in slice3:
                    one_chunk = padded_vol[i[0]:i[1],j[0]:j[1],k[0]:k[1]]
                    chunks_list.append(one_chunk)
        self.shape = vol.shape
        self.padded_shape = sp
        self.slice1 = slice1
        self.slice2 = slice2
        self.slice3 = slice3
        self.overlap_len = overlap_len
        return chunks_list

    def restore(self,new_list):
        overlap_len = self.overlap_len
        new_vol = np.zeros(self.padded_shape,dtype=type(new_list[0][0,0,0]))
        factor_vol = np.zeros(self.padded_shape,dtype=type(new_list[0][0,0,0]))
        for n1,i in enumerate(self.slice1):
            for n2,j in enumerate(self.slice2):
                for n3,k in enumerate(self.slice3):
                    print(n1,n2,n3)
                    one_chuck = new_list[n1*len(self.slice2)*len(self.slice3)+n2*len(self.slice3)+n3]
                    print(one_chuck[overlap_len[0]//2:-(overlap_len[0]//2),overlap_len[1]//2:-(overlap_len[1]//2),overlap_len[2]//2:-(overlap_len[2]//2)].shape)
                    print(one_chuck.shape)
                    new_vol[i[0]+overlap_len[0]//2:i[1]-overlap_len[0]//2,j[0]+overlap_len[1]//2:j[1]-overlap_len[1]//2, 
                    k[0]+overlap_len[2]//2:k[1]-overlap_len[2]//2] = one_chuck[overlap_len[0]//2:-(overlap_len[0]//2),overlap_len[1]//2:-(overlap_len[1]//2),overlap_len[2]//2:-(overlap_len[2]//2)]
                    # factor_vol[i[0]:i[1],j[0]:j[1],k[0]:k[1]] += np.ones(factor_vol[i[0]:i[1],j[0]:j[1],k[0]:k[1]].shape)
        
         
        # return np.multiply(new_vol,1/factor_vol)
        return new_vol[overlap_len[0]//2:-(overlap_len[0]//2),
                    overlap_len[1]//2:-(overlap_len[1]//2),
                    overlap_len[2]//2:-(overlap_len[2]//2)]

                    
if __name__ == '__main__':
    import mrcfile
    import sys
    import time
    start = time.time()
    args = sys.argv
    mrcFile = args[1]
    outFile = args[2]
    defocus = float(args[3]) #CTFFIND
    pixsize = float(args[4])
    snrfalloff = float(args[5]) # Recommand: 1
    num_cpu = int(args[6])
    deconvstrength = 1
    
    with mrcfile.open(mrcFile) as mrc:
        vol = mrc.data
    c = Chunks(num=(1,4,4),overlap=0.25)
    chunks_list = c.get_chunks(vol)
    chunks_gpu_num_list = [[array,j%num_cpu] for j,array in enumerate(chunks_list)]
    with Pool(num_cpu) as p:
        partial_func = partial(tom_deconv_tomo,angpix=pixsize, defocus=defocus, snrfalloff=snrfalloff, 
        deconvstrength=deconvstrength, highpassnyquist=0.1, phaseflipped=False, phaseshift=0 )
        results = p.map(partial_func,chunks_gpu_num_list,chunksize=1)
    chunks_deconv_list = list(results)
    # for i in chunks_list:
    #     result = tom_deconv_tomo(i, angpix=pixsize, defocus=defocus, snrfalloff=snrfalloff, deconvstrength=deconvstrength, highpassnyquist=0.1, phaseflipped=False, phaseshift=0 )
    #     chunks_deconv_list.append(result)
    vol_restored = c.restore(chunks_deconv_list)
    outname = outFile.split('.')[0] +'-snr{}-strnth{}.rec'.format(str(snrfalloff),str(deconvstrength))
    print("out_name: ",outname)
    # result = tom_deconv_tomo(vol, angpix=pixsize, defocus=defocus, snrfalloff=snrfalloff, deconvstrength=deconvstrength, highpassnyquist=0.1, phaseflipped=False, phaseshift=0 )
    with mrcfile.new(outname, overwrite=True) as mrc:
        mrc.set_data(vol_restored)
    end = time.time()
    print('excution time: ',end-start)
    #np.savetxt('plot.txt', res, type=float)
    
#####test for tuning ctf deconv #######
'''
    import matplotlib.pyplot as plt
    ctf53,winer53 = wiener1d(5.0, 3.0, 1, 1,highpassnyquist=0.1, phaseflipped=False, phaseshift=0  )
    fig,axs = plt.subplots(3)
    fig.suptitle('pixle 5.0 defocus 3.0 use defocus 1.0 wiener') 
    axs[0].plot(data,ctf53) 
    axs[1].plot(data,winer51) 
    axs[2].plot(data,ctf53*winer51) 
    plt.show()
'''
    
    
'''
cupy code buckup:

    s1 = - int(np.shape(vol)[1] / 2)
    f1 = s1 + np.shape(vol)[1] - 1
    m1 = cp.arange(s1,f1+1)     

    s2 = - int(np.shape(vol)[0] / 2)
    f2 = s2 + np.shape(vol)[0] - 1
    m2 = cp.arange(s2,f2+1)     

    s3 = - int(np.shape(vol)[2] / 2)
    f3 = s3 + np.shape(vol)[2] - 1
    m3 = cp.arange(s3,f3+1)     

#s3 = -floor(size(vol,3)/2);
#f3 = s3 + size(vol,3) - 1;
    x, y, z = cp.meshgrid(m1,m2,m3);
    print(x.shape)
    x = x.astype(cp.float32) / cp.abs(s1);
    y = y.astype(cp.float32) / cp.abs(s2);
    z = z.astype(cp.float32) / cp.maximum(1, cp.abs(s3));
    #z = z.astype(float) / np.abs(s3);
    r = cp.sqrt(x**2+y**2+z**2);
    del x,y,z
    r = cp.minimum(1, r);
    r = cp.fft.ifftshift(r);

    #x = 0:1/2047:1;
    print(data.shape, wiener.shape, r.shape)
    ramp = cp.interp(r, cp.asarray(data),cp.asarray(wiener)).astype(cp.float32);
    del r
    #ramp = np.interp(data,wiener,r);
    deconv = cp.real(cp.fft.ifftn(cp.fft.fftn(cp.asarray(vol)) * ramp))

    #return real(ifftn(fftn(single(vol)).*ramp));
    return cp.asnumpy(deconv).astype(np.float32)
    '''
