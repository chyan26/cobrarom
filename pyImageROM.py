import numpy as np
import glob
import os
import time

from numpy.lib import recfunctions as rfn

import cv2
from astropy.io import fits
import sep
from astropy.table import Table, Column, Row, vstack
from astropy.io import ascii
import matplotlib.pyplot as plt


def findFiberSpot(data, sigma):
    image=data
    #m, s = np.mean(image), np.std(image)
    bkg = sep.Background(image, bw=32, bh=32, fw=3, fh=3)
    objs = sep.extract(image-bkg, sigma, err=bkg.globalrms)
    aper_radius=3
    
    # Calculate the Kron Radius
    kronrad, krflag = sep.kron_radius(image, objs['x'], objs['y'], \
        objs['a'], objs['b'], objs['theta'], aper_radius)

    r_min = 3
    use_circle = kronrad * np.sqrt(objs['a'] * objs['b'])
    cinx=np.where(use_circle <= r_min)
    einx=np.where(use_circle > r_min)

    # Calculate the equivalent of FLUX_AUTO
    flux, fluxerr, flag = sep.sum_ellipse(image, objs['x'][einx], objs['y'][einx], \
        objs['a'][einx], objs['b'][einx], objs['theta'][einx], 2.5*kronrad[einx],subpix=1)		

    cflux, cfluxerr, cflag = sep.sum_circle(image, objs['x'][cinx], objs['y'][cinx],
                                    objs['a'][cinx], subpix=1)

    # Adding half pixel to measured coordinate.  
    objs['x'] =  objs['x']+0.5
    objs['y'] =  objs['y']+0.5

    objs['flux'][einx]=flux
    objs['flux'][cinx]=cflux


    r, flag = sep.flux_radius(image, objs['x'], objs['y'], \
        aper_radius*objs['a'], 0.5,normflux=objs['flux'], subpix=5)

    flag |= krflag
    
    rfn.append_fields(objs, 'r', r)

    objects=objs[:]

    
    return objects

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print()



datapath='/Volumes/Science/20180906/'
tblist=['G1_500ms', 'G2_500ms','G3_500ms', 'all_500ms',\
        'G1_100ms', 'G2_100ms', 'G3_100ms', 'all_100ms',\
        'G1_1s', 'G2_1s', 'G3_1s', 'all_1s',\
        'G1_5s', 'G2_5s', 'G3_5s', 'all_5s',\
        'G1_10s', 'G2_10s', 'G3_10s', 'all_10s']

for tb in tblist:
    filelist=glob.glob(datapath+'/'+tb+'_*.fits')
    outputcsv='/Volumes/Disk/Data/20180906/'+tb+'.csv'
    print("Running "+tb)

    # Excludine the last file because we interrupted the last exposure.
    filelist = filelist[:-1]

    n_frame=len(filelist)
    count = 0

    frame_id=np.array([])
    x=np.array([])
    y=np.array([])


    pxx=None
    pyy=None

    for f in filelist:
        if os.stat(f).st_size != 0:
            hdu1=fits.open(f)
            img = np.flipud(hdu1[0].data[400:700,100:1300])
            pim = np.zeros([*img.shape,3])
            pim[:,:,0] = np.flipud(img)
            pim[:,:,1] = np.flipud(img)
            pim[:,:,2] = np.flipud(img)
            if count == 0:
                obj = findFiberSpot(img.astype(np.float), sigma=4.0)
                new=np.sort(obj, order='x')
                new[:] = new[::-1]
                pxx = new['x']
                pyy = new['y']
                
                txx = new['x']
                tyy = new['y']
                frame=np.zeros(len(new['x']))
            else:
                txx, tyy, tff = sep.winpos(img.astype(np.float), pxx, pyy ,3)
                pxx = txx
                pyy = tyy
                frame=np.zeros(len(txx))+count
            # new=np.sort(obj, order='x')
            # new[:] = new[::-1]
            # fid = [str(e) for e in list(range(2,54)) if e not in {39, 43}]

            img = img.astype(np.uint8)
            
            for i in range(len(txx)):
                #txx=new['x'].data[i]
                #tyy=new['y'].data[i]
                cv2.circle(pim,(int(txx[i]),int(300-tyy[i])), 5, (0,0,255), 2)
                #cv2.putText(pim,fid[i],(int(txx),300-int(tyy)), font, 0.4,(255,255,255),1,cv2.LINE_AA)

            cv2.imshow('frame',pim)
            

        
            frame_id=np.append(frame_id,frame)
            x=np.append(x,txx)
            y=np.append(y,tyy)
            #pxx = new['x'].data
            #pyy = new['y'].data
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            hdu1.close()

        count += 1
        printProgressBar(count, n_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)

    data = Table([frame_id, x, y], names=('Frame ID','X', 'Y'))
    ascii.write(data, outputcsv, format='csv', overwrite=True)  
    

    cv2.destroyAllWindows()
