import numpy as np
from numpy.lib import recfunctions as rfn

import cv2
from astropy.io import fits
import sep
from astropy.table import Table, Column, Row, vstack
from astropy.io import ascii

def findFiberSpot(data,sigma):
    image=data
    #m, s = np.mean(image), np.std(image)
    bkg = sep.Background(image, bw=32, bh=32, fw=2, fh=2)
    objs = sep.extract(image-bkg, sigma, err=bkg.globalrms)
    aper_radius=2

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
    #print(r)
    #dir(objs)
    #objs.append({'r':r})
    #objs['r']=r
    #objs['flag'][einx]=flag
    #objs['flag'][cinx]=cflag
    rfn.append_fields(objs, 'r', r)
    #index=np.logical_and(objs['a'] < 2.5,flag >= 0)

    objects=objs[:]

    #objects=objs[:][np.where(index == True)]

    
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

def sepOnAvi(avifile, csvfile, sigma=10.0):
    
    cap = cv2.VideoCapture(avifile)
    n_frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    #n_frame=20
    count = 0
    frame_id=np.array([])
    x=np.array([])
    y=np.array([])
    a=np.array([])
    b=np.array([])

    while(cap.isOpened()):
        if count != n_frame:
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = frame[900:1200,0:2500,:]
            gim = gray[900:1200,0:2500]
            obj=findFiberSpot(gim.astype(np.float), sigma=10.0)


            #img=np.array([img]).astype(np.uint8)
            #for x,y in obj['x'].data,obj['y'].data:
            for i in range(0,len(obj['x'].data)):
                xx=obj['x'].data[i]
                yy=obj['y'].data[i]
            
                cv2.circle(img,(int(xx),int(yy)), 10, (0,0,255), 2)

            cv2.imshow('frame',img)
            
            if count == 0:
                frame=np.zeros(len(obj['x']))
            else:
                frame=np.zeros(len(obj['x']))+count
            frame_id=np.append(frame_id,frame)
            x=np.append(x,obj['x'].data)
            y=np.append(y,obj['y'].data)
            a=np.append(a,obj['a'].data)
            b=np.append(b,obj['b'].data)

            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

        count += 1
        printProgressBar(count, n_frame, prefix = 'Progress:', suffix = 'Complete', length = 50)

    data = Table([frame_id, x, y, a, b], names=('Frame ID','X', 'Y', 'a', 'b'))
    ascii.write(data, csvfile, format='csv', overwrite=True)  
    
    cap.release()
    cv2.destroyAllWindows()


def main():

    datapath='/Volumes/Science/20180927/'
    logpath='/Volumes/Disk/Data/20180927/'
    figpath='/Volumes/Disk/Data/20180927/Figures'

    motor = ['theta','phi']
    interval = ['35','94','125','156','188']
    fibergroup = ['one','group','all']


    for m in motor:
        for itv in interval:
            for fg in fibergroup:
                aviname = datapath+m+'/'+'20180927_'+itv+'int_'+fg+'fiber.avi'
                outputcsv = logpath+'20180927_'+m+'_'+itv+'int_'+fg+'fiber.csv'
                sepOnAvi(aviname, outputcsv)
                print(aviname)
                print(outputcsv)


#print(__name__)

if __name__ == "__main__":
    #print(__name__)
    main()
