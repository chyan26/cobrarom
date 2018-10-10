import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from numpy.lib import recfunctions as rfn
import math
import xml.etree.cElementTree as ET
import sep
import cv2
import pandas as pd
import glob
import random

from scipy import optimize, stats
from astropy.table import Table, Column, Row, vstack
from astropy.io import ascii

from bokeh.io import output_notebook, show, export_png
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, Title
from bokeh.models.glyphs import Text

hover = HoverTool(tooltips=[
                ("index", "$index"),
                ("data (x,y)", "($x, $y)"),
                ("canvas (x,y)", "($sx, $sy)"),
                ("value", "@image"),
                ])

TOOLS = ['pan','box_zoom','wheel_zoom', 'save' ,'reset',hover]
N = 4000
x = np.random.random(size=N) * 100
y = np.random.random(size=N) * 100
colors = [
    "#%02x%02x%02x" % (int(r), int(g), 150) for r, g in zip(50+2*x, 30+2*y)
]



def rotate(x, y, rad):
    nx = math.cos(rad)*x - math.sin(rad)*y
    ny = math.sin(rad)*x + math.cos(rad)*y
    return nx, ny

def calc_R(x,y, xc, yc):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((x-xc)**2 + (y-yc)**2)

def f(c, x, y):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_R(x, y, *c)
    return Ri - Ri.mean()

def leastsq_circle(x,y):
    # coordinates of the barycenter
    x_m = np.mean(x)
    y_m = np.mean(y)
    center_estimate = x_m, y_m
    center, ier = optimize.leastsq(f, center_estimate, args=(x,y))
    xc, yc = center
    Ri       = calc_R(x, y, *center)
    R        = Ri.mean()
    residu   = np.sum((Ri - R)**2)
    return xc, yc, R, residu

def polar(x, y):
    theta = np.arctan2(y, x)* 180 / np.pi
    rho = np.hypot(x, y)
    ind=np.where(theta<0)
    
    theta[ind]=theta[ind]+360
    return theta, rho

def findCenter(xp, yp):
    xc1,yc1,rbest1,res1=leastsq_circle(xp,yp)
    return xc1, yc1, rbest1

def rotate(x, y, rad):
    nx = math.cos(rad)*x - math.sin(rad)*y
    ny = math.sin(rad)*x + math.cos(rad)*y
    return nx, ny

def distance(a, b):
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    return np.sqrt(dx**2 + dy**2)


def groupROMdata(inputcsv, outputcsv, fnoid):
    
    imageYsize = 300
    imageXsize = 2500 


    # g1fid=np.array([])
    # g2fid=np.array([])
    # g3fid=np.array([])
    # oddfid = np.array([])
    # evenfid = np.array([])
    # allfid = np.array([])

    # allfid = [e for e in list(range(2,57)) if e not in {39, 43}]
    # for i in allfid:
    #     if i % 3 == 1:
    #         g1fid=np.append(g1fid,i)
    #     if i % 3 == 2:
    #         g2fid=np.append(g2fid,i)
    #     if i % 3 == 0:
    #         #g3fid=np.append(g3fid,i)
    #         #g3fid = [ 3. , 6.,  9., 12., 15., 21., 24., 27., 30., 33., 36., 42., 45., 48., 51., 54.]
    #     if i % 2 == 0:
    #         evenfid = np.append(evenfid,i)
    #     if i % 2 == 1:
    #         oddfid = np.append(oddfid,i)

    #onefid = [21]
    # print(g3fid)

    #fnoid=onefid
    #for tinx, tb in enumerate(tablelist):
        # if tinx==0: fnoid=onefid
        # if tinx==1: fnoid=g3fid
        # if tinx==2: fnoid=oddfid
        #if tinx==3: fnoid=allfid

    tablefile=inputcsv
    output = outputcsv


    groupTable = Table.read(tablefile, format='csv')
    #output = 'g3_10s_clean.csv'

    fid = np.array([])
    xx = np.array([])
    yy = np.array([])
    xc = np.array([])
    yc = np.array([])
    r = np.array([])
    fno = np.array([])

    H1, xedges, yedges = np.histogram2d(groupTable['Y'].data, groupTable['X'].data, \
                bins=(imageYsize, imageXsize), range=([0,imageYsize],[0,imageXsize]))

    for _ in range(10):
        index = np.where(H1 == H1.max())
        H1[index]=0

    H = H1
    ret,thresh1 = cv2.threshold(H.astype(np.uint8),0,255,cv2.THRESH_BINARY)

    # Flip the image if it is not done before. Note, origain is at lower-left.
    thresh1 = np.flipud(thresh1)
    groupTable['Y']=300-groupTable['Y']

    p2 = 15
    theta_circles = cv2.HoughCircles(thresh1,\
            cv2.HOUGH_GRADIENT,2,50,param1=80,param2=p2,minRadius=0,maxRadius=20) 
 
    center_x=theta_circles[0,:,0]
    center_y=theta_circles[0,:,1]
    radius = theta_circles[0,:,2]

    while (len(radius) < len(fnoid)):
        p2 -= 1
        theta_circles = cv2.HoughCircles(thresh1,\
            cv2.HOUGH_GRADIENT,2,50,param1=50,param2=p2,minRadius=0,maxRadius=20)     
        
        center_x=theta_circles[0,:,0]
        center_y=theta_circles[0,:,1]
        radius = theta_circles[0,:,2]
        
    while (len(radius) > len(fnoid)):
        p2 += 1
        theta_circles = cv2.HoughCircles(thresh1,\
            cv2.HOUGH_GRADIENT,2,50,param1=50,param2=p2,minRadius=0,maxRadius=20)     
        
        center_x=theta_circles[0,:,0]
        center_y=theta_circles[0,:,1]
        radius = theta_circles[0,:,2]

    cxx,cyy,crr =zip(*(sorted(zip(*(center_x,center_y,radius)), key=lambda x: x[0], reverse=True)))
    center_x=np.array(cxx)
    center_y=np.array(cyy)
    radius=np.array(crr)

   
    p = figure(x_range=(0, imageXsize), y_range=(0, imageYsize),plot_width=imageXsize, plot_height=imageYsize,tools=TOOLS)
    cmap=LinearColorMapper(palette="Viridis256", low=0.001, high=0.5)
    p.image(image=[thresh1], x=0, y=0, dw=imageXsize, dh=imageYsize, color_mapper=cmap)
    # #p.circle(groupTable['X'],groupTable['Y'], radius=0.2,color='blue',fill_color=None)
    p.circle(center_x,center_y, radius=radius+30,color='blue',fill_color=None)
    # #source = ColumnDataSource(dict(x=center_x, y=center_y, text=fid))
    #show(p)
    basename=os.path.basename(inputcsv)
    export_png(p,figpath+basename[0:-4]+'.png')

    for i in range(len(center_x)):
        d = np.sqrt((groupTable['Y'].data-center_y[i])**2+(groupTable['X'].data-center_x[i])**2)
        inx = np.where(d < radius[i]+30)
        fid = np.append(fid,groupTable['Frame ID'].data[inx])
        xx = np.append(xx,groupTable['X'].data[inx])
        yy = np.append(yy,groupTable['Y'].data[inx])    
        

        xc_tmp, yc_tmp, rbest1 = findCenter(groupTable['X'].data[inx],groupTable['Y'].data[inx])
        xc = np.append(xc,np.zeros(len(groupTable['X'].data[inx]))+xc_tmp)
        yc = np.append(yc,np.zeros(len(groupTable['X'].data[inx]))+yc_tmp)
        r = np.append(r,np.zeros(len(groupTable['X'].data[inx]))+rbest1)
        fno = np.append(fno,np.zeros(len(groupTable['X'].data[inx]))+fnoid[i])

    dist = np.zeros(len(xx))
    for ffid in np.unique(fno):
        inx = np.where(fno == ffid)
        sxx = xx[inx]
        syy = yy[inx]
        sid = fid[inx]
        diff = np.zeros(len(sxx))
        for i in range(len(sxx)-1):
            diff[i]=distance((sxx[i],syy[i]),(sxx[i+1],syy[i+1]))
        dist[inx] = diff 

    data = Table([fid, xx, yy, xc,yc,r, fno, dist], names=('Frame ID','X', 'Y','Xcenter','Ycenter','R', 'Fno','Distance'))
    ascii.write(data,output, format='csv', overwrite=True) 
            

figpath='/Volumes/Disk/Data/20180927/Figures/'


def main():
    
    logpath='/Volumes/Disk/Data/20180927/'

    g3fid=np.array([])
    allfid = [e for e in list(range(2,57)) if e not in {39, 43}]
    for i in allfid:
        if i % 3 == 0:
            g3fid=np.append(g3fid,i)
    
    #fnoid=allfid
    fnoid=g3fid
    #fnoid=[21]

    motor = ['theta','phi']
    interval = ['35','94','125','156','188']
    fibergroup = 'group'

    for m in motor:
        for itv in interval:
            csvname = logpath+'20180927_'+m+'_'+itv+'int_'+fibergroup+'fiber.csv'
            cleancsv = logpath+'20180927_'+m+'_'+itv+'int_'+fibergroup+'fiber_clean.csv'
            groupROMdata(csvname, cleancsv, fnoid)
            print(csvname)
            print(cleancsv)


if __name__ == "__main__":
    #print(__name__)
    main()

