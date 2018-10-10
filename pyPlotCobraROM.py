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

from scipy import optimize
from astropy.table import Table, Column, Row, vstack
from astropy.io import ascii

from bokeh.io import output_notebook, show, export_png
from bokeh.plotting import figure, show, output_file
import bokeh.palettes
from bokeh.layouts import column,gridplot
from bokeh.models import HoverTool, ColumnDataSource, LinearColorMapper, \
                        Title, Plot, Line, DataRange1d, LinearAxis
from bokeh.models.glyphs import Text
from bokeh.models import PanTool, WheelZoomTool,BoxZoomTool,ResetTool, SaveTool

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

def polar(x, y):
    theta = np.arctan2(y, x)* 180 / np.pi
    rho = np.hypot(x, y)
    ind=np.where(theta<0)
    
    theta[ind]=theta[ind]+360
    return theta, rho

def getAngleVariation(theta):
    angle=[]
    for a in theta:
        a -= theta[0]
        angle.append(0-a)
    for i, element in enumerate(angle):
        if element < -4:
            angle[i] += 360
       
    return angle

def makeDistancePlot(table, fiber_list=None):
    source = ColumnDataSource(dict(x=table['Frame ID'], y=table['Distance']))
    xdr = DataRange1d()
    ydr = DataRange1d()
    
    p = Plot(x_range=xdr, y_range=ydr,plot_width=1000, plot_height=300)
    xaxis = LinearAxis()
    p.add_layout(xaxis, 'below')

    yaxis = LinearAxis()
    p.add_layout(yaxis, 'left')
   
    if (fiber_list.any() == None):
        for fno in np.unique(table['Fno'].data):
            
            inx = np.where(table['Fno'].data == fno)
            source = ColumnDataSource(dict(x=table['Frame ID'].data[inx], y=table['Distance'].data[inx]))
            
            line=Line(x='x',y='y', line_color=random.choice(colors))
            p.add_glyph(source, line)
    else:
        for fno in fiber_list:
            inx = np.where(table['Fno'].data == fno)
            source = ColumnDataSource(dict(x=table['Frame ID'].data[inx], y=table['Distance'].data[inx]))
            
            line=Line(x='x',y='y', line_color=random.choice(colors))
            p.add_glyph(source, line)

    p.add_tools(PanTool(), WheelZoomTool(),BoxZoomTool(),ResetTool(), SaveTool(), HoverTool())

    return p



def plotCobraROM(tab1, tab2, tab1stop, tab1step, tab2stop, tab2step, fiber_list=None):
    plots=[]
    d_angle=[]
    if (fiber_list.any() != None):
        for fno in fiber_list:
            index = np.logical_and(np.logical_and(tab1['Frame ID'] > tab1stop, tab1['Frame ID'] < tab1step), tab1['Fno'].data == fno)
            ind = np.where(index == True)
            
            xc = float(tab1['Xcenter'].data[ind[0][0]])
            yc = float(tab1['Ycenter'].data[ind[0][0]])
            r = float(tab1['R'].data[ind[0][0]])

            theta, rho = polar(tab1['X'].data[ind]-xc, tab1['Y'].data[ind]-yc)
            angle1=getAngleVariation(theta)

            xx=[xc+0.2*r]
            yy=[yc+1.8*r]
            string=['group: {:5.2f}  {:5.2f}'.format(theta[0],np.max(angle1))]

            source = ColumnDataSource(dict(x=xx, y=yy, text=string))
            glyph = Text(x="x", y="y", text="text", angle=0, text_color="red")


            p = figure(x_range=[xc-2*r,xc+2*r], y_range=[yc-2*r,yc+2*r] ,plot_width=400, plot_height=400,\
                tools=TOOLS, title="Fiber No. "+str(int(fno)))
            p.circle(xc,yc,radius=r, color="black",fill_color=None)
            p.circle(tab1['X'].data[ind],tab1['Y'].data[ind], radius=1.0, color="red",fill_color="red")
            p.add_glyph(source, glyph)


            index = np.logical_and(np.logical_and(tab2['Frame ID'] > tab2stop, tab2['Frame ID'] < tab2step), tab2['Fno'].data == fno)
            ind = np.where(index == True)
            
            xc = float(tab2['Xcenter'].data[ind[0][0]])
            yc = float(tab2['Ycenter'].data[ind[0][0]])
            r = float(tab2['R'].data[ind[0][0]])

            theta, rho = polar(tab2['X'].data[ind]-xc, tab2['Y'].data[ind]-yc)
            angle2=getAngleVariation(theta)
            #print(theta[0],np.max(angle2))

            xx=[xc+0.2*r]
            yy=[yc+1.5*r]
            string=['all: {:5.2f}  {:5.2f}'.format(theta[0],np.max(angle2))]

            source = ColumnDataSource(dict(x=xx, y=yy, text=string))
            glyph = Text(x="x", y="y", text="text", angle=0, x_offset= 25, text_color="blue")
     
            p.circle(tab2['X'].data[ind],tab2['Y'].data[ind], radius=1.0, color="blue",fill_color=None)
            p.add_glyph(source, glyph)

            d_angle.append(np.max(angle2)-np.max(angle1))
            plots.append(p)


    grid = gridplot(*plots, ncols=4)
    return grid, d_angle

    

def main():
    datapath='/Volumes/Disk/Data/20180927/'
    figpath='/Volumes/Disk/Data/20180927/Figures/'

    m='phi'
    interval = ['35','94','125','156','188']
    for itv in interval:

        g3fid=np.array([])
        allfid = [e for e in list(range(2,57)) if e not in {39, 43, 54}]
        for i in allfid:
            if i % 3 == 0:
                g3fid=np.append(g3fid,i)
        
        onefib=np.array([6])
        groupTag = ['one','group','all',]
        plots = []
        for group in groupTag:
            tablefile = datapath+'20180927_'+m+'_'+itv+'int_'+group+'fiber_clean.csv'
            #print(tablefile)
            tab = Table.read(tablefile, format='csv')
            p = makeDistancePlot(tab, fiber_list=g3fid)
            plots.append(p)

        grid = gridplot(*plots, ncols=1)
        output_file(figpath+itv+"_fiberMove.html", title="Plot of Distance movment")
        #export_png(grid, filename=figpath+itv+"_fiberMove.png")
        show(grid)

        tab1=Table.read('/Volumes/Disk/Data/20180927/20180927_'+m+'_'+itv+'int_groupfiber_clean.csv')
        tab2=Table.read('/Volumes/Disk/Data/20180927/20180927_'+m+'_'+itv+'int_allfiber_clean.csv')

        if m is 'theta':
            if itv is '35':
                g1, darray =plotCobraROM(tab1, tab2, 413, 450, 480, 522, fiber_list=g3fid)  #35int
            if itv is '94':
                g1, darray =plotCobraROM(tab1, tab2, 369, 1071, 394, 1168, fiber_list=g3fid)  #35int
            if itv is '125':
                g1, darray =plotCobraROM(tab1, tab2, 502, 1263, 357, 779, fiber_list=g3fid)  #35int
            if itv is '156':
                g1, darray =plotCobraROM(tab1, tab2, 926, 1070, 524, 671, fiber_list=g3fid)  #156int
            if itv is '188':
                g1, darray =plotCobraROM(tab1, tab2, 992, 1151, 447, 612, fiber_list=g3fid)  #156int
        if m is 'phi':
            if itv is '35':
                g1, darray =plotCobraROM(tab1, tab2, 72, 211, 73, 220, fiber_list=g3fid)
            if itv is '94':
                g1, darray =plotCobraROM(tab1, tab2, 146, 402, 142, 398, fiber_list=g3fid)
            if itv is '125':
                g1, darray =plotCobraROM(tab1, tab2, 223, 591, 220, 580, fiber_list=g3fid)
            if itv is '156':
                g1, darray =plotCobraROM(tab1, tab2, 267, 389, 308, 433, fiber_list=g3fid)
            if itv is '188':
                g1, darray =plotCobraROM(tab1, tab2, 319, 810, 307, 451, fiber_list=g3fid)

        output_file(figpath+m+'_'+itv+"_fiber.html", title="Plot of Distance movment")
        export_png(g1, filename=figpath+m+'_'+itv+"_fiber.png")
        #show(g1)
        
        if float(itv) > 100:
            xmin = -25
            xmax = 35
            nn = 10
        else:
            xmin=int(np.min(darray))
            xmax=int(np.max(darray))
            nn=int((xmax-xmin)/10)


        hist, edges = np.histogram(darray, bins=range(xmin,xmax,nn))
        
        p = figure(y_range=[0,16],background_fill_color="#E8DDCB")
        p.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                fill_color="#036564", line_color="#033649")

        p.xaxis.axis_label = 'Angle Difference (all - group)'
        p.yaxis.axis_label = 'Counts'
        output_file(figpath+m+'_'+itv+"histo.html", title="Plot of Distance movment")
        export_png(p, filename=figpath+m+'_'+itv+"int_histo.png")
        #show(p)
 

if __name__ == "__main__":
    #print(__name__)
    main()
