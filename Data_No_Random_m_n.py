# -*- coding: utf-8 -*-
""" Data_Ex_Normal_m_n.py
Generates Magnitudes and Phases of electric and magnetic fields of a rectangular waveguide
using equations from Table 3.2 from [1].
Normal Distribution noise was added to magnitudes and phases using [2]
Modal Numbers are generated using random distribution
Date: February 21 2022
Author: Kate Antonov

Sources:
********************************************************************************
NOTE: The use of this program is limited to NON-COMMERCIAL usage only.
If the program code (or a modified version) is used in a scientific work,
then reference should be made to the following:
[1] ECE 504 Machine Learning for Electromagnetics, Instructor: Dr. Ata Zadehgol, Spring 2022.
[2] Pozar, D. M.(2005) Microwave Engineering. John Wiley and Sons.
[3] NumPy, "numpy.random.normal",
https://numpy.org/doc/stable/reference/random/generated/numpy.random.normal.html
[4] NumPy, "numpy.random.randint",
https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
********************************************************************************
"""

from math import pi,sqrt, exp, cos, sin
import numpy as np
from matplotlib import pyplot as plt
from numba import jit
import cmath
from scipy.stats import kde
from matplotlib.colors import LogNorm
import csv
import pandas as pd  
#import pickles

#%%Randomization Function
a = 1.07e-2
b = 0.43e-2
#x = np.zeros(9000)
#y = np.zeros(9000)
#m = np.zeros(9000)
#n = np.zeros(9000)
#f = np.zeros(9000)
#for i in range(len(f))
from numpy.random import seed
from numpy.random import normal
from numpy.random import shuffle
from numpy.random import randint
# seed random number generator
seed(42)
# m_1 = np.random.normal(2,0.3,128)
# m = np.zeros(128)
# for i in range(len(m_1)):
#     m[i] = round(m_1[i],0)
# #m = round(m_1,0)
# #print(m)
# print(m.max())
# print(m.min())
m = np.random.randint(1,4,9000)
np.random.shuffle(m)
print(m)
seed(42)
# n_1 = np.random.normal(2,0.3,128)
# n = np.zeros(128)
# for i in range(len(n_1)):
#     n[i] = round(n_1[i],0)
# #m = round(m_1,0)
# #print(n)
# shuffle(n)
# print(n.max())
# print(n.min())
n = np.random.randint(1,4,9000)
print(n)
#seed(42)

pre_x = np.zeros(9000)
for i in range(9000):
    seed(i)
    x_1 = np.random.random()
    pre_x[i] = round(x_1,4)
#m = round(m_1,0)
#print(x)
print(pre_x.max())
print(pre_x.min())
x = 0 + (pre_x * (0.00107 - 0))
print(x)
print(x.max())
print(x.min())
#c = round(5.76543, 0)
#y_1 = np.random.normal(5,1.27,9000)
pre_y = np.zeros(9000)
for i in range(9000):
    seed(i)
    y_1 = np.random.random()
    pre_y[i] = round(y_1,4)
#m = round(m_1,0)
#print(y)
print(pre_y.max())
print(pre_y.min())
y = 0 + (pre_y * (4.3e-4 - 0))
print(y)
print(y.max())
print(y.min())
np.random.shuffle(y)
# f_1 = np.random.normal(5,1.3,9000)
# pre_f = np.zeros(9000)
# for i in range(len(f_1)):
#     pre_f[i] = round(f_1[i],0)
# #m = round(m_1,0)
# print(pre_f)
# print(pre_f.max())
# print(pre_f.min())
# f = 0 + (pre_f * (2e9- 0))
# print(f)
# print(f.max())
# print(f.min())
f = 15e9
w = 2*np.pi*f
A = 1
z = 0
#y0 = b/2
mu0 = 4*np.pi*10**-7
mu = 1*mu0
eps0 = (1e-9)/(36*np.pi)
eps = 2.1*eps0

#%% Function Equations
kc = np.sqrt(((m*np.pi)/a)**2 + ((n*np.pi)/b)**2)
beta = w*np.sqrt(mu*eps)
def Ex_TE(x,y,m,n,w):
    y = ((1j*w*mu*n*np.pi)/(kc**2*b))*A*np.cos((m*np.pi*x)/a)*np.sin((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Ey_TE(x,y,m,n,w):
    y = ((-1j*w*mu*m*np.pi)/(kc**2*a))*A*np.sin((m*np.pi*x)/a)*np.cos((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Ez_TE(x,y,m,n,w):
    y = np.zeros(9000)
    return y
def Hx_TE(x,y,m,n,w):
    y = ((1j*beta*m*np.pi)/(kc**2*a))*A*np.sin((m*np.pi*x)/a)*np.cos((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Hy_TE(x,y,m,n,w):
    y = ((1j*beta*n*np.pi)/(kc**2*b))*A*np.cos((m*np.pi*x)/a)*np.sin((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Hz_TE(x,y,m,n,w):
    y = A*np.cos((m*np.pi*x)/a)*np.cos((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Ex_TM(x,y,m,n,w):
    y = ((-1j*beta*m*np.pi)/(kc**2*a))*A*np.cos((m*np.pi*x)/a)*np.sin((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Ey_TM(x,y,m,n,w):
    y = ((-1j*beta*n*np.pi)/(kc**2*b))*A*np.sin((m*np.pi*x)/a)*np.cos((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Ez_TM(x,y,m,n,w):
    y = A*np.sin((m*np.pi*x)/a)*np.sin((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Hx_TM(x,y,m,n,w):
    y = ((1j*w*eps*n*np.pi)/(kc**2*b))*A*np.sin((m*np.pi*x)/a)*np.cos((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Hy_TM(x,y,m,n,w):
    y = ((-1j*w*eps*m*np.pi)/(kc**2*a))*A*np.cos((m*np.pi*x)/a)*np.sin((n*np.pi*y)/b)*np.exp(-1j*beta*z)
    return y
def Hz_TM(x,y,m,n,w):
    y = np.zeros(9000)
    return y
#%%Actual Data Generation
def generate_dataset_abs(x,y,m,n,w):
    y1 = np.abs(Ex_TE(x,y,m,n,w))
    y3 = np.abs(Ey_TE(x,y,m,n,w))
    y5 = np.abs(Ez_TE(x,y,m,n,w))
    y7 = np.abs(Hx_TE(x,y,m,n,w))
    y9 = np.abs(Hy_TE(x,y,m,n,w))
    y11 = np.abs(Hz_TE(x,y,m,n,w))
    y13 = np.abs(Ex_TM(x,y,m,n,w))
    y15 = np.abs(Ey_TM(x,y,m,n,w))
    y17 = np.abs(Ez_TM(x,y,m,n,w))
    y19 = np.abs(Hx_TM(x,y,m,n,w))
    y21 = np.abs(Hy_TM(x,y,m,n,w))
    y23 = np.abs(Hz_TM(x,y,m,n,w))
    return y1,y3,y5,y7,y9,y11,y13,y15,y17,y19,y21,y23
def generate_dataset_angle(x,y,m,n,w):
    y2 = np.angle(Ex_TE(x,y,m,n,w))
    y4 = np.angle(Ey_TE(x,y,m,n,w))
    y6 = np.angle(Ez_TE(x,y,m,n,w))
    y8 = np.angle(Hx_TE(x,y,m,n,w))
    y10 = np.angle(Hy_TE(x,y,m,n,w))
    y12 = np.angle(Hz_TE(x,y,m,n,w))
    y14 = np.angle(Ex_TM(x,y,m,n,w))
    y16 = np.angle(Ey_TM(x,y,m,n,w))
    y18 = np.angle(Ez_TM(x,y,m,n,w))
    y20 = np.angle(Hx_TM(x,y,m,n,w))
    y22 = np.angle(Hy_TM(x,y,m,n,w))
    y24 = np.angle(Hz_TM(x,y,m,n,w))
    return y2,y4,y6,y8,y10,y12,y14,y16,y18,y20,y22,y24

def generate_dataset_abs_with_noise(x,y,m,n,w):
    y1 = np.abs(Ex_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y3 = np.abs(Ey_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y5 = np.abs(Ez_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y7 = np.abs(Hx_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y9 = np.abs(Hy_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y11 = np.abs(Hz_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y13 = np.abs(Ex_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y15 = np.abs(Ey_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y17 = np.abs(Ez_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y19 = np.abs(Hx_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y21 = np.abs(Hy_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y23 = np.abs(Hz_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    return y1,y3,y5,y7,y9,y11,y13,y15,y17,y19,y21,y23
def generate_dataset_angle_with_noise(x,y,m,n,w):
    y2 = np.angle(Ex_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y4 = np.angle(Ey_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y6 = np.angle(Ez_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y8 = np.angle(Hx_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y10 = np.angle(Hy_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y12 = np.angle(Hz_TE(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y14 = np.angle(Ex_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y16 = np.angle(Ey_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y18 = np.angle(Ez_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y20 = np.angle(Hx_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y22 = np.angle(Hy_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    y24 = np.angle(Hz_TM(x,y,m,n,w)) + np.random.normal(0,np.amax(np.abs(Ex_TE(x,y,m,n,w)))*0.05)
    return y2,y4,y6,y8,y10,y12,y14,y16,y18,y20,y22,y24

y1,y3,y5,y7,y9,y11,y13,y15,y17,y19,y21,y23 = generate_dataset_abs_with_noise(x,y,m,n,w)
y2,y4,y6,y8,y10,y12,y14,y16,y18,y20,y22,y24 = generate_dataset_angle_with_noise(x,y,m,n,w) 

#%% Loading Data into DataFrame
pre_magn = np.array([y1,y3,y5,y7,y9,y11,y13,y15,y17,y19,y21,y23])
#print(pre_magn)

#magn = pre_magn.T
magn2 = pre_magn.flatten()
magn3 = magn2.T
#premagn = np.array([f])
#magn_1 = pre_magn.T
#print(magn)
pre_phase = np.array([y2,y4,y6,y8,y10,y12,y14,y16,y18,y20,y22,y24])
phase = pre_phase.T
phase2 = pre_phase.flatten()
phase3 = phase2.T
magn1 = y1

#print(phase2)
# pre_magn = np.array([f])
# magn = pre_magn.T
x4 = np.array([x,x,x,x,x,x,x,x,x,x,x,x])
pre_x = x4.flatten()
x2 = pre_x.T
y4 = np.array([y,y,y,y,y,y,y,y,y,y,y,y])
pre_y = y4.flatten()
y2 = pre_y.T
m4 = np.array([m,m,m,m,m,m,m,m,m,m,m,m])
pre_m = m4.flatten()
m2 = pre_m.T
n4 = np.array([n,n,n,n,n,n,n,n,n,n,n,n])
pre_n = n4.flatten()
n2 = pre_n.T
#phase = pre_phase.T
phase2 = pre_phase.flatten()
phase3 = phase2.T
"""
Divides each function into the magnitude and phase parts using np.real() and np.angle(). 
Each array for the input parameters were made by making 12 by 9000 arrays and flattening 
them so that they can be written to a .csv file. The program was ran 6 times, with 
each input parameter array copied and pasted into a .csv file in this manner:
"""
np.savetxt("my_output_file.csv",n2,delimiter=",")
