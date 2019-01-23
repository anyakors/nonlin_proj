# this python file is used together with the bash script to calculate ten Lypunov factors

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
import sys
  
#parameters
N = 4
D = 16

# parameter s is either set explicitly or taken from the command line arguments

#s = 0.15
s = float(sys.argv[1])
tmax = 10000
epsilon = 0.05
deltax_new = 4*epsilon

w = np.zeros((N,D), dtype=np.longdouble)
B = np.zeros((N,), dtype=np.longdouble)
x = np.zeros((N,), dtype=np.longdouble)
xn = np.zeros((N,), dtype=np.longdouble)
y = np.zeros((D,), dtype=np.longdouble)
yn = np.zeros((D,), dtype=np.longdouble)

max_lambda = []

for i in np.arange(0,N):
    for j in np.arange(0,D):
        w[i,j] = 1.0 - np.multiply(2.0,np.random.random(),dtype=np.longdouble)
    B[i] = np.multiply(s,np.random.random(),dtype=np.longdouble)
    x[i] = 0.001
    xn[i] = x[i] + epsilon


for t in np.arange(0,tmax):
    y0 = 0.0
    for i in np.arange(0,N):
        y[0] += np.multiply(B[i],x[i],dtype=np.longdouble)
        yn[0] += np.multiply(B[i],xn[i],dtype=np.longdouble)
    for j in np.arange(D-1,0,-1):
        y[j] = y[j-1]
        yn[j] = yn[j-1]
    for i in np.arange(0,N):
        u, un = 0.0, 0.0
        for j in np.arange(0,D):
            u += np.multiply(w[i,j],y[j],dtype=np.longdouble)
            un += np.multiply(w[i,j],yn[j],dtype=np.longdouble)
        x[i] = np.tanh(u,dtype=np.longdouble)
        xn[i] = np.tanh(un,dtype=np.longdouble)

for t in np.arange(tmax,5*tmax):
    y0 = 0.0
    deltax = copy.copy(deltax_new)
    deltax_new = 0
    for i in np.arange(0,N):
        y[0] += np.multiply(B[i],x[i],dtype=np.longdouble)
        yn[0] += np.multiply(B[i],xn[i],dtype=np.longdouble)
    for j in np.arange(D-1,0,-1):
        y[j] = y[j-1]
        yn[j] = yn[j-1]
    for i in np.arange(0,N):
        u, un = 0.0, 0.0
        for j in np.arange(0,D):
            u += np.multiply(w[i,j],y[j],dtype=np.longdouble)
            un += np.multiply(w[i,j],yn[j],dtype=np.longdouble)
        x[i] = np.tanh(u,dtype=np.longdouble)
        xn[i] = np.tanh(un,dtype=np.longdouble)
        deltax_new += np.square(x[i]-xn[i])
    if deltax!=0 and deltax_new!=0:
        max_lambda.append(np.sqrt(deltax_new)/np.sqrt(deltax))


av_lambda = sum(np.log(max_lambda))/(4*tmax)
print(av_lambda)