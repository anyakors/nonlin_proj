import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
    
#parameters
N = 4
D = 16
s = 0.75
tmax = 100000

w = np.zeros((N,D), dtype=np.longdouble)
B = np.zeros((N,), dtype=np.longdouble)
x = np.zeros((N,), dtype=np.longdouble)
y = np.zeros((D,), dtype=np.longdouble)

x1_t = []
x2_t = []

for i in np.arange(0,N):
    for j in np.arange(0,D):
        w[i,j] = 1.0 - np.multiply(2.0,np.random.random(),dtype=np.longdouble)
    B[i] = np.multiply(s,np.random.random(),dtype=np.longdouble)
    x[i] = 0.001

for t in np.arange(0,tmax):
    y0 = 0.0
    for i in np.arange(0,N):
        y[0] += np.multiply(B[i],x[i],dtype=np.longdouble)
    for j in np.arange(D-1,0,-1):
        y[j] = y[j-1]
    for i in np.arange(0,N):
        u = 0.0
        for j in np.arange(0,D):
            u += np.multiply(w[i,j],y[j],dtype=np.longdouble)
        x[i] = np.tanh(u,dtype=np.longdouble)
    x1_t.append(x[0])
    #x1_t.append(x[2])
    x2_t.append(x[1])
    #x2_t.append(x[3])

np.savetxt('x1', x1_t, delimiter=' ', fmt="%s")
np.savetxt('x2', x2_t, delimiter=' ', fmt="%s")

t = x1_t[-1000:]

#plt.subplot(1,2,1)
#plt.scatter(x1_t[-1000:], x2_t[-1000:], c=t, cmap=cm.rainbow)
#plt.colorbar()

#plt.subplot(1,2,2)
plt.plot(np.arange(0,1000), x1_t[-1000:])
plt.show()