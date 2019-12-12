import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
#from mpl_toolkits.mplot3d import Axes3D
from scipy.fftpack import fft, ifft, diff
from scipy.integrate import odeint, complex_ode

alpha = -0.1
beta = -1.4
grid = 0.5
dom = 100
t0 = 0
t1 = 200
dt = 0.5

def soliton(x, c):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    return u

def dAdt(t, A):

	L = len(A)
	Re_A_xx = diff(A.real, order=2, period=L)	
	Im_A_xx = diff(A.imag, order=2, period=L)	
	A_xx = Re_A_xx + 1j*Im_A_xx
	abs_A2 = abs(A)**2
	A_t = (1 + 1j*alpha)*A_xx + A - (1 + 1j*beta)*abs_A2*A
	A_t_ = 1j*(A_xx + 2*abs(A)*A)
	return A_t

def dAdx(A):

	L = len(A)
	Re_A_x = diff(A.real, order=1, period=L)	
	Im_A_x = diff(A.imag, order=1, period=L)	
	A_x = Re_A_x + 1j*Im_A_x
	return A_x


#sol = np.array([], dtype = np.complex64)
sol = []
t   = np.array([], dtype = np.complex64)
x = np.arange(-dom,dom,grid)
L = len(x)

mu = 0
sigma = 0.1

#A_0 = 0.2*np.sin(2*np.pi*x) + 0.1*1j*np.cos(2*np.pi*x+np.pi/6) + 0.1*np.random.random()
#A_0 = 1*x + np.random.random() + 0.01
#A_0 = 1/np.cos((2*np.pi*x/L)**2) + 0.8/np.cos((2*np.pi*x/L)**2) + 0.01*np.random.random()
#A_0 = np.cos(16*np.pi*x/L) + np.sin(8*np.pi*x/L) + 0.1*np.random.normal(mu, sigma, dom*2/grid)
A_0 = 0.1*np.random.normal(mu, sigma, dom*2/grid) # noise
#A_0 = soliton(x-0.33*L, 0.75) + soliton(x-0.66*L, 0.4) + 0.1*np.random.normal(mu, sigma, dom*2/grid) # soliton solution
#A_0 = np.sqrt(1 - (20*np.pi/L)**2)*np.exp(1j*20*np.pi*x/L) + 0.1*np.random.random()
#A_0 = 0.25*np.exp(0.5*1j*x/L) + 0.05*np.random.normal(mu, sigma, dom*2/grid) # stationary wave

r = complex_ode(dAdt)
r.set_initial_value(A_0, t0)

while r.successful() and r.t < t1:
	t = np.append(t,r.t+dt)
	sol.append(r.integrate(r.t+dt))

sol = np.array(sol)

X,T = np.meshgrid(x,t)

#fig = plt.figure(figsize=(15,5))

# `ax` is a 3D-aware axis instance, because of the projection='3d' keyword argument to add_subplot
#ax = fig.add_subplot(1, 2, 1, projection='3d')
# surface_plot with color grading and color bar
#p = ax.plot_surface(X, T, abs(sol), rstride=1, cstride=1, cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False)
#cb = fig.colorbar(p, shrink=0.5)

plt.figure(figsize=(16,5.5))
ax = plt.subplot(121)
#ax = fig.add_subplot(1, 2, 2)
plt.imshow(abs(sol[::-1,:]), aspect='equal')
#plt.imshow(sol[::-1,:].real)
#fig.colorbar(p, shrink=0.5)
plt.xticks([0, 200, 400], ('-100', '0', '100'))
plt.yticks([0, 200, 400], ('200', '100', '0'))
plt.colorbar(shrink=0.8)

ax = plt.subplot(122)
#plt.plot(abs(sol[:,100]), abs(dAdx(sol[:,100])))
#print np.shape(sol[:,100])
plt.plot(x,abs(sol[-1,:]))
#plt.plot(x,abs(sol[2,:]))
plt.show()