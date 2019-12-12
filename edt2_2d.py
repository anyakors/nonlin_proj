import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.fftpack import fft, ifft, diff, fftn, ifftn
from scipy.integrate import odeint, complex_ode

alpha = 1.0
beta = 2.0
grid = 1
dom = 50
t0 = 0
t1 = 100
dt = 1

def dAdt(t, A):

	L = len(A)
	Re_A_xx = diff(A.real, order=2, period=L)	
	Im_A_xx = diff(A.imag, order=2, period=L)	
	A_xx = Re_A_xx + 1j*Im_A_xx
	abs_A2 = abs(A)**2

	A_t = (1 + 1j*alpha)*A_xx + A - (1 + 1j*beta)*abs_A2*A
	A_t_ = 1j*(A_xx + 2*abs(A)*A)

	return A_t

def soliton(x, c):
    """Profile of the exact solution to the KdV for a single soliton on the real line."""
    u = 0.5*c*np.cosh(0.5*np.sqrt(c)*x)**(-2)
    return u

def solve_fft(A_0, t):

	step = 0
	sol = []

	cA 	    	= 1 - del2*(1+1j*alpha);
	expA 	  	= exp(dT*cA);
	nlfacA  	= (exp(dT*cA).*(1+1./cA/dT)-1./cA/dT-2)./cA;
	nlfacAp 	= (exp(dT*cA).*(-1./cA/dT)+1./cA/dT+1)./cA;

	if step = 0:
		A_hat = fftn(A_0)
	else:
		A_hat = fftn(A)



	return


#sol = np.array([], dtype = np.complex64)
sol = []
t   = np.array([], dtype = np.complex64)
x = np.arange(-dom,dom,grid)
y = np.arange(-dom,dom,grid)
L = len(x)

mu = 0
sigma = 0.1

X, Y = np.meshgrid(x,y)

A_0 = np.cos(4*np.pi*X/L) + np.sin(8*np.pi*Y/L) + np.random.normal(mu, sigma)
#A_0 = np.sqrt(1 - (20*np.pi/L)**2)*np.exp(1j*20*np.pi*x/L) + 0.01*np.random.random()



#A_t = dAdt(t, A_0)
#plt.imshow(abs(A_t))


#plt.show()