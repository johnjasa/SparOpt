import numpy as np
from scipy.sparse import diags
from scipy import linalg
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

x = np.array([10.0, 20.5, 31.0, 41.5, 52.0, 62.5, 73.0, 83.5, 94.0, 104.5, 115.63])

y = np.array([-0.41301261, -0.44246513, -0.43706879, -0.39464275, -0.31330533, -0.19168688, -0.02922765, 0.1733981, 0.41366632, 0.6862874, 1.])

diagonals = [np.ones(len(x)) * 4, np.ones(len(x) - 1), np.ones(len(x) - 1)]

A = diags(diagonals, [0, -1, 1]).toarray()
A[0,0] = 2.
A[-1,-1] = 2

b = np.zeros(len(x))
for i in xrange(1,len(x)-1):
	b[i] = 3. * (y[i+1] - y[i-1])

y_d = linalg.solve(A,b)

y_dd = np.zeros(len(x))
for i in xrange(1,len(x)-1):
	y_dd[i] = 6. * (y[i+1] - y[i]) - 4. * y_d[i] - 2. * y_d[i+1]



print y_d[6:]
print y_dd[6:]

d = np.zeros(len(x))
for i in xrange(1,len(x)-1):
	d[i] = 2. * (y[i] - y[i+1]) + y_d[i] + y_d[i+1]

ys = CubicSpline(x,y)

ys_d = ys.derivative(nu=1)
ys_dd = ys.derivative(nu=2)
print ys_d(x)
print ys_dd(x)
#[-0.00439351 -0.00118102  0.00224437  0.00586707  0.00964835  0.01352693 0.01740898  0.02114715  0.02454355  0.02721847  0.02896766]
#[0.00029581 0.00031609 0.00033637 0.00035367 0.00036657 0.00037221 0.00036723 0.0003448  0.00030213 0.00020738 0.00010694]
t = np.linspace(0,1,50)
plt.plot([0., 1.],[y[-2], y[-1]])
plt.plot(t, y[-2] + y_d[-2]*t + y_dd[-2] / 2.*t**2. + d[-2]*t**3.)
plt.plot(t,ys(x[-2] + (x[-1] - x[-2])*t))
plt.show()