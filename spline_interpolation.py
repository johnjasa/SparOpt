import numpy as np
from scipy.sparse import diags
from scipy import linalg
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt

x = np.array([10.0, 20.5, 31.0, 41.5, 52.0, 62.5, 73.0, 83.5, 94.0, 104.5, 115.63])

y = np.array([-0.41301261, -0.44246513, -0.43706879, -0.39464275, -0.31330533, -0.19168688, -0.02922765, 0.1733981, 0.41366632, 0.6862874, 1.])

n = len(x)

x_elem = np.zeros(n - 1)

h = np.zeros(n - 1)
delta = np.zeros(n - 1)
for i in xrange(n - 1):
	h[i] = x[i+1] - x[i]
	delta[i] = (y[i+1] - y[i]) / (x[i+1] - x[i])
	x_elem[i] = (x[i+1] + x[i]) / 2

A = np.zeros((n,n))
r = np.zeros(n)
for i in xrange(1,n-1):
	A[i,i] = 2. * (h[i] + h[i-1])
	A[i,i-1] = h[i]
	A[i,i+1] = h[i-1]

	r[i] = 3. * (h[i-1] * delta[i] + h[i] * delta[i-1])

A[0,0] = h[1]
A[0,1] = h[0] + h[1]
A[-1,-1] = h[-2]
A[-1,-2] = h[-1] + h[-2]

r[0] = ((2. * h[1] + 3. * h[0]) * h[1] * delta[0] + h[0]**2. * delta[1]) / (h[0] + h[1])
r[-1] = ((2. * h[-2] + 3. * h[-1]) * h[-2] * delta[-1] + h[-1]**2. * delta[-2]) / (h[-1] + h[-2])

d = linalg.solve(A,r)

dd = np.zeros(n)
y_elem = np.zeros(n - 1)
y_d_elem = np.zeros(n - 1)
y_dd_elem = np.zeros(n - 1)
for i in xrange(n - 1):
	dd[i] = (6. * delta[i] - 2. * d[i+1] - 4. * d[i]) / h[i]
	y_elem[i] = (y[i+1] + y[i]) / 2. - 1. / 8. * h[i] * (d[i+1] - d[i])
	y_d_elem[i] = 3. / (2. * h[i]) * (y[i+1] - y[i]) - 1. / 4. * (d[i+1] + d[i])
	y_dd_elem[i] = 1. / h[i] * (d[i+1] - d[i])


dd[-1] = (-6. * delta[-1] + 4. * d[-1] + 2. * d[-2]) / h[-1]

#print d
#print dd

ys = CubicSpline(x,y, bc_type='not-a-knot')
ys_d = ys.derivative(nu=1)
ys_dd = ys.derivative(nu=2)

#print ys_d(x)
#print ys_dd(x)

print y_dd_elem
print ys_dd(x_elem)

"""
t = np.linspace(0,1,50)

plt.plot(t, y[-2] + y_d[-2]*t + y_dd[-2] / 2.*t**2. + d[-2]*t**3.)
plt.plot(t,ys(x[-2] + (x[-1] - x[-2])*t))
plt.show()
"""