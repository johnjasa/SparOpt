import numpy as np
from scipy import linalg

from openmdao.api import ImplicitComponent

class TowerNode1Deriv(ImplicitComponent):

	def setup(self):
		self.add_input('x_towernode', val=np.zeros(11), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')

		self.add_output('x_d_towernode', val=np.zeros(11), units='m/m')

		#self.declare_partials('*', '*')

	def apply_nonlinear(self, inputs, outputs, residuals):
		z = inputs['z_towernode']
		x = inputs['x_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		delta = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]
			delta[i] = (x[i+1] - x[i]) / (z[i+1] - z[i])

		A = np.zeros((N_tower,N_tower))
		r = np.zeros(N_tower)
		for i in xrange(1,N_tower - 1):
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

		residuals['x_d_towernode'] = A.dot(outputs['x_d_towernode']) - r

	def solve_nonlinear(self, inputs, outputs):
		z = inputs['z_towernode']
		x = inputs['x_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		delta = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]
			delta[i] = (x[i+1] - x[i]) / (z[i+1] - z[i])

		A = np.zeros((N_tower,N_tower))
		r = np.zeros(N_tower)
		for i in xrange(1,N_tower - 1):
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

		outputs['x_d_towernode'] = linalg.solve(A,r)