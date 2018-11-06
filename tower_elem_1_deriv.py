import numpy as np

from openmdao.api import ExplicitComponent

class TowerElem1Deriv(ExplicitComponent):

	def setup(self):
		self.add_input('x_towernode', val=np.zeros(11), units='m')
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_towernode', val=np.zeros(11), units='m/m')

		self.add_output('x_d_towerelem', val=np.zeros(10), units='m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_towernode']
		x = inputs['x_towernode']
		x_d = inputs['x_d_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]

		outputs['x_d_towerelem'] = np.zeros(N_tower - 1)
		
		for i in xrange(N_tower - 1):
			outputs['x_d_towerelem'][i] = 3. / (2. * h[i]) * (x[i+1] - x[i]) - 1. / 4. * (x_d[i+1] + x_d[i])

	def compute_partials(self, inputs, partials):
		z = inputs['z_towernode']
		x = inputs['x_towernode']
		x_d = inputs['x_d_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]

		partials['x_d_towerelem', 'z_towernode'] = np.zeros((N_tower - 1, N_tower))
		partials['x_d_towerelem', 'x_towernode'] = np.zeros((N_tower - 1, N_tower))
		partials['x_d_towerelem', 'x_d_towernode'] = np.zeros((N_tower - 1, N_tower))
		
		for i in xrange(N_tower - 1):
			partials['x_d_towerelem', 'z_towernode'][i,i] = 3. / (2. * h[i]**2.) * (x[i+1] - x[i])
			partials['x_d_towerelem', 'z_towernode'][i,i+1] = -3. / (2. * h[i]**2.) * (x[i+1] - x[i])
		
			partials['x_d_towerelem', 'x_towernode'][i,i] = -3. / (2. * h[i])
			partials['x_d_towerelem', 'x_towernode'][i,i+1] = 3. / (2. * h[i])

			partials['x_d_towerelem', 'x_d_towernode'][i,i] = - 1. / 4.
			partials['x_d_towerelem', 'x_d_towernode'][i,i+1] = - 1. / 4.