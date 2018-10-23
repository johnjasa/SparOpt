import numpy as np

from openmdao.api import ExplicitComponent

class TowerElem2Deriv(ExplicitComponent):

	def setup(self):
		self.add_input('z_towernode', val=np.zeros(11), units='m')
		self.add_input('x_d_towernode', val=np.zeros(11), units='m/m')

		self.add_output('x_dd_towerelem', val=np.zeros(10), units='1/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_towernode']
		x_d = inputs['x_d_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]

		outputs['x_dd_towerelem'] = np.zeros(N_tower - 1)
		
		for i in xrange(N_tower - 1):
			outputs['x_dd_towerelem'][i] = 1. / h[i] * (x_d[i+1] - x_d[i])