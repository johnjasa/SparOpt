import numpy as np

from openmdao.api import ExplicitComponent

class TowerNode1LHS(ExplicitComponent):

	def setup(self):
		self.add_input('z_towernode', val=np.zeros(11), units='m')

		self.add_output('tower_spline_lhs', val=np.zeros((11,11)), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		z = inputs['z_towernode']

		N_tower = len(z)

		h = np.zeros(N_tower - 1)
		for i in xrange(N_tower - 1):
			h[i] = z[i+1] - z[i]

		outputs['tower_spline_lhs'] = np.zeros((N_tower,N_tower))
		
		for i in xrange(1,N_tower - 1):
			outputs['tower_spline_lhs'][i,i] = 2. * (h[i] + h[i-1])
			outputs['tower_spline_lhs'][i,i-1] = h[i]
			outputs['tower_spline_lhs'][i,i+1] = h[i-1]

		outputs['tower_spline_lhs'][0,0] = h[1]
		outputs['tower_spline_lhs'][0,1] = h[0] + h[1]
		outputs['tower_spline_lhs'][-1,-1] = h[-2]
		outputs['tower_spline_lhs'][-1,-2] = h[-1] + h[-2]

	def compute_partials(self, inputs, partials):
		z = inputs['z_towernode']

		N_tower = len(z)

		partials['tower_spline_lhs', 'z_towernode'] = np.zeros((N_tower * N_tower,N_tower))

		for i in xrange(1,N_tower - 1):
			partials['tower_spline_lhs', 'z_towernode'][i*N_tower-1+i,i] = -1.
			partials['tower_spline_lhs', 'z_towernode'][i*N_tower-1+i,i+1] = 1.

			partials['tower_spline_lhs', 'z_towernode'][i*N_tower+i,i-1] = -2.
			partials['tower_spline_lhs', 'z_towernode'][i*N_tower+i,i] = 0.
			partials['tower_spline_lhs', 'z_towernode'][i*N_tower+i,i+1] = 2.

			partials['tower_spline_lhs', 'z_towernode'][i*N_tower+1+i,i-1] = -1.
			partials['tower_spline_lhs', 'z_towernode'][i*N_tower+1+i,i] = 1.

		partials['tower_spline_lhs', 'z_towernode'][0,1] = -1.
		partials['tower_spline_lhs', 'z_towernode'][0,2] = 1.
		partials['tower_spline_lhs', 'z_towernode'][1,0] = -1.
		partials['tower_spline_lhs', 'z_towernode'][1,2] = 1.
		partials['tower_spline_lhs', 'z_towernode'][-1,-3] = -1.
		partials['tower_spline_lhs', 'z_towernode'][-1,-2] = 1.
		partials['tower_spline_lhs', 'z_towernode'][-2,-3] = -1.
		partials['tower_spline_lhs', 'z_towernode'][-2,-1] = 1.
