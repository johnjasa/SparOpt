import numpy as np

from openmdao.api import ExplicitComponent

class TowerBucklingResist(ExplicitComponent):

	def setup(self):
		self.add_input('chi_x', val=np.zeros(10))
		self.add_input('f_y', val=0., units='MPa')
		self.add_input('gamma_M_tower', val=0.)
		self.add_input('gamma_F_tower', val=0.)

		self.add_output('maxval_tower_buckling', val=np.zeros(10), units='MPa')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		#material and load factors ref DNVGL-OS-J101 and DNVGL-OS-J103 respectively: gamma_M = 1.1, gamma_F = 1.35

		outputs['maxval_tower_buckling'] = inputs['chi_x'] * inputs['f_y'] / (inputs['gamma_M_tower'] * inputs['gamma_F_tower'])

	def compute_partials(self, inputs, partials):
		
		partials['maxval_tower_buckling', 'chi_x'] = np.zeros((10,10))
		partials['maxval_tower_buckling', 'f_y'] = np.zeros((10,1))
		partials['maxval_tower_buckling', 'gamma_M_tower'] = np.zeros((10,1))
		partials['maxval_tower_buckling', 'gamma_F_tower'] = np.zeros((10,1))

		for i in xrange(10):
			partials['maxval_tower_buckling', 'chi_x'][i,i] += inputs['f_y'] / (inputs['gamma_M_tower'] * inputs['gamma_F_tower'])
			partials['maxval_tower_buckling', 'f_y'][i,0] += inputs['chi_x'][i] / (inputs['gamma_M_tower'] * inputs['gamma_F_tower'])
			partials['maxval_tower_buckling', 'gamma_M_tower'][i,0] += -inputs['chi_x'][i] * inputs['f_y'] / (inputs['gamma_M_tower']**2. * inputs['gamma_F_tower'])
			partials['maxval_tower_buckling', 'gamma_F_tower'][i,0] += -inputs['chi_x'][i] * inputs['f_y'] / (inputs['gamma_M_tower'] * inputs['gamma_F_tower']**2.)