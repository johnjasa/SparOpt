import numpy as np

from openmdao.api import ExplicitComponent

class TowerCoG(ExplicitComponent):

	def setup(self):
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('tot_M_tower', val=0., units='kg')

		self.add_output('CoG_tower', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_tower  = inputs['L_tower']
		M_tower  = inputs['M_tower']
		tot_M_tower  = inputs['tot_M_tower']

		CoG_t_mass = 0.0

		for i in xrange(len(L_tower)):
			CoG_sec = 10. + np.sum(L_tower[0:i]) + L_tower[i] / 2.
			CoG_t_mass += M_tower[i] * CoG_sec

		outputs['CoG_tower'] = CoG_t_mass / tot_M_tower

	def compute_partials(self, inputs, partials):
		L_tower  = inputs['L_tower']
		M_tower  = inputs['M_tower']
		tot_M_tower  = inputs['tot_M_tower']

		partials['CoG_tower', 'L_tower'] = np.zeros((1,10))
		partials['CoG_tower', 'M_tower'] = np.zeros((1,10))
		partials['CoG_tower', 'tot_M_tower'] = 0.

		CoG_t_mass = 0.

		for i in xrange(10):
			CoG_sec = 10. + np.sum(L_tower[0:i]) + L_tower[i] / 2.
			
			CoG_t_mass += M_tower[i] * CoG_sec

			partials['CoG_tower', 'L_tower'][0,i] += 0.5 * M_tower[i] / tot_M_tower
			partials['CoG_tower', 'M_tower'][0,i] += CoG_sec / tot_M_tower

			for j in xrange(i):
				partials['CoG_tower', 'L_tower'][0,j] += M_tower[i] / tot_M_tower

		partials['CoG_tower', 'tot_M_tower'] = -CoG_t_mass / tot_M_tower**2.