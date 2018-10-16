import numpy as np

from openmdao.api import ExplicitComponent

class TowerCoG(ExplicitComponent):

	def setup(self):
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')

		self.add_output('CoG_tower', val=0., units='m')

	def compute(self, inputs, outputs):
		L_tower  = inputs['L_tower']
		M_tower  = inputs['M_tower']

		CoG_t_mass = 0.0

		for i in xrange(len(L_tower)):
			CoG_sec = 10. + np.sum(L_tower[0:i]) + L_tower[i] / 2.
			CoG_t_mass += M_tower[i] * CoG_sec

		outputs['CoG_tower'] = CoG_t_mass / np.sum(M_tower)