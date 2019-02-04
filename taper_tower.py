import numpy as np

from openmdao.api import ExplicitComponent

class TaperTower(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower_p', val=np.zeros(11), units='m')

		self.add_output('taper_tower', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower_p = inputs['D_tower_p']

		outputs['taper_tower'] = D_tower_p[1:] / D_tower_p[:-1]
	
	def compute_partials(self, inputs, partials):
		D_tower_p = inputs['D_tower_p']

		for i in xrange(10):
			partials['taper_tower', 'D_tower_p'][i,i] = -D_tower_p[i+1] / D_tower_p[i]**2.
			partials['taper_tower', 'D_tower_p'][i,i+1] = 1. / D_tower_p[i]