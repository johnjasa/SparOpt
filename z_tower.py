import numpy as np

from openmdao.api import ExplicitComponent

class ZTower(ExplicitComponent):

	def setup(self):
		self.add_input('L_tower', val=np.zeros(10), units='m')

		self.add_output('Z_tower', val=np.zeros(11), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_tower = inputs['L_tower']
		
		outputs['Z_tower'] = np.zeros(len(L_tower) + 1)

		outputs['Z_tower'][0] = 10.

		for i in xrange(len(L_tower)):
			outputs['Z_tower'][i+1] = 10. + np.sum(L_tower[:i+1])

	def compute_partials(self, inputs, partials):
		partials['Z_tower', 'L_tower'] = np.concatenate((np.tril(np.ones((10,10)),-1),np.ones((1,10))),0)