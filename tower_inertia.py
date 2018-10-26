import numpy as np

from openmdao.api import ExplicitComponent

class TowerInertia(ExplicitComponent):

	def setup(self):
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')

		self.add_output('I_tower', val=0., units='kg*m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_tower  = inputs['L_tower']
		M_tower  = inputs['M_tower']

		outputs['I_tower'] = 0.

		for i in xrange(len(L_tower)):
			CoG_sec = 10. + np.sum(L_tower[0:i]) + L_tower[i] / 2.
			outputs['I_tower'] += M_tower[i] * CoG_sec**2.

	def compute_partials(self, inputs, partials):
		L_tower  = inputs['L_tower']
		M_tower  = inputs['M_tower']

		partials['I_tower', 'L_tower'] = np.zeros((1,10))
		partials['I_tower', 'M_tower'] = np.zeros((1,10))

		for i in xrange(len(L_tower)):
			CoG_sec = 10. + np.sum(L_tower[0:i]) + L_tower[i] / 2.
			partials['I_tower', 'M_tower'][0,i] += CoG_sec**2.
			partials['I_tower', 'L_tower'][0,i] += 2. *  M_tower[i] * CoG_sec * 0.5

			for j in xrange(i):
				partials['I_tower', 'L_tower'][0,j] += 2. *  M_tower[i] * CoG_sec