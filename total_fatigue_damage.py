import numpy as np

from openmdao.api import ExplicitComponent

class TotalFatigueDamage(ExplicitComponent):

	def setup(self):
		for i in xrange(2):
			self.add_input('fatigue_damage%d' % i, val=0.)

		self.add_output('total_fatigue_damage', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		for i in xrange(2):
			outputs['total_fatigue_damage'] += np.sum(inputs['fatigue_damage%d' % i])
	
	def compute_partials(self, inputs, partials):
		for i in xrange(2):
			partials['total_fatigue_damage', 'fatigue_damage%d' % i] = 1.