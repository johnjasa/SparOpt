import numpy as np

from openmdao.api import ExplicitComponent

class TotalFatigueDamage(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('fatigue_damage%d' % i, val=np.zeros(11))
			self.add_input('p%d' % i, val=0.)

		self.add_output('total_fatigue_damage', val=np.zeros(11))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		for i in xrange(self.N_EC):
			outputs['total_fatigue_damage'] += inputs['fatigue_damage%d' % i] * inputs['p%d' % i]
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			partials['total_fatigue_damage', 'fatigue_damage%d' % i] = np.ones(11) * inputs['p%d' % i]
			partials['total_fatigue_damage', 'p%d' % i] = inputs['fatigue_damage%d' % i]