import numpy as np

from openmdao.api import ExplicitComponent

class TotalHullFatigueDamage(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('hull_fatigue_damage%d' % i, val=np.zeros(10))
			self.add_input('p%d' % i, val=0.)
		
		self.add_input('DFF_hull', val=0.)

		self.add_output('total_hull_fatigue_damage', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		for i in xrange(self.N_EC):
			outputs['total_hull_fatigue_damage'] += inputs['hull_fatigue_damage%d' % i] * inputs['p%d' % i] * inputs['DFF_hull'] * 24. * 365.25 * 20. #damage in 20 years
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			partials['total_hull_fatigue_damage', 'hull_fatigue_damage%d' % i] = np.ones(11) * inputs['p%d' % i] * inputs['DFF_hull'] * 24. * 365.25 * 20.
			partials['total_hull_fatigue_damage', 'p%d' % i] = inputs['hull_fatigue_damage%d' % i] * inputs['DFF_hull'] * 24. * 365.25 * 20.