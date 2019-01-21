import numpy as np

from openmdao.api import ExplicitComponent

class TotalTowerFatigueDamage(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('tower_fatigue_damage%d' % i, val=np.zeros(11))
			self.add_input('p%d' % i, val=0.)

		self.add_input('DFF_tower', val=0.)

		self.add_output('total_tower_fatigue_damage', val=np.zeros(11))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		for i in xrange(self.N_EC):
			outputs['total_tower_fatigue_damage'] += inputs['tower_fatigue_damage%d' % i] * inputs['p%d' % i] * inputs['DFF_tower'] * 24. * 365.25 * 20. #damage in 20 years
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			for j in xrange(11):
				partials['total_tower_fatigue_damage', 'tower_fatigue_damage%d' % i][j,j] += inputs['p%d' % i] * inputs['DFF_tower'] * 24. * 365.25 * 20.
			
			partials['total_tower_fatigue_damage', 'p%d' % i][:,0] += inputs['tower_fatigue_damage%d' % i] * inputs['DFF_tower'] * 24. * 365.25 * 20.
			partials['total_tower_fatigue_damage', 'DFF_tower'][:,0] += inputs['tower_fatigue_damage%d' % i] * inputs['p%d' % i] * 24. * 365.25 * 20.