import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodTowerStress(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_tower_stress_CDF', val=0.)

		self.add_output('T_tower_stress', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		outputs['T_tower_stress'] = 1. / ((1. - inputs['long_term_tower_stress_CDF']) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		partials['T_tower_stress', 'long_term_tower_stress_CDF'] = 1. / ((1. - inputs['long_term_tower_stress_CDF']) * m1h)**2. * m1h