import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodSurge(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_surge_CDF', val=0.)

		self.add_output('T_surge', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		outputs['T_surge'] = 1. / ((1. - inputs['long_term_surge_CDF']) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		partials['T_surge', 'long_term_surge_CDF'] = 1. / ((1. - inputs['long_term_surge_CDF']) * m1h)**2. * m1h