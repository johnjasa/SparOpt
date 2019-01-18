import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodSurge(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_surge_CDF', val=0.)

		self.add_output('T_surge', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		if inputs['long_term_surge_CDF'] == 0.:
			outputs['T_surge'] = 10000.
		else:
			outputs['T_surge'] = 1. / ((1. - inputs['long_term_surge_CDF']) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		if inputs['long_term_surge_CDF'] == 0.:
			partials['T_surge', 'long_term_surge_CDF'][i,i] = 0.
		else:
			partials['T_surge', 'long_term_surge_CDF'] = 1. / ((1. - inputs['long_term_surge_CDF']) * m1h)**2. * m1h