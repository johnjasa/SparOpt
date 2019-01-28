import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodMoorTen(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_moor_ten_CDF', val=0.)

		self.add_output('T_moor_ten', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		if inputs['long_term_moor_ten_CDF'] == 1.:
			outputs['T_moor_ten'] = 10000.
		else:
			outputs['T_moor_ten'] = 1. / ((1. - inputs['long_term_moor_ten_CDF']) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		if inputs['long_term_moor_ten_CDF'] == 1.:
			partials['T_moor_ten', 'long_term_moor_ten_CDF'] = 0.
		else:
			partials['T_moor_ten', 'long_term_moor_ten_CDF'] = 1. / ((1. - inputs['long_term_moor_ten_CDF']) * m1h)**2. * m1h