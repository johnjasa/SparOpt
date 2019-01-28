import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodPitch(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_pitch_CDF', val=0.)

		self.add_output('T_pitch', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		if inputs['long_term_pitch_CDF'] == 1.:
			outputs['T_pitch'] = 10000.
		else:
			outputs['T_pitch'] = 1. / ((1. - inputs['long_term_pitch_CDF']) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		if inputs['long_term_pitch_CDF'] == 1.:
			partials['T_pitch', 'long_term_pitch_CDF'] = 0.
		else:
			partials['T_pitch', 'long_term_pitch_CDF'] = 1. / ((1. - inputs['long_term_pitch_CDF']) * m1h)**2. * m1h