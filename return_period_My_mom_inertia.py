import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodMyMomInertia(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_My_mom_inertia_CDF', val=np.zeros(10))

		self.add_output('T_My_mom_inertia', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		for i in xrange(10):
			if inputs['long_term_My_mom_inertia_CDF'][i] == 1.:
				outputs['T_My_mom_inertia'][i] = 10000.
			else:
				outputs['T_My_mom_inertia'][i] = 1. / ((1. - inputs['long_term_My_mom_inertia_CDF'][i]) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		for i in xrange(10):
			if inputs['long_term_My_mom_inertia_CDF'][i] == 1.:
				partials['T_My_mom_inertia', 'long_term_My_mom_inertia_CDF'][i,i] = 0.
			else:
				partials['T_My_mom_inertia', 'long_term_My_mom_inertia_CDF'][i,i] = 1. / ((1. - inputs['long_term_My_mom_inertia_CDF'][i]) * m1h)**2. * m1h