import numpy as np

from openmdao.api import ExplicitComponent

class ReturnPeriodMyShellBuckling(ExplicitComponent):

	def setup(self):
		self.add_input('long_term_My_shell_buckling_CDF', val=np.zeros(10))

		self.add_output('T_My_shell_buckling', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		m1h = 365.25 * 24. #number of 1h sea states in a year

		for i in xrange(10):
			if inputs['long_term_My_shell_buckling_CDF'][i] == 0.:
				outputs['T_My_shell_buckling'][i] = 10000. * np.ones(10)
			else:
				outputs['T_My_shell_buckling'][i] = 1. / ((1. - inputs['long_term_My_shell_buckling_CDF'][i]) * m1h)
	
	def compute_partials(self, inputs, partials):
		m1h = 365.25 * 24. #number of 1h sea states in a year
		
		for i in xrange(10):
			if inputs['long_term_My_shell_buckling_CDF'][i] == 0.:
				partials['T_My_shell_buckling', 'long_term_My_shell_buckling_CDF'][i,i] = 0.
			else:
				partials['T_My_shell_buckling', 'long_term_My_shell_buckling_CDF'][i,i] = 1. / ((1. - inputs['long_term_My_shell_buckling_CDF'][i]) * m1h)**2. * m1h