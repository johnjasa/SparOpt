import numpy as np

from openmdao.api import ExplicitComponent

class LongTermFairleadCDF(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('short_term_fairlead_CDF%d' % i, val=0.)
			self.add_input('p%d' % i, val=0.)

		self.add_output('long_term_fairlead_CDF', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['long_term_fairlead_CDF'] = 0.
		for i in xrange(self.N_EC):
			outputs['long_term_fairlead_CDF'] += inputs['short_term_fairlead_CDF%d' % i] * inputs['p%d' % i]
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			partials['long_term_fairlead_CDF', 'short_term_fairlead_CDF%d' % i] = inputs['p%d' % i]
			partials['long_term_fairlead_CDF', 'p%d' % i] = inputs['short_term_fairlead_CDF%d' % i]