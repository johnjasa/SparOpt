import numpy as np

from openmdao.api import ExplicitComponent

class LongTermMyMomInertiaCDF(ExplicitComponent):

	def initialize(self):
		self.options.declare('EC', types=dict)

	def setup(self):
		EC = self.options['EC']
		self.N_EC = EC['N_EC']
		
		for i in xrange(self.N_EC):
			self.add_input('short_term_My_mom_inertia_CDF%d' % i, val=np.zeros(10))
			self.add_input('p%d' % i, val=0.)

		self.add_output('long_term_My_mom_inertia_CDF', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['long_term_My_mom_inertia_CDF'] = 0.
		for i in xrange(self.N_EC):
			outputs['long_term_My_mom_inertia_CDF'] += inputs['short_term_My_mom_inertia_CDF%d' % i] * inputs['p%d' % i]
	
	def compute_partials(self, inputs, partials):
		for i in xrange(self.N_EC):
			for j in xrange(10):
				partials['long_term_My_mom_inertia_CDF', 'short_term_My_mom_inertia_CDF%d' % i][j,j] = inputs['p%d' % i]
			
			partials['long_term_My_mom_inertia_CDF', 'p%d' % i][:,0] = inputs['short_term_My_mom_inertia_CDF%d' % i]