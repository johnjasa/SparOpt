import numpy as np

from openmdao.api import ExplicitComponent

class HullAlpha(ExplicitComponent):

	def setup(self):
		self.add_input('A_R', val=np.zeros(10), units='m**2')
		self.add_input('l_eo', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('alpha', val=np.zeros(10))

		self.declare_partials('alpha', 'A_R', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('alpha', 'l_eo', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('alpha', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['alpha'] = inputs['A_R'] / (inputs['l_eo'] * inputs['wt_spar_p'][:-1])

	def compute_partials(self, inputs, partials):
		partials['alpha', 'A_R'] = 1. / (inputs['l_eo'] * inputs['wt_spar_p'][:-1])
		partials['alpha', 'l_eo'] = -inputs['A_R'] / (inputs['l_eo']**2. * inputs['wt_spar_p'][:-1])
		partials['alpha', 'wt_spar_p'] = -inputs['A_R'] / (inputs['l_eo'] * inputs['wt_spar_p'][:-1]**2.)