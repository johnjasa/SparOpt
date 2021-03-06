import numpy as np

from openmdao.api import ExplicitComponent

class HullFKsd(ExplicitComponent):

	def setup(self):
		self.add_input('f_ks', val=np.zeros(10), units='MPa')
		self.add_input('gamma_M_hull', val=np.zeros(10))

		self.add_output('f_ksd', val=np.ones(10), units='MPa')

		self.declare_partials('f_ksd', 'f_ks', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('f_ksd', 'gamma_M_hull', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['f_ksd'] = inputs['f_ks'] / inputs['gamma_M_hull']

	def compute_partials(self, inputs, partials):
		partials['f_ksd', 'f_ks'] = 1. / inputs['gamma_M_hull']
		partials['f_ksd', 'gamma_M_hull'] = -inputs['f_ks'] / inputs['gamma_M_hull']**2.