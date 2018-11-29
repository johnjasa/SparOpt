import numpy as np

from openmdao.api import ExplicitComponent

class ShellBuckling(ExplicitComponent):

	def setup(self):
		self.add_input('sigma_j', val=np.zeros(10), units='MPa')
		self.add_input('f_ksd', val=np.zeros(10), units='MPa')

		self.add_output('shell_buckling', val=np.zeros(10))

		self.declare_partials('shell_buckling', 'sigma_j', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('shell_buckling', 'f_ksd', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['shell_buckling'] = inputs['sigma_j'] / inputs['f_ksd'] - 1. #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		partials['shell_buckling', 'sigma_j'] = 1. / inputs['f_ksd']
		partials['shell_buckling', 'f_ksd'] = -inputs['sigma_j'] / inputs['f_ksd']**2.
		