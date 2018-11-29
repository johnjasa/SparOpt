import numpy as np

from openmdao.api import ExplicitComponent

class HullZT(ExplicitComponent):

	def setup(self):
		self.add_input('r_0', val=np.zeros(10), units='m')
		self.add_input('r_f', val=np.zeros(10), units='m')

		self.add_output('z_t', val=np.zeros(10), units='m')

		self.declare_partials('z_t', 'r_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('z_t', 'r_f', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['z_t'] = inputs['r_0'] - inputs['r_f']

	def compute_partials(self, inputs, partials):
		partials['z_t', 'r_0'] = np.ones(10)
		partials['z_t', 'r_f'] = -np.ones(10)