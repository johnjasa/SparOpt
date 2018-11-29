import numpy as np

from openmdao.api import ExplicitComponent

class HullGyrRadius(ExplicitComponent):

	def setup(self):
		self.add_input('I_C', val=np.zeros(10), units='m**4')
		self.add_input('A_C', val=np.zeros(10), units='m**2')

		self.add_output('i_C', val=np.zeros(10), units='m')

		self.declare_partials('i_C', 'I_C', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('i_C', 'A_C', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['i_C'] = np.sqrt(inputs['I_C'] / inputs['A_C'])

	def compute_partials(self, inputs, partials):
		partials['i_C', 'I_C'] = 0.5 / np.sqrt(inputs['I_C'] / inputs['A_C']) * 1. / inputs['A_C']
		partials['i_C', 'A_C'] = -0.5 / np.sqrt(inputs['I_C'] / inputs['A_C']) * inputs['I_C'] / inputs['A_C']**2.