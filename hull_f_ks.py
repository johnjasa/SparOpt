import numpy as np

from openmdao.api import ExplicitComponent

class HullFKs(ExplicitComponent):

	def setup(self):
		self.add_input('f_y', val=0., units='MPa')
		self.add_input('lambda_s', val=np.zeros(10))

		self.add_output('f_ks', val=np.zeros(10), units='MPa')

		self.declare_partials('f_ks', 'f_y')
		self.declare_partials('f_ks', 'lambda_s', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['f_ks'] = inputs['f_y'] / np.sqrt(1. + inputs['lambda_s']**4.)

	def compute_partials(self, inputs, partials):

		partials['f_ks', 'f_y'] = 1. / np.sqrt(1. + inputs['lambda_s']**4.)
		partials['f_ks', 'lambda_s'] = -2. * inputs['f_y'] / (1. + inputs['lambda_s']**4.)**(3./2.) * inputs['lambda_s']**3.