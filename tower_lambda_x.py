import numpy as np

from openmdao.api import ExplicitComponent

class TowerLambdaX(ExplicitComponent):

	def setup(self):
		self.add_input('f_y', val=0., units='MPa')
		self.add_input('sigma_x_Rcr', val=np.zeros(10), units='MPa')

		self.add_output('lambda_x', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):

		outputs['lambda_x'] = np.sqrt(inputs['f_y'] / inputs['sigma_x_Rcr'])

	def compute_partials(self, inputs, partials):

		partials['lambda_x', 'f_y'] = np.zeros((10,1))
		partials['lambda_x', 'sigma_x_Rcr'] = np.zeros((10,10))

		for i in xrange(10):
			partials['lambda_x', 'f_y'][i,0] += 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_x_Rcr'][i]) * 1. / inputs['sigma_x_Rcr'][i]
			partials['lambda_x', 'sigma_x_Rcr'][i,i] += -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_x_Rcr'][i]) * inputs['f_y'] / inputs['sigma_x_Rcr'][i]**2.