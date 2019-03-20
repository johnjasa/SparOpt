import numpy as np

from openmdao.api import ExplicitComponent

class HullLambdaS(ExplicitComponent):

	def setup(self):
		self.add_input('f_y', val=0., units='MPa')
		self.add_input('sigma_j', val=np.zeros(10), units='MPa')
		self.add_input('sigma_a0', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m0', val=np.zeros(10), units='MPa')
		self.add_input('sigma_h0', val=np.zeros(10), units='MPa')
		#self.add_input('tau', val=np.ones(10), units='MPa')
		self.add_input('f_Ea', val=np.zeros(10), units='MPa')
		self.add_input('f_Em', val=np.zeros(10), units='MPa')
		self.add_input('f_Eh', val=np.zeros(10), units='MPa')
		#self.add_input('f_Etau', val=np.zeros(10), units='MPa')

		self.add_output('lambda_s', val=np.ones(10))

		self.declare_partials('lambda_s', 'f_y')
		self.declare_partials('lambda_s', 'sigma_j', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'sigma_a0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'sigma_m0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'sigma_h0', rows=np.arange(10), cols=np.arange(10))
		#self.declare_partials('lambda_s', 'tau', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'f_Ea', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'f_Em', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('lambda_s', 'f_Eh', rows=np.arange(10), cols=np.arange(10))
		#self.declare_partials('lambda_s', 'f_Etau', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		#outputs['lambda_s'] = np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau']))
		outputs['lambda_s'] = np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh']))

	def compute_partials(self, inputs, partials):
		#partials['lambda_s', 'f_y'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * 1. / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])
		#partials['lambda_s', 'sigma_j'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j']**2. * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])
		#partials['lambda_s', 'sigma_a0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Ea']
		#partials['lambda_s', 'sigma_m0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Em']
		#partials['lambda_s', 'sigma_h0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Eh']
		#partials['lambda_s', 'tau'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Etau']
		#partials['lambda_s', 'f_Ea'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_a0'] / inputs['f_Ea']**2.
		#partials['lambda_s', 'f_Em'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_m0'] / inputs['f_Em']**2.
		#partials['lambda_s', 'f_Eh'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_h0'] / inputs['f_Eh']**2.
		#partials['lambda_s', 'f_Etau'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'] + inputs['tau'] / inputs['f_Etau'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['tau'] / inputs['f_Etau']**2.
		partials['lambda_s', 'f_y'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * 1. / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])
		partials['lambda_s', 'sigma_j'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j']**2. * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])
		partials['lambda_s', 'sigma_a0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Ea']
		partials['lambda_s', 'sigma_m0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Em']
		partials['lambda_s', 'sigma_h0'] = 0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * 1. / inputs['f_Eh']
		partials['lambda_s', 'f_Ea'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_a0'] / inputs['f_Ea']**2.
		partials['lambda_s', 'f_Em'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_m0'] / inputs['f_Em']**2.
		partials['lambda_s', 'f_Eh'] = -0.5 / np.sqrt(inputs['f_y'] / inputs['sigma_j'] * (inputs['sigma_a0'] / inputs['f_Ea'] + inputs['sigma_m0'] / inputs['f_Em'] + inputs['sigma_h0'] / inputs['f_Eh'])) * inputs['f_y'] / inputs['sigma_j'] * inputs['sigma_h0'] / inputs['f_Eh']**2.