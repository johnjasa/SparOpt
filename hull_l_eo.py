import numpy as np

from openmdao.api import ExplicitComponent

class HullLEo(ExplicitComponent):

	def setup(self):
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('beta', val=np.zeros(10))

		self.add_output('l_eo', val=np.ones(10), units='m')

		self.declare_partials('l_eo', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('l_eo', 'beta', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		l_stiff = inputs['l_stiff']
		beta = inputs['beta']

		outputs['l_eo'] = l_stiff / beta * ((np.cosh(2. * beta) - np.cos(2. * beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))

	def compute_partials(self, inputs, partials):
		l_stiff = inputs['l_stiff']
		beta = inputs['beta']

		partials['l_eo', 'l_stiff'] = 1. / beta * ((np.cosh(2. * beta) - np.cos(2. * beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))
		partials['l_eo', 'beta'] = -l_stiff / beta**2. * ((np.cosh(2. * beta) - np.cos(2. * beta)) / (np.sinh(2. * beta) + np.sin(2. * beta))) + l_stiff / beta * (2. - (np.cosh(2. * beta) - np.cos(2. * beta)) * (2. * np.cos(2. * beta) + 2. * np.cosh(2. * beta)) / (np.sinh(2.*beta) + np.sin(2.*beta))**2.)