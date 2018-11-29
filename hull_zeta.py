import numpy as np

from openmdao.api import ExplicitComponent

class HullZeta(ExplicitComponent):

	def setup(self):
		self.add_input('beta', val=np.zeros(10))

		self.add_output('zeta', val=np.zeros(10))

		self.declare_partials('zeta', 'beta', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		zeta = 2. * ((np.sinh(beta) * np.cos(beta) + np.cosh(beta) * np.sin(beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))
		if zeta < 0.:
			zeta = 0.

		outputs['zeta'] = zeta

	def compute_partials(self, inputs, partials):
		zeta = 2. * ((np.sinh(beta) * np.cos(beta) + np.cosh(beta) * np.sin(beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))
		if zeta < 0.:
			zeta = 0.

		dzeta = 2. * (2. * np.cos(beta) * np.cosh(beta) / (np.sinh(2. * beta) + np.sin(2. * beta)) - (2. * np.cos(2. * beta) + 2. * np.cosh(2. * beta)) * (np.cos(beta) * np.sinh(beta) + np.sin(beta) * np.cosh(beta)) / (np.sin(2. * beta) + np.sinh(2. * beta))**2.)
		if zeta == 0.:
			dzeta = 0

		partials['zeta', 'beta'] = dzeta