import numpy as np

from openmdao.api import ExplicitComponent

class HullZeta(ExplicitComponent):

	def setup(self):
		self.add_input('beta', val=np.zeros(10))

		self.add_output('zeta', val=np.zeros(10))

		self.declare_partials('zeta', 'beta', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		beta = inputs['beta']
		zeta = 2. * ((np.sinh(beta) * np.cos(beta) + np.cosh(beta) * np.sin(beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))

		for i in xrange(len(zeta)):
			if zeta[i] < 0.:
				zeta[i] = 0.

		outputs['zeta'] = zeta

	def compute_partials(self, inputs, partials):
		beta = inputs['beta']
		zeta = 2. * ((np.sinh(beta) * np.cos(beta) + np.cosh(beta) * np.sin(beta)) / (np.sinh(2. * beta) + np.sin(2. * beta)))

		for i in xrange(len(zeta)):
			if zeta[i] < 0.:
				zeta[i] = 0.

		dzeta = 2. * (2. * np.cos(beta) * np.cosh(beta) / (np.sinh(2. * beta) + np.sin(2. * beta)) - (2. * np.cos(2. * beta) + 2. * np.cosh(2. * beta)) * (np.cos(beta) * np.sinh(beta) + np.sin(beta) * np.cosh(beta)) / (np.sin(2. * beta) + np.sinh(2. * beta))**2.)
		for i in xrange(len(zeta)):
			if zeta[i] < 0.:
				dzeta[i] = 0

		partials['zeta', 'beta'] = dzeta