import numpy as np

from openmdao.api import ExplicitComponent

class TowerChiX(ExplicitComponent):

	def setup(self):
		self.add_input('lambda_x', val=np.zeros(10))
		self.add_input('alpha_x', val=np.zeros(10))

		self.add_output('chi_x', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		lambda_x = inputs['lambda_x']
		alpha_x = inputs['alpha_x']

		#Eurocode 3 EN 1993-1-6:2007 pp. 76
		lambda_0 = 0.2
		beta_x = 0.6
		eta_x = 1.

		lambda_p = np.sqrt(alpha_x / (1. - beta_x))

		for i in xrange(10):
			if lambda_x[i] <= lambda_0:
				chi_x = 1.
			elif lambda_x[i] >= lambda_p[i]:
				chi_x = alpha_x[i] / lambda_x[i]**2.
			else:
				chi_x = 1. - beta_x * ((lambda_x[i] - lambda_0) / (lambda_p[i] - lambda_0))**eta_x
		
			outputs['chi_x'][i] = chi_x

	def compute_partials(self, inputs, partials):
		lambda_x = inputs['lambda_x']
		alpha_x = inputs['alpha_x']

		#Eurocode 3 EN 1993-1-6:2007 pp. 76
		lambda_0 = 0.2
		beta_x = 0.6
		eta_x = 1.

		lambda_p = np.sqrt(alpha_x / (1. - beta_x))

		partials['chi_x', 'lambda_x'] = np.zeros((10,10))
		partials['chi_x', 'alpha_x'] = np.zeros((10,10))

		for i in xrange(10):
			if lambda_x[i] <= lambda_0:
				continue
			elif lambda_x[i] >= lambda_p[i]:
				partials['chi_x', 'alpha_x'][i,i] += 1. / lambda_x[i]**2.
				partials['chi_x', 'lambda_x'][i,i] += -2. * alpha_x[i] / lambda_x[i]**3.
			else:
				partials['chi_x', 'lambda_x'][i,i] += -eta_x * beta_x * ((lambda_x[i] - lambda_0) / (lambda_p[i] - lambda_0))**(eta_x - 1.) * 1. / (lambda_p[i] - lambda_0)
				partials['chi_x', 'alpha_x'][i,i] += eta_x * beta_x * ((lambda_x[i] - lambda_0) / (lambda_p[i] - lambda_0))**(eta_x - 1.) * (lambda_x[i] - lambda_0) / (lambda_p[i] - lambda_0)**2. * 0.5 / np.sqrt(alpha_x[i] / (1. - beta_x)) * 1. / (1. - beta_x) 