import numpy as np

from openmdao.api import ExplicitComponent

class TowerAlphaX(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')

		self.add_output('alpha_x', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		#"High" fabrication quality
		Q = 25.

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.
		delta_w_k = 1. / Q * np.sqrt(r / wt_tower_p[:-1]) * wt_tower_p[:-1]

		outputs['alpha_x'] = 0.62 / (1. + 1.91 * (delta_w_k / wt_tower_p[:-1])**1.44)

	def compute_partials(self, inputs, partials):
		D_tower_p = inputs['D_tower_p']
		wt_tower_p = inputs['wt_tower_p']

		#"High" fabrication quality
		Q = 25.

		r = (D_tower_p[:-1] - wt_tower_p[:-1]) / 2.
		delta_w_k = 1. / Q * np.sqrt(r / wt_tower_p[:-1]) * wt_tower_p[:-1]

		partials['alpha_x', 'D_tower_p'] = np.zeros((10,11))
		partials['alpha_x', 'wt_tower_p'] = np.zeros((10,11))

		for i in xrange(10):
			partials['alpha_x', 'D_tower_p'][i,i] += -0.62 / (1. + 1.91 * (delta_w_k[i] / wt_tower_p[i])**1.44)**2. * 1.44 * 1.91 * (delta_w_k[i] / wt_tower_p[i])**0.44 * 1. / wt_tower_p[i] * 1. / Q * 0.5 / np.sqrt(r[i] / wt_tower_p[i]) * 1. / wt_tower_p[i] * 0.5 * wt_tower_p[i]
			partials['alpha_x', 'wt_tower_p'][i,i] += -0.62 / (1. + 1.91 * (delta_w_k[i] / wt_tower_p[i])**1.44)**2. * 1.44 * 1.91 * (delta_w_k[i] / wt_tower_p[i])**0.44 * (-delta_w_k[i] / wt_tower_p[i]**2. + 1. / wt_tower_p[i] * (1. / Q * np.sqrt(r[i] / wt_tower_p[i]) + wt_tower_p[i] * 1. / Q * 0.5 / np.sqrt(r[i] / wt_tower_p[i]) * (-0.5 / wt_tower_p[i] - r[i] / wt_tower_p[i]**2.)))