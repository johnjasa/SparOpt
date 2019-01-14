import numpy as np

from openmdao.api import ExplicitComponent

class TowerCritBucklingStress(ExplicitComponent):

	def setup(self):
		self.add_input('D_tower_p', val=np.zeros(11), units='m')
		self.add_input('wt_tower_p', val=np.zeros(11), units='m')
		self.add_input('C_x', val=np.zeros(10))

		self.add_output('sigma_x_Rcr', val=np.zeros(10), units='MPa')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		r = (inputs['D_tower_p'][:-1] - inputs['wt_tower_p'][:-1]) / 2.

		outputs['sigma_x_Rcr'] = 0.605 * E * inputs['C_x'] * inputs['wt_tower_p'][:-1] / r

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		r = (inputs['D_tower_p'][:-1] - inputs['wt_tower_p'][:-1]) / 2.

		partials['sigma_x_Rcr', 'D_tower_p'] = np.zeros((10,11))
		partials['sigma_x_Rcr', 'wt_tower_p'] = np.zeros((10,11))
		partials['sigma_x_Rcr', 'C_x'] = np.zeros((10,10))

		for i in xrange(10):
			partials['sigma_x_Rcr', 'D_tower_p'][i,i] += -0.605 * E * inputs['C_x'][i] * inputs['wt_tower_p'][i] / r[i]**2. * 0.5
			partials['sigma_x_Rcr', 'wt_tower_p'][i,i] += 0.605 * E * inputs['C_x'][i] * (1. / r[i] + inputs['wt_tower_p'][i] / r[i]**2. * 0.5)
			partials['sigma_x_Rcr', 'C_x'][i,i] += 0.605 * E * inputs['wt_tower_p'][i] / r[i]