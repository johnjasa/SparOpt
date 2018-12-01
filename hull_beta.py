import numpy as np

from openmdao.api import ExplicitComponent

class HullBeta(ExplicitComponent):

	def setup(self):
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('beta', val=np.zeros(10))

		self.declare_partials('beta', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('beta', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('beta', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['beta'] = inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]))

	def compute_partials(self, inputs, partials):
		partials['beta', 'l_stiff'] = 1. / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]))
		partials['beta', 'r_hull'] = -inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]))**2. * 1.56 * 0.5 / np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * inputs['wt_spar_p'][:-1]
		partials['beta', 'wt_spar_p'] = -inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]))**2. * 1.56 * 0.5 / np.sqrt(inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * inputs['r_hull']