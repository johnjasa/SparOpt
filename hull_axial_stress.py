import numpy as np

from openmdao.api import ExplicitComponent

class HullAxialStress(ExplicitComponent):

	def setup(self):
		self.add_input('N_hull', val=np.zeros(10), units='N')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('sigma_a', val=np.ones(10), units='MPa')

		self.declare_partials('sigma_a', 'N_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_a', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_a', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['sigma_a'] = inputs['N_hull'] / (2. * np.pi * inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * 1e-6

	def compute_partials(self, inputs, partials):
		partials['sigma_a', 'N_hull'] = 1. / (2. * np.pi * inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * 1e-6
		partials['sigma_a', 'r_hull'] = -inputs['N_hull'] / (2. * np.pi * inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * 1e-6
		partials['sigma_a', 'wt_spar_p'] = -inputs['N_hull'] / (2. * np.pi * inputs['r_hull'] * inputs['wt_spar_p'][:-1]**2.) * 1e-6