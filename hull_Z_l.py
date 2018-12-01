import numpy as np

from openmdao.api import ExplicitComponent

class HullZL(ExplicitComponent):

	def setup(self):
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('Z_l', val=np.zeros(10))

		self.declare_partials('Z_l', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('Z_l', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('Z_l', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		nu = 0.3

		outputs['Z_l'] = inputs['l_stiff']**2. / (inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * np.sqrt(1. - nu**2.)

	def compute_partials(self, inputs, partials):
		nu = 0.3

		partials['Z_l', 'l_stiff'] = 2. * inputs['l_stiff'] / (inputs['r_hull'] * inputs['wt_spar_p'][:-1]) * np.sqrt(1. - nu**2.)
		partials['Z_l', 'r_hull'] = -inputs['l_stiff']**2. / (inputs['r_hull']**2. * inputs['wt_spar_p'][:-1]) * np.sqrt(1. - nu**2.)
		partials['Z_l', 'wt_spar_p'] = -inputs['l_stiff']**2. / (inputs['r_hull'] * inputs['wt_spar_p'][:-1]**2.) * np.sqrt(1. - nu**2.)