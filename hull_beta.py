import numpy as np

from openmdao.api import ExplicitComponent

class HullBeta(ExplicitComponent):

	def setup(self):
		self.add_input('l_stiff', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')

		self.add_output('beta', val=np.zeros(10))

		self.declare_partials('beta', 'l_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('beta', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('beta', 'wt_spar', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['beta'] = inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar']))

	def compute_partials(self, inputs, partials):
		partials['beta', 'l_stiff'] = 1. / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar']))
		partials['beta', 'r_hull'] = -inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar']))**2. * 1.56 * 0.5 / np.sqrt(inputs['r_hull'] * inputs['wt_spar']) * inputs['wt_spar']
		partials['beta', 'wt_spar'] = -inputs['l_stiff'] / (1.56 * np.sqrt(inputs['r_hull'] * inputs['wt_spar']))**2. * 1.56 * 0.5 / np.sqrt(inputs['r_hull'] * inputs['wt_spar']) * inputs['r_hull']