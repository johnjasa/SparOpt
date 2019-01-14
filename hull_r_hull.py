import numpy as np

from openmdao.api import ExplicitComponent

class HullRHull(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('r_hull', val=np.ones(10), units='m')

		self.declare_partials('r_hull', 'D_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_hull', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['r_hull'] = 0.5 * inputs['D_spar_p'][:-1] - 0.5 * inputs['wt_spar_p'][:-1]

	def compute_partials(self, inputs, partials):
		partials['r_hull', 'D_spar_p'] = 0.5 * np.ones(10)
		partials['r_hull', 'wt_spar_p'] = -0.5 * np.ones(10)