import numpy as np

from openmdao.api import ExplicitComponent

class HullAC(ExplicitComponent):

	def setup(self):
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')

		self.add_output('A_C', val=np.zeros(10), units='m**2')

		self.declare_partials('A_C', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('A_C', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['A_C'] = 2. * np.pi * inputs['r_hull'] * inputs['wt_spar_p'][:-1]

	def compute_partials(self, inputs, partials):
		partials['A_C', 'r_hull'] = 2. * np.pi * inputs['wt_spar_p'][:-1]
		partials['A_C', 'wt_spar_p'] = 2. * np.pi * inputs['r_hull']