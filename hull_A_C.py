import numpy as np

from openmdao.api import ExplicitComponent

class HullAC(ExplicitComponent):

	def setup(self):
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')

		self.add_output('A_C', val=np.zeros(10), units='m**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['A_C'] = 2. * np.pi * inputs['r_hull'] * inputs['wt_spar']

	def compute_partials(self, inputs, partials):
		partials['A_C', 'r_hull'] = 2. * np.pi * inputs['wt_spar']
		partials['A_C', 'wt_spar'] = 2. * np.pi * inputs['r_hull']