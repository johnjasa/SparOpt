import numpy as np

from openmdao.api import ExplicitComponent

class HullDelta0(ExplicitComponent):

	def setup(self):
		self.add_input('r_hull', val=np.zeros(10), units='m')

		self.add_output('delta_0', val=np.zeros(10), units='m')

		self.declare_partials('delta_0', 'r_hull', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['delta_0'] = 0.005 * inputs['r_hull']

	def compute_partials(self, inputs, partials):
		partials['delta_0', 'r_hull'] = 0.005 * np.ones(10)