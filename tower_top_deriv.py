import numpy as np

from openmdao.api import ExplicitComponent

class TowerTopDeriv(ExplicitComponent):

	def setup(self):
		self.add_input('x_d_towernode', val=np.zeros(11), units='m/m')

		self.add_output('x_d_towertop', val=0., units='m/m')

		self.declare_partials('x_d_towertop', 'x_d_towernode', val=np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]))

	def compute(self, inputs, outputs):
		outputs['x_d_towertop'] = inputs['x_d_towernode'][-1]