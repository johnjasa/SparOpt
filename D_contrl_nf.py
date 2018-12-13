import numpy as np

from openmdao.api import ExplicitComponent

class Dcontrl(ExplicitComponent):

	def setup(self):
		self.add_input('k_t', val=0., units='rad*s/m')

		self.add_output('D_contrl', val=np.zeros((2,2)))

		self.declare_partials('D_contrl', 'k_t', val=np.array([[0., 0., 0., 1.]]))

	def compute(self, inputs, outputs):
		outputs['D_contrl'] = np.array([[0., 0.],[0., inputs['k_t']]])