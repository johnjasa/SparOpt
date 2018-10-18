import numpy as np

from openmdao.api import ExplicitComponent

class Acontrl(ExplicitComponent):

	def setup(self):
		self.add_input('omega_lowpass', val=0., units='rad/s')

		self.add_output('A_contrl', val=np.zeros((2,2)))

		#self.declare_partials('A_contrl', 'omega_lowpass', val=np.array([[0., 0.],[0., -1.]]))

	def compute(self, inputs, outputs):
		omega_lowpass = inputs['omega_lowpass']

		outputs['A_contrl'] = np.array([[0., 1.],[0., -omega_lowpass]])