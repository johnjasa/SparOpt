import numpy as np

from openmdao.api import ExplicitComponent

class Bcontrl(ExplicitComponent):

	def setup(self):
		self.add_input('omega_lowpass', val=0., units='rad/s')

		self.add_output('B_contrl', val=np.zeros((2,2)))

		self.declare_partials('B_contrl', 'omega_lowpass', val=np.array([[0., 0., 1., 0.]]))

	def compute(self, inputs, outputs):
		outputs['B_contrl'] = np.array([[0., 0.],[inputs['omega_lowpass'], 0.]])