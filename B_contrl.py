import numpy as np

from openmdao.api import ExplicitComponent

class Bcontrl(ExplicitComponent):

	def setup(self):
		self.add_input('omega_lowpass', val=0., units='rad/s')

		self.add_output('B_contrl', val=np.zeros((2,1)))

		#self.declare_partials('B_contrl', 'omega_lowpass', val=np.array([[0.],[1.]]))

	def compute(self, inputs, outputs):
		omega_lowpass = inputs['omega_lowpass']

		outputs['B_contrl'] = np.array([[0.],[omega_lowpass]])