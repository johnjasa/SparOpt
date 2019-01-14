import numpy as np

from openmdao.api import ExplicitComponent

class Bcontrl(ExplicitComponent):

	def setup(self):
		self.add_input('omega_lowpass', val=0., units='rad/s')
		self.add_input('k_t', val=0., units='rad*s/m')

		self.add_output('B_contrl', val=np.zeros((4,2)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['B_contrl'] = np.array([[0., 0.],[inputs['omega_lowpass'], -inputs['omega_lowpass'] * inputs['k_t']],[0., 0.],[1., -inputs['k_t']]])

	def compute_partials(self, inputs, partials): #TODO check
		partials['B_contrl', 'omega_lowpass'] = np.array([[0., 0., 1., 0., 0., 0., 0., 0.]]).T
		partials['B_contrl', 'k_t'] = np.array([[0., 0., 0., -inputs['omega_lowpass'], 0., 0., 0., -1.]]).T