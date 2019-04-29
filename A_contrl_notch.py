import numpy as np

from openmdao.api import ExplicitComponent

class Acontrl(ExplicitComponent):

	def setup(self):
		self.add_input('omega_lowpass', val=0., units='rad/s')
		self.add_input('omega_notch', val=0., units='rad/s')
		self.add_input('bandwidth_notch', val=0., units='rad/s')

		self.add_output('A_contrl', val=np.zeros((4,4)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega_lowpass = inputs['omega_lowpass']
		omega_notch = inputs['omega_notch']
		bandwidth_notch = inputs['bandwidth_notch']

		outputs['A_contrl'] = np.array([[0., 1., 0., 0.],[0., -omega_lowpass, 0., -bandwidth_notch * omega_lowpass], [0., 0., 0., 1.], [0., 0., -omega_notch**2., -bandwidth_notch]])

	def compute_partials(self, inputs, partials):
		omega_lowpass = inputs['omega_lowpass']
		omega_notch = inputs['omega_notch']
		bandwidth_notch = inputs['bandwidth_notch']

		partials['A_contrl', 'omega_lowpass'] = np.array([[0., 0., 0., 0., 0., -1., 0., -bandwidth_notch, 0., 0., 0., 0., 0., 0., 0., 0.]], dtype='float').T
		partials['A_contrl', 'omega_notch'] = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., -2. * omega_notch, 0.]], dtype='float').T
		partials['A_contrl', 'bandwidth_notch'] = np.array([[0., 0., 0., 0., 0., 0., 0., -omega_lowpass, 0., 0., 0., 0., 0., 0., 0., -1.]], dtype='float').T
