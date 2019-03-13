import numpy as np

from openmdao.api import ExplicitComponent

class StdDevMoorTanVel(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_moor_tan_vel', val=np.zeros(N_omega), units='(m/s)**2*s/rad')

		self.add_output('stddev_moor_tan_vel', val=0., units='m/s')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_moor_tan_vel'] = np.sqrt(np.trapz(inputs['resp_moor_tan_vel'], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_moor_tan_vel', 'resp_moor_tan_vel'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_moor_tan_vel'], omega)) * domega
		partials['stddev_moor_tan_vel', 'resp_moor_tan_vel'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_moor_tan_vel'], omega)) * domega / 2.
		partials['stddev_moor_tan_vel', 'resp_moor_tan_vel'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_moor_tan_vel'], omega)) * domega / 2.