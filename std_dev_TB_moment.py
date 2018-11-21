import numpy as np

from openmdao.api import ExplicitComponent

class StdDevTBMoment(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_TB_moment', val=np.zeros(N_omega), units='(N*m)**2*s/rad')

		self.add_output('stddev_TB_moment', val=0., units='N*m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_TB_moment'] = np.sqrt(np.trapz(inputs['resp_TB_moment'], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_TB_moment', 'resp_TB_moment'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_TB_moment'], omega)) * domega