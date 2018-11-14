import numpy as np

from openmdao.api import ExplicitComponent

class NormVelWaveSurge(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')

		self.add_output('Re_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_output('Im_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wave_vel_surge'] = -inputs['Im_RAO_wave_surge'] * omega
		outputs['Im_RAO_wave_vel_surge'] = inputs['Re_RAO_wave_surge'] * omega