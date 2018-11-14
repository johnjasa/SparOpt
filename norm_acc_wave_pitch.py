import numpy as np

from openmdao.api import ExplicitComponent

class NormAccWavePitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')

		self.add_output('Re_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/m')
		self.add_output('Im_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wave_acc_pitch'] = -inputs['Re_RAO_wave_pitch'] * omega**2.
		outputs['Im_RAO_wave_acc_pitch'] = -inputs['Im_RAO_wave_pitch'] * omega**2.