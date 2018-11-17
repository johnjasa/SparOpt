import numpy as np

from openmdao.api import ExplicitComponent

class NormVelWaveBend(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')

		self.add_output('Re_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_output('Im_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')

		self.declare_partials('Re_RAO_wave_vel_bend', 'Re_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_vel_bend', 'Im_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_vel_bend', 'Re_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_vel_bend', 'Im_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wave_vel_bend'] = -inputs['Im_RAO_wave_bend'] * omega
		outputs['Im_RAO_wave_vel_bend'] = inputs['Re_RAO_wave_bend'] * omega

	def compute_partials(self, inputs, partials): #TODO check
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_wave_vel_bend', 'Re_RAO_wave_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wave_vel_bend', 'Im_RAO_wave_bend'] = -omega
		partials['Im_RAO_wave_vel_bend', 'Re_RAO_wave_bend'] = omega
		partials['Im_RAO_wave_vel_bend', 'Im_RAO_wave_bend'] = np.zeros(N_omega)