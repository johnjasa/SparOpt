import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWaveRotspeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_input('Im_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_input('Re_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Im_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Re_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_input('Im_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))

		self.add_output('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		wave_force_surge = inputs['Re_wave_force_surge'] + 1j * inputs['Im_wave_force_surge']
		wave_force_pitch = inputs['Re_wave_force_pitch'] + 1j * inputs['Im_wave_force_pitch']
		wave_force_bend = inputs['Re_wave_force_bend'] + 1j * inputs['Im_wave_force_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_rotspeed = H_feedbk[:,6,3] * wave_force_surge + H_feedbk[:,6,4] * wave_force_pitch + H_feedbk[:,6,5] * wave_force_bend

		outputs['Re_RAO_wave_rotspeed'] = np.real(RAO_wave_rotspeed)
		outputs['Im_RAO_wave_rotspeed'] = np.imag(RAO_wave_rotspeed)

	def compute_partials(self, inputs, partials):
		pass