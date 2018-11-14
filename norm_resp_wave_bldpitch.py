import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWaveBldpitch(ExplicitComponent):

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
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)

		self.add_output('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		wave_force_surge = inputs['Re_wave_force_surge'] + 1j * inputs['Im_wave_force_surge']
		wave_force_pitch = inputs['Re_wave_force_pitch'] + 1j * inputs['Im_wave_force_pitch']
		wave_force_bend = inputs['Re_wave_force_bend'] + 1j * inputs['Im_wave_force_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_rot_lp = H_feedbk[:,7,3] * wave_force_surge + H_feedbk[:,7,4] * wave_force_pitch + H_feedbk[:,7,5] * wave_force_bend
		RAO_wave_rotspeed_lp = H_feedbk[:,8,3] * wave_force_surge + H_feedbk[:,8,4] * wave_force_pitch + H_feedbk[:,8,5] * wave_force_bend
		RAO_wave_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_wave_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_wave_rotspeed_lp

		outputs['Re_RAO_wave_bldpitch'] = np.real(RAO_wave_bldpitch)
		outputs['Im_RAO_wave_bldpitch'] = np.imag(RAO_wave_bldpitch)

	def compute_partials(self, inputs, partials):
		pass