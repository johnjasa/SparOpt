import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWave(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		self.omega_wave = freqs['omega_wave']
		N_omega = len(self.omega)
		N_omega_wave = len(self.omega_wave)

		self.add_input('Re_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')
		self.add_input('Im_wave_forces', val=np.zeros((N_omega_wave,3,1)), units='N/m')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)

		self.add_output('Re_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_output('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_output('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Re_RAO_wave_rot_lp', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Re_RAO_wave_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Im_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_output('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_output('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Im_RAO_wave_rot_lp', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Im_RAO_wave_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega
		omega_wave = self.omega_wave

		wave_forces = inputs['Re_wave_forces'] + 1j * inputs['Im_wave_forces']
		wave_force_surge = wave_forces[:,0,0]
		wave_force_pitch = wave_forces[:,1,0]
		wave_force_bend = wave_forces[:,2,0]

		wave_force_surge = np.interp(omega, omega_wave, wave_force_surge)
		wave_force_pitch = np.interp(omega, omega_wave, wave_force_pitch)
		wave_force_bend = np.interp(omega, omega_wave, wave_force_bend)

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_surge = H_feedbk[:,0,3] * wave_force_surge + H_feedbk[:,0,4] * wave_force_pitch + H_feedbk[:,0,5] * wave_force_bend
		RAO_wave_pitch = H_feedbk[:,1,3] * wave_force_surge + H_feedbk[:,1,4] * wave_force_pitch + H_feedbk[:,1,5] * wave_force_bend
		RAO_wave_bend = H_feedbk[:,2,3] * wave_force_surge + H_feedbk[:,2,4] * wave_force_pitch + H_feedbk[:,2,5] * wave_force_bend
		RAO_wave_rotspeed = H_feedbk[:,6,3] * wave_force_surge + H_feedbk[:,6,4] * wave_force_pitch + H_feedbk[:,6,5] * wave_force_bend
		RAO_wave_rot_lp = H_feedbk[:,7,3] * wave_force_surge + H_feedbk[:,7,4] * wave_force_pitch + H_feedbk[:,7,5] * wave_force_bend
		RAO_wave_rotspeed_lp = H_feedbk[:,8,3] * wave_force_surge + H_feedbk[:,8,4] * wave_force_pitch + H_feedbk[:,8,5] * wave_force_bend
		RAO_wave_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_wave_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_wave_rotspeed_lp

		outputs['Re_RAO_wave_surge'] = np.real(RAO_wave_surge)
		outputs['Re_RAO_wave_pitch'] = np.real(RAO_wave_pitch)
		outputs['Re_RAO_wave_bend'] = np.real(RAO_wave_bend)
		outputs['Re_RAO_wave_rotspeed'] = np.real(RAO_wave_rotspeed)
		outputs['Re_RAO_wave_rot_lp'] = np.real(RAO_wave_rot_lp)
		outputs['Re_RAO_wave_rotspeed_lp'] = np.real(RAO_wave_rotspeed_lp)
		outputs['Re_RAO_wave_bldpitch'] = np.real(RAO_wave_bldpitch)

		outputs['Im_RAO_wave_surge'] = np.imag(RAO_wave_surge)
		outputs['Im_RAO_wave_pitch'] = np.imag(RAO_wave_pitch)
		outputs['Im_RAO_wave_bend'] = np.imag(RAO_wave_bend)
		outputs['Im_RAO_wave_rotspeed'] = np.imag(RAO_wave_rotspeed)
		outputs['Im_RAO_wave_rot_lp'] = np.imag(RAO_wave_rot_lp)
		outputs['Im_RAO_wave_rotspeed_lp'] = np.imag(RAO_wave_rotspeed_lp)
		outputs['Im_RAO_wave_bldpitch'] = np.imag(RAO_wave_bldpitch)