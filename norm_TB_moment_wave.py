import numpy as np

from openmdao.api import ExplicitComponent

class NormTBMomentWave(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_vel_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_vel_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_vel_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_vel_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_acc_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_acc_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_acc_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_acc_bend', val=np.zeros(N_omega), units='m/m')

		self.add_output('Re_RAO_wave_TB_moment', val=np.zeros(N_omega), units='N*m/m')
		self.add_output('Im_RAO_wave_TB_moment', val=np.zeros(N_omega), units='N*m/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		RAO_wave_acc_surge = inputs['Re_RAO_wave_acc_surge'] + 1j * inputs['Im_RAO_wave_acc_surge']
		RAO_wave_acc_pitch = inputs['Re_RAO_wave_acc_pitch'] + 1j * inputs['Im_RAO_wave_acc_pitch']
		RAO_wave_acc_bend = inputs['Re_RAO_wave_acc_bend'] + 1j * inputs['Im_RAO_wave_acc_bend']
		RAO_wave_vel_surge = inputs['Re_RAO_wave_vel_surge'] + 1j * inputs['Im_RAO_wave_vel_surge']
		RAO_wave_vel_pitch = inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch']
		RAO_wave_vel_bend = inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend']
		RAO_wave_pitch = inputs['Re_RAO_wave_pitch'] + 1j * inputs['Im_RAO_wave_pitch']
		RAO_wave_bend = inputs['Re_RAO_wave_bend'] + 1j * inputs['Im_RAO_wave_bend']
		RAO_wave_rotspeed = inputs['Re_RAO_wave_rotspeed'] + 1j * inputs['Im_RAO_wave_rotspeed']
		RAO_wave_bldpitch = inputs['Re_RAO_wave_bldpitch'] + 1j * inputs['Im_RAO_wave_bldpitch']

		RAO_wave_TB_moment = -mom_acc_surge * RAO_wave_acc_surge[i] - mom_acc_pitch * RAO_wave_acc_pitch[i] - mom_acc_bend * RAO_wave_acc_bend[i] - mom_damp_surge * RAO_wave_vel_surge[i] - mom_damp_pitch * RAO_wave_vel_pitch[i] - mom_damp_bend * RAO_wave_vel_bend[i] + mom_grav_pitch * RAO_wave_pitch[i] + mom_grav_bend * RAO_wave_bend[i] + mom_rotspeed * RAO_wave_rotspeed[i] + mom_bldpitch * RAO_wave_bldpitch[i]

		outputs['Re_RAO_wave_TB_moment'] = np.real(RAO_wave_TB_moment)

		outputs['Im_RAO_wave_TB_moment'] = np.imag(RAO_wave_TB_moment)