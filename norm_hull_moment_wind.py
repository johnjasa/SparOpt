import numpy as np

from openmdao.api import ExplicitComponent

class NormHullMomentWind(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('hull_mom_acc_surge', val=np.zeros(10), units='kg*m')
		self.add_input('hull_mom_acc_pitch', val=np.zeros(10), units='kg*m**2/rad')
		self.add_input('hull_mom_acc_bend', val=np.zeros(10), units='kg*m')
		self.add_input('hull_mom_damp_surge', val=np.zeros(10), units='N*s')
		self.add_input('hull_mom_damp_pitch', val=np.zeros(10), units='N*m*s/rad')
		self.add_input('hull_mom_damp_bend', val=np.zeros(10), units='N*s')
		self.add_input('hull_mom_grav_pitch', val=np.zeros(10), units='N*m/rad')
		self.add_input('hull_mom_grav_bend', val=np.zeros(10), units='N')
		self.add_input('hull_mom_rotspeed', val=np.zeros(10), units='N*m*s/rad')
		self.add_input('hull_mom_bldpitch', val=np.zeros(10), units='N*m/rad')
		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_input('Re_RAO_wind_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/(m/s)')
		self.add_input('Re_RAO_wind_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_input('Im_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_input('Im_RAO_wind_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/(m/s)')
		self.add_input('Im_RAO_wind_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_input('CoG_rotor', val=0., units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('thrust_wind', val=np.zeros(N_omega), units='m/s')

		self.add_output('Re_RAO_wind_hull_moment', val=np.zeros((N_omega,10)), units='N*m/(m/s)')
		self.add_output('Im_RAO_wind_hull_moment', val=np.zeros((N_omega,10)), units='N*m/(m/s)')

	def compute(self, inputs, outputs):
		hull_mom_acc_surge = inputs['hull_mom_acc_surge']
		hull_mom_acc_pitch = inputs['hull_mom_acc_pitch']
		hull_mom_acc_bend = inputs['hull_mom_acc_bend']
		hull_mom_damp_surge = inputs['hull_mom_damp_surge']
		hull_mom_damp_pitch = inputs['hull_mom_damp_pitch']
		hull_mom_damp_bend = inputs['hull_mom_damp_bend']
		hull_mom_grav_pitch = inputs['hull_mom_grav_pitch']
		hull_mom_grav_bend = inputs['hull_mom_grav_bend']
		hull_mom_rotspeed = inputs['hull_mom_rotspeed']
		hull_mom_bldpitch = inputs['hull_mom_bldpitch']

		CoG_rotor = inputs['CoG_rotor'][0]
		Z_spar = inputs['Z_spar']
		dthrust_dv = inputs['dthrust_dv'][0]
		thrust_wind = inputs['thrust_wind']

		RAO_wind_acc_surge = inputs['Re_RAO_wind_acc_surge'] + 1j * inputs['Im_RAO_wind_acc_surge']
		RAO_wind_acc_pitch = inputs['Re_RAO_wind_acc_pitch'] + 1j * inputs['Im_RAO_wind_acc_pitch']
		RAO_wind_acc_bend = inputs['Re_RAO_wind_acc_bend'] + 1j * inputs['Im_RAO_wind_acc_bend']
		RAO_wind_vel_surge = inputs['Re_RAO_wind_vel_surge'] + 1j * inputs['Im_RAO_wind_vel_surge']
		RAO_wind_vel_pitch = inputs['Re_RAO_wind_vel_pitch'] + 1j * inputs['Im_RAO_wind_vel_pitch']
		RAO_wind_vel_bend = inputs['Re_RAO_wind_vel_bend'] + 1j * inputs['Im_RAO_wind_vel_bend']
		RAO_wind_pitch = inputs['Re_RAO_wind_pitch'] + 1j * inputs['Im_RAO_wind_pitch']
		RAO_wind_bend = inputs['Re_RAO_wind_bend'] + 1j * inputs['Im_RAO_wind_bend']
		RAO_wind_rotspeed = inputs['Re_RAO_wind_rotspeed'] + 1j * inputs['Im_RAO_wind_rotspeed']
		RAO_wind_bldpitch = inputs['Re_RAO_wind_bldpitch'] + 1j * inputs['Im_RAO_wind_bldpitch']

		for i in xrange(len(hull_mom_acc_surge)):
			RAO_wind_hull_moment = -hull_mom_acc_surge[i] * RAO_wind_acc_surge - hull_mom_acc_pitch[i] * RAO_wind_acc_pitch - hull_mom_acc_bend[i] * RAO_wind_acc_bend - hull_mom_damp_surge[i] * RAO_wind_vel_surge - hull_mom_damp_pitch[i] * RAO_wind_vel_pitch - hull_mom_damp_bend[i] * RAO_wind_vel_bend + hull_mom_grav_pitch[i] * RAO_wind_pitch + hull_mom_grav_bend[i] * RAO_wind_bend + hull_mom_rotspeed[i] * RAO_wind_rotspeed + hull_mom_bldpitch[i] * RAO_wind_bldpitch + (CoG_rotor - Z_spar[i]) * dthrust_dv * thrust_wind

			outputs['Re_RAO_wind_hull_moment'][:,i] = np.real(RAO_wind_hull_moment)
			outputs['Im_RAO_wind_hull_moment'][:,i] = np.imag(RAO_wind_hull_moment)
"""
	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		mom_acc_surge = inputs['mom_acc_surge'][0]
		mom_acc_pitch = inputs['mom_acc_pitch'][0]
		mom_acc_bend = inputs['mom_acc_bend'][0]
		mom_damp_surge = inputs['mom_damp_surge'][0]
		mom_damp_pitch = inputs['mom_damp_pitch'][0]
		mom_damp_bend = inputs['mom_damp_bend'][0]
		mom_grav_pitch = inputs['mom_grav_pitch'][0]
		mom_grav_bend = inputs['mom_grav_bend'][0]
		mom_rotspeed = inputs['mom_rotspeed'][0]
		mom_bldpitch = inputs['mom_bldpitch'][0]

		CoG_rotor = inputs['CoG_rotor'][0]
		Z_tower = inputs['Z_tower']
		dthrust_dv = inputs['dthrust_dv'][0]
		thrust_wind = inputs['thrust_wind']

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

		(CoG_rotor - Z_tower[0]) * dthrust_dv * thrust_wind
		
		partials['Re_RAO_wave_TB_moment', 'mom_acc_surge'] = np.real(-RAO_wave_acc_surge)
"""