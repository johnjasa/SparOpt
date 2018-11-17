import numpy as np

from openmdao.api import ExplicitComponent

class NormTBMomentWave(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('mom_acc_surge', val=np.zeros(11), units='kg*m')
		self.add_input('mom_acc_pitch', val=np.zeros(11), units='kg*m**2/rad')
		self.add_input('mom_acc_bend', val=np.zeros(11), units='kg*m')
		self.add_input('mom_damp_surge', val=np.zeros(11), units='N*s')
		self.add_input('mom_damp_pitch', val=np.zeros(11), units='N*m*s/rad')
		self.add_input('mom_damp_bend', val=np.zeros(11), units='N*s')
		self.add_input('mom_grav_pitch', val=np.zeros(11), units='N*m/rad')
		self.add_input('mom_grav_bend', val=np.zeros(11), units='N')
		self.add_input('mom_rotspeed', val=np.zeros(11), units='N*m*s/rad')
		self.add_input('mom_bldpitch', val=np.zeros(11), units='N*m/rad')
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

		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_acc_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_acc_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_acc_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_damp_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_damp_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_damp_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_grav_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_grav_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_rotspeed', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'mom_bldpitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))

		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_acc_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_acc_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_acc_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_damp_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_damp_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_damp_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_grav_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_grav_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_rotspeed', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'mom_bldpitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))

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
		partials['Re_RAO_wave_TB_moment', 'mom_acc_pitch'] = np.real(-RAO_wave_acc_pitch)
		partials['Re_RAO_wave_TB_moment', 'mom_acc_bend'] = np.real(-RAO_wave_acc_bend)
		partials['Re_RAO_wave_TB_moment', 'mom_damp_surge'] = np.real(-RAO_wave_vel_surge)
		partials['Re_RAO_wave_TB_moment', 'mom_damp_pitch'] = np.real(-RAO_wave_vel_pitch)
		partials['Re_RAO_wave_TB_moment', 'mom_damp_bend'] = np.real(-RAO_wave_vel_bend)
		partials['Re_RAO_wave_TB_moment', 'mom_grav_pitch'] = np.real(RAO_wave_pitch)
		partials['Re_RAO_wave_TB_moment', 'mom_grav_bend'] = np.real(RAO_wave_bend)
		partials['Re_RAO_wave_TB_moment', 'mom_rotspeed'] = np.real(RAO_wave_rotspeed)
		partials['Re_RAO_wave_TB_moment', 'mom_bldpitch'] = np.real(RAO_wave_bldpitch)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_pitch'] = mom_grav_pitch * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_bend'] = mom_grav_bend * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_rotspeed'] = mom_rotspeed * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_bldpitch'] = mom_bldpitch * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_rotspeed'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_bldpitch'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_surge'] = -mom_damp_surge * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_pitch'] = -mom_damp_pitch * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_vel_bend'] = -mom_damp_bend * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_surge'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_vel_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_surge'] = -mom_acc_surge * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_pitch'] = -mom_acc_pitch * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Re_RAO_wave_acc_bend'] = -mom_acc_bend * np.ones(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_surge'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wave_TB_moment', 'Im_RAO_wave_acc_bend'] = np.zeros(N_omega)

		partials['Im_RAO_wave_TB_moment', 'mom_acc_surge'] = np.imag(-RAO_wave_acc_surge)
		partials['Im_RAO_wave_TB_moment', 'mom_acc_pitch'] = np.imag(-RAO_wave_acc_pitch)
		partials['Im_RAO_wave_TB_moment', 'mom_acc_bend'] = np.imag(-RAO_wave_acc_bend)
		partials['Im_RAO_wave_TB_moment', 'mom_damp_surge'] = np.imag(-RAO_wave_vel_surge)
		partials['Im_RAO_wave_TB_moment', 'mom_damp_pitch'] = np.imag(-RAO_wave_vel_pitch)
		partials['Im_RAO_wave_TB_moment', 'mom_damp_bend'] = np.imag(-RAO_wave_vel_bend)
		partials['Im_RAO_wave_TB_moment', 'mom_grav_pitch'] = np.imag(RAO_wave_pitch)
		partials['Im_RAO_wave_TB_moment', 'mom_grav_bend'] = np.imag(RAO_wave_bend)
		partials['Im_RAO_wave_TB_moment', 'mom_rotspeed'] = np.imag(RAO_wave_rotspeed)
		partials['Im_RAO_wave_TB_moment', 'mom_bldpitch'] = np.imag(RAO_wave_bldpitch)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_rotspeed'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_bldpitch'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_pitch'] = mom_grav_pitch * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_bend'] = mom_grav_bend * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_rotspeed'] = mom_rotspeed * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_bldpitch'] = mom_bldpitch * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_vel_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_surge'] = -mom_damp_surge * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_pitch'] = -mom_damp_pitch * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_vel_bend'] = -mom_damp_bend * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Re_RAO_wave_acc_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_surge'] = -mom_acc_surge * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_pitch'] = -mom_acc_pitch * np.ones(N_omega)
		partials['Im_RAO_wave_TB_moment', 'Im_RAO_wave_acc_bend'] = -mom_acc_bend * np.ones(N_omega)