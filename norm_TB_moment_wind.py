import numpy as np

from openmdao.api import ExplicitComponent

class NormTBMomentWind(ExplicitComponent):

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
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('dthrust_dv', val=0., units='N*s/m')
		self.add_input('thrust_wind', val=np.zeros(N_omega), units='m/s')

		self.add_output('Re_RAO_wind_TB_moment', val=np.zeros(N_omega), units='N*m/(m/s)')
		self.add_output('Im_RAO_wind_TB_moment', val=np.zeros(N_omega), units='N*m/(m/s)')

		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_acc_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_acc_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_acc_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_damp_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_damp_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_damp_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_grav_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_grav_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_rotspeed', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'mom_bldpitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'CoG_rotor')
		self.declare_partials('Re_RAO_wind_TB_moment', 'Z_tower', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Re_RAO_wind_TB_moment', 'dthrust_dv')
		self.declare_partials('Re_RAO_wind_TB_moment', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_acc_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_acc_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_acc_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_damp_surge', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_damp_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_damp_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_grav_pitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_grav_bend', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_rotspeed', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'mom_bldpitch', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'CoG_rotor')
		self.declare_partials('Im_RAO_wind_TB_moment', 'Z_tower', rows=np.arange(N_omega), cols=np.zeros(N_omega))
		self.declare_partials('Im_RAO_wind_TB_moment', 'dthrust_dv')
		self.declare_partials('Im_RAO_wind_TB_moment', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
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

		RAO_wind_TB_moment = -mom_acc_surge * RAO_wind_acc_surge - mom_acc_pitch * RAO_wind_acc_pitch - mom_acc_bend * RAO_wind_acc_bend - mom_damp_surge * RAO_wind_vel_surge - mom_damp_pitch * RAO_wind_vel_pitch - mom_damp_bend * RAO_wind_vel_bend + mom_grav_pitch * RAO_wind_pitch + mom_grav_bend * RAO_wind_bend + mom_rotspeed * RAO_wind_rotspeed + mom_bldpitch * RAO_wind_bldpitch + (CoG_rotor - Z_tower[0]) * dthrust_dv * thrust_wind

		outputs['Re_RAO_wind_TB_moment'] = np.real(RAO_wind_TB_moment)

		outputs['Im_RAO_wind_TB_moment'] = np.imag(RAO_wind_TB_moment)

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

		(CoG_rotor - Z_tower[0]) * dthrust_dv * thrust_wind
		
		partials['Re_RAO_wind_TB_moment', 'mom_acc_surge'] = np.real(-RAO_wind_acc_surge)
		partials['Re_RAO_wind_TB_moment', 'mom_acc_pitch'] = np.real(-RAO_wind_acc_pitch)
		partials['Re_RAO_wind_TB_moment', 'mom_acc_bend'] = np.real(-RAO_wind_acc_bend)
		partials['Re_RAO_wind_TB_moment', 'mom_damp_surge'] = np.real(-RAO_wind_vel_surge)
		partials['Re_RAO_wind_TB_moment', 'mom_damp_pitch'] = np.real(-RAO_wind_vel_pitch)
		partials['Re_RAO_wind_TB_moment', 'mom_damp_bend'] = np.real(-RAO_wind_vel_bend)
		partials['Re_RAO_wind_TB_moment', 'mom_grav_pitch'] = np.real(RAO_wind_pitch)
		partials['Re_RAO_wind_TB_moment', 'mom_grav_bend'] = np.real(RAO_wind_bend)
		partials['Re_RAO_wind_TB_moment', 'mom_rotspeed'] = np.real(RAO_wind_rotspeed)
		partials['Re_RAO_wind_TB_moment', 'mom_bldpitch'] = np.real(RAO_wind_bldpitch)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_pitch'] = mom_grav_pitch * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_bend'] = mom_grav_bend * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_rotspeed'] = mom_rotspeed * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_bldpitch'] = mom_bldpitch * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_rotspeed'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_bldpitch'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_surge'] = -mom_damp_surge * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_pitch'] = -mom_damp_pitch * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_vel_bend'] = -mom_damp_bend * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_surge'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_vel_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_surge'] = -mom_acc_surge * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_pitch'] = -mom_acc_pitch * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Re_RAO_wind_acc_bend'] = -mom_acc_bend * np.ones(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_surge'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'Im_RAO_wind_acc_bend'] = np.zeros(N_omega)
		partials['Re_RAO_wind_TB_moment', 'CoG_rotor'] = dthrust_dv * thrust_wind
		partials['Re_RAO_wind_TB_moment', 'Z_tower'] = -dthrust_dv * thrust_wind
		partials['Re_RAO_wind_TB_moment', 'dthrust_dv'] = (CoG_rotor - Z_tower[0]) * thrust_wind
		partials['Re_RAO_wind_TB_moment', 'thrust_wind'] = (CoG_rotor - Z_tower[0]) * dthrust_dv * np.ones(N_omega)

		partials['Im_RAO_wind_TB_moment', 'mom_acc_surge'] = np.imag(-RAO_wind_acc_surge)
		partials['Im_RAO_wind_TB_moment', 'mom_acc_pitch'] = np.imag(-RAO_wind_acc_pitch)
		partials['Im_RAO_wind_TB_moment', 'mom_acc_bend'] = np.imag(-RAO_wind_acc_bend)
		partials['Im_RAO_wind_TB_moment', 'mom_damp_surge'] = np.imag(-RAO_wind_vel_surge)
		partials['Im_RAO_wind_TB_moment', 'mom_damp_pitch'] = np.imag(-RAO_wind_vel_pitch)
		partials['Im_RAO_wind_TB_moment', 'mom_damp_bend'] = np.imag(-RAO_wind_vel_bend)
		partials['Im_RAO_wind_TB_moment', 'mom_grav_pitch'] = np.imag(RAO_wind_pitch)
		partials['Im_RAO_wind_TB_moment', 'mom_grav_bend'] = np.imag(RAO_wind_bend)
		partials['Im_RAO_wind_TB_moment', 'mom_rotspeed'] = np.imag(RAO_wind_rotspeed)
		partials['Im_RAO_wind_TB_moment', 'mom_bldpitch'] = np.imag(RAO_wind_bldpitch)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_rotspeed'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_bldpitch'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_pitch'] = mom_grav_pitch * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_bend'] = mom_grav_bend * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_rotspeed'] = mom_rotspeed * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_bldpitch'] = mom_bldpitch * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_vel_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_surge'] = -mom_damp_surge * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_pitch'] = -mom_damp_pitch * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_vel_bend'] = -mom_damp_bend * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_pitch'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Re_RAO_wind_acc_bend'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_surge'] = -mom_acc_surge * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_pitch'] = -mom_acc_pitch * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Im_RAO_wind_acc_bend'] = -mom_acc_bend * np.ones(N_omega)
		partials['Im_RAO_wind_TB_moment', 'CoG_rotor'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'Z_tower'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'dthrust_dv'] = np.zeros(N_omega)
		partials['Im_RAO_wind_TB_moment', 'thrust_wind'] = np.zeros(N_omega)