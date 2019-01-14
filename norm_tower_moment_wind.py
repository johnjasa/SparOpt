import numpy as np

from openmdao.api import ExplicitComponent

class NormTowerMomentWind(ExplicitComponent):

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

		self.add_output('Re_RAO_wind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')
		self.add_output('Im_RAO_wind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')

		Rows = Rows1 = np.arange(0,11*N_omega,11)
		for i in xrange(1,11):
			Rows = np.concatenate((Rows,Rows1 + i),0)

		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_rotspeed', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_bldpitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wind_tower_moment', 'CoG_rotor')
		self.declare_partials('Re_RAO_wind_tower_moment', 'Z_tower', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wind_tower_moment', 'dthrust_dv')
		self.declare_partials('Re_RAO_wind_tower_moment', 'thrust_wind', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))

		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_rotspeed', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_bldpitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))

	def compute(self, inputs, outputs):
		mom_acc_surge = inputs['mom_acc_surge']
		mom_acc_pitch = inputs['mom_acc_pitch']
		mom_acc_bend = inputs['mom_acc_bend']
		mom_damp_surge = inputs['mom_damp_surge']
		mom_damp_pitch = inputs['mom_damp_pitch']
		mom_damp_bend = inputs['mom_damp_bend']
		mom_grav_pitch = inputs['mom_grav_pitch']
		mom_grav_bend = inputs['mom_grav_bend']
		mom_rotspeed = inputs['mom_rotspeed']
		mom_bldpitch = inputs['mom_bldpitch']

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

		for i in xrange(len(mom_acc_surge)):
			RAO_wind_tower_moment = -mom_acc_surge[i] * RAO_wind_acc_surge - mom_acc_pitch[i] * RAO_wind_acc_pitch - mom_acc_bend[i] * RAO_wind_acc_bend - mom_damp_surge[i] * RAO_wind_vel_surge - mom_damp_pitch[i] * RAO_wind_vel_pitch - mom_damp_bend[i] * RAO_wind_vel_bend + mom_grav_pitch[i] * RAO_wind_pitch + mom_grav_bend[i] * RAO_wind_bend + mom_rotspeed[i] * RAO_wind_rotspeed + mom_bldpitch[i] * RAO_wind_bldpitch + (CoG_rotor - Z_tower[i]) * dthrust_dv * thrust_wind

			outputs['Re_RAO_wind_tower_moment'][:,i] = np.real(RAO_wind_tower_moment)
			outputs['Im_RAO_wind_tower_moment'][:,i] = np.imag(RAO_wind_tower_moment)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		mom_acc_surge = inputs['mom_acc_surge']
		mom_acc_pitch = inputs['mom_acc_pitch']
		mom_acc_bend = inputs['mom_acc_bend']
		mom_damp_surge = inputs['mom_damp_surge']
		mom_damp_pitch = inputs['mom_damp_pitch']
		mom_damp_bend = inputs['mom_damp_bend']
		mom_grav_pitch = inputs['mom_grav_pitch']
		mom_grav_bend = inputs['mom_grav_bend']
		mom_rotspeed = inputs['mom_rotspeed']
		mom_bldpitch = inputs['mom_bldpitch']

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
		
		for i in xrange(len(mom_acc_surge)):
			partials['Re_RAO_wind_tower_moment', 'mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_surge)
			partials['Re_RAO_wind_tower_moment', 'mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_pitch)
			partials['Re_RAO_wind_tower_moment', 'mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_bend)
			partials['Re_RAO_wind_tower_moment', 'mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_surge)
			partials['Re_RAO_wind_tower_moment', 'mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_pitch)
			partials['Re_RAO_wind_tower_moment', 'mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_bend)
			partials['Re_RAO_wind_tower_moment', 'mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_pitch)
			partials['Re_RAO_wind_tower_moment', 'mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_bend)
			partials['Re_RAO_wind_tower_moment', 'mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_rotspeed)
			partials['Re_RAO_wind_tower_moment', 'mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_bldpitch)
			partials['Re_RAO_wind_tower_moment', 'Z_tower'][i*N_omega:i*N_omega+N_omega] = -dthrust_dv * thrust_wind

			partials['Im_RAO_wind_tower_moment', 'mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_surge)
			partials['Im_RAO_wind_tower_moment', 'mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_pitch)
			partials['Im_RAO_wind_tower_moment', 'mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_bend)
			partials['Im_RAO_wind_tower_moment', 'mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_surge)
			partials['Im_RAO_wind_tower_moment', 'mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_pitch)
			partials['Im_RAO_wind_tower_moment', 'mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_bend)
			partials['Im_RAO_wind_tower_moment', 'mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_pitch)
			partials['Im_RAO_wind_tower_moment', 'mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_bend)
			partials['Im_RAO_wind_tower_moment', 'mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_rotspeed)
			partials['Im_RAO_wind_tower_moment', 'mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_bldpitch)

		for i in xrange(len(RAO_wind_acc_surge)):
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_pitch'][i*11:i*11+11] = mom_grav_pitch
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_bend'][i*11:i*11+11] = mom_grav_bend
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_rotspeed'][i*11:i*11+11] = mom_rotspeed
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_bldpitch'][i*11:i*11+11] = mom_bldpitch
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_surge'][i*11:i*11+11] = -mom_damp_surge
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_pitch'][i*11:i*11+11] = -mom_damp_pitch
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_vel_bend'][i*11:i*11+11] = -mom_damp_bend
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_surge'][i*11:i*11+11] = -mom_acc_surge
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_pitch'][i*11:i*11+11] = -mom_acc_pitch
			partials['Re_RAO_wind_tower_moment', 'Re_RAO_wind_acc_bend'][i*11:i*11+11] = -mom_acc_bend
			partials['Re_RAO_wind_tower_moment', 'thrust_wind'][i*11:i*11+11] = (CoG_rotor - Z_tower) * dthrust_dv
			partials['Re_RAO_wind_tower_moment', 'CoG_rotor'][i*11:i*11+11,0] = dthrust_dv * thrust_wind[i]
			partials['Re_RAO_wind_tower_moment', 'dthrust_dv'][i*11:i*11+11,0] = (CoG_rotor - Z_tower) * thrust_wind[i]

			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_pitch'][i*11:i*11+11] = mom_grav_pitch
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_bend'][i*11:i*11+11] = mom_grav_bend
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_rotspeed'][i*11:i*11+11] = mom_rotspeed
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_bldpitch'][i*11:i*11+11] = mom_bldpitch
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_surge'][i*11:i*11+11] = -mom_damp_surge
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_pitch'][i*11:i*11+11] = -mom_damp_pitch
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_vel_bend'][i*11:i*11+11] = -mom_damp_bend
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_surge'][i*11:i*11+11] = -mom_acc_surge
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_pitch'][i*11:i*11+11] = -mom_acc_pitch
			partials['Im_RAO_wind_tower_moment', 'Im_RAO_wind_acc_bend'][i*11:i*11+11] = -mom_acc_bend