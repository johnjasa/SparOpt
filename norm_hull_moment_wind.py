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
		self.add_input('hull_mom_fairlead', val=np.zeros(10), units='N*m/m')
		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
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

		Rows = Rows1 = np.arange(0,10*N_omega,10)
		for i in xrange(1,10):
			Rows = np.concatenate((Rows,Rows1 + i),0)

		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'hull_mom_fairlead', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_rotspeed', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_bldpitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_fairlead', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wind_hull_moment', 'CoG_rotor')
		self.declare_partials('Re_RAO_wind_hull_moment', 'Z_spar', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wind_hull_moment', 'dthrust_dv')
		self.declare_partials('Re_RAO_wind_hull_moment', 'thrust_wind', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))

		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'hull_mom_fairlead', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_rotspeed', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_bldpitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_fairlead', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))

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
		hull_mom_fairlead = inputs['hull_mom_fairlead']

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
		RAO_wind_fairlead = inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead']

		for i in xrange(len(hull_mom_acc_surge)):
			RAO_wind_hull_moment = -hull_mom_acc_surge[i] * RAO_wind_acc_surge - hull_mom_acc_pitch[i] * RAO_wind_acc_pitch - hull_mom_acc_bend[i] * RAO_wind_acc_bend - hull_mom_damp_surge[i] * RAO_wind_vel_surge - hull_mom_damp_pitch[i] * RAO_wind_vel_pitch - hull_mom_damp_bend[i] * RAO_wind_vel_bend + hull_mom_grav_pitch[i] * RAO_wind_pitch + hull_mom_grav_bend[i] * RAO_wind_bend + hull_mom_rotspeed[i] * RAO_wind_rotspeed + hull_mom_bldpitch[i] * RAO_wind_bldpitch + hull_mom_fairlead[i] * RAO_wind_fairlead + (CoG_rotor - Z_spar[i]) * dthrust_dv * thrust_wind

			outputs['Re_RAO_wind_hull_moment'][:,i] = np.real(RAO_wind_hull_moment)
			outputs['Im_RAO_wind_hull_moment'][:,i] = np.imag(RAO_wind_hull_moment)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

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
		hull_mom_fairlead = inputs['hull_mom_fairlead']

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
		RAO_wind_fairlead = inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead']
		
		for i in xrange(len(hull_mom_acc_surge)):
			partials['Re_RAO_wind_hull_moment', 'hull_mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_surge)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_pitch)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_acc_bend)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_surge)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_pitch)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wind_vel_bend)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_pitch)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_bend)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_rotspeed)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_bldpitch)
			partials['Re_RAO_wind_hull_moment', 'hull_mom_fairlead'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wind_fairlead)
			partials['Re_RAO_wind_hull_moment', 'Z_spar'][i*N_omega:i*N_omega+N_omega] = -dthrust_dv * thrust_wind

			partials['Im_RAO_wind_hull_moment', 'hull_mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_surge)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_pitch)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_acc_bend)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_surge)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_pitch)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wind_vel_bend)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_pitch)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_bend)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_rotspeed)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_bldpitch)
			partials['Im_RAO_wind_hull_moment', 'hull_mom_fairlead'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wind_fairlead)

		for i in xrange(len(RAO_wind_acc_surge)):
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_pitch'][i*10:i*10+10] = hull_mom_grav_pitch
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_bend'][i*10:i*10+10] = hull_mom_grav_bend
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_rotspeed'][i*10:i*10+10] = hull_mom_rotspeed
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_bldpitch'][i*10:i*10+10] = hull_mom_bldpitch
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_fairlead'][i*10:i*10+10] = hull_mom_fairlead
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_surge'][i*10:i*10+10] = -hull_mom_damp_surge
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_pitch'][i*10:i*10+10] = -hull_mom_damp_pitch
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_vel_bend'][i*10:i*10+10] = -hull_mom_damp_bend
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_surge'][i*10:i*10+10] = -hull_mom_acc_surge
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_pitch'][i*10:i*10+10] = -hull_mom_acc_pitch
			partials['Re_RAO_wind_hull_moment', 'Re_RAO_wind_acc_bend'][i*10:i*10+10] = -hull_mom_acc_bend
			partials['Re_RAO_wind_hull_moment', 'thrust_wind'][i*10:i*10+10] = (CoG_rotor - Z_spar[:-1]) * dthrust_dv
			partials['Re_RAO_wind_hull_moment', 'CoG_rotor'][i*10:i*10+10,0] = dthrust_dv * thrust_wind[i]
			partials['Re_RAO_wind_hull_moment', 'dthrust_dv'][i*10:i*10+10,0] = (CoG_rotor - Z_spar[:-1]) * thrust_wind[i]

			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_pitch'][i*10:i*10+10] = hull_mom_grav_pitch
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_bend'][i*10:i*10+10] = hull_mom_grav_bend
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_rotspeed'][i*10:i*10+10] = hull_mom_rotspeed
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_bldpitch'][i*10:i*10+10] = hull_mom_bldpitch
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_fairlead'][i*10:i*10+10] = hull_mom_fairlead
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_surge'][i*10:i*10+10] = -hull_mom_damp_surge
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_pitch'][i*10:i*10+10] = -hull_mom_damp_pitch
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_vel_bend'][i*10:i*10+10] = -hull_mom_damp_bend
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_surge'][i*10:i*10+10] = -hull_mom_acc_surge
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_pitch'][i*10:i*10+10] = -hull_mom_acc_pitch
			partials['Im_RAO_wind_hull_moment', 'Im_RAO_wind_acc_bend'][i*10:i*10+10] = -hull_mom_acc_bend