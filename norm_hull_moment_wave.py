import numpy as np

from openmdao.api import ExplicitComponent

class NormHullMomentWave(ExplicitComponent):

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
		self.add_input('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wave_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/m')
		self.add_input('Re_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/m')
		self.add_input('Re_RAO_wave_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/m')
		self.add_input('Im_RAO_wave_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/m')
		self.add_input('Im_RAO_wave_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/m')
		self.add_input('Im_RAO_wave_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/m')
		self.add_input('Re_hull_wave_excit_mom', val=np.zeros((N_omega,10)), units='N*m/m')
		self.add_input('Im_hull_wave_excit_mom', val=np.zeros((N_omega,10)), units='N*m/m')

		self.add_output('Re_RAO_wave_hull_moment', val=np.zeros((N_omega,10)), units='N*m/m')
		self.add_output('Im_RAO_wave_hull_moment', val=np.zeros((N_omega,10)), units='N*m/m')

		Rows = Rows1 = np.arange(0,10*N_omega,10)
		for i in xrange(1,10):
			Rows = np.concatenate((Rows,Rows1 + i),0)

		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'hull_mom_fairlead', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_rotspeed', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_bldpitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_fairlead', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Re_RAO_wave_hull_moment', 'Re_hull_wave_excit_mom', rows=np.arange(10*N_omega), cols=np.arange(10*N_omega))

		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'hull_mom_fairlead', rows=Rows, cols=np.repeat(np.arange(10),N_omega))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_rotspeed', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_bldpitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_fairlead', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_surge', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_pitch', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_bend', rows=np.arange(10*N_omega), cols=np.repeat(np.arange(N_omega),10))
		self.declare_partials('Im_RAO_wave_hull_moment', 'Im_hull_wave_excit_mom', rows=np.arange(10*N_omega), cols=np.arange(10*N_omega))

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

		hull_wave_excit_mom = inputs['Re_hull_wave_excit_mom'] + 1j * inputs['Im_hull_wave_excit_mom']
		
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
		RAO_wave_fairlead = inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead']

		for i in xrange(len(hull_mom_acc_surge)):
			RAO_wave_hull_moment = -hull_mom_acc_surge[i] * RAO_wave_acc_surge - hull_mom_acc_pitch[i] * RAO_wave_acc_pitch - hull_mom_acc_bend[i] * RAO_wave_acc_bend - hull_mom_damp_surge[i] * RAO_wave_vel_surge - hull_mom_damp_pitch[i] * RAO_wave_vel_pitch - hull_mom_damp_bend[i] * RAO_wave_vel_bend + hull_mom_grav_pitch[i] * RAO_wave_pitch + hull_mom_grav_bend[i] * RAO_wave_bend + hull_mom_rotspeed[i] * RAO_wave_rotspeed + hull_mom_bldpitch[i] * RAO_wave_bldpitch + hull_mom_fairlead[i] * RAO_wave_fairlead + hull_wave_excit_mom[:,i]
			
			outputs['Re_RAO_wave_hull_moment'][:,i] = np.real(RAO_wave_hull_moment)
			outputs['Im_RAO_wave_hull_moment'][:,i] = np.imag(RAO_wave_hull_moment)

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

		hull_wave_excit_mom = inputs['Re_hull_wave_excit_mom'] + 1j * inputs['Im_hull_wave_excit_mom']

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
		RAO_wave_fairlead = inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead']
		
		for i in xrange(len(hull_mom_acc_surge)):
			partials['Re_RAO_wave_hull_moment', 'hull_mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_surge)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_pitch)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_bend)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_surge)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_pitch)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_bend)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_pitch)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_bend)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_rotspeed)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_bldpitch)
			partials['Re_RAO_wave_hull_moment', 'hull_mom_fairlead'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_fairlead)
			partials['Re_RAO_wave_hull_moment', 'Re_hull_wave_excit_mom'][i*N_omega:i*N_omega+N_omega] = np.ones(N_omega)

			partials['Im_RAO_wave_hull_moment', 'hull_mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_surge)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_pitch)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_bend)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_surge)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_pitch)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_bend)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_pitch)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_bend)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_rotspeed)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_bldpitch)
			partials['Im_RAO_wave_hull_moment', 'hull_mom_fairlead'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_fairlead)
			partials['Im_RAO_wave_hull_moment', 'Im_hull_wave_excit_mom'][i*N_omega:i*N_omega+N_omega] = np.ones(N_omega)

		for i in xrange(len(RAO_wave_acc_surge)):
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_pitch'][i*10:i*10+10] = hull_mom_grav_pitch
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_bend'][i*10:i*10+10] = hull_mom_grav_bend
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_rotspeed'][i*10:i*10+10] = hull_mom_rotspeed
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_bldpitch'][i*10:i*10+10] = hull_mom_bldpitch
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_fairlead'][i*10:i*10+10] = hull_mom_fairlead
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_surge'][i*10:i*10+10] = -hull_mom_damp_surge
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_pitch'][i*10:i*10+10] = -hull_mom_damp_pitch
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_vel_bend'][i*10:i*10+10] = -hull_mom_damp_bend
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_surge'][i*10:i*10+10] = -hull_mom_acc_surge
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_pitch'][i*10:i*10+10] = -hull_mom_acc_pitch
			partials['Re_RAO_wave_hull_moment', 'Re_RAO_wave_acc_bend'][i*10:i*10+10] = -hull_mom_acc_bend

			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_pitch'][i*10:i*10+10] = hull_mom_grav_pitch
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_bend'][i*10:i*10+10] = hull_mom_grav_bend
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_rotspeed'][i*10:i*10+10] = hull_mom_rotspeed
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_bldpitch'][i*10:i*10+10] = hull_mom_bldpitch
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_fairlead'][i*10:i*10+10] = hull_mom_fairlead
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_surge'][i*10:i*10+10] = -hull_mom_damp_surge
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_pitch'][i*10:i*10+10] = -hull_mom_damp_pitch
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_vel_bend'][i*10:i*10+10] = -hull_mom_damp_bend
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_surge'][i*10:i*10+10] = -hull_mom_acc_surge
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_pitch'][i*10:i*10+10] = -hull_mom_acc_pitch
			partials['Im_RAO_wave_hull_moment', 'Im_RAO_wave_acc_bend'][i*10:i*10+10] = -hull_mom_acc_bend
