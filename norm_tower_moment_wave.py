import numpy as np

from openmdao.api import ExplicitComponent

class NormTowerMomentWave(ExplicitComponent):

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

		self.add_output('Re_RAO_wave_tower_moment', val=np.zeros((N_omega,11)), units='N*m/m')
		self.add_output('Im_RAO_wave_tower_moment', val=np.zeros((N_omega,11)), units='N*m/m')

		Rows = Rows1 = np.arange(0,11*N_omega,11)
		for i in xrange(1,11):
			Rows = np.concatenate((Rows,Rows1 + i),0)

		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_rotspeed', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_bldpitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))

		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_acc_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_acc_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_acc_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_damp_surge', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_damp_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_damp_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_grav_pitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_grav_bend', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_rotspeed', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'mom_bldpitch', rows=Rows, cols=np.repeat(np.arange(11),N_omega))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_rotspeed', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_bldpitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_surge', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_pitch', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_bend', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))

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

		for i in xrange(len(mom_acc_surge)):
			RAO_wave_tower_moment = -mom_acc_surge[i] * RAO_wave_acc_surge - mom_acc_pitch[i] * RAO_wave_acc_pitch - mom_acc_bend[i] * RAO_wave_acc_bend - mom_damp_surge[i] * RAO_wave_vel_surge - mom_damp_pitch[i] * RAO_wave_vel_pitch - mom_damp_bend[i] * RAO_wave_vel_bend + mom_grav_pitch[i] * RAO_wave_pitch + mom_grav_bend[i] * RAO_wave_bend + mom_rotspeed[i] * RAO_wave_rotspeed + mom_bldpitch[i] * RAO_wave_bldpitch
			
			outputs['Re_RAO_wave_tower_moment'][:,i] = np.real(RAO_wave_tower_moment)
			outputs['Im_RAO_wave_tower_moment'][:,i] = np.imag(RAO_wave_tower_moment)

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
		
		for i in xrange(len(mom_acc_surge)):
			partials['Re_RAO_wave_tower_moment', 'mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_surge)
			partials['Re_RAO_wave_tower_moment', 'mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_pitch)
			partials['Re_RAO_wave_tower_moment', 'mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_acc_bend)
			partials['Re_RAO_wave_tower_moment', 'mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_surge)
			partials['Re_RAO_wave_tower_moment', 'mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_pitch)
			partials['Re_RAO_wave_tower_moment', 'mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.real(-RAO_wave_vel_bend)
			partials['Re_RAO_wave_tower_moment', 'mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_pitch)
			partials['Re_RAO_wave_tower_moment', 'mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_bend)
			partials['Re_RAO_wave_tower_moment', 'mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_rotspeed)
			partials['Re_RAO_wave_tower_moment', 'mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.real(RAO_wave_bldpitch)

			partials['Im_RAO_wave_tower_moment', 'mom_acc_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_surge)
			partials['Im_RAO_wave_tower_moment', 'mom_acc_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_pitch)
			partials['Im_RAO_wave_tower_moment', 'mom_acc_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_acc_bend)
			partials['Im_RAO_wave_tower_moment', 'mom_damp_surge'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_surge)
			partials['Im_RAO_wave_tower_moment', 'mom_damp_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_pitch)
			partials['Im_RAO_wave_tower_moment', 'mom_damp_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(-RAO_wave_vel_bend)
			partials['Im_RAO_wave_tower_moment', 'mom_grav_pitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_pitch)
			partials['Im_RAO_wave_tower_moment', 'mom_grav_bend'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_bend)
			partials['Im_RAO_wave_tower_moment', 'mom_rotspeed'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_rotspeed)
			partials['Im_RAO_wave_tower_moment', 'mom_bldpitch'][i*N_omega:i*N_omega+N_omega] = np.imag(RAO_wave_bldpitch)

		for i in xrange(len(RAO_wave_acc_surge)):
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_pitch'][i*11:i*11+11] = mom_grav_pitch
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_bend'][i*11:i*11+11] = mom_grav_bend
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_rotspeed'][i*11:i*11+11] = mom_rotspeed
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_bldpitch'][i*11:i*11+11] = mom_bldpitch
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_surge'][i*11:i*11+11] = -mom_damp_surge
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_pitch'][i*11:i*11+11] = -mom_damp_pitch
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_vel_bend'][i*11:i*11+11] = -mom_damp_bend
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_surge'][i*11:i*11+11] = -mom_acc_surge
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_pitch'][i*11:i*11+11] = -mom_acc_pitch
			partials['Re_RAO_wave_tower_moment', 'Re_RAO_wave_acc_bend'][i*11:i*11+11] = -mom_acc_bend

			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_pitch'][i*11:i*11+11] = mom_grav_pitch
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_bend'][i*11:i*11+11] = mom_grav_bend
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_rotspeed'][i*11:i*11+11] = mom_rotspeed
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_bldpitch'][i*11:i*11+11] = mom_bldpitch
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_surge'][i*11:i*11+11] = -mom_damp_surge
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_pitch'][i*11:i*11+11] = -mom_damp_pitch
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_vel_bend'][i*11:i*11+11] = -mom_damp_bend
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_surge'][i*11:i*11+11] = -mom_acc_surge
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_pitch'][i*11:i*11+11] = -mom_acc_pitch
			partials['Im_RAO_wave_tower_moment', 'Im_RAO_wave_acc_bend'][i*11:i*11+11] = -mom_acc_bend