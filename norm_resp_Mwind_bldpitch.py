import numpy as np

from openmdao.api import ExplicitComponent

class NormRespMWindBldpitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_RAO_Mwind_vel_surge', val=np.ones(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_surge', val=np.ones(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_pitch', val=np.ones(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_pitch', val=np.ones(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_bend', val=np.ones(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_bend', val=np.ones(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('k_t', val=0., units='rad*s/m')
		self.add_input('gain_corr_factor', val=0.)
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('Re_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_output('Im_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')

		Hcols = Hcols1 = np.array([43,49])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*9*6+Hcols1),0)

		self.declare_partials('Re_RAO_Mwind_bldpitch', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'k_i')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'k_p')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'k_t')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'gain_corr_factor')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'CoG_rotor')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'k_i')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'k_p')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'k_t')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'gain_corr_factor')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'CoG_rotor')

	def compute(self, inputs, outputs):
		omega = self.omega

		moment_wind = inputs['moment_wind']

		RAO_Mwind_vel_surge = inputs['Re_RAO_Mwind_vel_surge'] + 1j * inputs['Im_RAO_Mwind_vel_surge']
		RAO_Mwind_vel_pitch = inputs['Re_RAO_Mwind_vel_pitch'] + 1j * inputs['Im_RAO_Mwind_vel_pitch']
		RAO_Mwind_vel_bend = inputs['Re_RAO_Mwind_vel_bend'] + 1j * inputs['Im_RAO_Mwind_vel_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_Mwind_rot_lp = H_feedbk[:,7,1] * moment_wind
		RAO_Mwind_rotspeed_lp = H_feedbk[:,8,1] * moment_wind
		RAO_Mwind_vel_towertop = RAO_Mwind_vel_surge + inputs['CoG_rotor'] * RAO_Mwind_vel_pitch + RAO_Mwind_vel_bend
		RAO_Mwind_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_Mwind_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_Mwind_rotspeed_lp + inputs['k_t'] * RAO_Mwind_vel_towertop

		outputs['Re_RAO_Mwind_bldpitch'] = np.real(RAO_Mwind_bldpitch)
		outputs['Im_RAO_Mwind_bldpitch'] = np.imag(RAO_Mwind_bldpitch)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_Mwind_bldpitch', 'moment_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,1] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,1]
		partials['Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_surge'] = inputs['k_t'] * np.ones(N_omega)
		partials['Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_pitch'] = inputs['k_t'] * inputs['CoG_rotor'] * np.ones(N_omega)
		partials['Re_RAO_Mwind_bldpitch', 'Re_RAO_Mwind_vel_bend'] = inputs['k_t'] * np.ones(N_omega)
		partials['Im_RAO_Mwind_bldpitch', 'moment_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,1] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,1]
		partials['Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_surge'] = inputs['k_t'] * np.ones(N_omega)
		partials['Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_pitch'] = inputs['k_t'] * inputs['CoG_rotor'] * np.ones(N_omega)
		partials['Im_RAO_Mwind_bldpitch', 'Im_RAO_Mwind_vel_bend'] = inputs['k_t'] * np.ones(N_omega)

		partials['Re_RAO_Mwind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * inputs['Re_H_feedbk'][:,7,1] * inputs['moment_wind']
		partials['Im_RAO_Mwind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * inputs['Im_H_feedbk'][:,7,1] * inputs['moment_wind']
		partials['Re_RAO_Mwind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * inputs['Re_H_feedbk'][:,8,1] * inputs['moment_wind']
		partials['Im_RAO_Mwind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * inputs['Im_H_feedbk'][:,8,1] * inputs['moment_wind']
		partials['Re_RAO_Mwind_bldpitch', 'k_t'] = inputs['Re_RAO_Mwind_vel_surge'] + inputs['CoG_rotor'] * inputs['Re_RAO_Mwind_vel_pitch'] + inputs['Re_RAO_Mwind_vel_bend']
		partials['Im_RAO_Mwind_bldpitch', 'k_t'] = inputs['Im_RAO_Mwind_vel_surge'] + inputs['CoG_rotor'] * inputs['Im_RAO_Mwind_vel_pitch'] + inputs['Im_RAO_Mwind_vel_bend']
		partials['Re_RAO_Mwind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * inputs['Re_H_feedbk'][:,7,1] * inputs['moment_wind'] + inputs['k_p'] * inputs['Re_H_feedbk'][:,8,1] * inputs['moment_wind']
		partials['Im_RAO_Mwind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * inputs['Im_H_feedbk'][:,7,1] * inputs['moment_wind'] + inputs['k_p'] * inputs['Im_H_feedbk'][:,8,1] * inputs['moment_wind']
		partials['Re_RAO_Mwind_bldpitch', 'CoG_rotor'] = inputs['k_t'] * inputs['Re_RAO_Mwind_vel_pitch']
		partials['Im_RAO_Mwind_bldpitch', 'CoG_rotor'] = inputs['k_t'] * inputs['Im_RAO_Mwind_vel_pitch']

		partials['Re_RAO_Mwind_bldpitch', 'Im_H_feedbk'] = np.zeros(2*N_omega)
		partials['Im_RAO_Mwind_bldpitch', 'Re_H_feedbk'] = np.zeros(2*N_omega)

		for i in xrange(N_omega):
			partials['Re_RAO_Mwind_bldpitch', 'Re_H_feedbk'][2*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['moment_wind'][i]
			partials['Re_RAO_Mwind_bldpitch', 'Re_H_feedbk'][2*i+1] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['moment_wind'][i]
			partials['Im_RAO_Mwind_bldpitch', 'Im_H_feedbk'][2*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['moment_wind'][i]
			partials['Im_RAO_Mwind_bldpitch', 'Im_H_feedbk'][2*i+1] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['moment_wind'][i]