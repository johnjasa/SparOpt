import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWaveBldpitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_input('Im_wave_force_surge', val=np.zeros(N_omega), units='N/m')
		self.add_input('Re_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Im_wave_force_pitch', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Re_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_input('Im_wave_force_bend', val=np.zeros(N_omega), units='N/m')
		self.add_input('Re_RAO_wave_vel_surge', val=np.ones(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_surge', val=np.ones(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wave_vel_pitch', val=np.ones(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_vel_pitch', val=np.ones(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_vel_bend', val=np.ones(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_bend', val=np.ones(N_omega), units='(m/s)/m')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('k_t', val=0., units='rad*s/m')
		self.add_input('gain_corr_factor', val=0.)
		self.add_input('CoG_rotor', val=0., units='m')

		self.add_output('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_output('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')

		Hcols = Hcols1 = np.array([45,46,47,51,52,53])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*9*6+Hcols1),0)

		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Im_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Im_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Im_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),6), cols=Hcols)
		self.declare_partials('Re_RAO_wave_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),6), cols=Hcols)
		self.declare_partials('Im_RAO_wave_bldpitch', 'Re_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Re_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Re_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),6), cols=Hcols)
		self.declare_partials('Im_RAO_wave_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),6), cols=Hcols)
		self.declare_partials('Re_RAO_wave_bldpitch', 'k_i')
		self.declare_partials('Re_RAO_wave_bldpitch', 'k_p')
		self.declare_partials('Re_RAO_wave_bldpitch', 'k_t')
		self.declare_partials('Re_RAO_wave_bldpitch', 'gain_corr_factor')
		self.declare_partials('Re_RAO_wave_bldpitch', 'CoG_rotor')
		self.declare_partials('Im_RAO_wave_bldpitch', 'k_i')
		self.declare_partials('Im_RAO_wave_bldpitch', 'k_p')
		self.declare_partials('Im_RAO_wave_bldpitch', 'k_t')
		self.declare_partials('Im_RAO_wave_bldpitch', 'gain_corr_factor')
		self.declare_partials('Im_RAO_wave_bldpitch', 'CoG_rotor')

	def compute(self, inputs, outputs):
		omega = self.omega

		wave_force_surge = inputs['Re_wave_force_surge'] + 1j * inputs['Im_wave_force_surge']
		wave_force_pitch = inputs['Re_wave_force_pitch'] + 1j * inputs['Im_wave_force_pitch']
		wave_force_bend = inputs['Re_wave_force_bend'] + 1j * inputs['Im_wave_force_bend']

		RAO_wave_vel_surge = inputs['Re_RAO_wave_vel_surge'] + 1j * inputs['Im_RAO_wave_vel_surge']
		RAO_wave_vel_pitch = inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch']
		RAO_wave_vel_bend = inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_rot_lp = H_feedbk[:,7,3] * wave_force_surge + H_feedbk[:,7,4] * wave_force_pitch + H_feedbk[:,7,5] * wave_force_bend
		RAO_wave_rotspeed_lp = H_feedbk[:,8,3] * wave_force_surge + H_feedbk[:,8,4] * wave_force_pitch + H_feedbk[:,8,5] * wave_force_bend
		RAO_wave_vel_towertop = RAO_wave_vel_surge + inputs['CoG_rotor'] * RAO_wave_vel_pitch + RAO_wave_vel_bend
		RAO_wave_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_wave_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_wave_rotspeed_lp + inputs['k_t'] * RAO_wave_vel_towertop

		outputs['Re_RAO_wave_bldpitch'] = np.real(RAO_wave_bldpitch)
		outputs['Im_RAO_wave_bldpitch'] = np.imag(RAO_wave_bldpitch)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		wave_force_surge = inputs['Re_wave_force_surge'] + 1j * inputs['Im_wave_force_surge']
		wave_force_pitch = inputs['Re_wave_force_pitch'] + 1j * inputs['Im_wave_force_pitch']
		wave_force_bend = inputs['Re_wave_force_bend'] + 1j * inputs['Im_wave_force_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_rot_lp = H_feedbk[:,7,3] * wave_force_surge + H_feedbk[:,7,4] * wave_force_pitch + H_feedbk[:,7,5] * wave_force_bend
		RAO_wave_rotspeed_lp = H_feedbk[:,8,3] * wave_force_surge + H_feedbk[:,8,4] * wave_force_pitch + H_feedbk[:,8,5] * wave_force_bend

		partials['Re_RAO_wave_bldpitch', 'Re_wave_force_surge'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,3] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,3]
		partials['Re_RAO_wave_bldpitch', 'Im_wave_force_surge'] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,3] - inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,3]
		partials['Re_RAO_wave_bldpitch', 'Re_wave_force_pitch'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,4] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,4]
		partials['Re_RAO_wave_bldpitch', 'Im_wave_force_pitch'] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,4] - inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,4]
		partials['Re_RAO_wave_bldpitch', 'Re_wave_force_bend'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,5] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,5]
		partials['Re_RAO_wave_bldpitch', 'Im_wave_force_bend'] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,5] - inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,5]
		partials['Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_surge'] = inputs['k_t'] * np.ones(N_omega)
		partials['Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_pitch'] = inputs['k_t'] * inputs['CoG_rotor'] * np.ones(N_omega)
		partials['Re_RAO_wave_bldpitch', 'Re_RAO_wave_vel_bend'] = inputs['k_t'] * np.ones(N_omega)
		partials['Im_RAO_wave_bldpitch', 'Re_wave_force_surge'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,3] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,3]
		partials['Im_RAO_wave_bldpitch', 'Im_wave_force_surge'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,3] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,3]
		partials['Im_RAO_wave_bldpitch', 'Re_wave_force_pitch'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,4] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,4]
		partials['Im_RAO_wave_bldpitch', 'Im_wave_force_pitch'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,4] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,4]
		partials['Im_RAO_wave_bldpitch', 'Re_wave_force_bend'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,5] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,5]
		partials['Im_RAO_wave_bldpitch', 'Im_wave_force_bend'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,5] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,5]
		partials['Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_surge'] = inputs['k_t'] * np.ones(N_omega)
		partials['Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_pitch'] = inputs['k_t'] * inputs['CoG_rotor'] * np.ones(N_omega)
		partials['Im_RAO_wave_bldpitch', 'Im_RAO_wave_vel_bend'] = inputs['k_t'] * np.ones(N_omega)

		partials['Re_RAO_wave_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * np.real(RAO_wave_rot_lp)
		partials['Re_RAO_wave_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * np.real(RAO_wave_rotspeed_lp)
		partials['Re_RAO_wave_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * np.real(RAO_wave_rot_lp) + inputs['k_p'] * np.real(RAO_wave_rotspeed_lp)
		partials['Re_RAO_wave_bldpitch', 'k_t'] = inputs['Re_RAO_wave_vel_surge'] + inputs['CoG_rotor'] * inputs['Re_RAO_wave_vel_pitch'] + inputs['Re_RAO_wave_vel_bend']
		partials['Re_RAO_wave_bldpitch', 'CoG_rotor'] =  inputs['k_t'] * inputs['Re_RAO_wave_vel_pitch']
		partials['Im_RAO_wave_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * np.imag(RAO_wave_rot_lp)
		partials['Im_RAO_wave_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * np.imag(RAO_wave_rotspeed_lp)
		partials['Im_RAO_wave_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * np.imag(RAO_wave_rot_lp) + inputs['k_p'] * np.imag(RAO_wave_rotspeed_lp)
		partials['Im_RAO_wave_bldpitch', 'k_t'] = inputs['Im_RAO_wave_vel_surge'] + inputs['CoG_rotor'] * inputs['Im_RAO_wave_vel_pitch'] + inputs['Im_RAO_wave_vel_bend']
		partials['Im_RAO_wave_bldpitch', 'CoG_rotor'] =  inputs['k_t'] * inputs['Im_RAO_wave_vel_pitch']

		for i in xrange(N_omega):
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_surge'][i]
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+1] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_pitch'][i]
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+2] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_bend'][i]
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+3] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_surge'][i]
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+4] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_pitch'][i]
			partials['Re_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+5] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_bend'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_surge'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+1] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_pitch'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+2] = -inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_bend'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+3] = -inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_surge'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+4] = -inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_pitch'][i]
			partials['Re_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+5] = -inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_bend'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_surge'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+1] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_pitch'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+2] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_wave_force_bend'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+3] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_surge'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+4] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_pitch'][i]
			partials['Im_RAO_wave_bldpitch', 'Re_H_feedbk'][6*i+5] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_wave_force_bend'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_surge'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+1] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_pitch'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+2] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_wave_force_bend'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+3] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_surge'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+4] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_pitch'][i]
			partials['Im_RAO_wave_bldpitch', 'Im_H_feedbk'][6*i+5] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_wave_force_bend'][i]