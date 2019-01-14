import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWaveRotspeed(ExplicitComponent):

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
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,11,6)))

		self.add_output('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_output('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')

		Hcols = Hcols1 = np.array([39,40,41])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*11*6+Hcols1),0)

		self.declare_partials('Re_RAO_wave_rotspeed', 'Re_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Im_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Re_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Im_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Re_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Im_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wave_rotspeed', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),3), cols=Hcols)
		self.declare_partials('Re_RAO_wave_rotspeed', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),3), cols=Hcols)
		self.declare_partials('Im_RAO_wave_rotspeed', 'Re_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Im_wave_force_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Re_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Im_wave_force_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Re_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Im_wave_force_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wave_rotspeed', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),3), cols=Hcols)
		self.declare_partials('Im_RAO_wave_rotspeed', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),3), cols=Hcols)

	def compute(self, inputs, outputs):
		omega = self.omega

		wave_force_surge = inputs['Re_wave_force_surge'] + 1j * inputs['Im_wave_force_surge']
		wave_force_pitch = inputs['Re_wave_force_pitch'] + 1j * inputs['Im_wave_force_pitch']
		wave_force_bend = inputs['Re_wave_force_bend'] + 1j * inputs['Im_wave_force_bend']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wave_rotspeed = H_feedbk[:,6,3] * wave_force_surge + H_feedbk[:,6,4] * wave_force_pitch + H_feedbk[:,6,5] * wave_force_bend

		outputs['Re_RAO_wave_rotspeed'] = np.real(RAO_wave_rotspeed)
		outputs['Im_RAO_wave_rotspeed'] = np.imag(RAO_wave_rotspeed)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_wave_rotspeed', 'Re_wave_force_surge'] = inputs['Re_H_feedbk'][:,6,3]
		partials['Re_RAO_wave_rotspeed', 'Im_wave_force_surge'] = -inputs['Im_H_feedbk'][:,6,3]
		partials['Re_RAO_wave_rotspeed', 'Re_wave_force_pitch'] = inputs['Re_H_feedbk'][:,6,4]
		partials['Re_RAO_wave_rotspeed', 'Im_wave_force_pitch'] = -inputs['Im_H_feedbk'][:,6,4]
		partials['Re_RAO_wave_rotspeed', 'Re_wave_force_bend'] = inputs['Re_H_feedbk'][:,6,5]
		partials['Re_RAO_wave_rotspeed', 'Im_wave_force_bend'] = -inputs['Im_H_feedbk'][:,6,5]
		partials['Im_RAO_wave_rotspeed', 'Re_wave_force_surge'] = inputs['Im_H_feedbk'][:,6,3]
		partials['Im_RAO_wave_rotspeed', 'Im_wave_force_surge'] = inputs['Re_H_feedbk'][:,6,3]
		partials['Im_RAO_wave_rotspeed', 'Re_wave_force_pitch'] = inputs['Im_H_feedbk'][:,6,4]
		partials['Im_RAO_wave_rotspeed', 'Im_wave_force_pitch'] = inputs['Re_H_feedbk'][:,6,4]
		partials['Im_RAO_wave_rotspeed', 'Re_wave_force_bend'] = inputs['Im_H_feedbk'][:,6,5]
		partials['Im_RAO_wave_rotspeed', 'Im_wave_force_bend'] = inputs['Re_H_feedbk'][:,6,5]

		for i in xrange(N_omega):
			partials['Re_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i] = inputs['Re_wave_force_surge'][i]
			partials['Re_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i+1] = inputs['Re_wave_force_pitch'][i]
			partials['Re_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i+2] = inputs['Re_wave_force_bend'][i]
			partials['Re_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i] = -inputs['Im_wave_force_surge'][i]
			partials['Re_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i+1] = -inputs['Im_wave_force_pitch'][i]
			partials['Re_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i+2] = -inputs['Im_wave_force_bend'][i]
			partials['Im_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i] = inputs['Im_wave_force_surge'][i]
			partials['Im_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i+1] = inputs['Im_wave_force_pitch'][i]
			partials['Im_RAO_wave_rotspeed', 'Re_H_feedbk'][3*i+2] = inputs['Im_wave_force_bend'][i]
			partials['Im_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i] = inputs['Re_wave_force_surge'][i]
			partials['Im_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i+1] = inputs['Re_wave_force_pitch'][i]
			partials['Im_RAO_wave_rotspeed', 'Im_H_feedbk'][3*i+2] = inputs['Re_wave_force_bend'][i]