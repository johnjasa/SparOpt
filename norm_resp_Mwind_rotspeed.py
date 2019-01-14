import numpy as np

from openmdao.api import ExplicitComponent

class NormRespMWindRotspeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,11,6)))

		self.add_output('Re_RAO_Mwind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_output('Im_RAO_Mwind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')

		self.declare_partials('Re_RAO_Mwind_rotspeed', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_rotspeed', 'Re_H_feedbk', rows=np.arange(N_omega), cols=np.arange(37,N_omega*11*6,11*6))
		self.declare_partials('Re_RAO_Mwind_rotspeed', 'Im_H_feedbk', rows=np.arange(N_omega), cols=np.arange(37,N_omega*11*6,11*6))
		self.declare_partials('Im_RAO_Mwind_rotspeed', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_rotspeed', 'Re_H_feedbk', rows=np.arange(N_omega), cols=np.arange(37,N_omega*11*6,11*6))
		self.declare_partials('Im_RAO_Mwind_rotspeed', 'Im_H_feedbk', rows=np.arange(N_omega), cols=np.arange(37,N_omega*11*6,11*6))

	def compute(self, inputs, outputs):
		omega = self.omega

		moment_wind = inputs['moment_wind']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_Mwind_rotspeed = H_feedbk[:,6,1] * moment_wind

		outputs['Re_RAO_Mwind_rotspeed'] = np.real(RAO_Mwind_rotspeed)
		outputs['Im_RAO_Mwind_rotspeed'] = np.imag(RAO_Mwind_rotspeed)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_Mwind_rotspeed', 'moment_wind'] = inputs['Re_H_feedbk'][:,6,1]
		partials['Im_RAO_Mwind_rotspeed', 'moment_wind'] = inputs['Im_H_feedbk'][:,6,1]

		partials['Re_RAO_Mwind_rotspeed', 'Im_H_feedbk'] = np.zeros(N_omega)
		partials['Im_RAO_Mwind_rotspeed', 'Re_H_feedbk'] = np.zeros(N_omega)

		partials['Re_RAO_Mwind_rotspeed', 'Re_H_feedbk'] = inputs['moment_wind']
		partials['Im_RAO_Mwind_rotspeed', 'Im_H_feedbk'] = inputs['moment_wind']