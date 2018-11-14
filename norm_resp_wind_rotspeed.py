import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWindRotspeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('thrust_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('torque_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))

		self.add_output('Re_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_output('Im_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		thrust_wind = inputs['thrust_wind']
		torque_wind = inputs['torque_wind']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wind_rotspeed = H_feedbk[:,6,0] * thrust_wind + H_feedbk[:,6,2] * torque_wind

		outputs['Re_RAO_wind_rotspeed'] = np.real(RAO_wind_rotspeed)
		outputs['Im_RAO_wind_rotspeed'] = np.imag(RAO_wind_rotspeed)

	def compute_partials(self, inputs, partials):
		pass