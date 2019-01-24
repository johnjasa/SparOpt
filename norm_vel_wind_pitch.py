import numpy as np

from openmdao.api import ExplicitComponent

class NormVelWindPitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')

		self.add_output('Re_RAO_wind_vel_pitch', val=np.ones(N_omega), units='(rad/s)/(m/s)')
		self.add_output('Im_RAO_wind_vel_pitch', val=np.ones(N_omega), units='(rad/s)/(m/s)')

		self.declare_partials('Re_RAO_wind_vel_pitch', 'Re_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_vel_pitch', 'Im_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_vel_pitch', 'Re_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_vel_pitch', 'Im_RAO_wind_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wind_vel_pitch'] = -inputs['Im_RAO_wind_pitch'] * omega
		outputs['Im_RAO_wind_vel_pitch'] = inputs['Re_RAO_wind_pitch'] * omega

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		partials['Re_RAO_wind_vel_pitch', 'Re_RAO_wind_pitch'] = np.zeros(N_omega)
		partials['Re_RAO_wind_vel_pitch', 'Im_RAO_wind_pitch'] = -omega
		partials['Im_RAO_wind_vel_pitch', 'Re_RAO_wind_pitch'] = omega
		partials['Im_RAO_wind_vel_pitch', 'Im_RAO_wind_pitch'] = np.zeros(N_omega)