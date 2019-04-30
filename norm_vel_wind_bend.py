import numpy as np

from openmdao.api import ExplicitComponent

class NormVelWindBend(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')

		self.add_output('Re_RAO_wind_vel_bend', val=np.ones(N_omega), units='(m/s)/(m/s)')
		self.add_output('Im_RAO_wind_vel_bend', val=np.ones(N_omega), units='(m/s)/(m/s)')

		self.declare_partials('Re_RAO_wind_vel_bend', 'Re_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega), val=0.)
		self.declare_partials('Re_RAO_wind_vel_bend', 'Im_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega), val=-self.omega)
		self.declare_partials('Im_RAO_wind_vel_bend', 'Re_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega), val=self.omega)
		self.declare_partials('Im_RAO_wind_vel_bend', 'Im_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega), val=0.)

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wind_vel_bend'] = -inputs['Im_RAO_wind_bend'] * omega
		outputs['Im_RAO_wind_vel_bend'] = inputs['Re_RAO_wind_bend'] * omega
