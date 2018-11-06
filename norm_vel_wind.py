import numpy as np

from openmdao.api import ExplicitComponent

class NormVelWind(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')

		self.add_output('Re_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_output('Re_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_output('Re_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_output('Im_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_output('Im_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_output('Im_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wind_vel_surge'] = -inputs['Im_RAO_wind_surge'] * omega
		outputs['Re_RAO_wind_vel_pitch'] = -inputs['Im_RAO_wind_pitch'] * omega
		outputs['Re_RAO_wind_vel_bend'] = -inputs['Im_RAO_wind_bend'] * omega

		outputs['Im_RAO_wind_vel_surge'] = inputs['Re_RAO_wind_surge'] * omega
		outputs['Im_RAO_wind_vel_pitch'] = inputs['Re_RAO_wind_pitch'] * omega
		outputs['Im_RAO_wind_vel_bend'] = inputs['Re_RAO_wind_bend'] * omega