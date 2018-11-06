import numpy as np

from openmdao.api import ExplicitComponent

class NormAccWind(ExplicitComponent):

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

		self.add_output('Re_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_output('Re_RAO_wind_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/(m/s)')
		self.add_output('Re_RAO_wind_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_output('Im_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_output('Im_RAO_wind_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)/(m/s)')
		self.add_output('Im_RAO_wind_acc_bend', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wind_acc_surge'] = -inputs['Re_RAO_wind_surge'] * omega**2.
		outputs['Re_RAO_wind_acc_pitch'] = -inputs['Re_RAO_wind_pitch'] * omega**2.
		outputs['Re_RAO_wind_acc_bend'] = -inputs['Re_RAO_wind_bend'] * omega**2.

		outputs['Im_RAO_wind_acc_surge'] = -inputs['Im_RAO_wind_surge'] * omega**2.
		outputs['Im_RAO_wind_acc_pitch'] = -inputs['Im_RAO_wind_pitch'] * omega**2.
		outputs['Im_RAO_wind_acc_bend'] = -inputs['Im_RAO_wind_bend'] * omega**2.