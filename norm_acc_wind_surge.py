import numpy as np

from openmdao.api import ExplicitComponent

class NormAccWindSurge(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')

		self.add_output('Re_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')
		self.add_output('Im_RAO_wind_acc_surge', val=np.zeros(N_omega), units='(m/s**2)/(m/s)')

		self.declare_partials('Re_RAO_wind_acc_surge', 'Re_RAO_wind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_acc_surge', 'Im_RAO_wind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_acc_surge', 'Re_RAO_wind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_acc_surge', 'Im_RAO_wind_surge', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['Re_RAO_wind_acc_surge'] = -inputs['Re_RAO_wind_surge'] * omega**2.
		outputs['Im_RAO_wind_acc_surge'] = -inputs['Im_RAO_wind_surge'] * omega**2.

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(self.omega)

		partials['Re_RAO_wind_acc_surge', 'Re_RAO_wind_surge'] = -omega**2.
		partials['Re_RAO_wind_acc_surge', 'Im_RAO_wind_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wind_acc_surge', 'Re_RAO_wind_surge'] = np.zeros(N_omega)
		partials['Im_RAO_wind_acc_surge', 'Im_RAO_wind_surge'] = -omega**2.