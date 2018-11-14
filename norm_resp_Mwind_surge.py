import numpy as np

from openmdao.api import ExplicitComponent

class NormRespMWindSurge(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))

		self.add_output('Re_RAO_Mwind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_output('Im_RAO_Mwind_surge', val=np.zeros(N_omega), units='m/(m/s)')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		moment_wind = inputs['moment_wind']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_Mwind_surge = H_feedbk[:,0,1] * moment_wind

		outputs['Re_RAO_Mwind_surge'] = np.real(RAO_Mwind_surge)
		outputs['Im_RAO_Mwind_surge'] = np.imag(RAO_Mwind_surge)

	def compute_partials(self, inputs, partials):
		pass