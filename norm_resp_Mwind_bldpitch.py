import numpy as np

from openmdao.api import ExplicitComponent

class NormRespMWindBldpitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('moment_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,9,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)

		self.add_output('Re_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_output('Im_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		moment_wind = inputs['moment_wind']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_Mwind_rot_lp = H_feedbk[:,7,1] * moment_wind
		RAO_Mwind_rotspeed_lp = H_feedbk[:,8,1] * moment_wind
		RAO_Mwind_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_Mwind_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_Mwind_rotspeed_lp

		outputs['Re_RAO_Mwind_bldpitch'] = np.real(RAO_Mwind_bldpitch)
		outputs['Im_RAO_Mwind_bldpitch'] = np.imag(RAO_Mwind_bldpitch)

	def compute_partials(self, inputs, partials):
		pass