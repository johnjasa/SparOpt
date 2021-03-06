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
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)
		self.add_input('windspeed_0', val=0., units='m/s')

		self.add_output('Re_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_output('Im_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')

		Hcols = Hcols1 = np.array([43,49])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*11*6+Hcols1),0)

		self.declare_partials('Re_RAO_Mwind_bldpitch', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'moment_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),2), cols=Hcols)
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'k_i')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'k_p')
		self.declare_partials('Re_RAO_Mwind_bldpitch', 'gain_corr_factor')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'k_i')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'k_p')
		self.declare_partials('Im_RAO_Mwind_bldpitch', 'gain_corr_factor')

	def compute(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		moment_wind = inputs['moment_wind']
		windspeed_0 = inputs['windspeed_0']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_Mwind_rot_lp = H_feedbk[:,7,1] * moment_wind
		RAO_Mwind_rotspeed_lp = H_feedbk[:,8,1] * moment_wind

		if (windspeed_0 <= 25.) and (windspeed_0 >= 11.4):
			RAO_Mwind_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_Mwind_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_Mwind_rotspeed_lp
		else:
			RAO_Mwind_bldpitch = np.zeros(N_omega)

		outputs['Re_RAO_Mwind_bldpitch'] = np.real(RAO_Mwind_bldpitch)
		outputs['Im_RAO_Mwind_bldpitch'] = np.imag(RAO_Mwind_bldpitch)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		windspeed_0 = inputs['windspeed_0']

		if (windspeed_0 <= 25.) and (windspeed_0 >= 11.4):
			partials['Re_RAO_Mwind_bldpitch', 'moment_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,1] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,1]
			partials['Im_RAO_Mwind_bldpitch', 'moment_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,1] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,1]

			partials['Re_RAO_Mwind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * inputs['Re_H_feedbk'][:,7,1] * inputs['moment_wind']
			partials['Im_RAO_Mwind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * inputs['Im_H_feedbk'][:,7,1] * inputs['moment_wind']
			partials['Re_RAO_Mwind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * inputs['Re_H_feedbk'][:,8,1] * inputs['moment_wind']
			partials['Im_RAO_Mwind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * inputs['Im_H_feedbk'][:,8,1] * inputs['moment_wind']
			partials['Re_RAO_Mwind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * inputs['Re_H_feedbk'][:,7,1] * inputs['moment_wind'] + inputs['k_p'] * inputs['Re_H_feedbk'][:,8,1] * inputs['moment_wind']
			partials['Im_RAO_Mwind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * inputs['Im_H_feedbk'][:,7,1] * inputs['moment_wind'] + inputs['k_p'] * inputs['Im_H_feedbk'][:,8,1] * inputs['moment_wind']

			partials['Re_RAO_Mwind_bldpitch', 'Im_H_feedbk'] = np.zeros(2*N_omega)
			partials['Im_RAO_Mwind_bldpitch', 'Re_H_feedbk'] = np.zeros(2*N_omega)

			for i in xrange(N_omega):
				partials['Re_RAO_Mwind_bldpitch', 'Re_H_feedbk'][2*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['moment_wind'][i]
				partials['Re_RAO_Mwind_bldpitch', 'Re_H_feedbk'][2*i+1] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['moment_wind'][i]
				partials['Im_RAO_Mwind_bldpitch', 'Im_H_feedbk'][2*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['moment_wind'][i]
				partials['Im_RAO_Mwind_bldpitch', 'Im_H_feedbk'][2*i+1] = inputs['gain_corr_factor'] * inputs['k_p'] * inputs['moment_wind'][i]