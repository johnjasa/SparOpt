import numpy as np

from openmdao.api import ExplicitComponent

class NormRespWindBldpitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('thrust_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('torque_wind', val=np.zeros(N_omega), units='m/s')
		self.add_input('Re_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('Im_H_feedbk', val=np.zeros((N_omega,11,6)))
		self.add_input('k_i', val=0., units='rad/rad')
		self.add_input('k_p', val=0., units='rad*s/rad')
		self.add_input('gain_corr_factor', val=0.)
		self.add_input('windspeed_0', val=0., units='m/s')

		self.add_output('Re_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_output('Im_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')

		Hcols = Hcols1 = np.array([42,44,48,50])
		for i in xrange(1,N_omega):
			Hcols = np.concatenate((Hcols,i*11*6+Hcols1),0)

		self.declare_partials('Re_RAO_wind_bldpitch', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_bldpitch', 'torque_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Re_RAO_wind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),4), cols=Hcols)
		self.declare_partials('Re_RAO_wind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),4), cols=Hcols)
		self.declare_partials('Im_RAO_wind_bldpitch', 'thrust_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_bldpitch', 'torque_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('Im_RAO_wind_bldpitch', 'Re_H_feedbk', rows=np.repeat(np.arange(N_omega),4), cols=Hcols)
		self.declare_partials('Im_RAO_wind_bldpitch', 'Im_H_feedbk', rows=np.repeat(np.arange(N_omega),4), cols=Hcols)
		self.declare_partials('Re_RAO_wind_bldpitch', 'k_i')
		self.declare_partials('Re_RAO_wind_bldpitch', 'k_p')
		self.declare_partials('Re_RAO_wind_bldpitch', 'gain_corr_factor')
		self.declare_partials('Im_RAO_wind_bldpitch', 'k_i')
		self.declare_partials('Im_RAO_wind_bldpitch', 'k_p')
		self.declare_partials('Im_RAO_wind_bldpitch', 'gain_corr_factor')

	def compute(self, inputs, outputs):
		omega = self.omega
		N_omega = len(omega)

		thrust_wind = inputs['thrust_wind']
		torque_wind = inputs['torque_wind']
		windspeed_0 = inputs['windspeed_0']

		H_feedbk = inputs['Re_H_feedbk'] + 1j * inputs['Im_H_feedbk']

		RAO_wind_rot_lp = H_feedbk[:,7,0] * thrust_wind + H_feedbk[:,7,2] * torque_wind
		RAO_wind_rotspeed_lp = H_feedbk[:,8,0] * thrust_wind + H_feedbk[:,8,2] * torque_wind

		if (windspeed_0 <= 25.) and (windspeed_0 >= 11.4):
			RAO_wind_bldpitch = inputs['gain_corr_factor'] * inputs['k_i'] * RAO_wind_rot_lp + inputs['gain_corr_factor'] * inputs['k_p'] * RAO_wind_rotspeed_lp
		else:
			RAO_wind_bldpitch = np.zeros(N_omega)

		outputs['Re_RAO_wind_bldpitch'] = np.real(RAO_wind_bldpitch)
		outputs['Im_RAO_wind_bldpitch'] = np.imag(RAO_wind_bldpitch)

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)

		windspeed_0 = inputs['windspeed_0']

		if (windspeed_0 <= 25.) and (windspeed_0 >= 11.4):
			partials['Re_RAO_wind_bldpitch', 'thrust_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,0] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,0]
			partials['Re_RAO_wind_bldpitch', 'torque_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Re_H_feedbk'][:,7,2] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Re_H_feedbk'][:,8,2]
			partials['Im_RAO_wind_bldpitch', 'thrust_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,0] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,0]
			partials['Im_RAO_wind_bldpitch', 'torque_wind'] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['Im_H_feedbk'][:,7,2] + inputs['gain_corr_factor'] * inputs['k_p'] * inputs['Im_H_feedbk'][:,8,2]

			partials['Re_RAO_wind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * (inputs['Re_H_feedbk'][:,7,0] * inputs['thrust_wind'] + inputs['Re_H_feedbk'][:,7,2] * inputs['torque_wind'])
			partials['Im_RAO_wind_bldpitch', 'k_i'] = inputs['gain_corr_factor'] * (inputs['Im_H_feedbk'][:,7,0] * inputs['thrust_wind'] + inputs['Im_H_feedbk'][:,7,2] * inputs['torque_wind'])
			partials['Re_RAO_wind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * (inputs['Re_H_feedbk'][:,8,0] * inputs['thrust_wind'] + inputs['Re_H_feedbk'][:,8,2] * inputs['torque_wind'])
			partials['Im_RAO_wind_bldpitch', 'k_p'] = inputs['gain_corr_factor'] * (inputs['Im_H_feedbk'][:,8,0] * inputs['thrust_wind'] + inputs['Im_H_feedbk'][:,8,2] * inputs['torque_wind'])
			partials['Re_RAO_wind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * (inputs['Re_H_feedbk'][:,7,0] * inputs['thrust_wind'] + inputs['Re_H_feedbk'][:,7,2] * inputs['torque_wind']) + inputs['k_p'] * (inputs['Re_H_feedbk'][:,8,0] * inputs['thrust_wind'] + inputs['Re_H_feedbk'][:,8,2] * inputs['torque_wind'])
			partials['Im_RAO_wind_bldpitch', 'gain_corr_factor'] = inputs['k_i'] * (inputs['Im_H_feedbk'][:,7,0] * inputs['thrust_wind'] + inputs['Im_H_feedbk'][:,7,2] * inputs['torque_wind']) + inputs['k_p'] * (inputs['Im_H_feedbk'][:,8,0] * inputs['thrust_wind'] + inputs['Im_H_feedbk'][:,8,2] * inputs['torque_wind'])

			partials['Re_RAO_wind_bldpitch', 'Im_H_feedbk'] = np.zeros(4*N_omega)
			partials['Im_RAO_wind_bldpitch', 'Re_H_feedbk'] = np.zeros(4*N_omega)

			for i in xrange(N_omega):
				partials['Re_RAO_wind_bldpitch', 'Re_H_feedbk'][4*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['thrust_wind'][i]
				partials['Re_RAO_wind_bldpitch', 'Re_H_feedbk'][4*i+1] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['torque_wind'][i]
				partials['Re_RAO_wind_bldpitch', 'Re_H_feedbk'][4*i+2] = inputs['gain_corr_factor'] * inputs['k_p'] *inputs['thrust_wind'][i]
				partials['Re_RAO_wind_bldpitch', 'Re_H_feedbk'][4*i+3] = inputs['gain_corr_factor'] * inputs['k_p'] *inputs['torque_wind'][i]
				partials['Im_RAO_wind_bldpitch', 'Im_H_feedbk'][4*i] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['thrust_wind'][i]
				partials['Im_RAO_wind_bldpitch', 'Im_H_feedbk'][4*i+1] = inputs['gain_corr_factor'] * inputs['k_i'] * inputs['torque_wind'][i]
				partials['Im_RAO_wind_bldpitch', 'Im_H_feedbk'][4*i+2] = inputs['gain_corr_factor'] * inputs['k_p'] *inputs['thrust_wind'][i]
				partials['Im_RAO_wind_bldpitch', 'Im_H_feedbk'][4*i+3] = inputs['gain_corr_factor'] * inputs['k_p'] *inputs['torque_wind'][i]