import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectrumRotspeed(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_rotspeed', val=np.zeros(N_omega), units='rad**2*s/(rad*s)')

		self.declare_partials('resp_rotspeed', 'Re_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'Im_RAO_wave_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'Re_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'Im_RAO_wind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'Re_RAO_Mwind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'Im_RAO_Mwind_rotspeed', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_rotspeed', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_rotspeed'] = np.abs(inputs['Re_RAO_wave_rotspeed'] + 1j * inputs['Im_RAO_wave_rotspeed'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_rotspeed'] + 1j * inputs['Im_RAO_wind_rotspeed'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_rotspeed'] + 1j * inputs['Im_RAO_Mwind_rotspeed'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials): #TODO check
		partials['resp_rotspeed', 'Re_RAO_wave_rotspeed'] = 2. * inputs['Re_RAO_wave_rotspeed'] * inputs['S_wave']
		partials['resp_rotspeed', 'Im_RAO_wave_rotspeed'] = 2. * inputs['Im_RAO_wave_rotspeed'] * inputs['S_wave']
		partials['resp_rotspeed', 'Re_RAO_wind_rotspeed'] = 2. * inputs['Re_RAO_wind_rotspeed'] * inputs['S_wind']
		partials['resp_rotspeed', 'Im_RAO_wind_rotspeed'] = 2. * inputs['Im_RAO_wind_rotspeed'] * inputs['S_wind']
		partials['resp_rotspeed', 'Re_RAO_Mwind_rotspeed'] = 2. * inputs['Re_RAO_Mwind_rotspeed'] * inputs['S_wind']
		partials['resp_rotspeed', 'Im_RAO_Mwind_rotspeed'] = 2. * inputs['Im_RAO_Mwind_rotspeed'] * inputs['S_wind']
		partials['resp_rotspeed', 'S_wave'] = np.abs(inputs['Re_RAO_wave_rotspeed'] + 1j * inputs['Im_RAO_wave_rotspeed'])**2.
		partials['resp_rotspeed', 'S_wind'] = np.abs(inputs['Re_RAO_wind_rotspeed'] + 1j * inputs['Im_RAO_wind_rotspeed'])**2. + np.abs(inputs['Re_RAO_Mwind_rotspeed'] + 1j * inputs['Im_RAO_Mwind_rotspeed'])**2.