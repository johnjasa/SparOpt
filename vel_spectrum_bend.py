import numpy as np

from openmdao.api import ExplicitComponent

class VelSpectrumBend(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_vel_bend', val=np.zeros(N_omega), units='(m/s)**2*s/rad')

		self.declare_partials('resp_vel_bend', 'Re_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'Im_RAO_wave_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'Re_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'Im_RAO_wind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'Re_RAO_Mwind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'Im_RAO_Mwind_vel_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_bend', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_vel_bend'] = np.abs(inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_vel_bend'] + 1j * inputs['Im_RAO_wind_vel_bend'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_vel_bend'] + 1j * inputs['Im_RAO_Mwind_vel_bend'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials):
		partials['resp_vel_bend', 'Re_RAO_wave_vel_bend'] = 2. * inputs['Re_RAO_wave_vel_bend'] * inputs['S_wave']
		partials['resp_vel_bend', 'Im_RAO_wave_vel_bend'] = 2. * inputs['Im_RAO_wave_vel_bend'] * inputs['S_wave']
		partials['resp_vel_bend', 'Re_RAO_wind_vel_bend'] = 2. * inputs['Re_RAO_wind_vel_bend'] * inputs['S_wind']
		partials['resp_vel_bend', 'Im_RAO_wind_vel_bend'] = 2. * inputs['Im_RAO_wind_vel_bend'] * inputs['S_wind']
		partials['resp_vel_bend', 'Re_RAO_Mwind_vel_bend'] = 2. * inputs['Re_RAO_Mwind_vel_bend'] * inputs['S_wind']
		partials['resp_vel_bend', 'Im_RAO_Mwind_vel_bend'] = 2. * inputs['Im_RAO_Mwind_vel_bend'] * inputs['S_wind']
		partials['resp_vel_bend', 'S_wave'] = np.abs(inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend'])**2.
		partials['resp_vel_bend', 'S_wind'] = np.abs(inputs['Re_RAO_wind_vel_bend'] + 1j * inputs['Im_RAO_wind_vel_bend'])**2. + np.abs(inputs['Re_RAO_Mwind_vel_bend'] + 1j * inputs['Im_RAO_Mwind_vel_bend'])**2.