import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectrumBend(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_Mwind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_Mwind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_bend', val=np.zeros(N_omega), units='m**2*s/rad')

		self.declare_partials('resp_bend', 'Re_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'Im_RAO_wave_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'Re_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'Im_RAO_wind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'Re_RAO_Mwind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'Im_RAO_Mwind_bend', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bend', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_bend'] = np.abs(inputs['Re_RAO_wave_bend'] + 1j * inputs['Im_RAO_wave_bend'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_bend'] + 1j * inputs['Im_RAO_wind_bend'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_bend'] + 1j * inputs['Im_RAO_Mwind_bend'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials): #TODO check
		partials['resp_bend', 'Re_RAO_wave_bend'] = 2. * inputs['Re_RAO_wave_bend'] * inputs['S_wave']
		partials['resp_bend', 'Im_RAO_wave_bend'] = 2. * inputs['Im_RAO_wave_bend'] * inputs['S_wave']
		partials['resp_bend', 'Re_RAO_wind_bend'] = 2. * inputs['Re_RAO_wind_bend'] * inputs['S_wind']
		partials['resp_bend', 'Im_RAO_wind_bend'] = 2. * inputs['Im_RAO_wind_bend'] * inputs['S_wind']
		partials['resp_bend', 'Re_RAO_Mwind_bend'] = 2. * inputs['Re_RAO_Mwind_bend'] * inputs['S_wind']
		partials['resp_bend', 'Im_RAO_Mwind_bend'] = 2. * inputs['Im_RAO_Mwind_bend'] * inputs['S_wind']
		partials['resp_bend', 'S_wave'] = np.abs(inputs['Re_RAO_wave_bend'] + 1j * inputs['Im_RAO_wave_bend'])**2.
		partials['resp_bend', 'S_wind'] = np.abs(inputs['Re_RAO_wind_bend'] + 1j * inputs['Im_RAO_wind_bend'])**2. + np.abs(inputs['Re_RAO_Mwind_bend'] + 1j * inputs['Im_RAO_Mwind_bend'])**2.