import numpy as np

from openmdao.api import ExplicitComponent

class MoorTenSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_moor_ten', val=np.zeros(N_omega), units='N/m')
		self.add_input('Im_RAO_wave_moor_ten', val=np.zeros(N_omega), units='N/m')
		self.add_input('Re_RAO_wind_moor_ten', val=np.zeros(N_omega), units='N/(m/s)')
		self.add_input('Im_RAO_wind_moor_ten', val=np.zeros(N_omega), units='N/(m/s)')
		self.add_input('Re_RAO_Mwind_moor_ten', val=np.zeros(N_omega), units='N/(m/s)')
		self.add_input('Im_RAO_Mwind_moor_ten', val=np.zeros(N_omega), units='N/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_moor_ten', val=np.zeros(N_omega), units='N**2*s/rad')

		self.declare_partials('resp_moor_ten', 'Re_RAO_wave_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'Im_RAO_wave_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'Re_RAO_wind_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'Im_RAO_wind_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'Re_RAO_Mwind_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'Im_RAO_Mwind_moor_ten', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_moor_ten', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_moor_ten'] = np.abs(inputs['Re_RAO_wave_moor_ten'] + 1j * inputs['Im_RAO_wave_moor_ten'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_moor_ten'] + 1j * inputs['Im_RAO_wind_moor_ten'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_moor_ten'] + 1j * inputs['Im_RAO_Mwind_moor_ten'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials):
		partials['resp_moor_ten', 'Re_RAO_wave_moor_ten'] = 2. * inputs['Re_RAO_wave_moor_ten'] * inputs['S_wave']
		partials['resp_moor_ten', 'Im_RAO_wave_moor_ten'] = 2. * inputs['Im_RAO_wave_moor_ten'] * inputs['S_wave']
		partials['resp_moor_ten', 'Re_RAO_wind_moor_ten'] = 2. * inputs['Re_RAO_wind_moor_ten'] * inputs['S_wind']
		partials['resp_moor_ten', 'Im_RAO_wind_moor_ten'] = 2. * inputs['Im_RAO_wind_moor_ten'] * inputs['S_wind']
		partials['resp_moor_ten', 'Re_RAO_Mwind_moor_ten'] = 2. * inputs['Re_RAO_Mwind_moor_ten'] * inputs['S_wind']
		partials['resp_moor_ten', 'Im_RAO_Mwind_moor_ten'] = 2. * inputs['Im_RAO_Mwind_moor_ten'] * inputs['S_wind']
		partials['resp_moor_ten', 'S_wave'] = np.abs(inputs['Re_RAO_wave_moor_ten'] + 1j * inputs['Im_RAO_wave_moor_ten'])**2.
		partials['resp_moor_ten', 'S_wind'] = np.abs(inputs['Re_RAO_wind_moor_ten'] + 1j * inputs['Im_RAO_wind_moor_ten'])**2. + np.abs(inputs['Re_RAO_Mwind_moor_ten'] + 1j * inputs['Im_RAO_Mwind_moor_ten'])**2.