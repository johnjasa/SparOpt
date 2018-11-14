import numpy as np

from openmdao.api import ExplicitComponent

class TBMomentSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_TB_moment', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Im_RAO_wave_TB_moment', val=np.zeros(N_omega), units='N*m/m')
		self.add_input('Re_RAO_wind_TB_moment', val=np.zeros(N_omega), units='N*m/(m/s)')
		self.add_input('Im_RAO_wind_TB_moment', val=np.zeros(N_omega), units='N*m/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_TB_moment', val=np.zeros(N_omega), units='(N*m)**2*s/rad')

		self.declare_partials('resp_TB_moment', 'Re_RAO_wave_TB_moment', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_moment', 'Im_RAO_wave_TB_moment', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_moment', 'Re_RAO_wind_TB_moment', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_moment', 'Im_RAO_wind_TB_moment', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_moment', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_TB_moment', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_TB_moment'] = np.abs(inputs['Re_RAO_wave_TB_moment'] + 1j * inputs['Im_RAO_wave_TB_moment'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_TB_moment'] + 1j * inputs['Im_RAO_wind_TB_moment'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials):
		partials['resp_TB_moment', 'Re_RAO_wave_TB_moment'] = 2. * inputs['Re_RAO_wave_TB_moment'] * inputs['S_wave']
		partials['resp_TB_moment', 'Im_RAO_wave_TB_moment'] = 2. * inputs['Im_RAO_wave_TB_moment'] * inputs['S_wave']
		partials['resp_TB_moment', 'Re_RAO_wind_TB_moment'] = 2. * inputs['Re_RAO_wind_TB_moment'] * inputs['S_wind']
		partials['resp_TB_moment', 'Im_RAO_wind_TB_moment'] = 2. * inputs['Im_RAO_wind_TB_moment'] * inputs['S_wind']
		partials['resp_TB_moment', 'S_wave'] = np.abs(inputs['Re_RAO_wave_TB_moment'] + 1j * inputs['Im_RAO_wave_TB_moment'])**2.
		partials['resp_TB_moment', 'S_wind'] = np.abs(inputs['Re_RAO_wind_TB_moment'] + 1j * inputs['Im_RAO_wind_TB_moment'])**2.