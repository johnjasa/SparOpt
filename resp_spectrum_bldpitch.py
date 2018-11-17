import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectrumBldpitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_Mwind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_bldpitch', val=np.zeros(N_omega), units='rad**2*s/rad')

		self.declare_partials('resp_bldpitch', 'Re_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'Im_RAO_wave_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'Re_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'Im_RAO_wind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'Re_RAO_Mwind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'Im_RAO_Mwind_bldpitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_bldpitch', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_bldpitch'] = np.abs(inputs['Re_RAO_wave_bldpitch'] + 1j * inputs['Im_RAO_wave_bldpitch'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_bldpitch'] + 1j * inputs['Im_RAO_wind_bldpitch'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_bldpitch'] + 1j * inputs['Im_RAO_Mwind_bldpitch'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials): #TODO check
		partials['resp_bldpitch', 'Re_RAO_wave_bldpitch'] = 2. * inputs['Re_RAO_wave_bldpitch'] * inputs['S_wave']
		partials['resp_bldpitch', 'Im_RAO_wave_bldpitch'] = 2. * inputs['Im_RAO_wave_bldpitch'] * inputs['S_wave']
		partials['resp_bldpitch', 'Re_RAO_wind_bldpitch'] = 2. * inputs['Re_RAO_wind_bldpitch'] * inputs['S_wind']
		partials['resp_bldpitch', 'Im_RAO_wind_bldpitch'] = 2. * inputs['Im_RAO_wind_bldpitch'] * inputs['S_wind']
		partials['resp_bldpitch', 'Re_RAO_Mwind_bldpitch'] = 2. * inputs['Re_RAO_Mwind_bldpitch'] * inputs['S_wind']
		partials['resp_bldpitch', 'Im_RAO_Mwind_bldpitch'] = 2. * inputs['Im_RAO_Mwind_bldpitch'] * inputs['S_wind']
		partials['resp_bldpitch', 'S_wave'] = np.abs(inputs['Re_RAO_wave_bldpitch'] + 1j * inputs['Im_RAO_wave_bldpitch'])**2.
		partials['resp_bldpitch', 'S_wind'] = np.abs(inputs['Re_RAO_wind_bldpitch'] + 1j * inputs['Im_RAO_wind_bldpitch'])**2. + np.abs(inputs['Re_RAO_Mwind_bldpitch'] + 1j * inputs['Im_RAO_Mwind_bldpitch'])**2.