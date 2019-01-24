import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectrumFairlead(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_fairlead', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_Mwind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_Mwind_fairlead', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_fairlead', val=np.zeros(N_omega), units='m**2*s/rad')

		self.declare_partials('resp_fairlead', 'Re_RAO_wave_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'Im_RAO_wave_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'Re_RAO_wind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'Im_RAO_wind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'Re_RAO_Mwind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'Im_RAO_Mwind_fairlead', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_fairlead', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_fairlead'] = np.abs(inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_fairlead'] + 1j * inputs['Im_RAO_Mwind_fairlead'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials):
		partials['resp_fairlead', 'Re_RAO_wave_fairlead'] = 2. * inputs['Re_RAO_wave_fairlead'] * inputs['S_wave']
		partials['resp_fairlead', 'Im_RAO_wave_fairlead'] = 2. * inputs['Im_RAO_wave_fairlead'] * inputs['S_wave']
		partials['resp_fairlead', 'Re_RAO_wind_fairlead'] = 2. * inputs['Re_RAO_wind_fairlead'] * inputs['S_wind']
		partials['resp_fairlead', 'Im_RAO_wind_fairlead'] = 2. * inputs['Im_RAO_wind_fairlead'] * inputs['S_wind']
		partials['resp_fairlead', 'Re_RAO_Mwind_fairlead'] = 2. * inputs['Re_RAO_Mwind_fairlead'] * inputs['S_wind']
		partials['resp_fairlead', 'Im_RAO_Mwind_fairlead'] = 2. * inputs['Im_RAO_Mwind_fairlead'] * inputs['S_wind']
		partials['resp_fairlead', 'S_wave'] = np.abs(inputs['Re_RAO_wave_fairlead'] + 1j * inputs['Im_RAO_wave_fairlead'])**2.
		partials['resp_fairlead', 'S_wind'] = np.abs(inputs['Re_RAO_wind_fairlead'] + 1j * inputs['Im_RAO_wind_fairlead'])**2. + np.abs(inputs['Re_RAO_Mwind_fairlead'] + 1j * inputs['Im_RAO_Mwind_fairlead'])**2.