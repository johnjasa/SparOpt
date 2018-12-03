import numpy as np

from openmdao.api import ExplicitComponent

class TowerMomentSpectrum(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_tower_moment', val=np.zeros((N_omega,11)), units='N*m/m')
		self.add_input('Im_RAO_wave_tower_moment', val=np.zeros((N_omega,11)), units='N*m/m')
		self.add_input('Re_RAO_wind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')
		self.add_input('Im_RAO_wind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')
		self.add_input('Re_RAO_Mwind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')
		self.add_input('Im_RAO_Mwind_tower_moment', val=np.zeros((N_omega,11)), units='N*m/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_tower_moment', val=np.zeros((N_omega,11)), units='(N*m)**2*s/rad')

		self.declare_partials('resp_tower_moment', 'Re_RAO_wave_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'Im_RAO_wave_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'Re_RAO_wind_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'Im_RAO_wind_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'Re_RAO_Mwind_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'Im_RAO_Mwind_tower_moment', rows=np.arange(11*N_omega), cols=np.arange(11*N_omega))
		self.declare_partials('resp_tower_moment', 'S_wave', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))
		self.declare_partials('resp_tower_moment', 'S_wind', rows=np.arange(11*N_omega), cols=np.repeat(np.arange(N_omega),11))

	def compute(self, inputs, outputs):
		for i in xrange(11):
			outputs['resp_tower_moment'][:,i] = np.abs(inputs['Re_RAO_wave_tower_moment'][:,i] + 1j * inputs['Im_RAO_wave_tower_moment'][:,i])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_tower_moment'][:,i] + 1j * inputs['Im_RAO_wind_tower_moment'][:,i])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_tower_moment'][:,i] + 1j * inputs['Im_RAO_Mwind_tower_moment'][:,i])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials):
		N_omega = len(self.omega)

		for i in xrange(N_omega):
			partials['resp_tower_moment', 'Re_RAO_wave_tower_moment'][i*11:i*11+11] = 2. * inputs['Re_RAO_wave_tower_moment'][i] * inputs['S_wave'][i]
			partials['resp_tower_moment', 'Im_RAO_wave_tower_moment'][i*11:i*11+11] = 2. * inputs['Im_RAO_wave_tower_moment'][i] * inputs['S_wave'][i]
			partials['resp_tower_moment', 'Re_RAO_wind_tower_moment'][i*11:i*11+11] = 2. * inputs['Re_RAO_wind_tower_moment'][i] * inputs['S_wind'][i]
			partials['resp_tower_moment', 'Im_RAO_wind_tower_moment'][i*11:i*11+11] = 2. * inputs['Im_RAO_wind_tower_moment'][i] * inputs['S_wind'][i]
			partials['resp_tower_moment', 'Re_RAO_Mwind_tower_moment'][i*11:i*11+11] = 2. * inputs['Re_RAO_Mwind_tower_moment'][i] * inputs['S_wind'][i]
			partials['resp_tower_moment', 'Im_RAO_Mwind_tower_moment'][i*11:i*11+11] = 2. * inputs['Im_RAO_Mwind_tower_moment'][i] * inputs['S_wind'][i]
			partials['resp_tower_moment', 'S_wave'][i*11:i*11+11] = np.abs(inputs['Re_RAO_wave_tower_moment'][i] + 1j * inputs['Im_RAO_wave_tower_moment'][i])**2.
			partials['resp_tower_moment', 'S_wind'][i*11:i*11+11] = np.abs(inputs['Re_RAO_wind_tower_moment'][i] + 1j * inputs['Im_RAO_wind_tower_moment'][i])**2. + np.abs(inputs['Re_RAO_Mwind_tower_moment'][i] + 1j * inputs['Im_RAO_Mwind_tower_moment'][i])**2.