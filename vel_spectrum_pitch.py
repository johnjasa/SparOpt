import numpy as np

from openmdao.api import ExplicitComponent

class VelSpectrumPitch(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_vel_pitch', val=np.zeros(N_omega), units='(rad/s)**2*s/rad')

		self.declare_partials('resp_vel_pitch', 'Re_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'Im_RAO_wave_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'Re_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'Im_RAO_wind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'Re_RAO_Mwind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'Im_RAO_Mwind_vel_pitch', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'S_wave', rows=np.arange(N_omega), cols=np.arange(N_omega))
		self.declare_partials('resp_vel_pitch', 'S_wind', rows=np.arange(N_omega), cols=np.arange(N_omega))

	def compute(self, inputs, outputs):
		outputs['resp_vel_pitch'] = np.abs(inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_vel_pitch'] + 1j * inputs['Im_RAO_wind_vel_pitch'])**2. * inputs['S_wind'] + np.abs(inputs['Re_RAO_Mwind_vel_pitch'] + 1j * inputs['Im_RAO_Mwind_vel_pitch'])**2. * inputs['S_wind']

	def compute_partials(self, inputs, partials): #TODO check
		partials['resp_vel_pitch', 'Re_RAO_wave_vel_pitch'] = 2. * inputs['Re_RAO_wave_vel_pitch'] * inputs['S_wave']
		partials['resp_vel_pitch', 'Im_RAO_wave_vel_pitch'] = 2. * inputs['Im_RAO_wave_vel_pitch'] * inputs['S_wave']
		partials['resp_vel_pitch', 'Re_RAO_wind_vel_pitch'] = 2. * inputs['Re_RAO_wind_vel_pitch'] * inputs['S_wind']
		partials['resp_vel_pitch', 'Im_RAO_wind_vel_pitch'] = 2. * inputs['Im_RAO_wind_vel_pitch'] * inputs['S_wind']
		partials['resp_vel_pitch', 'Re_RAO_Mwind_vel_pitch'] = 2. * inputs['Re_RAO_Mwind_vel_pitch'] * inputs['S_wind']
		partials['resp_vel_pitch', 'Im_RAO_Mwind_vel_pitch'] = 2. * inputs['Im_RAO_Mwind_vel_pitch'] * inputs['S_wind']
		partials['resp_vel_pitch', 'S_wave'] = np.abs(inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch'])**2.
		partials['resp_vel_pitch', 'S_wind'] = np.abs(inputs['Re_RAO_wind_vel_pitch'] + 1j * inputs['Im_RAO_wind_vel_pitch'])**2. + np.abs(inputs['Re_RAO_Mwind_vel_pitch'] + 1j * inputs['Im_RAO_Mwind_vel_pitch'])**2.