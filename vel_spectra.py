import numpy as np

from openmdao.api import ExplicitComponent

class VelSpectra(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_surge', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Im_RAO_wave_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_vel_bend', val=np.zeros(N_omega), units='(m/s)/m')
		self.add_input('Re_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_vel_surge', val=np.zeros(N_omega), units='(m/s)**2*s/rad')
		self.add_output('resp_vel_pitch', val=np.zeros(N_omega), units='(rad/s)**2*s/rad')
		self.add_output('resp_vel_bend', val=np.zeros(N_omega), units='(m/s)**2*s/rad')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['resp_vel_surge'] = np.abs(inputs['Re_RAO_wave_vel_surge'] + 1j * inputs['Im_RAO_wave_vel_surge'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_vel_surge'] + 1j * inputs['Im_RAO_wind_vel_surge'])**2. * inputs['S_wind']
		outputs['resp_vel_pitch'] = np.abs(inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_vel_pitch'] + 1j * inputs['Im_RAO_wind_vel_pitch'])**2. * inputs['S_wind']
		outputs['resp_vel_bend'] = np.abs(inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_vel_bend'] + 1j * inputs['Im_RAO_wind_vel_bend'])**2. * inputs['S_wind']