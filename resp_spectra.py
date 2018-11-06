import numpy as np

from openmdao.api import ExplicitComponent

class RespSpectra(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('Re_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Re_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_rot_lp', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wave_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Re_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_surge', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_pitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_bend', val=np.zeros(N_omega), units='m/m')
		self.add_input('Im_RAO_wave_rotspeed', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_rot_lp', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Im_RAO_wave_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/m')
		self.add_input('Im_RAO_wave_bldpitch', val=np.zeros(N_omega), units='rad/m')
		self.add_input('Re_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Re_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_rot_lp', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Re_RAO_wind_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_surge', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_pitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_bend', val=np.zeros(N_omega), units='m/(m/s)')
		self.add_input('Im_RAO_wind_rotspeed', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_rot_lp', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('Im_RAO_wind_rotspeed_lp', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_wind_bldpitch', val=np.zeros(N_omega), units='rad/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')

		self.add_output('resp_surge', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_output('resp_pitch', val=np.zeros(N_omega), units='rad**2*s/rad')
		self.add_output('resp_bend', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_output('resp_rotspeed', val=np.zeros(N_omega), units='rad**2*s/(rad*s)')
		self.add_output('resp_rot_lp', val=np.zeros(N_omega), units='rad**2*s/rad')
		self.add_output('resp_rotspeed_lp', val=np.zeros(N_omega), units='rad**2*s/(rad*s)')
		self.add_output('resp_bldpitch', val=np.zeros(N_omega), units='rad**2*s/rad')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['resp_surge'] = np.abs(inputs['Re_RAO_wave_surge'] + 1j * inputs['Im_RAO_wave_surge'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_surge'] + 1j * inputs['Im_RAO_wind_surge'])**2. * inputs['S_wind']
		outputs['resp_pitch'] = np.abs(inputs['Re_RAO_wave_pitch'] + 1j * inputs['Im_RAO_wave_pitch'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_pitch'] + 1j * inputs['Im_RAO_wind_pitch'])**2. * inputs['S_wind']
		outputs['resp_bend'] = np.abs(inputs['Re_RAO_wave_bend'] + 1j * inputs['Im_RAO_wave_bend'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_bend'] + 1j * inputs['Im_RAO_wind_bend'])**2. * inputs['S_wind']
		outputs['resp_rotspeed'] = np.abs(inputs['Re_RAO_wave_rotspeed'] + 1j * inputs['Im_RAO_wave_rotspeed'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_rotspeed'] + 1j * inputs['Im_RAO_wind_rotspeed'])**2. * inputs['S_wind']
		outputs['resp_rot_lp'] = np.abs(inputs['Re_RAO_wave_rot_lp'] + 1j * inputs['Im_RAO_wave_rot_lp'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_rot_lp'] + 1j * inputs['Im_RAO_wind_rot_lp'])**2. * inputs['S_wind']
		outputs['resp_rotspeed_lp'] = np.abs(inputs['Re_RAO_wave_rotspeed_lp'] + 1j * inputs['Im_RAO_wave_rotspeed_lp'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_rotspeed_lp'] + 1j * inputs['Im_RAO_wind_rotspeed_lp'])**2. * inputs['S_wind']
		outputs['resp_bldpitch'] = np.abs(inputs['Re_RAO_wave_bldpitch'] + 1j * inputs['Im_RAO_wave_bldpitch'])**2. * inputs['S_wave'] + np.abs(inputs['Re_RAO_wind_bldpitch'] + 1j * inputs['Im_RAO_wind_bldpitch'])**2. * inputs['S_wind']