import numpy as np

from openmdao.api import ExplicitComponent

class VelDistr(ExplicitComponent):

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
		self.add_input('Re_RAO_Mwind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Re_RAO_Mwind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_surge', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_pitch', val=np.zeros(N_omega), units='(rad/s)/(m/s)')
		self.add_input('Im_RAO_Mwind_vel_bend', val=np.zeros(N_omega), units='(m/s)/(m/s)')
		self.add_input('S_wave', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('S_wind', val=np.zeros(N_omega), units='m**2/(rad*s)')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('x_sparelem', val=np.zeros(13), units='m')

		self.add_output('stddev_vel_distr', val=np.zeros(13), units='m/s')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega
		
		z_sparnode = inputs['z_sparnode']
		x_sparelem = inputs['x_sparelem']

		for i in xrange(len(x_sparelem)):
			z = (z_sparnode[i] + z_sparnode[i+1]) / 2

			resp_vel_disp = np.abs((inputs['Re_RAO_wave_vel_surge'] + 1j * inputs['Im_RAO_wave_vel_surge']) + z * (inputs['Re_RAO_wave_vel_pitch'] + 1j * inputs['Im_RAO_wave_vel_pitch']) + x_sparelem[i] * (inputs['Re_RAO_wave_vel_bend'] + 1j * inputs['Im_RAO_wave_vel_bend']))**2. * inputs['S_wave'] + np.abs((inputs['Re_RAO_wind_vel_surge'] + 1j * inputs['Im_RAO_wind_vel_surge']) + z * (inputs['Re_RAO_wind_vel_pitch'] + 1j * inputs['Im_RAO_wind_vel_pitch']) + x_sparelem[i] * (inputs['Re_RAO_wind_vel_bend'] + 1j * inputs['Im_RAO_wind_vel_bend']))**2. * inputs['S_wind'] + np.abs((inputs['Re_RAO_Mwind_vel_surge'] + 1j * inputs['Im_RAO_Mwind_vel_surge']) + z * (inputs['Re_RAO_Mwind_vel_pitch'] + 1j * inputs['Im_RAO_Mwind_vel_pitch']) + x_sparelem[i] * (inputs['Re_RAO_Mwind_vel_bend'] + 1j * inputs['Im_RAO_Mwind_vel_bend']))**2. * inputs['S_wind']

			outputs['stddev_vel_distr'][i] = np.sqrt(np.trapz(resp_vel_disp, omega))

	#TODO