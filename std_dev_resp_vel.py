import numpy as np

from openmdao.api import ExplicitComponent

class StdDevRespVel(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_vel_surge', val=np.zeros(N_omega), units='(m/s)**2*s/rad')
		self.add_input('resp_vel_pitch', val=np.zeros(N_omega), units='(rad/s)**2*s/rad')
		self.add_input('resp_vel_bend', val=np.zeros(N_omega), units='(m/s)**2*s/rad')

		self.add_output('stddev_vel_surge', val=0., units='m/s')
		self.add_output('stddev_vel_pitch', val=0., units='rad/s')
		self.add_output('stddev_vel_bend', val=0., units='m/s')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_vel_surge'] = np.sqrt(np.trapz(inputs['resp_vel_surge'], omega))
		outputs['stddev_vel_pitch'] = np.sqrt(np.trapz(inputs['resp_vel_pitch'], omega))
		outputs['stddev_vel_bend'] = np.sqrt(np.trapz(inputs['resp_vel_bend'], omega))