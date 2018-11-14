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

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_vel_surge'] = np.sqrt(np.trapz(inputs['resp_vel_surge'], omega))
		outputs['stddev_vel_pitch'] = np.sqrt(np.trapz(inputs['resp_vel_pitch'], omega))
		outputs['stddev_vel_bend'] = np.sqrt(np.trapz(inputs['resp_vel_bend'], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_vel_surge', 'resp_vel_surge'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_vel_surge'], omega)) * domega
		partials['stddev_vel_surge', 'resp_vel_pitch'] = np.zeros((1,N_omega))
		partials['stddev_vel_surge', 'resp_vel_bend'] = np.zeros((1,N_omega))

		partials['stddev_vel_pitch', 'resp_vel_surge'] = np.zeros((1,N_omega))
		partials['stddev_vel_pitch', 'resp_vel_pitch'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_vel_pitch'], omega)) * domega
		partials['stddev_vel_pitch', 'resp_vel_bend'] = np.zeros((1,N_omega))

		partials['stddev_vel_bend', 'resp_vel_surge'] = np.zeros((1,N_omega))
		partials['stddev_vel_bend', 'resp_vel_pitch'] = np.zeros((1,N_omega))
		partials['stddev_vel_bend', 'resp_vel_bend'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_vel_bend'], omega)) * domega

		partials['stddev_vel_surge', 'resp_vel_surge'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_surge'], omega)) * domega / 2.
		partials['stddev_vel_pitch', 'resp_vel_pitch'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_pitch'], omega)) * domega / 2.
		partials['stddev_vel_bend', 'resp_vel_bend'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_bend'], omega)) * domega / 2.

		partials['stddev_vel_surge', 'resp_vel_surge'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_surge'], omega)) * domega / 2.
		partials['stddev_vel_pitch', 'resp_vel_pitch'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_pitch'], omega)) * domega / 2.
		partials['stddev_vel_bend', 'resp_vel_bend'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_vel_bend'], omega)) * domega / 2.