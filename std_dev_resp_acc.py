import numpy as np

from openmdao.api import ExplicitComponent

class StdDevRespAcc(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_acc_surge', val=np.zeros(N_omega), units='(m/s**2)**2*s/rad')
		self.add_input('resp_acc_pitch', val=np.zeros(N_omega), units='(rad/s**2)**2*s/rad')
		self.add_input('resp_acc_bend', val=np.zeros(N_omega), units='(m/s**2)**2*s/rad')

		self.add_input('stddev_acc_surge', val=0., units='m/s**2')
		self.add_input('stddev_acc_pitch', val=0., units='rad/s**2')
		self.add_input('stddev_acc_bend', val=0., units='m/s**2')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_acc_surge'] = np.sqrt(np.trapz(inputs['resp_acc_surge'], omega))
		outputs['stddev_acc_pitch'] = np.sqrt(np.trapz(inputs['resp_acc_pitch'], omega))
		outputs['stddev_acc_bend'] = np.sqrt(np.trapz(inputs['resp_acc_bend'], omega))