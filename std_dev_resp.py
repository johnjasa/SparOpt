import numpy as np

from openmdao.api import ExplicitComponent

class StdDevResp(ExplicitComponent):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']
		self.omega = freqs['omega']
		N_omega = len(self.omega)

		self.add_input('resp_surge', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('resp_pitch', val=np.zeros(N_omega), units='rad**2*s/rad')
		self.add_input('resp_bend', val=np.zeros(N_omega), units='m**2*s/rad')
		self.add_input('resp_rotspeed', val=np.zeros(N_omega), units='rad**2*s/(rad*s)')
		self.add_input('resp_rot_lp', val=np.zeros(N_omega), units='rad**2*s/rad')
		self.add_input('resp_rotspeed_lp', val=np.zeros(N_omega), units='rad**2*s/(rad*s)')
		self.add_input('resp_bldpitch', val=np.zeros(N_omega), units='rad**2*s/rad')

		self.add_output('stddev_surge', val=0., units='m')
		self.add_output('stddev_pitch', val=0., units='rad')
		self.add_output('stddev_bend', val=0., units='m')
		self.add_output('stddev_rotspeed', val=0., units='rad/s')
		self.add_output('stddev_rot_lp', val=0., units='rad')
		self.add_output('stddev_rotspeed_lp', val=0., units='rad/s')
		self.add_output('stddev_bldpitch', val=0., units='rad')

		#self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_surge'] = np.sqrt(np.trapz(inputs['resp_surge'], omega))
		outputs['stddev_pitch'] = np.sqrt(np.trapz(inputs['resp_pitch'], omega))
		outputs['stddev_bend'] = np.sqrt(np.trapz(inputs['resp_bend'], omega))
		outputs['stddev_rotspeed'] = np.sqrt(np.trapz(inputs['resp_rotspeed'], omega))
		outputs['stddev_rot_lp'] = np.sqrt(np.trapz(inputs['resp_rot_lp'], omega))
		outputs['stddev_rotspeed_lp'] = np.sqrt(np.trapz(inputs['resp_rotspeed_lp'], omega))
		outputs['stddev_bldpitch'] = np.sqrt(np.trapz(inputs['resp_bldpitch'], omega))