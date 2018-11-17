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
		self.add_input('resp_bldpitch', val=np.zeros(N_omega), units='rad**2*s/rad')

		self.add_output('stddev_surge', val=0., units='m')
		self.add_output('stddev_pitch', val=0., units='rad')
		self.add_output('stddev_bend', val=0., units='m')
		self.add_output('stddev_rotspeed', val=0., units='rad/s')
		self.add_output('stddev_bldpitch', val=0., units='rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		omega = self.omega

		outputs['stddev_surge'] = np.sqrt(np.trapz(inputs['resp_surge'], omega))
		outputs['stddev_pitch'] = np.sqrt(np.trapz(inputs['resp_pitch'], omega))
		outputs['stddev_bend'] = np.sqrt(np.trapz(inputs['resp_bend'], omega))
		outputs['stddev_rotspeed'] = np.sqrt(np.trapz(inputs['resp_rotspeed'], omega))
		outputs['stddev_bldpitch'] = np.sqrt(np.trapz(inputs['resp_bldpitch'], omega))

	def compute_partials(self, inputs, partials):
		omega = self.omega
		N_omega = len(omega)
		domega = omega[1] - omega[0]

		partials['stddev_surge', 'resp_surge'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_surge'], omega)) * domega
		partials['stddev_surge', 'resp_pitch'] = np.zeros((1,N_omega))
		partials['stddev_surge', 'resp_bend'] = np.zeros((1,N_omega))
		partials['stddev_surge', 'resp_rotspeed'] = np.zeros((1,N_omega))
		partials['stddev_surge', 'resp_bldpitch'] = np.zeros((1,N_omega))

		partials['stddev_pitch', 'resp_surge'] = np.zeros((1,N_omega))
		partials['stddev_pitch', 'resp_pitch'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_pitch'], omega)) * domega
		partials['stddev_pitch', 'resp_bend'] = np.zeros((1,N_omega))
		partials['stddev_pitch', 'resp_rotspeed'] = np.zeros((1,N_omega))
		partials['stddev_pitch', 'resp_bldpitch'] = np.zeros((1,N_omega))

		partials['stddev_bend', 'resp_surge'] = np.zeros((1,N_omega))
		partials['stddev_bend', 'resp_pitch'] = np.zeros((1,N_omega))
		partials['stddev_bend', 'resp_bend'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_bend'], omega)) * domega
		partials['stddev_bend', 'resp_rotspeed'] = np.zeros((1,N_omega))
		partials['stddev_bend', 'resp_bldpitch'] = np.zeros((1,N_omega))

		partials['stddev_rotspeed', 'resp_surge'] = np.zeros((1,N_omega))
		partials['stddev_rotspeed', 'resp_pitch'] = np.zeros((1,N_omega))
		partials['stddev_rotspeed', 'resp_bend'] = np.zeros((1,N_omega))
		partials['stddev_rotspeed', 'resp_rotspeed'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_rotspeed'], omega)) * domega
		partials['stddev_rotspeed', 'resp_bldpitch'] = np.zeros((1,N_omega))

		partials['stddev_bldpitch', 'resp_surge'] = np.zeros((1,N_omega))
		partials['stddev_bldpitch', 'resp_pitch'] = np.zeros((1,N_omega))
		partials['stddev_bldpitch', 'resp_bend'] = np.zeros((1,N_omega))
		partials['stddev_bldpitch', 'resp_rotspeed'] = np.zeros((1,N_omega))
		partials['stddev_bldpitch', 'resp_bldpitch'] = np.ones((1,N_omega)) * 0.5 / np.sqrt(np.trapz(inputs['resp_bldpitch'], omega)) * domega

		partials['stddev_surge', 'resp_surge'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_surge'], omega)) * domega / 2.
		partials['stddev_pitch', 'resp_pitch'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_pitch'], omega)) * domega / 2.
		partials['stddev_bend', 'resp_bend'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_bend'], omega)) * domega / 2.
		partials['stddev_rotspeed', 'resp_rotspeed'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_rotspeed'], omega)) * domega / 2.
		partials['stddev_bldpitch', 'resp_bldpitch'][0,0] += -0.5 / np.sqrt(np.trapz(inputs['resp_bldpitch'], omega)) * domega / 2.

		partials['stddev_surge', 'resp_surge'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_surge'], omega)) * domega / 2.
		partials['stddev_pitch', 'resp_pitch'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_pitch'], omega)) * domega / 2.
		partials['stddev_bend', 'resp_bend'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_bend'], omega)) * domega / 2.
		partials['stddev_rotspeed', 'resp_rotspeed'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_rotspeed'], omega)) * domega / 2.
		partials['stddev_bldpitch', 'resp_bldpitch'][0,-1] += -0.5 / np.sqrt(np.trapz(inputs['resp_bldpitch'], omega)) * domega / 2.