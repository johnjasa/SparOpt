import numpy as np

from openmdao.api import ExplicitComponent

class SteadyRotSpeed(ExplicitComponent):

	def setup(self):
		self.add_input('windspeed_0', val=0., units='m/s')

		self.add_output('rotspeed_0', val=1., units='rad/s')

	def compute(self, inputs, outputs):
		windspeed_0 = inputs['windspeed_0']

		if windspeed_0 >= 11.:
			outputs['rotspeed_0'] = 9.6 * 2. * np.pi / 60.
		elif windspeed_0 <= 7.:
			outputs['rotspeed_0'] = 6.0 * 2. * np.pi / 60.
		else:
			data_windspeed = np.array([7.0, 8.0, 9.0, 10.0, 11.0])
			data_rotspeed = np.array([6.0, 6.43, 7.23, 8.03, 9.6]) * 2. * np.pi / 60.

			outputs['rotspeed_0'] = np.interp(windspeed_0, data_windspeed, data_rotspeed)
			