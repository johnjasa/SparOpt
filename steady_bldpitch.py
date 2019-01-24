import numpy as np

from openmdao.api import ExplicitComponent

class SteadyBladePitch(ExplicitComponent):

	def setup(self):
		self.add_input('windspeed_0', val=0., units='m/s')

		self.add_output('bldpitch_0', val=0., units='rad')

	def compute(self, inputs, outputs):
		windspeed_0 = inputs['windspeed_0']

		if windspeed_0 < 11.4:
			outputs['bldpitch_0'] = 0.
		elif windspeed_0 > 25.:
			outputs['bldpitch_0'] = 90. * np.pi / 180.
		else:
			data_windspeed = np.array([12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25.])
			data_bldpitch = np.array([6.09 , 8.33 , 10.10, 11.67, 13.09, 14.41, 15.66, 16.85, 17.99, 19.08, 20.14, 21.18, 22.19, 23.17]) * np.pi / 180.
		
			outputs['bldpitch_0'] = np.interp(windspeed_0, data_windspeed, data_bldpitch)