import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermPitchCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_pitch', val=0., units='1/s')
		self.add_input('mean_pitch', val=0., units='rad')
		self.add_input('stddev_pitch', val=0., units='rad')
		self.add_input('maxval_pitch', val=0., units='rad')

		self.add_output('short_term_pitch_CDF', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_pitch']
		mean_extreme = inputs['mean_pitch']
		stddev_extreme = inputs['stddev_pitch']
		value_extreme = inputs['maxval_pitch']

		T = 3600. #seconds

		outputs['short_term_pitch_CDF'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_pitch']
		mean_extreme = inputs['mean_pitch']
		stddev_extreme = inputs['stddev_pitch']
		value_extreme = inputs['maxval_pitch']

		T = 3600. #seconds

		partials['short_term_pitch_CDF', 'v_z_pitch'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
		partials['short_term_pitch_CDF', 'mean_pitch'] = -np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)
		partials['short_term_pitch_CDF', 'stddev_pitch'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * ((value_extreme - mean_extreme)**2. / stddev_extreme**3.)
		partials['short_term_pitch_CDF', 'maxval_pitch'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)