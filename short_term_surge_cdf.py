import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermSurgeCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_surge', val=0., units='1/s')
		self.add_input('mean_surge', val=0., units='m')
		self.add_input('stddev_surge', val=0., units='m')
		self.add_input('max_value_surge', val=0., units='m')

		self.add_output('short_term_surge_CDF', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_surge']
		mean_extreme = inputs['mean_surge']
		stddev_extreme = inputs['stddev_surge']
		value_extreme = inputs['max_value_surge']

		T = 3600. #seconds

		outputs['short_term_surge_CDF'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_surge']
		mean_extreme = inputs['mean_surge']
		stddev_extreme = inputs['stddev_surge']
		value_extreme = inputs['max_value_surge']

		T = 3600. #seconds

		partials['short_term_surge_CDF', 'v_z_surge'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
		partials['short_term_surge_CDF', 'mean_surge'] = -np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)
		partials['short_term_surge_CDF', 'stddev_surge'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * ((value_extreme - mean_extreme)**2. / stddev_extreme**3.)
		partials['short_term_surge_CDF', 'max_value_surge'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)