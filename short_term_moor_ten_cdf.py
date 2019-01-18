import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermMoorTenCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_moor_ten', val=0., units='1/s')
		self.add_input('mean_moor_ten', val=0., units='N')
		self.add_input('stddev_moor_ten', val=0., units='N')
		self.add_input('maxval_moor_ten', val=0., units='N')

		self.add_output('short_term_moor_ten_CDF', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_moor_ten']
		mean_extreme = inputs['mean_moor_ten']
		stddev_extreme = inputs['stddev_moor_ten']
		value_extreme = inputs['maxval_moor_ten']

		T = 3600. #seconds

		outputs['short_term_moor_ten_CDF'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_moor_ten']
		mean_extreme = inputs['mean_moor_ten']
		stddev_extreme = inputs['stddev_moor_ten']
		value_extreme = inputs['maxval_moor_ten']

		T = 3600. #seconds

		partials['short_term_moor_ten_CDF', 'v_z_moor_ten'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
		partials['short_term_moor_ten_CDF', 'mean_moor_ten'] = -np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)
		partials['short_term_moor_ten_CDF', 'stddev_moor_ten'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * ((value_extreme - mean_extreme)**2. / stddev_extreme**3.)
		partials['short_term_moor_ten_CDF', 'maxval_moor_ten'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)