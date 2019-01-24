import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermFairleadCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_fairlead', val=0., units='1/s')
		self.add_input('moor_offset', val=0., units='m')
		self.add_input('stddev_fairlead', val=0., units='m')
		self.add_input('maxval_fairlead', val=0., units='m')

		self.add_output('short_term_fairlead_CDF', val=0.)

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_fairlead']
		mean_extreme = inputs['moor_offset']
		stddev_extreme = inputs['stddev_fairlead']
		value_extreme = inputs['maxval_fairlead']

		T = 3600. #seconds

		outputs['short_term_fairlead_CDF'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_fairlead']
		mean_extreme = inputs['moor_offset']
		stddev_extreme = inputs['stddev_fairlead']
		value_extreme = inputs['maxval_fairlead']

		T = 3600. #seconds

		partials['short_term_fairlead_CDF', 'v_z_fairlead'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)))
		partials['short_term_fairlead_CDF', 'moor_offset'] = -np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)
		partials['short_term_fairlead_CDF', 'stddev_fairlead'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * (-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * ((value_extreme - mean_extreme)**2. / stddev_extreme**3.)
		partials['short_term_fairlead_CDF', 'maxval_fairlead'] = np.exp(-v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.))) * v_z_extreme * T * np.exp(-(value_extreme - mean_extreme)**2. / (2. * stddev_extreme**2.)) * (value_extreme - mean_extreme) / (stddev_extreme**2.)