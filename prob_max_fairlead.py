import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxFairlead(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_fairlead', val=0., units='1/s')
		self.add_input('moor_offset', val=0., units='m')
		self.add_input('stddev_fairlead', val=0., units='m')

		self.add_output('prob_max_fairlead', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_fairlead']
		mean_extreme = inputs['moor_offset']
		stddev_extreme = inputs['stddev_fairlead']

		T = 3600. #seconds

		outputs['prob_max_fairlead'] = mean_extreme + stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_fairlead']
		mean_extreme = inputs['moor_offset']
		stddev_extreme = inputs['stddev_fairlead']

		T = 3600. #seconds

		partials['prob_max_fairlead', 'v_z_fairlead'] = stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme
		partials['prob_max_fairlead', 'moor_offset'] = 1.
		partials['prob_max_fairlead', 'stddev_fairlead'] = np.sqrt(2. * np.log(v_z_extreme * T))