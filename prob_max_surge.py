import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxSurge(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_surge', val=0., units='1/s')
		self.add_input('mean_surge', val=0., units='m')
		self.add_input('stddev_surge', val=0., units='m')

		self.add_output('prob_max_surge', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_surge']
		mean_extreme = inputs['mean_surge']
		stddev_extreme = inputs['stddev_surge']

		T = 3600. #seconds

		outputs['prob_max_surge'] = mean_extreme + stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_surge']
		mean_extreme = inputs['mean_surge']
		stddev_extreme = inputs['stddev_surge']

		T = 3600. #seconds

		partials['prob_max_surge', 'v_z_surge'] = stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme
		partials['prob_max_surge', 'mean_surge'] = 1.
		partials['prob_max_surge', 'stddev_surge'] = np.sqrt(2. * np.log(v_z_extreme * T))