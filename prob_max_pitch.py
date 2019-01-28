import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxPitch(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_pitch', val=0., units='1/s')
		self.add_input('mean_pitch', val=0., units='rad')
		self.add_input('stddev_pitch', val=0., units='rad')

		self.add_output('prob_max_pitch', val=0., units='rad')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_pitch']
		mean_extreme = inputs['mean_pitch']
		stddev_extreme = inputs['stddev_pitch']

		T = 3600. #seconds

		outputs['prob_max_pitch'] = mean_extreme + stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_pitch']
		mean_extreme = inputs['mean_pitch']
		stddev_extreme = inputs['stddev_pitch']

		T = 3600. #seconds

		partials['prob_max_pitch', 'v_z_pitch'] = stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme
		partials['prob_max_pitch', 'mean_pitch'] = 1.
		partials['prob_max_pitch', 'stddev_pitch'] = np.sqrt(2. * np.log(v_z_extreme * T))