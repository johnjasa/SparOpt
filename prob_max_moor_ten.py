import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxMoorTen(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_moor_ten', val=0., units='1/s')
		self.add_input('mean_moor_ten', val=0., units='N')
		self.add_input('stddev_moor_ten', val=0., units='N')
		self.add_input('gamma_F_moor_mean', val=0.)
		self.add_input('gamma_F_moor_dyn', val=0.)

		self.add_output('prob_max_moor_ten', val=0., units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_moor_ten']
		mean_extreme = inputs['mean_moor_ten']
		stddev_extreme = inputs['stddev_moor_ten']
		gamma_F_moor_mean = inputs['gamma_F_moor_mean']
		gamma_F_moor_dyn = inputs['gamma_F_moor_dyn']

		T = 3600. #seconds

		outputs['prob_max_moor_ten'] = mean_extreme * gamma_F_moor_mean + stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T)) * gamma_F_moor_dyn
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_moor_ten']
		mean_extreme = inputs['mean_moor_ten']
		stddev_extreme = inputs['stddev_moor_ten']
		gamma_F_moor_mean = inputs['gamma_F_moor_mean']
		gamma_F_moor_dyn = inputs['gamma_F_moor_dyn']

		T = 3600. #seconds

		partials['prob_max_moor_ten', 'v_z_moor_ten'] = stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme * gamma_F_moor_dyn
		partials['prob_max_moor_ten', 'mean_moor_ten'] = gamma_F_moor_mean
		partials['prob_max_moor_ten', 'stddev_moor_ten'] = np.sqrt(2. * np.log(v_z_extreme * T)) * gamma_F_moor_dyn
		partials['prob_max_moor_ten', 'gamma_F_moor_mean'] = mean_extreme
		partials['prob_max_moor_ten', 'gamma_F_moor_dyn'] = stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))