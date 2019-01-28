import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxTowerStress(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_tower_stress', val=np.zeros(10), units='1/s')
		self.add_input('mean_tower_stress', val=np.zeros(10), units='MPa')
		self.add_input('stddev_tower_stress', val=np.zeros(10), units='MPa')

		self.add_output('prob_max_tower_stress', val=np.zeros(10), units='MPa')

		self.declare_partials('prob_max_tower_stress', 'v_z_tower_stress', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('prob_max_tower_stress', 'mean_tower_stress', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('prob_max_tower_stress', 'stddev_tower_stress', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_tower_stress']
		mean_extreme = inputs['mean_tower_stress']
		stddev_extreme = inputs['stddev_tower_stress']

		T = 3600. #seconds

		outputs['prob_max_tower_stress'] = mean_extreme - stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_tower_stress']
		mean_extreme = inputs['mean_tower_stress']
		stddev_extreme = inputs['stddev_tower_stress']

		T = 3600. #seconds

		partials['prob_max_tower_stress', 'v_z_tower_stress'] = -stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme
		partials['prob_max_tower_stress', 'mean_tower_stress'] = np.ones(10)
		partials['prob_max_tower_stress', 'stddev_tower_stress'] = -np.sqrt(2. * np.log(v_z_extreme * T))