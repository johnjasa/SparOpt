import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermTowerStressCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_tower_stress', val=np.zeros(10), units='1/s')
		self.add_input('mean_tower_stress', val=np.zeros(10), units='MPa')
		self.add_input('stddev_tower_stress', val=np.zeros(10), units='MPa')
		self.add_input('maxval_tower_stress', val=np.zeros(10), units='MPa')

		self.add_output('short_term_tower_stress_CDF', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_tower_stress']
		mean_extreme = inputs['mean_tower_stress']
		stddev_extreme = inputs['stddev_tower_stress']
		value_extreme = inputs['maxval_tower_stress']

		T = 3600. #seconds

		for i in xrange(10):
			outputs['short_term_tower_stress_CDF'][i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)))
	
	def compute_partials(self, inputs, partials): #TODO check
		v_z_extreme = inputs['v_z_tower_stress']
		mean_extreme = inputs['mean_tower_stress']
		stddev_extreme = inputs['stddev_tower_stress']
		value_extreme = inputs['maxval_tower_stress']

		T = 3600. #seconds

		for i in xrange(10):
			partials['short_term_tower_stress_CDF', 'v_z_tower_stress'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * (-T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)))
			partials['short_term_tower_stress_CDF', 'mean_tower_stress'][i,i] = -np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)) * (value_extreme[i] - mean_extreme[i]) / (stddev_extreme[i]**2.)
			partials['short_term_tower_stress_CDF', 'stddev_tower_stress'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * (-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * ((value_extreme[i] - mean_extreme[i])**2. / stddev_extreme[i]**3.)
			partials['short_term_tower_stress_CDF', 'maxval_tower_stress'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)) * (value_extreme[i] - mean_extreme[i]) / (stddev_extreme[i]**2.)