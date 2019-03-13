import numpy as np

from openmdao.api import ExplicitComponent

class ProbMaxHullMoment(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_hull_moment', val=np.zeros(10), units='1/s')
		self.add_input('mean_hull_moment', val=np.zeros(10), units='N*m')
		self.add_input('stddev_hull_moment', val=np.zeros(10), units='N*m')

		self.add_output('My_hull', val=np.zeros(10), units='N*m')

		self.declare_partials('My_hull', 'v_z_hull_moment', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('My_hull', 'mean_hull_moment', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('My_hull', 'stddev_hull_moment', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_hull_moment']
		mean_extreme = inputs['mean_hull_moment']
		stddev_extreme = inputs['stddev_hull_moment']

		T = 3600. #seconds

		outputs['My_hull'] = mean_extreme - stddev_extreme * np.sqrt(2. * np.log(v_z_extreme * T))
	
	def compute_partials(self, inputs, partials):
		v_z_extreme = inputs['v_z_hull_moment']
		mean_extreme = inputs['mean_hull_moment']
		stddev_extreme = inputs['stddev_hull_moment']

		T = 3600. #seconds

		partials['My_hull', 'v_z_hull_moment'] = -stddev_extreme * 0.5 / np.sqrt(2. * np.log(v_z_extreme * T)) * 2. / v_z_extreme
		partials['My_hull', 'mean_hull_moment'] = np.ones(10)
		partials['My_hull', 'stddev_hull_moment'] = -np.sqrt(2. * np.log(v_z_extreme * T))