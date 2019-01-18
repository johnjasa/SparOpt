import numpy as np

from openmdao.api import ExplicitComponent

class ShortTermMyMomInertiaCDF(ExplicitComponent):

	def setup(self):
		self.add_input('v_z_hull_moment', val=np.zeros(10), units='1/s')
		self.add_input('mean_hull_moment', val=np.zeros(10), units='N*m')
		self.add_input('stddev_hull_moment', val=np.zeros(10), units='N*m')
		self.add_input('maxval_My_mom_inertia', val=np.zeros(10), units='N*m')

		self.add_output('short_term_My_mom_inertia_CDF', val=np.zeros(10))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		v_z_extreme = inputs['v_z_hull_moment']
		mean_extreme = inputs['mean_hull_moment']
		stddev_extreme = inputs['stddev_hull_moment']
		value_extreme = inputs['maxval_My_mom_inertia']

		T = 3600. #seconds

		for i in xrange(10):
			outputs['short_term_My_mom_inertia_CDF'][i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)))
	
	def compute_partials(self, inputs, partials): #TODO check
		v_z_extreme = inputs['v_z_hull_moment']
		mean_extreme = inputs['mean_hull_moment']
		stddev_extreme = inputs['stddev_hull_moment']
		value_extreme = inputs['maxval_My_mom_inertia']

		T = 3600. #seconds

		for i in xrange(10):
			partials['short_term_My_mom_inertia_CDF', 'v_z_hull_moment'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * (-T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)))
			partials['short_term_My_mom_inertia_CDF', 'mean_hull_moment'][i,i] = -np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)) * (value_extreme[i] - mean_extreme[i]) / (stddev_extreme[i]**2.)
			partials['short_term_My_mom_inertia_CDF', 'stddev_hull_moment'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * (-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * ((value_extreme[i] - mean_extreme[i])**2. / stddev_extreme[i]**3.)
			partials['short_term_My_mom_inertia_CDF', 'maxval_My_mom_inertia'][i,i] = np.exp(-v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.))) * v_z_extreme[i] * T * np.exp(-(value_extreme[i] - mean_extreme[i])**2. / (2. * stddev_extreme[i]**2.)) * (value_extreme[i] - mean_extreme[i]) / (stddev_extreme[i]**2.)