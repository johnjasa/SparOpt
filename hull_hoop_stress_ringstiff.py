import numpy as np

from openmdao.api import ExplicitComponent

class HullHoopStressRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('net_pressure', val=np.zeros(10), units='MPa')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('alpha', val=np.zeros(10))
		self.add_input('r_r', val=np.zeros(10), units='m')
		self.add_input('sigma_a', val=np.zeros(10), units='MPa')
		self.add_input('sigma_m', val=np.zeros(10), units='MPa')

		self.add_output('sigma_h', val=np.zeros(10))

		self.declare_partials('sigma_hR', 'net_pressure', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'alpha', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'r_r', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'sigma_a', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('sigma_hR', 'sigma_m', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		nu = 0.3

		outputs['sigma_hR'] = (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m'])) * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']

	def compute_partials(self, inputs, partials):
		partials['sigma_hR', 'net_pressure'] = inputs['r_hull'] / inputs['wt_spar'] * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']
		partials['sigma_hR', 'r_hull'] = inputs['net_pressure'] / inputs['wt_spar'] * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r'] + (inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m'])) * 1. / (1. + inputs['alpha']) * 1. / inputs['r_r']
		partials['sigma_hR', 'wt_spar'] = -inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar']**2. * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']
		partials['sigma_hR', 'alpha'] = -(inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m'])) * 1. / (1. + inputs['alpha'])**2. * inputs['r_hull'] / inputs['r_r']
		partials['sigma_hR', 'r_r'] = -(inputs['net_pressure'] * inputs['r_hull'] / inputs['wt_spar'] - nu * (inputs['sigma_a'] + inputs['sigma_m'])) * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']**2.
		partials['sigma_hR', 'sigma_a'] = -nu * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']
		partials['sigma_hR', 'sigma_m'] = -nu * 1. / (1. + inputs['alpha']) * inputs['r_hull'] / inputs['r_r']