import numpy as np

from openmdao.api import ExplicitComponent

class HullRF(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')

		self.add_output('r_f', val=np.zeros(10), units='m')

		self.declare_partials('r_f', 'D_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_f', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_f', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_f', 't_f_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['r_f'] = 0.5 * inputs['D_spar'] - inputs['wt_spar'] - inputs['h_stiff'] - inputs['t_f_stiff']

	def compute_partials(self, inputs, partials):
		partials['r_f', 'D_spar'] = 0.5 * np.ones(10)
		partials['r_f', 'wt_spar'] = -1. * np.ones(10)
		partials['r_f', 'h_stiff'] = -1. * np.ones(10)
		partials['r_f', 't_f_stiff'] = -1. * np.ones(10)