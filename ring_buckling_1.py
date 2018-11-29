import numpy as np

from openmdao.api import ExplicitComponent

class RingBuckling1(ExplicitComponent):

	def setup(self):
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('t_w_stiff', val=np.zeros(10), units='m')
		self.add_input('f_y', val=0., units='MPa')

		self.add_output('ring_buckling_1', val=np.zeros(10))

		self.declare_partials('ring_buckling_1', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('ring_buckling_1', 't_w_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('ring_buckling_1', 'f_y', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		outputs['ring_buckling_1'] = inputs['h_stiff'] / (1.35 * inputs['t_w_stiff'] * np.sqrt(E / inputs['f_y'])) - 1.0 #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		partials['ring_buckling_1', 'h_stiff'] = 1. / (1.35 * inputs['t_w_stiff'] * np.sqrt(E / inputs['f_y']))
		partials['ring_buckling_1', 't_w_stiff'] = -inputs['h_stiff'] / (1.35 * inputs['t_w_stiff']**2. * np.sqrt(E / inputs['f_y']))
		partials['ring_buckling_1', 'f_y'] = 0.5 * inputs['h_stiff'] / (1.35 * inputs['t_w_stiff'] * (E / inputs['f_y'])**(3./2.)) * E / inputs['f_y']**2.
		