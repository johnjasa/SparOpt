import numpy as np

from openmdao.api import ExplicitComponent

class RingBuckling2(ExplicitComponent):

	def setup(self):
		self.add_input('h_stiff', val=np.zeros(10))
		self.add_input('b_stiff', val=np.zeros(10))
		self.add_input('r_hull', val=np.zeros(10))
		self.add_input('f_y', val=0.)

		self.add_output('ring_buckling_2', val=np.zeros(10))

		self.declare_partials('ring_buckling_2', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('ring_buckling_2', 'b_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('ring_buckling_2', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('ring_buckling_2', 'f_y', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		outputs['ring_buckling_2'] = (7. * inputs['h_stiff'] / np.sqrt(10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])) / inputs['b_stiff'] - 1. #less than 0 to satisfy constraint

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		partials['ring_buckling_2', 'h_stiff'] = (7. / np.sqrt(10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])) / inputs['b_stiff'] - 0.5 * (7. * inputs['h_stiff'] / (10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])**(3./2.) * E / (inputs['f_y'] * inputs['r_hull'])) / inputs['b_stiff']
		partials['ring_buckling_2', 'b_stiff'] = -(7. * inputs['h_stiff'] / np.sqrt(10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])) / inputs['b_stiff']**2.
		partials['ring_buckling_2', 'r_hull'] = 0.5 * (7. * inputs['h_stiff'] / (10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])**(3./2.) * E * inputs['h_stiff'] / (inputs['f_y'] * inputs['r_hull']**2.)) / inputs['b_stiff']
		partials['ring_buckling_2', 'f_y'] = 0.5 * (7. * inputs['h_stiff'] / (10. + E / inputs['f_y'] * inputs['h_stiff'] / inputs['r_hull'])**(3./2.) * E * inputs['h_stiff'] / (inputs['f_y']**2. * inputs['r_hull'])) / inputs['b_stiff']