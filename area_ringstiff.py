import numpy as np

from openmdao.api import ExplicitComponent

class AreaRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('t_w_stiff', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')
		self.add_input('b_stiff', val=np.zeros(10), units='m')

		self.add_output('A_R', val=np.zeros(10), units='m')

		self.declare_partials('A_R', 't_w_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('A_R', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('A_R', 't_f_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('A_R', 'b_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		outputs['A_R'] = inputs['t_w_stiff'] * inputs['h_stiff'] + inputs['t_f_stiff'] * inputs['b_stiff']

	def compute_partials(self, inputs, partials):
		partials['A_R', 't_w_stiff'] = inputs['h_stiff']
		partials['A_R', 'h_stiff'] = inputs['t_w_stiff']
		partials['A_R', 't_f_stiff'] = inputs['b_stiff']
		partials['A_R', 'b_stiff'] = inputs['t_f_stiff']