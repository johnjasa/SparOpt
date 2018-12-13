import numpy as np

from openmdao.api import ExplicitComponent

class AsFb(ExplicitComponent):

	def setup(self):
		self.add_input('A_struct', val=np.zeros((7,7)))
		self.add_input('BsDcCs', val=np.zeros((7,7)))

		self.add_output('A_s_fb', val=np.zeros((7,7)))

		self.declare_partials('A_s_fb', 'A_struct', rows=np.arange(7*7), cols=np.arange(7*7))
		self.declare_partials('A_s_fb', 'BsDcCs', rows=np.arange(7*7), cols=np.arange(7*7))

	def compute(self, inputs, outputs):
		outputs['A_s_fb'] = inputs['A_struct'] + inputs['BsDcCs']

	def compute_partials(self, inputs, partials):
		partials['A_s_fb', 'A_struct'] = np.ones(np.size(inputs['A_struct']))
		partials['A_s_fb', 'BsDcCs'] = np.ones(np.size(inputs['BsDcCs']))