import numpy as np

from openmdao.api import ExplicitComponent

class BcCs(ExplicitComponent):

	def setup(self):
		self.add_input('B_contrl', val=np.zeros((4,2)))
		self.add_input('C_struct', val=np.zeros((2,7)))

		self.add_output('BcCs', val=np.zeros((4,7)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['BcCs'] = np.matmul(inputs['B_contrl'],inputs['C_struct'])

	def compute_partials(self, inputs, partials):
		partials['BcCs', 'B_contrl'] = np.kron(np.identity(2),np.transpose(inputs['C_struct']))
		partials['BcCs', 'C_struct'] = np.kron(inputs['B_contrl'],np.identity(7))