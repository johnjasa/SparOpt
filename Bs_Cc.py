import numpy as np

from openmdao.api import ExplicitComponent

class BsCc(ExplicitComponent):

	def setup(self):
		self.add_input('B_struct', val=np.zeros((7,2)))
		self.add_input('C_contrl', val=np.zeros((2,2)))

		self.add_output('BsCc', val=np.zeros((7,2)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['BsCc'] = np.matmul(inputs['B_struct'],inputs['C_contrl'])

	def compute_partials(self, inputs, partials):
		partials['BsCc', 'B_struct'] = np.kron(np.identity(7),np.transpose(inputs['C_contrl']))
		partials['BsCc', 'C_contrl'] = np.kron(inputs['B_struct'],np.identity(2))