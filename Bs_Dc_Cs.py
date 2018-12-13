import numpy as np

from openmdao.api import ExplicitComponent

class BsDcCs(ExplicitComponent):

	def setup(self):
		self.add_input('B_struct', val=np.zeros((7,2)))
		self.add_input('D_contrl', val=np.zeros((2,2)))
		self.add_input('C_struct', val=np.zeros((2,7)))

		self.add_output('BsDcCs', val=np.zeros((7,7)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['BsDcCs'] = np.linalg.multi_dot([inputs['B_struct'], inputs['D_contrl'], inputs['C_struct']])

	def compute_partials(self, inputs, partials):
		partials['BsDcCs', 'B_struct'] = np.kron(np.identity(7),np.matmul(inputs['D_contrl'], inputs['C_struct']).T)
		partials['BsDcCs', 'D_contrl'] = np.kron(inputs['B_struct'], inputs['C_struct'].T)
		partials['BsDcCs', 'C_struct'] = np.kron(np.matmul(inputs['B_struct'], inputs['D_contrl']),np.identity(7))