import numpy as np

from openmdao.api import ExplicitComponent

class Afeedbk(ExplicitComponent):

	def setup(self):
		self.add_input('A_struct', val=np.zeros((7,7)))
		self.add_input('A_contrl', val=np.zeros((2,2)))
		self.add_input('B_struct', val=np.zeros((7,2)))
		self.add_input('B_contrl', val=np.zeros((2,1)))
		self.add_input('C_struct', val=np.zeros((1,7)))
		self.add_input('C_contrl', val=np.zeros((2,2)))

		self.add_output('A_feedbk', val=np.zeros((9,9)))

	def compute(self, inputs, outputs):
		outputs['A_feedbk'] = np.concatenate((np.concatenate((inputs['A_struct'],np.matmul(inputs['B_struct'],inputs['C_contrl'])),1),np.concatenate((np.matmul(inputs['B_contrl'],inputs['C_struct']),inputs['A_contrl']),1)),0)