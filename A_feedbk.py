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

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		outputs['A_feedbk'] = np.concatenate((np.concatenate((inputs['A_struct'],np.matmul(inputs['B_struct'],inputs['C_contrl'])),1),np.concatenate((np.matmul(inputs['B_contrl'],inputs['C_struct']),inputs['A_contrl']),1)),0)

	def compute_partials(self, inputs, partials):
		partials['A_feedbk', 'A_struct'] = np.concatenate((np.concatenate((np.ones((7,7)),np.zeros((7,2))),1),np.zeros((2,9))),0)
		partials['A_feedbk', 'A_contrl'] = np.concatenate((np.zeros((7,9)),np.concatenate((np.zeros((7,7)),np.ones((2,2))),1)),0)
		partials['A_feedbk', 'B_struct'] = np.concatenate((np.concatenate((np.zeros((7,7)),np.matmul(np.ones((7,2)),inputs['C_contrl'])),1),np.zeros((2,9))),0)
		partials['A_feedbk', 'B_contrl'] = np.concatenate((np.zeros((7,9)),np.concatenate((np.matmul(np.ones((2,1)),inputs['C_struct']),np.zeros((2,2))),1)),0)
		partials['A_feedbk', 'C_struct'] = np.concatenate((np.zeros((7,9)),np.concatenate((np.matmul(inputs['B_contrl'],np.ones((1,7))),np.zeros((2,2))),1)),0)
		partials['A_feedbk', 'C_contrl'] = np.concatenate((np.concatenate((np.zeros((7,7)),np.matmul(inputs['B_struct'],np.ones((2,2)))),1),np.zeros((2,9))),0)