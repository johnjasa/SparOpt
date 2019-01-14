import numpy as np

from openmdao.api import ExplicitComponent

class Bfeedbk(ExplicitComponent):

	def setup(self):
		self.add_input('Bfb_ext', val=np.zeros((3,6)))
		self.add_input('I_d', val=0., units='kg*m**2')
		self.add_input('dtorque_dv', val=0., units='N*s')

		self.add_output('B_feedbk', val=np.zeros((11,6)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		I_d = inputs['I_d']
		dtorque_dv = inputs['dtorque_dv'][0]

		B1 = np.zeros((3,6))
		B2 = inputs['Bfb_ext']
		B3 = np.array([[0., 0., dtorque_dv / I_d, 0., 0., 0.]])
		B4 = np.zeros((4,6))

		outputs['B_feedbk'] = np.concatenate((B1,B2,B3,B4),0)

	def compute_partials(self, inputs, partials):
		I_d = inputs['I_d']
		dtorque_dv = inputs['dtorque_dv'][0]

		partials['B_feedbk', 'Bfb_ext'] = np.concatenate((np.zeros((18,18)),np.identity(18),np.zeros((30,18))),0)
		partials['B_feedbk', 'I_d'] = np.concatenate((np.zeros((36,1)),np.array([[0.], [0.], -dtorque_dv / I_d**2., [0.], [0.], [0.]]),np.zeros((24,1))),0)
		partials['B_feedbk', 'dtorque_dv'] = np.concatenate((np.zeros((36,1)),np.array([[0.], [0.], 1. / I_d, [0.], [0.], [0.]]),np.zeros((24,1))),0)