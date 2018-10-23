import numpy as np

from openmdao.api import ExplicitComponent

class Bstruct(ExplicitComponent):

	def setup(self):
		self.add_input('Bstr_ext', val=np.zeros((3,2)))
		self.add_input('I_d', val=0., units='kg*m**2')
		self.add_input('dtorque_dbldpitch', val=0., units='N*m/rad')

		self.add_output('B_struct', val=np.zeros((7,2)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		I_d = inputs['I_d']
		dtorque_dbldpitch = inputs['dtorque_dbldpitch']

		B1 = np.zeros((3,2))
		B2 = inputs['Bstr_ext']
		B3 = np.array([[0., dtorque_dbldpitch / I_d]])

		outputs['B_struct'] = np.concatenate((B1,B2,B3),0)

	def compute_partials(self, inputs, partials):
		I_d = inputs['I_d']
		dtorque_dbldpitch = inputs['dtorque_dbldpitch']

		partials['B_struct', 'Bstr_ext'] = np.concatenate((np.zeros((6,6)),np.identity(6),np.zeros((2,6))),0)
		partials['B_struct', 'I_d'] = np.concatenate((np.zeros((12,1)),np.array([[0.], -dtorque_dbldpitch / I_d**2.])),0)
		partials['B_struct', 'dtorque_dbldpitch'] = np.concatenate((np.zeros((12,1)),np.array([[0.], 1. / I_d])),0)