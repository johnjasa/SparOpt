import numpy as np

from openmdao.api import ExplicitComponent

class Bstruct(ExplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')

		self.add_input('CoG_rotor', val=0., units='m')

		self.add_input('I_d', val=0., units='kg*m**2')

		self.add_input('dthrust_dbldpitch', val=0., units='N/rad')
		self.add_input('dtorque_dbldpitch', val=0., units='N*m/rad')

		self.add_output('B_struct', val=np.zeros((7,2)))

	def compute(self, inputs, outputs):
		M_global = inputs['M_global']
		A_global = inputs['A_global']

		CoG_rotor = inputs['CoG_rotor']

		I_d = inputs['I_d']

		dthrust_dbldpitch = inputs['dthrust_dbldpitch']
		dtorque_dbldpitch = inputs['dtorque_dbldpitch']

		B1 = np.zeros((3,2))
		B2 = np.matmul(np.linalg.inv(M_global + A_global), np.array([[0., dthrust_dbldpitch],[0., CoG_rotor * dthrust_dbldpitch],[0., dthrust_dbldpitch]]))
		B3 = np.array([[0., dtorque_dbldpitch / I_d]])

		outputs['B_struct'] = np.concatenate((B1,B2,B3),0)