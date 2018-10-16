import numpy as np

from openmdao.api import ExplicitComponent

class Astruct(ExplicitComponent):

	def setup(self):
		self.add_input('M_global', val=np.zeros((3,3)), units='kg')
		self.add_input('A_global', val=np.zeros((3,3)), units='kg')
		self.add_input('B_global', val=np.zeros((3,3)), units='N*s/m')
		self.add_input('K_global', val=np.zeros((3,3)), units='N/m')

		self.add_input('CoG_rotor', val=0., units='m')

		self.add_input('I_d', val=0., units='kg*m**2')

		self.add_input('dtorque_dv', val=0., units='N*s')
		self.add_input('dthrust_drotspeed', 0., units='N*s/rad')
		self.add_input('dtorque_drotspeed', 0., units='N*m*s/rad')

		self.add_output('A_struct', val=np.zeros((7,7)))

	def compute(self, inputs, outputs):
		M_global = inputs['M_global']
		A_global = inputs['A_global']
		B_global = inputs['B_global']
		K_global = inputs['K_global']

		CoG_rotor = inputs['CoG_rotor'][0]

		I_d = inputs['I_d']

		dtorque_dv = inputs['dtorque_dv'][0]
		dthrust_drotspeed = inputs['dthrust_drotspeed'][0]
		dtorque_drotspeed = inputs['dtorque_drotspeed'][0]


		A1 = np.concatenate((np.zeros((3,3)),np.identity(3),np.zeros((3,1))),1)
		A_stiff = np.matmul(-np.linalg.inv(M_global + A_global), K_global)
		A_damp = np.matmul(-np.linalg.inv(M_global + A_global), B_global)
		A_wind = np.matmul(np.linalg.inv(M_global + A_global), np.array([[dthrust_drotspeed],[CoG_rotor * dthrust_drotspeed],[dthrust_drotspeed]]))
		A2 = np.concatenate((A_stiff,A_damp,A_wind),1)
		A3 = np.array([[0., 0., 0., -dtorque_dv / I_d, -CoG_rotor * dtorque_dv / I_d, -dtorque_dv / I_d, dtorque_drotspeed / I_d]])

		outputs['A_struct'] = np.concatenate((A1,A2,A3),0)