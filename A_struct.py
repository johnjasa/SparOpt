import numpy as np

from openmdao.api import ExplicitComponent

class Astruct(ExplicitComponent):

	def setup(self):
		self.add_input('Astr_stiff', val=np.zeros((3,3)))
		self.add_input('Astr_damp', val=np.zeros((3,3)))
		self.add_input('Astr_ext', val=np.zeros((3,1)))

		self.add_input('CoG_rotor', val=0., units='m')

		self.add_input('I_d', val=0., units='kg*m**2')

		self.add_input('dtorque_dv', val=0., units='N*s')
		self.add_input('dtorque_drotspeed', 0., units='N*m*s/rad')

		self.add_output('A_struct', val=np.zeros((7,7)))

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		Astr_stiff = inputs['Astr_stiff']
		Astr_damp = inputs['Astr_damp']
		Astr_ext = inputs['Astr_ext']

		CoG_rotor = inputs['CoG_rotor'][0]

		I_d = inputs['I_d']

		dtorque_dv = inputs['dtorque_dv'][0]
		dtorque_drotspeed = inputs['dtorque_drotspeed'][0]

		A1 = np.concatenate((np.zeros((3,3)),np.identity(3),np.zeros((3,1))),1)
		A2 = np.concatenate((-Astr_stiff,-Astr_damp,Astr_ext),1)
		A3 = np.array([[0., 0., 0., -dtorque_dv / I_d, -CoG_rotor * dtorque_dv / I_d, -dtorque_dv / I_d, dtorque_drotspeed / I_d]])

		outputs['A_struct'] = np.concatenate((A1,A2,A3),0)

	def compute_partials(self, inputs, partials):
		Astr_stiff = inputs['Astr_stiff']
		Astr_damp = inputs['Astr_damp']
		Astr_ext = inputs['Astr_ext']

		CoG_rotor = inputs['CoG_rotor'][0]

		I_d = inputs['I_d']

		dtorque_dv = inputs['dtorque_dv'][0]
		dtorque_drotspeed = inputs['dtorque_drotspeed'][0]

		Astiff_arr1 = np.concatenate((np.zeros((21,3)),-np.identity(3),np.zeros((25,3))),0)
		Astiff_arr2 = np.concatenate((np.zeros((28,3)),-np.identity(3),np.zeros((18,3))),0)
		Astiff_arr3 = np.concatenate((np.zeros((35,3)),-np.identity(3),np.zeros((11,3))),0)

		Adamp_arr1 = np.concatenate((np.zeros((24,3)),-np.identity(3),np.zeros((22,3))),0)
		Adamp_arr2 = np.concatenate((np.zeros((31,3)),-np.identity(3),np.zeros((15,3))),0)
		Adamp_arr3 = np.concatenate((np.zeros((38,3)),-np.identity(3),np.zeros((8,3))),0)

		Aext_arr1 = np.concatenate((np.zeros((27,1)),np.identity(1),np.zeros((21,1))),0)
		Aext_arr2 = np.concatenate((np.zeros((34,1)),np.identity(1),np.zeros((14,1))),0)
		Aext_arr3 = np.concatenate((np.zeros((41,1)),np.identity(1),np.zeros((7,1))),0)

		partials['A_struct', 'Astr_stiff'] = np.concatenate((Astiff_arr1,Astiff_arr2,Astiff_arr3),1)
		partials['A_struct', 'Astr_damp'] = np.concatenate((Adamp_arr1,Adamp_arr2,Adamp_arr3),1)
		partials['A_struct', 'Astr_ext'] = np.concatenate((Aext_arr1,Aext_arr2,Aext_arr3),1)
		partials['A_struct', 'CoG_rotor'] = np.concatenate((np.zeros((46,1)),np.array([-dtorque_dv / I_d]),np.zeros((2,1))),0)
		partials['A_struct', 'I_d'] = np.concatenate((np.zeros((45,1)),np.array([dtorque_dv / I_d**2., CoG_rotor * dtorque_dv / I_d**2., dtorque_dv / I_d**2., -dtorque_drotspeed / I_d**2.])),0)
		partials['A_struct', 'dtorque_dv'] = np.concatenate((np.zeros((45,1)),np.array([-1. / I_d, -CoG_rotor / I_d, -1. / I_d, [0.]])),0)
		partials['A_struct', 'dtorque_drotspeed'] = np.concatenate((np.zeros((48,1)),np.array([1. / I_d])),0)