import numpy as np
from scipy import linalg
import scipy.sparse as ss

from openmdao.api import ExplicitComponent

class Modeshape(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(3), units='m')
		self.add_input('L_spar', val=np.zeros(3), units='m')
		self.add_input('wt_spar', val=np.zeros(3), units='m')
		self.add_input('M_spar', val=np.zeros(3), units='kg')
		self.add_input('Z_spar', val=np.zeros(4), units='m')
		self.add_input('CoG_spar', val=0., units='m')
		self.add_input('D_tower', val=np.zeros(10), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('wt_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('CoG_ball', val=0., units='m')
		self.add_input('wt_ball', val=0., units='m')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('buoy_spar', val=0., units='N')
		self.add_input('CoB', val=0., units='m')

		self.add_output('x_sparnode', val=np.zeros(7), units='m')
		self.add_output('x_towernode', val=np.zeros(11), units='m')
		self.add_output('z_sparnode', val=np.zeros(7), units='m')
		self.add_output('z_towernode', val=np.zeros(11), units='m')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		wt_spar = inputs['wt_spar']
		M_spar = inputs['M_spar']
		Z_spar = inputs['Z_spar']
		D_tower = inputs['D_tower']
		L_tower = inputs['L_tower']
		wt_tower = inputs['wt_tower']
		M_tower = inputs['M_tower']
		Z_tower = inputs['Z_tower']
		M_ball = inputs['M_ball']
		CoG_ball = inputs['CoG_ball']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + inputs['wt_ball'][0] + L_ball[0] #top of ballast
		z_moor = inputs['z_moor'][0]
		z_SWL = 0. #SWL
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		I_rotor = inputs['I_rotor']
		K_moor = inputs['K_moor']
		K_hydrostatic = inputs['buoy_spar'] * inputs['CoB'] - np.sum(M_spar) * 9.80665 * inputs['CoG_spar'] - inputs['M_ball'] * 9.80665 * inputs['CoG_ball'] - inputs['M_moor'] * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_spar[-1]**4.

		z_aux = np.array([z_ball, z_moor, z_SWL])

		z_sparnode = np.concatenate((Z_spar, z_aux),0)
		z_sparnode = np.unique(z_sparnode)
		z_sparnode = np.sort(z_sparnode)

		outputs['x_sparnode'] = [0.1671773, 0.01717872, -0.02452651, -0.31540944, -0.35093444, -0.36868245, -0.41301261]
		outputs['x_towernode'] = [-0.41301261, -0.44246513, -0.43706879, -0.39464275, -0.31330533, -0.19168688, -0.02922765, 0.1733981, 0.41366632, 0.6862874, 1.]

		outputs['z_sparnode'] = z_sparnode
		outputs['z_towernode'] = Z_tower