import numpy as np
from scipy.sparse import linalg
from scipy.linalg import det

from openmdao.api import ExplicitComponent

class ModeshapeCoords(ExplicitComponent):

	def setup(self):
		self.add_input('Z_spar', val=np.zeros(4), units='m')
		self.add_input('Z_tower', val=np.zeros(11), units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('wt_ball', val=0., units='m')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('eig_vector', val=np.zeros(34), units='m')

		self.add_output('x_sparnode', val=np.zeros(7), units='m')
		self.add_output('z_sparnode', val=np.zeros(7), units='m')
		self.add_output('x_towernode', val=np.zeros(11), units='m')
		self.add_output('z_towernode', val=np.zeros(11), units='m')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		Z_tower = inputs['Z_tower']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + inputs['wt_ball'][0] + L_ball[0] #top of ballast
		z_moor = inputs['z_moor'][0]
		z_SWL = 0. #SWL

		z_aux = np.array([z_ball, z_moor, z_SWL])

		z_sparnode = np.concatenate((Z_spar, z_aux),0)
		z_sparnode = np.unique(z_sparnode)
		z_sparnode = np.sort(z_sparnode)

		N_spar = len(Z_spar) - 1
		N_tower = len(Z_tower) - 1

		N_sparelem = len(z_sparnode) - 1
		N_elem = N_sparelem + N_tower

		x_sparnode = inputs['eig_vector'][0:(N_sparelem+1)*2:2]
		x_towernode = inputs['eig_vector'][(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		outputs['x_sparnode'] = x_sparnode / x_towernode[-1]
		outputs['x_towernode'] = x_towernode / x_towernode[-1]

		outputs['z_sparnode'] = z_sparnode
		outputs['z_towernode'] = inputs['Z_tower']