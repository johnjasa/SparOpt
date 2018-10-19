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

		self.add_output('x_sparmode', val=np.zeros(7), units='m')
		self.add_output('z_sparmode', val=np.zeros(7), units='m')
		self.add_output('x_towermode', val=np.zeros(11), units='m')
		self.add_output('z_towermode', val=np.zeros(11), units='m')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		Z_tower = inputs['Z_tower']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + inputs['wt_ball'][0] + L_ball[0] #top of ballast
		z_moor = inputs['z_moor'][0]
		z_SWL = 0. #SWL

		z_aux = np.array([z_ball, z_moor, z_SWL])

		z_sparmode = np.concatenate((Z_spar, z_aux),0)
		z_sparmode = np.unique(z_sparmode)
		z_sparmode = np.sort(z_sparmode)

		N_spar = len(Z_spar) - 1
		N_tower = len(Z_tower) - 1

		N_sparelem = len(z_sparmode) - 1
		N_elem = N_sparelem + N_tower

		x_sparmode = inputs['eig_vector'][0:(N_sparelem+1)*2:2]
		x_towermode = inputs['eig_vector'][(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		outputs['x_sparmode'] = x_sparmode / x_towermode[-1]
		outputs['x_towermode'] = x_towermode / x_towermode[-1]

		outputs['z_sparmode'] = z_sparmode
		outputs['z_towermode'] = inputs['Z_tower']