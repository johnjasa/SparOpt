import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemNormforce(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('M_moor', val=0., units='kg')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('tot_M_tower', val=0., units='kg')

		self.add_output('normforce_mode_elem', val=np.zeros(23), units='N')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		L_ball = inputs['L_ball']
		M_ball = inputs['M_ball']
		M_ball_elem = inputs['M_ball_elem']
		spar_draft = inputs['spar_draft'][0]
		z_ball = -spar_draft + L_ball[0]
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor'][0]
		z_sparnode = inputs['z_sparnode']
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		tot_M_tower = inputs['tot_M_tower']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(M_tower)

		for i in xrange(N_sparelem):
			for j in xrange(len(D_spar)):
				if z_sparnode[i] >= Z_spar[j] and z_sparnode[i] < Z_spar[j+1]:
					outputs['normforce_mode_elem'][i] = (np.sum(M_spar[:j]) + M_spar[j] / L_spar[j] * (z_sparnode[i] - Z_spar[j])) * 9.80665 - 1025. * 9.80665 * np.pi / 4. * (np.sum(D_spar[:j]**2. * L_spar[:j]) + D_spar[j]**2. * (z_sparnode[i] - Z_spar[j]))
					break

			if z_sparnode[i] >= z_ball:
				outputs['normforce_mode_elem'][i] += M_ball * 9.80665
			else:
				outputs['normforce_mode_elem'][i] += np.sum(M_ball_elem[:j]) * 9.80665

			if z_sparnode[i] >= z_moor:
				outputs['normforce_mode_elem'][i] += M_moor * 9.80665
		
		for i in xrange(N_towerelem):
			outputs['normforce_mode_elem'][N_sparelem+i] = (-M_nacelle - M_rotor - tot_M_tower + np.sum(M_tower[:i])) * 9.80665

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		Z_spar = inputs['Z_spar']
		M_spar = inputs['M_spar']
		L_ball = inputs['L_ball']
		M_ball = inputs['M_ball']
		M_ball_elem = inputs['M_ball_elem']
		spar_draft = inputs['spar_draft'][0]
		z_ball = -spar_draft + L_ball[0]
		M_moor = inputs['M_moor']
		z_moor = inputs['z_moor'][0]
		z_sparnode = inputs['z_sparnode']
		M_tower = inputs['M_tower']
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		tot_M_tower = inputs['tot_M_tower']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(M_tower)

		partials['normforce_mode_elem', 'D_spar'] = np.zeros((23,10))
		partials['normforce_mode_elem', 'L_spar'] = np.zeros((23,10))
		partials['normforce_mode_elem', 'Z_spar'] = np.zeros((23,11))
		partials['normforce_mode_elem', 'M_spar'] = np.zeros((23,10))
		partials['normforce_mode_elem', 'L_ball'] = np.zeros(23)
		partials['normforce_mode_elem', 'M_ball'] = np.zeros(23)
		partials['normforce_mode_elem', 'M_ball_elem'] = np.zeros((23,10))
		partials['normforce_mode_elem', 'spar_draft'] = np.zeros(23)
		partials['normforce_mode_elem', 'M_moor'] = np.zeros(23)
		partials['normforce_mode_elem', 'z_moor'] = np.zeros(23)
		partials['normforce_mode_elem', 'z_sparnode'] = np.zeros((23,14))

		for i in xrange(N_sparelem):
			for j in xrange(len(D_spar)):
				if z_sparnode[i] >= Z_spar[j] and z_sparnode[i] < Z_spar[j+1]:
					partials['normforce_mode_elem', 'D_spar'][i,j] += -1025. * 9.80665 * np.pi / 2. * D_spar[j] * (z_sparnode[i] - Z_spar[j])
					partials['normforce_mode_elem', 'L_spar'][i,j] += -M_spar[j] / L_spar[j]**2. * (z_sparnode[i] - Z_spar[j]) * 9.80665
					partials['normforce_mode_elem', 'Z_spar'][i,j] += -M_spar[j] / L_spar[j] * 9.80665 + 1025. * 9.80665 * np.pi / 4. * D_spar[j]**2.
					partials['normforce_mode_elem', 'M_spar'][i,j] += 9.80665 / L_spar[j] * (z_sparnode[i] - Z_spar[j])
					partials['normforce_mode_elem', 'z_sparnode'][i,i] += M_spar[j] / L_spar[j] * 9.80665 - 1025. * 9.80665 * np.pi / 4. * D_spar[j]**2.

					for k in xrange(j):
						partials['normforce_mode_elem', 'D_spar'][i,k] += -1025. * 9.80665 * np.pi / 2. * D_spar[k] * L_spar[k]
						partials['normforce_mode_elem', 'L_spar'][i,k] += -1025. * 9.80665 * np.pi / 4. * D_spar[k]**2.
						partials['normforce_mode_elem', 'M_spar'][i,k] += 9.80665

					break

			if z_sparnode[i] >= z_ball:
				partials['normforce_mode_elem', 'M_ball'][i] += 9.80665
			else:
				partials['normforce_mode_elem', 'M_ball_elem'][0,:j] += 9.80665

			if z_sparnode[i] >= z_moor:
				partials['normforce_mode_elem', 'M_moor'][i] += 9.80665
		
		partials['normforce_mode_elem', 'M_tower'] = np.zeros((23,10))
		partials['normforce_mode_elem', 'M_nacelle'] = np.zeros(23)
		partials['normforce_mode_elem', 'M_rotor'] = np.zeros(23)
		partials['normforce_mode_elem', 'tot_M_tower'] = np.zeros(23)

		for i in xrange(N_towerelem):
			partials['normforce_mode_elem', 'M_nacelle'][N_sparelem+i] = -9.80665
			partials['normforce_mode_elem', 'M_rotor'][N_sparelem+i] = -9.80665
			partials['normforce_mode_elem', 'tot_M_tower'][N_sparelem+i] = -9.80665

			for j in xrange(i):
				partials['normforce_mode_elem', 'M_tower'][N_sparelem+i,j] += 9.80665