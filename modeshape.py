import numpy as np

from openmdao.api import ExplicitComponent

class Modeshape(ExplicitComponent):

	def setup(self):
		self.add_input('D_secs', val=np.zeros(3), units='m')
		self.add_input('L_secs', val=np.zeros(3), units='m')
		self.add_input('wt_secs', val=np.zeros(3), units='m')
		self.add_input('M_secs', val=np.zeros(3), units='kg')
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

		self.add_output('x_sparmode', val=np.zeros(7), units='m')
		self.add_output('x_towermode', val=np.zeros(11), units='m')
		self.add_output('z_sparmode', val=np.zeros(7), units='m')
		self.add_output('z_towermode', val=np.zeros(11), units='m')

	def compute(self, inputs, outputs):
		D_secs = inputs['D_secs']
		L_secs = inputs['L_secs']
		wt_secs = inputs['wt_secs']
		M_secs = inputs['M_secs']
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
		K_hydrostatic = inputs['buoy_spar'] * inputs['CoB'] - np.sum(M_secs) * 9.80665 * inputs['CoG_spar'] - inputs['M_ball'] * 9.80665 * inputs['CoG_ball'] - inputs['M_moor'] * 9.80665 * z_moor + 1025. * 9.80665 * np.pi/64. * D_secs[-1]**4.

		z_aux = np.array([z_ball, z_moor, z_SWL])

		z_sparmode = np.concatenate((Z_spar, z_aux),0)
		z_sparmode = np.unique(z_sparmode)
		z_sparmode = np.sort(z_sparmode)

		N_spar = len(Z_spar) - 1
		N_tower = len(Z_tower) - 1

		N_sparelem = len(z_sparmode) - 1
		N_elem = N_sparelem + N_tower

		EI = np.zeros(N_elem)
		L = np.zeros(N_elem)
		m = np.zeros(N_elem) #kg/m

		for i in xrange(N_sparelem):
			L[i] = z_sparmode[i+1] - z_sparmode[i]
			for j in xrange(N_spar):
				if z_sparmode[i+1] <= Z_spar[j+1]:
					sparidx = j
					break
			EI[i] = 1e15 #np.pi / 64. * (D_secs[j]**4. - (D_secs[j] - 2. * wt_secs[j])**4.) * 2.1e11
			steelmass = M_secs[sparidx] / L_secs[sparidx]
			addedmass = 0.
			ballmass = 0.
			if z_sparmode[i+1] <= z_SWL:
				addedmass = 1025. * np.pi / 4. * D_secs[sparidx]**2.
			if z_sparmode[i+1] <= z_ball:
				ballmass = M_ball / L_ball
			m[i] = steelmass + addedmass + ballmass
		
		for i in xrange(N_tower):
			EI[N_sparelem+i] = np.pi / 64. * (D_tower[i]**4. - (D_tower[i] - 2. * wt_tower[i])**4.) * 2.1e11
			L[N_sparelem+i] = L_tower[i]
			m[N_sparelem+i] = M_tower[i] / L_tower[i]

		K = np.zeros(((N_elem+1)*2,(N_elem+1)*2))
		M = np.zeros(((N_elem+1)*2,(N_elem+1)*2))

		LD = np.zeros((N_elem,4))

		for i in xrange(N_elem):
			for j in xrange(4):
				LD[i,j] = j + 2 * i

		for i in xrange(N_elem):
			ke = np.zeros((4,4))
			kg = np.zeros((4,4))
			me = np.zeros((4,4))

			ke[0,0] = ke[2,2] = 12. / L[i]**3.
			ke[0,2] = ke[2,0] = -12. / L[i]**3.
			ke[0,1] = ke[1,0] = ke[0,3] = ke[3,0] = 6. / L[i]**2.
			ke[1,2] = ke[2,1] = ke[2,3] = ke[3,2] = -6. / L[i]**2.
			ke[1,1] = ke[3,3] = 4. / L[i]
			ke[1,3] = ke[3,1] = 2. / L[i]
			ke = ke * EI[i]

			if i >= N_sparelem:
				kg[0,0] = kg[2,2] = 6. / (5. * L[i])
				kg[0,2] = kg[2,0] = -6. / (5. * L[i])
				kg[0,1] = kg[1,0] = kg[0,3] = kg[3,0] = 1. / 10.
				kg[1,2] = kg[2,1] = kg[2,3] = kg[3,2] = -1. / 10.
				kg[1,1] = kg[3,3] = 2. * L[i] / 15.
				kg[1,3] = kg[3,1] = -L[i] / 30.
				kg = kg * (-M_nacelle - M_rotor - np.sum(m[N_sparelem:]*L[N_sparelem:]) + np.sum(m[N_sparelem:i]*L[N_sparelem:i])) * 9.80665

			me[0,0] = me[2,2] = 156.
			me[1,1] = me[3,3] = 4. * L[i]**2.
			me[0,1] = me[1,0] = 22. * L[i]
			me[2,3] = me[3,2] = -22. * L[i]
			me[0,2] = me[2,0] = 54.
			me[1,2] = me[2,1] = 13. * L[i]
			me[0,3] = me[3,0] = -13. * L[i]
			me[1,3] = me[3,1] = -3. * L[i]**2.
			me = me * m[i] * L[i] / 420.
			
			for j in xrange(4):
				row = int(LD[i][j])
				if row > -1:
					for p in xrange(4):
						col = int(LD[i][p])
						if col > -1:
							K[row][col] += (ke[j][p] + kg[j][p])
							M[row][col] += me[j][p]

		mooridx = np.concatenate(np.where(z_sparmode==z_moor))
		SWLidx = np.concatenate(np.where(z_sparmode==z_SWL))

		K[mooridx*2,mooridx*2] += K_moor
		K[SWLidx*2+1,SWLidx*2+1] += K_hydrostatic

		M[-2,-2] += (M_nacelle + M_rotor)
		M[-1,-1] += I_rotor

		eig_vector = np.linalg.eig(np.dot(np.linalg.inv(M), K))[1][:,-3]

		x_sparmode = eig_vector[0:(N_sparelem+1)*2:2]
		x_towermode = eig_vector[(N_sparelem+1)*2-2:(N_elem+1)*2:2]

		outputs['x_sparmode'] = x_sparmode / x_towermode[-1]
		outputs['x_towermode'] = x_towermode / x_towermode[-1]

		outputs['z_sparmode'] = z_sparmode
		outputs['z_towermode'] = Z_tower