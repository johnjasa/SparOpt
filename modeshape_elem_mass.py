import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('M_spar', val=np.zeros(10), units='kg')
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('L_ball_elem', val=np.zeros(10), units='m')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('L_mode_elem', val=np.zeros(23), units='m')

		self.add_output('mel', val=np.zeros((23,4,4)), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		M_spar = inputs['M_spar']
		Z_spar = inputs['Z_spar']
		L_tower = inputs['L_tower']
		M_tower = inputs['M_tower']
		M_ball_elem = inputs['M_ball_elem']
		L_ball_elem = inputs['L_ball_elem']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + L_ball[0]
		z_sparnode = inputs['z_sparnode']
		L = inputs['L_mode_elem']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(M_tower)
		N_elem = N_sparelem + N_towerelem

		m = np.zeros(N_elem) #kg/m

		for i in xrange(N_sparelem):
			for j in xrange(len(M_spar)):
				if z_sparnode[i+1] <= Z_spar[j+1]:
					sparidx = j
					break
			steelmass = M_spar[sparidx] / L_spar[sparidx]
			addedmass = 0.
			ballmass = 0.
			if z_sparnode[i+1] <= 0.:
				addedmass = 1025. * np.pi / 4. * D_spar[sparidx]**2.
			if z_sparnode[i+1] <= z_ball:
				ballmass = M_ball_elem[sparidx] / L_ball_elem[sparidx]
			m[i] = steelmass + addedmass + ballmass
		
		for i in xrange(N_towerelem):
			m[N_sparelem+i] = M_tower[i] / L_tower[i]

		outputs['mel'] = np.zeros((N_elem,4,4))
		for i in xrange(N_elem):
			outputs['mel'][i,0,0] = outputs['mel'][i,2,2] = 156.
			outputs['mel'][i,1,1] = outputs['mel'][i,3,3] = 4. * L[i]**2.
			outputs['mel'][i,0,1] = outputs['mel'][i,1,0] = 22. * L[i]
			outputs['mel'][i,2,3] = outputs['mel'][i,3,2] = -22. * L[i]
			outputs['mel'][i,0,2] = outputs['mel'][i,2,0] = 54.
			outputs['mel'][i,1,2] = outputs['mel'][i,2,1] = 13. * L[i]
			outputs['mel'][i,0,3] = outputs['mel'][i,3,0] = -13. * L[i]
			outputs['mel'][i,1,3] = outputs['mel'][i,3,1] = -3. * L[i]**2.
			outputs['mel'][i] = outputs['mel'][i] * m[i] * L[i] / 420.

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		M_spar = inputs['M_spar']
		Z_spar = inputs['Z_spar']
		L_tower = inputs['L_tower']
		M_tower = inputs['M_tower']
		M_ball_elem = inputs['M_ball_elem']
		L_ball_elem = inputs['L_ball_elem']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + L_ball[0]
		z_sparnode = inputs['z_sparnode']
		L = inputs['L_mode_elem']

		partials['mel', 'D_spar'] = np.zeros((368,10))
		partials['mel', 'L_spar'] = np.zeros((368,10))
		partials['mel', 'M_spar'] = np.zeros((368,10))
		partials['mel', 'L_tower'] = np.zeros((368,10))
		partials['mel', 'M_tower'] = np.zeros((368,10))
		partials['mel', 'spar_draft'] = np.zeros(368)
		partials['mel', 'M_ball_elem'] = np.zeros((368,10))
		partials['mel', 'L_ball_elem'] = np.zeros((368,10))
		partials['mel', 'L_ball'] = np.zeros(368)
		partials['mel', 'z_sparnode'] = np.zeros((368,14))
		partials['mel', 'L_mode_elem'] = np.zeros((368,23))

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(M_tower)
		N_elem = N_sparelem + N_towerelem

		dm_dD = np.zeros((N_elem,len(D_spar)))
		dm_dLs = np.zeros((N_elem,len(L_spar)))
		dm_dMs = np.zeros((N_elem,len(M_spar)))
		dm_dLt = np.zeros((N_elem,len(L_tower)))
		dm_dMt = np.zeros((N_elem,len(M_tower)))
		dm_dLb = np.zeros((N_elem,len(L_ball_elem)))
		dm_dMb = np.zeros((N_elem,len(M_ball_elem)))

		m = np.zeros(N_elem)

		for i in xrange(N_sparelem):
			for j in xrange(len(D_spar)):
				if z_sparnode[i+1] <= Z_spar[j+1]:
					sparidx = j
					break
			steelmass = M_spar[sparidx] / L_spar[sparidx]
			dm_dLs[i,sparidx] += -M_spar[sparidx] / L_spar[sparidx]**2.
			dm_dMs[i,sparidx] += 1. / L_spar[sparidx]
			addedmass = 0.
			ballmass = 0.
			if z_sparnode[i+1] <= 0.:
				addedmass = 1025. * np.pi / 4. * D_spar[sparidx]**2.
				dm_dD[i,sparidx] += 1025. * np.pi / 2. * D_spar[sparidx]
			if z_sparnode[i+1] <= z_ball:
				ballmass = M_ball_elem[sparidx] / L_ball_elem[sparidx]
				dm_dLb[i,sparidx] += -M_ball_elem[sparidx] / L_ball_elem[sparidx]**2.
				dm_dMb[i,sparidx] += 1. / L_ball_elem[sparidx]
			m[i] = steelmass + addedmass + ballmass
		
		for i in xrange(N_towerelem):
			m[N_sparelem+i] = M_tower[i] / L_tower[i]
			dm_dLt[N_sparelem+i,i] += -M_tower[i] / L_tower[i]**2.
			dm_dMt[N_sparelem+i,i] += 1. / L_tower[i]

		dmel_dm = np.zeros((N_elem,4,4))
		dmel_dLe = np.zeros((N_elem,4,4))
		for i in xrange(N_elem):
			dmel_dm[i,0,0] = dmel_dm[i,2,2] = 156.
			dmel_dm[i,1,1] = dmel_dm[i,3,3] = 4. * L[i]**2.
			dmel_dm[i,0,1] = dmel_dm[i,1,0] = 22. * L[i]
			dmel_dm[i,2,3] = dmel_dm[i,3,2] = -22. * L[i]
			dmel_dm[i,0,2] = dmel_dm[i,2,0] = 54.
			dmel_dm[i,1,2] = dmel_dm[i,2,1] = 13. * L[i]
			dmel_dm[i,0,3] = dmel_dm[i,3,0] = -13. * L[i]
			dmel_dm[i,1,3] = dmel_dm[i,3,1] = -3. * L[i]**2.
			dmel_dLe[i,0,0] = dmel_dLe[i,2,2] = 156. * m[i] / 420.
			dmel_dLe[i,1,1] = dmel_dLe[i,3,3] = 12. * L[i]**2. * m[i] / 420.
			dmel_dLe[i,0,1] = dmel_dLe[i,1,0] = 44. * m[i] * L[i] / 420.
			dmel_dLe[i,2,3] = dmel_dLe[i,3,2] = -44. * L[i] * m[i] / 420.
			dmel_dLe[i,0,2] = dmel_dLe[i,2,0] = 54. * m[i] / 420.
			dmel_dLe[i,1,2] = dmel_dLe[i,2,1] = 26. * m[i] * L[i] / 420.
			dmel_dLe[i,0,3] = dmel_dLe[i,3,0] = -26. * m[i] * L[i] / 420.
			dmel_dLe[i,1,3] = dmel_dLe[i,3,1] = -9. * L[i]**2. * m[i] / 420.
			dmel_dm[i] = dmel_dm[i] * L[i] / 420.

		for i in xrange(len(D_spar)):
			dmel_dD = []
			dmel_dLs = []
			dmel_dMs = []
			dmel_dLb = []
			dmel_dMb = []
			for j in xrange(N_elem):
				for k in xrange(4):
					for l in xrange(4):
						dmel_dD.append(dmel_dm[j,k,l] * dm_dD[j,i])
						dmel_dLs.append(dmel_dm[j,k,l] * dm_dLs[j,i])
						dmel_dMs.append(dmel_dm[j,k,l] * dm_dMs[j,i])
						dmel_dLb.append(dmel_dm[j,k,l] * dm_dLb[j,i])
						dmel_dMb.append(dmel_dm[j,k,l] * dm_dMb[j,i])
			
			partials['mel', 'D_spar'][:,i] = np.array(dmel_dD)
			partials['mel', 'L_spar'][:,i] = np.array(dmel_dLs)
			partials['mel', 'M_spar'][:,i] = np.array(dmel_dMs)
			partials['mel', 'L_ball_elem'][:,i] = np.array(dmel_dLb)
			partials['mel', 'M_ball_elem'][:,i] = np.array(dmel_dMb)

		for i in xrange(len(L_tower)):
			dmel_dLt = []
			dmel_dMt = []
			for j in xrange(N_elem):
				for k in xrange(4):
					for l in xrange(4):
						dmel_dLt.append(dmel_dm[j,k,l] * dm_dLt[j,i])
						dmel_dMt.append(dmel_dm[j,k,l] * dm_dMt[j,i])

			partials['mel', 'L_tower'][:,i] = np.array(dmel_dLt)
			partials['mel', 'M_tower'][:,i] = np.array(dmel_dMt)

		for i in xrange(N_elem):
			partials['mel', 'L_mode_elem'][16*i:16*i+16,i] = dmel_dLe[i].flatten()		