import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemMass(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(3), units='m')
		self.add_input('L_spar', val=np.zeros(3), units='m')
		self.add_input('M_spar', val=np.zeros(3), units='kg')
		self.add_input('Z_spar', val=np.zeros(4), units='m')
		self.add_input('L_tower', val=np.zeros(10), units='m')
		self.add_input('M_tower', val=np.zeros(10), units='kg')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('L_ball', val=0., units='m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')

		self.add_output('M_mode', val=np.zeros(23), units='kg/m')

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		L_spar = inputs['L_spar']
		M_spar = inputs['M_spar']
		Z_spar = inputs['Z_spar']
		L_tower = inputs['L_tower']
		M_tower = inputs['M_tower']
		M_ball = inputs['M_ball']
		L_ball = inputs['L_ball']
		z_ball = -inputs['spar_draft'][0] + L_ball[0]
		z_SWL = 0.
		z_sparnode = inputs['z_sparnode']

		N_sparelem = len(z_sparnode) - 1
		N_towerelem = len(M_tower) - 1
		N_elem = N_sparelem + N_towerelem

		m = np.zeros(N_elem) #kg/m

		for i in xrange(N_sparelem):
			for j in xrange(N_spar):
				if z_sparnode[i+1] <= Z_spar[j+1]:
					sparidx = j
					break
			steelmass = M_spar[sparidx] / L_spar[sparidx]
			addedmass = 0.
			ballmass = 0.
			if z_sparnode[i+1] <= z_SWL:
				addedmass = 1025. * np.pi / 4. * D_spar[sparidx]**2.
			if z_sparnode[i+1] <= z_ball:
				ballmass = M_ball / L_ball
			m[i] = steelmass + addedmass + ballmass
		
		for i in xrange(N_towerelem):
			m[N_sparelem+i] = M_tower[i] / L_tower[i]

		for i in xrange(N_elem):
			me = np.zeros((4,4))

			me[0,0] = me[2,2] = 156.
			me[1,1] = me[3,3] = 4. * L[i]**2.
			me[0,1] = me[1,0] = 22. * L[i]
			me[2,3] = me[3,2] = -22. * L[i]
			me[0,2] = me[2,0] = 54.
			me[1,2] = me[2,1] = 13. * L[i]
			me[0,3] = me[3,0] = -13. * L[i]
			me[1,3] = me[3,1] = -3. * L[i]**2.
			me = me * m[i] * L[i] / 420.