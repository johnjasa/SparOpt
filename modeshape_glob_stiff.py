import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeGlobStiff(ExplicitComponent):

	def setup(self):
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_sparnode', val=np.zeros(14), units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('kel', val=np.zeros((23,4,4)), units='N/m')

		self.add_output('K_mode', val=np.zeros((48,48)), units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		K_moor = inputs['K_moor']
		z_sparnode = inputs['z_sparnode']
		z_moor = inputs['z_moor']
		kel = inputs['kel']

		N_elem = len(kel)

		outputs['K_mode'] = np.zeros(((N_elem+1)*2,(N_elem+1)*2))

		LD = np.zeros((N_elem,4))

		for i in xrange(N_elem):
			for j in xrange(4):
				LD[i,j] = j + 2 * i

		for i in xrange(N_elem):
			for j in xrange(4):
				row = int(LD[i][j])
				if row > -1:
					for p in xrange(4):
						col = int(LD[i][p])
						if col > -1:
							outputs['K_mode'][row][col] += kel[i][j][p]

		mooridx = np.concatenate(np.where(z_sparnode==z_moor))

		outputs['K_mode'][mooridx*2,mooridx*2] += K_moor

	def compute_partials(self, inputs, partials):
		K_moor = inputs['K_moor']
		z_sparnode = inputs['z_sparnode']
		z_moor = inputs['z_moor']
		kel = inputs['kel']

		N_elem = len(kel)

		partials['K_mode', 'K_moor'] = np.zeros(2304)
		partials['K_mode', 'kel'] = np.zeros((2304,368))

		LD = np.zeros((N_elem,4))

		for i in xrange(N_elem):
			for j in xrange(4):
				LD[i,j] = j + 2 * i

		for i in xrange(N_elem):
			for j in xrange(4):
				row = int(LD[i][j])
				if row > -1:
					for p in xrange(4):
						col = int(LD[i][p])
						if col > -1:
							partials['K_mode', 'kel'][48*row+col][16*i+4*j+p] += 1.

		mooridx = np.concatenate(np.where(z_sparnode==z_moor))

		partials['K_mode', 'K_moor'][48*mooridx*2+mooridx*2] += 1.