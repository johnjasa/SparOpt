import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeGlobStiff(ExplicitComponent):

	def setup(self):
		self.add_input('K_moor', val=0., units='N/m')
		self.add_input('z_sparnode', val=np.zeros(13), units='m')
		self.add_input('z_moor', val=0., units='m')
		self.add_input('kel', val=np.zeros((22,4,4)), units='N/m')

		self.add_output('K_mode', val=np.zeros((46,46)), units='N/m')

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

		#mooridx = np.concatenate(np.where(z_sparnode==z_moor))

		#outputs['K_mode'][mooridx*2,mooridx*2] += K_moor

		moorel = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_moor > z_sparnode[i]) and (z_moor <= z_sparnode[i+1]):
				moorel = i
				break

		L = z_sparnode[moorel+1] - z_sparnode[moorel]
		x = z_moor - z_sparnode[moorel]

		outputs['K_mode'][moorel*2,moorel*2] += K_moor / L**3. * (L**3. - 3. * L * x**2. + 2. * x**3.)
		outputs['K_mode'][moorel*2+1,moorel*2+1] += K_moor / L**2. * (L**2. * x - 2. * L * x**2. + x**3.)
		outputs['K_mode'][moorel*2+2,moorel*2+2] += K_moor / L**3. * (3. * L * x**2. - 2. * x**3.)
		outputs['K_mode'][moorel*2+3,moorel*2+3] += K_moor / L**2. * (-L * x**2. + x**3.)

	def compute_partials(self, inputs, partials):
		K_moor = inputs['K_moor']
		z_sparnode = inputs['z_sparnode']
		z_moor = inputs['z_moor']
		kel = inputs['kel']

		N_elem = len(kel)

		partials['K_mode', 'K_moor'] = np.zeros((2116,1))
		partials['K_mode', 'z_moor'] = np.zeros((2116,1))
		partials['K_mode', 'z_sparnode'] = np.zeros((2116,13))
		partials['K_mode', 'kel'] = np.zeros((2116,352))

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
							partials['K_mode', 'kel'][46*row+col][16*i+4*j+p] += 1.

		#mooridx = np.concatenate(np.where(z_sparnode==z_moor))

		#partials['K_mode', 'K_moor'][48*mooridx*2+mooridx*2] += 1.

		moorel = 0

		for i in xrange(len(z_sparnode)-1):
			if (z_moor > z_sparnode[i]) and (z_moor <= z_sparnode[i+1]):
				moorel = i
				break

		L = z_sparnode[moorel+1] - z_sparnode[moorel]
		x = z_moor - z_sparnode[moorel]

		partials['K_mode', 'K_moor'][46*moorel*2+moorel*2] += 1. / L**3. * (L**3. - 3. * L * x**2. + 2. * x**3.)
		partials['K_mode', 'K_moor'][46*(moorel*2+1)+moorel*2+1] += 1. / L**2. * (L**2. * x - 2. * L * x**2. + x**3.)
		partials['K_mode', 'K_moor'][46*(moorel*2+2)+moorel*2+2] += 1. / L**3. * (3. * L * x**2. - 2. * x**3.)
		partials['K_mode', 'K_moor'][46*(moorel*2+3)+moorel*2+3] += 1. / L**2. * (-L * x**2. + x**3.)

		partials['K_mode', 'z_moor'][46*moorel*2+moorel*2] += K_moor / L**3. * (-6. * L * x + 6. * x**2.)
		partials['K_mode', 'z_moor'][46*(moorel*2+1)+moorel*2+1] += K_moor / L**2. * (L**2. - 4. * L * x + 3. * x**2.)
		partials['K_mode', 'z_moor'][46*(moorel*2+2)+moorel*2+2] += K_moor / L**3. * (6. * L * x - 6. * x**2.)
		partials['K_mode', 'z_moor'][46*(moorel*2+3)+moorel*2+3] += K_moor / L**2. * (-2. * L * x + 3. * x**2.)

		partials['K_mode', 'z_sparnode'][46*moorel*2+moorel*2, moorel] += 3. * K_moor / L**4. * (L**3. - 3. * L * x**2. + 2. * x**3.) - K_moor / L**3. * (3. * L**2. - 3. * x**2.) - K_moor / L**3. * (-6. * L * x + 6. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+1)+moorel*2+1, moorel] += 2. * K_moor / L**3. * (L**2. * x - 2. * L * x**2. + x**3.) - K_moor / L**2. * (2. * L * x - 2. * x**2.) - K_moor / L**2. * (L**2. - 4. * L * x + 3. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+2)+moorel*2+2, moorel] += 3. * K_moor / L**4. * (3. * L * x**2. - 2. * x**3.) - K_moor / L**3. * (3. * x**2.) - K_moor / L**3. * (6. * L * x - 6. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+3)+moorel*2+3, moorel] += 2. * K_moor / L**3. * (-L * x**2. + x**3.) - K_moor / L**2. * (-x**2.) - K_moor / L**2. * (-2. * L * x + 3. * x**2.)
		partials['K_mode', 'z_sparnode'][46*moorel*2+moorel*2, moorel+1] +=  -3. * K_moor / L**4. * (L**3. - 3. * L * x**2. + 2. * x**3.) + K_moor / L**3. * (3. * L**2. - 3. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+1)+moorel*2+1, moorel+1] += -2. * K_moor / L**3. * (L**2. * x - 2. * L * x**2. + x**3.) + K_moor / L**2. * (2. * L * x - 2. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+2)+moorel*2+2, moorel+1] += -3. * K_moor / L**4. * (3. * L * x**2. - 2. * x**3.) + K_moor / L**3. * (3. * x**2.)
		partials['K_mode', 'z_sparnode'][46*(moorel*2+3)+moorel*2+3, moorel+1] += -2. * K_moor / L**3. * (-L * x**2. + x**3.) + K_moor / L**2. * (-x**2.)