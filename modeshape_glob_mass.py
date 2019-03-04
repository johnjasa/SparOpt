import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeGlobMass(ExplicitComponent):

	def setup(self):
		self.add_input('M_nacelle', val=0., units='kg')
		self.add_input('M_rotor', val=0., units='kg')
		self.add_input('I_rotor', val=0., units='kg*m**2')
		self.add_input('mel', val=np.zeros((22,4,4)), units='kg')

		self.add_output('M_mode', val=np.zeros((46,46)), units='kg')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		I_rotor = inputs['I_rotor']
		mel = inputs['mel']
		N_elem = len(mel)

		outputs['M_mode'] = np.zeros(((N_elem+1)*2,(N_elem+1)*2))

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
							outputs['M_mode'][row][col] += mel[i][j][p]

		outputs['M_mode'][-2,-2] += (M_nacelle + M_rotor)
		outputs['M_mode'][-1,-1] += I_rotor

	def compute_partials(self, inputs, partials):
		M_nacelle = inputs['M_nacelle']
		M_rotor = inputs['M_rotor']
		I_rotor = inputs['I_rotor']
		mel = inputs['mel']
		N_elem = len(mel)

		partials['M_mode', 'M_nacelle'] = np.zeros(2116)
		partials['M_mode', 'M_rotor'] = np.zeros(2116)
		partials['M_mode', 'I_rotor'] = np.zeros(2116)
		partials['M_mode', 'mel'] = np.zeros((2116,352))

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
							partials['M_mode', 'mel'][46*row+col][16*i+4*j+p] += 1.

		partials['M_mode', 'M_nacelle'][-48] = 1.
		partials['M_mode', 'M_rotor'][-48] = 1.
		partials['M_mode', 'I_rotor'][-1] = 1.