import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeElemStiff(ExplicitComponent):

	def setup(self):
		self.add_input('', val=np.zeros(23), units='N*m**2')
		self.add_input('', val=np.zeros(23), units='m')
		self.add_input('', val=np.zeros(10), units='N')

		self.add_output('ke', val=0., units='N/m')
		self.add_output('kg', val=0., units='N/m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		EI = inputs['']
		L = inputs['']
		norm_force = inputs['']

		for i in xrange(N_elem):
			ke = np.zeros((4,4))
			kg = np.zeros((4,4))

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
				kg = kg * norm_force[i] * 9.80665

	def compute_partials(self, inputs, partials):
		partials['ke', ''] = 
		partials['ke', ''] = 
		partials['ke', ''] = 