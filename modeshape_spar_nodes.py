import numpy as np

from openmdao.api import ExplicitComponent

class ModeshapeSparNodes(ExplicitComponent):

	def setup(self):
		self.add_input('Z_spar', val=np.zeros(11), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('L_ball', val=0., units='m')

		self.add_output('z_sparnode', val=np.zeros(13), units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		Z_spar = inputs['Z_spar']
		L_ball = inputs['L_ball'][0]
		z_ball = -inputs['spar_draft'][0] + L_ball #top of ballast
		z_SWL = 0.

		if len(np.where(Z_spar==z_ball)[0]) != 0:
			z_ball += 0.1
		if len(np.where(Z_spar==z_SWL)[0]) != 0:
			z_SWL += 0.1
		if z_ball == z_SWL:
			z_ball += 0.1
		
		z_aux = np.array([z_ball, z_SWL])

		outputs['z_sparnode'] = np.concatenate((Z_spar, z_aux),0)
		outputs['z_sparnode'] = np.sort(outputs['z_sparnode'])

	def compute_partials(self, inputs, partials):
		Z_spar = inputs['Z_spar']
		L_ball = inputs['L_ball'][0]
		z_ball = -inputs['spar_draft'][0] + L_ball
		z_SWL = 0.

		if len(np.where(Z_spar==z_ball)[0]) != 0:
			z_ball += 0.1
		if len(np.where(Z_spar==z_SWL)[0]) != 0:
			z_SWL += 0.1
		if z_ball == z_SWL:
			z_ball += 0.1

		z_aux = np.array([z_ball, z_SWL])

		z_sparnode = np.concatenate((Z_spar, z_aux),0)
		z_sparnode = np.sort(z_sparnode)

		partials['z_sparnode', 'Z_spar'] = np.zeros((13,11))
		partials['z_sparnode', 'spar_draft'] = np.zeros(13)
		partials['z_sparnode', 'L_ball'] = np.zeros(13)

		ballidx = np.concatenate(np.where(z_sparnode==z_ball))
		SWLidx = np.concatenate(np.where(z_sparnode==z_SWL))

		partials['z_sparnode', 'spar_draft'][ballidx] = -1.
		partials['z_sparnode', 'L_ball'][ballidx] = 1.

		count = 0
		for i in xrange(13):
			if i != ballidx and i != SWLidx:
				partials['z_sparnode', 'Z_spar'][i,count] += 1.
				count += 1