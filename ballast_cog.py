import numpy as np

from openmdao.api import ExplicitComponent

class BallastCoG(ExplicitComponent):

	def setup(self):
		self.add_input('L_ball_elem', val=np.zeros(10), units='m')
		self.add_input('M_ball_elem', val=np.zeros(10), units='kg')
		self.add_input('M_ball', val=0., units='kg')
		self.add_input('spar_draft', val=0., units='m')

		self.add_output('CoG_ball', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		L_ball_elem  = inputs['L_ball_elem']
		M_ball_elem  = inputs['M_ball_elem']
		M_ball  = inputs['M_ball']
		spar_draft = inputs['spar_draft']

		CoG_t_mass = 0.

		for i in xrange(10):
			CoG_sec = -spar_draft + np.sum(L_ball_elem[0:i]) + L_ball_elem[i] / 2.
			CoG_t_mass += M_ball_elem[i] * CoG_sec
		
		outputs['CoG_ball'] = CoG_t_mass / M_ball

	def compute_partials(self, inputs, partials):
		L_ball_elem  = inputs['L_ball_elem']
		M_ball_elem  = inputs['M_ball_elem']
		M_ball  = inputs['M_ball']
		spar_draft = inputs['spar_draft']

		partials['CoG_ball', 'L_ball_elem'] = np.zeros((1,10))
		partials['CoG_ball', 'M_ball_elem'] = np.zeros((1,10))
		partials['CoG_ball', 'M_ball'] = 0.
		partials['CoG_ball', 'spar_draft'] = 0.

		CoG_t_mass = 0.

		for i in xrange(10):
			CoG_sec = -spar_draft + np.sum(L_ball_elem[0:i]) + L_ball_elem[i] / 2.
			
			CoG_t_mass += M_ball_elem[i] * CoG_sec

			partials['CoG_ball', 'L_ball_elem'][0,i] += 0.5 * M_ball_elem[i] / M_ball
			partials['CoG_ball', 'M_ball_elem'][0,i] += CoG_sec / M_ball
			partials['CoG_ball', 'spar_draft'] += -M_ball_elem[i] / M_ball

			for j in xrange(i):
				partials['CoG_ball', 'L_ball_elem'][0,j] += M_ball_elem[i] / M_ball

		partials['CoG_ball', 'M_ball'] = -CoG_t_mass / M_ball**2.