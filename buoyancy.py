import numpy as np

from openmdao.api import ExplicitComponent

class Buoyancy(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('L_spar', val=np.zeros(10), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('sub_vol', val=0., units='m**3')

		self.add_output('buoy_spar', val=0., units='N')
		self.add_output('CoB', val=0., units='m')

		self.declare_partials('*', '*')

	def compute(self, inputs, outputs):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		spar_draft = inputs['spar_draft']

		outputs['buoy_spar'] = inputs['sub_vol'] * 1025. * 9.80665

		CoB_t_vol = 0.

		for i in xrange(len(D_spar) - 1):
			CoB_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			CoB_t_vol += np.pi / 4. * D_spar[i]**2. * L_spar[i] * CoB_sec

		CoB_sec = -spar_draft + np.sum(L_spar[0:-1]) + (L_spar[-1] - 10.) / 2.
		CoB_t_vol += np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) * CoB_sec
		#TODO: last secton from 0 to 10

		outputs['CoB'] = CoB_t_vol / inputs['sub_vol']

	def compute_partials(self, inputs, partials):
		D_spar  = inputs['D_spar']
		L_spar  = inputs['L_spar']
		spar_draft = inputs['spar_draft']

		partials['buoy_spar', 'D_spar'] = 0.
		partials['buoy_spar', 'L_spar'] = 0.
		partials['buoy_spar', 'spar_draft'] = 0.
		partials['buoy_spar', 'sub_vol'] = 1025. * 9.80665

		partials['CoB', 'D_spar'] = np.zeros((1,10))
		partials['CoB', 'L_spar'] = np.zeros((1,10))
		partials['CoB', 'spar_draft'] = 0.

		CoB_t_vol = 0.

		for i in xrange(len(D_spar) - 1):
			CoB_sec = -spar_draft + np.sum(L_spar[0:i]) + L_spar[i] / 2.
			CoB_t_vol += np.pi / 4. * D_spar[i]**2. * L_spar[i] * CoB_sec

			partials['CoB', 'D_spar'][0,i] += np.pi / 2. * D_spar[i] * L_spar[i] * CoB_sec / inputs['sub_vol']
			
			partials['CoB', 'L_spar'][0,i] += (np.pi / 4. * D_spar[i]**2. * CoB_sec + np.pi / 4. * D_spar[i]**2. * L_spar[i] * 0.5) / inputs['sub_vol']
			for j in xrange(i):
				partials['CoB', 'L_spar'][0,j] += np.pi / 4. * D_spar[i]**2. * L_spar[i] / inputs['sub_vol']
			
			partials['CoB', 'spar_draft'] += -np.pi / 4. * D_spar[i]**2. * L_spar[i] / inputs['sub_vol']

		CoB_sec = -spar_draft + np.sum(L_spar[0:-1]) + (L_spar[-1] - 10.) / 2.
		CoB_t_vol += np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) * CoB_sec

		partials['CoB', 'D_spar'][0,-1] += np.pi / 2. * D_spar[-1] * (L_spar[-1] - 10.) * CoB_sec / inputs['sub_vol']

		partials['CoB', 'L_spar'][0,-1] += (np.pi / 4. * D_spar[-1]**2. * CoB_sec + np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) * 0.5) / inputs['sub_vol']
		for j in xrange(len(D_spar) - 1):
				partials['CoB', 'L_spar'][0,j] += np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) / inputs['sub_vol']

		partials['CoB', 'spar_draft'] += -np.pi / 4. * D_spar[-1]**2. * (L_spar[-1] - 10.) / inputs['sub_vol']

		partials['CoB', 'sub_vol'] += -CoB_t_vol / inputs['sub_vol']**2.