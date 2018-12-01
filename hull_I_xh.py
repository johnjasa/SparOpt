import numpy as np

from openmdao.api import ExplicitComponent

class HullIXh(ExplicitComponent):

	def setup(self):
		self.add_input('tau', val=np.zeros(10), units='MPa')
		self.add_input('r_0', val=np.zeros(10), units='m')
		self.add_input('spar_draft', val=0., units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')
		self.add_input('l_stiff', val=np.zeros(10), units='m')

		self.add_output('I_xh', val=np.zeros(10), units='m**4')

		self.declare_partials('I_xh', 'tau', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_xh', 'r_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_xh', 'spar_draft')
		self.declare_partials('I_xh', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_xh', 'l_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		E = 2.1e5 #MPa

		outputs['I_xh'] = (inputs['tau'] / E)**(8./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) * (inputs['spar_draft'] + 10.) * inputs['r_0'] * inputs['wt_spar_p'][:-1] * inputs['l_stiff']

	def compute_partials(self, inputs, partials):
		E = 2.1e5

		(inputs['tau'] / E)**(8./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) * (inputs['spar_draft'] + 10.) * inputs['r_0'] * inputs['wt_spar_p'][:-1] * inputs['l_stiff']

		partials['I_xh', 'tau'] = 8. / (5. * E) * (inputs['tau'] / E)**(3./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) * (inputs['spar_draft'] + 10.) * inputs['r_0'] * inputs['wt_spar_p'][:-1] * inputs['l_stiff']
		partials['I_xh', 'r_0'] = (inputs['tau'] / E)**(8./5.) * (inputs['spar_draft'] + 10.) * inputs['wt_spar_p'][:-1] * inputs['l_stiff'] * ((inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) + inputs['r_0'] * 1. / (5. * (inputs['spar_draft'] + 10.)) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(-4./5.))
		partials['I_xh', 'spar_draft'] = (inputs['tau'] / E)**(8./5.) * inputs['r_0'] * inputs['wt_spar_p'][:-1] * inputs['l_stiff'] * ((inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) - (inputs['spar_draft'] + 10.) * 1. / 5. * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(-4./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.)**2.))
		partials['I_xh', 'wt_spar_p'] = (inputs['tau'] / E)**(8./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) * (inputs['spar_draft'] + 10.) * inputs['r_0'] * inputs['l_stiff']
		partials['I_xh', 'l_stiff'] = (inputs['tau'] / E)**(8./5.) * (inputs['r_0'] / (inputs['spar_draft'] + 10.))**(1./5.) * (inputs['spar_draft'] + 10.) * inputs['r_0'] * inputs['wt_spar_p'][:-1]