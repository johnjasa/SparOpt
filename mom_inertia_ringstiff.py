import numpy as np

from openmdao.api import ExplicitComponent

class MomInertiaRingstiff(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar', val=np.zeros(10), units='m')
		self.add_input('wt_spar', val=np.zeros(10), units='m')
		self.add_input('t_w_stiff', val=np.zeros(10), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('b_stiff', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')
		self.add_input('r_0', val=np.zeros(10), units='m')
		self.add_input('l_ef', val=np.zeros(10), units='m')

		self.add_output('I_stiff', val=np.zeros(10), units='m**4')

		self.declare_partials('I_stiff', 'D_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'wt_spar', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 't_w_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 't_f_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'b_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'r_hull', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'r_0', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('I_stiff', 'l_ef', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		D_spar = inputs['D_spar']
		wt_spar = inputs['wt_spar']
		t_w_stiff = inputs['t_w_stiff']
		t_f_stiff = inputs['t_f_stiff']
		h_stiff = inputs['h_stiff']
		b_stiff = inputs['b_stiff']
		r_hull = inputs['r_hull']
		r_0 = inputs['r_0']
		l_ef = inputs['l_ef']

		outputs['I_stiff'] = 1. / 12. * (l_ef * wt_spar**3. + t_w_stiff * h_stiff**3. + b_stiff * t_f_stiff**3.) + l_ef * wt_spar * (r_hull - r_0)**2. + t_w_stiff * h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0)**2. + b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)**2.

	def compute_partials(self, inputs, partials):
		D_spar = inputs['D_spar']
		wt_spar = inputs['wt_spar']
		t_w_stiff = inputs['t_w_stiff']
		t_f_stiff = inputs['t_f_stiff']
		h_stiff = inputs['h_stiff']
		b_stiff = inputs['b_stiff']
		r_hull = inputs['r_hull']
		r_0 = inputs['r_0']
		l_ef = inputs['l_ef']

		partials['I_stiff', 'D_spar'] = t_w_stiff * h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0) + b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)
		partials['I_stiff', 'wt_spar'] = 1. / 4. * l_ef * wt_spar**2. + l_ef * (r_hull - r_0)**2. - 2. * t_w_stiff * h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0) - 2. * b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)
		partials['I_stiff', 't_w_stiff'] = 1. / 12. * h_stiff**3. + h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0)**2.
		partials['I_stiff', 't_f_stiff'] = 1. / 4. * b_stiff * t_f_stiff**2. + b_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)**2. - b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)
		partials['I_stiff', 'h_stiff'] = 1. / 4. * t_w_stiff * h_stiff**2. + t_w_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0)**2. - t_w_stiff * h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0) - 2. * b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)
		partials['I_stiff', 'b_stiff'] = 1. / 12. * t_f_stiff**3. + t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)**2.
		partials['I_stiff', 'r_hull'] = 2. * l_ef * wt_spar * (r_hull - r_0)
		partials['I_stiff', 'r_0'] = -2. * l_ef * wt_spar * (r_hull - r_0) - 2. * t_w_stiff * h_stiff * (D_spar / 2. - wt_spar - h_stiff / 2. - r_0) - 2. * b_stiff * t_f_stiff * (D_spar / 2. - wt_spar - h_stiff - t_f_stiff / 2. - r_0)
		partials['I_stiff', 'l_ef'] = 1. / 12. * wt_spar**3. + wt_spar * (r_hull - r_0)**2.