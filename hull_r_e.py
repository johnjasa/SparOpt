import numpy as np

from openmdao.api import ExplicitComponent

class HullRE(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')
		self.add_input('t_w_stiff', val=np.zeros(10), units='m')
		self.add_input('b_stiff', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')

		self.add_output('r_e', val=np.zeros(10), units='m')

		self.declare_partials('r_e', 'D_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_e', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_e', 't_f_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_e', 't_w_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_e', 'b_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_e', 'h_stiff', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]
		t_f_stiff = inputs['t_f_stiff']
		t_w_stiff = inputs['t_w_stiff']
		b_stiff = inputs['b_stiff']
		h_stiff = inputs['h_stiff']

		outputs['r_e'] = (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.)) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)

	def compute_partials(self, inputs, partials):
		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]
		t_f_stiff = inputs['t_f_stiff']
		t_w_stiff = inputs['t_w_stiff']
		b_stiff = inputs['b_stiff']
		h_stiff = inputs['h_stiff']

		partials['r_e', 'D_spar_p'] = 0.5 * (t_f_stiff * b_stiff + t_w_stiff * h_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)
		partials['r_e', 'wt_spar_p'] = (-t_f_stiff * b_stiff - t_w_stiff * h_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)
		partials['r_e', 't_f_stiff'] = (b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) - 0.5 * t_f_stiff * b_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.)) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)**2. * b_stiff
		partials['r_e', 't_w_stiff'] = h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.)) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)**2. * h_stiff
		partials['r_e', 'b_stiff'] = t_f_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.)) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)**2. * t_f_stiff
		partials['r_e', 'h_stiff'] = (-t_f_stiff * b_stiff + t_w_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) - 0.5 * t_w_stiff * h_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.)) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff)**2. * t_w_stiff