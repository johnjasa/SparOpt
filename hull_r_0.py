import numpy as np

from openmdao.api import ExplicitComponent

class HullR0(ExplicitComponent):

	def setup(self):
		self.add_input('D_spar_p', val=np.zeros(11), units='m')
		self.add_input('wt_spar_p', val=np.zeros(11), units='m')
		self.add_input('t_f_stiff', val=np.zeros(10), units='m')
		self.add_input('t_w_stiff', val=np.zeros(10), units='m')
		self.add_input('b_stiff', val=np.zeros(10), units='m')
		self.add_input('h_stiff', val=np.zeros(10), units='m')
		self.add_input('l_eo', val=np.zeros(10), units='m')
		self.add_input('r_hull', val=np.zeros(10), units='m')

		self.add_output('r_0', val=np.zeros(10), units='m')

		self.declare_partials('r_0', 'D_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 'wt_spar_p', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 't_f_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 't_w_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 'b_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 'h_stiff', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 'l_eo', rows=np.arange(10), cols=np.arange(10))
		self.declare_partials('r_0', 'r_hull', rows=np.arange(10), cols=np.arange(10))

	def compute(self, inputs, outputs):
		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]
		t_f_stiff = inputs['t_f_stiff']
		t_w_stiff = inputs['t_w_stiff']
		b_stiff = inputs['b_stiff']
		h_stiff = inputs['h_stiff']
		l_eo = inputs['l_eo']
		r_hull = inputs['r_hull']

		outputs['r_0'] = (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)

	def compute_partials(self, inputs, partials):
		D_spar_p = inputs['D_spar_p'][:-1]
		wt_spar_p = inputs['wt_spar_p'][:-1]
		t_f_stiff = inputs['t_f_stiff']
		t_w_stiff = inputs['t_w_stiff']
		b_stiff = inputs['b_stiff']
		h_stiff = inputs['h_stiff']
		l_eo = inputs['l_eo']
		r_hull = inputs['r_hull']

		(t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)

		partials['r_0', 'D_spar_p'] = 0.5 * (t_f_stiff * b_stiff + t_w_stiff * h_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)
		partials['r_0', 'wt_spar_p'] = (-t_f_stiff * b_stiff - t_w_stiff * h_stiff + l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * l_eo
		partials['r_0', 't_f_stiff'] = (b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) - 0.5 * t_f_stiff * b_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * b_stiff
		partials['r_0', 't_w_stiff'] = h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * h_stiff
		partials['r_0', 'b_stiff'] = t_f_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * t_f_stiff
		partials['r_0', 'h_stiff'] = (-t_f_stiff * b_stiff + t_w_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) - 0.5 * t_w_stiff * h_stiff) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * t_w_stiff
		partials['r_0', 'l_eo'] = wt_spar_p * r_hull / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo) - (t_f_stiff * b_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff - t_f_stiff / 2.) + t_w_stiff * h_stiff * (D_spar_p / 2. - wt_spar_p - h_stiff / 2.) + wt_spar_p * l_eo * r_hull) / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)**2. * wt_spar_p
		partials['r_0', 'r_hull'] =  wt_spar_p * l_eo / (t_f_stiff * b_stiff + t_w_stiff * h_stiff + wt_spar_p * l_eo)