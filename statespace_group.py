import numpy as np

from openmdao.api import Group

from A_str_stiff import AstrStiff
from A_str_ext import AstrExt
from B_str_ext import BstrExt
from B_struct import Bstruct
from C_struct_nf import Cstruct
from A_contrl_notch import Acontrl
from B_contrl_nf_notch import Bcontrl
from C_contrl_notch import Ccontrl
from Bs_Cc import BsCc
from Bc_Cs import BcCs
from B_fb_ext import BfbExt
from B_feedbk import Bfeedbk

class StateSpace(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('A_str_stiff', AstrStiff(), promotes_inputs=['M_global', 'A_global', 'K_global'], promotes_outputs=['Astr_stiff'])
		
		self.add_subsystem('A_str_ext', AstrExt(), promotes_inputs=['M_global', 'A_global', 'dthrust_drotspeed', 'CoG_rotor'], promotes_outputs=['Astr_ext'])

		self.add_subsystem('B_str_ext', BstrExt(), promotes_inputs=['M_global', 'A_global', 'dthrust_dbldpitch', 'CoG_rotor'], promotes_outputs=['Bstr_ext'])

		self.add_subsystem('B_struct', Bstruct(), promotes_inputs=['Bstr_ext', 'I_d', 'dtorque_dbldpitch'], promotes_outputs=['B_struct'])

		self.add_subsystem('C_struct', Cstruct(), promotes_inputs=['CoG_rotor'], promotes_outputs=['C_struct'])

		self.add_subsystem('A_contrl', Acontrl(), promotes_inputs=['omega_lowpass', 'omega_notch', 'bandwidth_notch'], promotes_outputs=['A_contrl'])

		self.add_subsystem('B_contrl', Bcontrl(), promotes_inputs=['omega_lowpass', 'k_t'], promotes_outputs=['B_contrl'])

		self.add_subsystem('C_contrl', Ccontrl(), promotes_inputs=['windspeed_0', 'rotspeed_0', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['C_contrl'])

		self.add_subsystem('Bs_Cc', BsCc(), promotes_inputs=['B_struct', 'C_contrl'], promotes_outputs=['BsCc'])

		self.add_subsystem('Bc_Cs', BcCs(), promotes_inputs=['B_contrl', 'C_struct'], promotes_outputs=['BcCs'])

		self.add_subsystem('B_fb_ext', BfbExt(), promotes_inputs=['M_global', 'A_global', 'CoG_rotor', 'dthrust_dv', 'dmoment_dv', 'x_d_towertop'], promotes_outputs=['Bfb_ext'])

		self.add_subsystem('B_feedbk', Bfeedbk(), promotes_inputs=['Bfb_ext', 'I_d', 'dtorque_dv'], promotes_outputs=['B_feedbk'])