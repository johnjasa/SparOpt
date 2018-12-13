import numpy as np

from openmdao.api import Group

from A_str_stiff import AstrStiff
#from A_str_damp import AstrDamp
from A_str_ext import AstrExt
#from A_struct import Astruct
from B_str_ext import BstrExt
from B_struct import Bstruct
from C_struct_nf import Cstruct
from A_contrl import Acontrl
from B_contrl_nf import Bcontrl
from C_contrl import Ccontrl
from D_contrl_nf import Dcontrl
from Bs_Dc_Cs import BsDcCs
from Bs_Cc import BsCc
from Bc_Cs import BcCs
#from A_feedbk import Afeedbk
from B_fb_ext import BfbExt
from B_feedbk import Bfeedbk
#from transfer_function import TransferFunction

class StateSpace(Group):

	def initialize(self):
		self.options.declare('freqs', types=dict)

	def setup(self):
		freqs = self.options['freqs']

		self.add_subsystem('A_str_stiff', AstrStiff(), promotes_inputs=['M_global', 'A_global', 'K_global'], promotes_outputs=['Astr_stiff'])
		
		#self.add_subsystem('A_str_damp', AstrDamp(), promotes_inputs=['M_global', 'A_global', 'B_global'], promotes_outputs=['Astr_damp'])
		
		self.add_subsystem('A_str_ext', AstrExt(), promotes_inputs=['M_global', 'A_global', 'dthrust_drotspeed', 'CoG_rotor'], promotes_outputs=['Astr_ext'])

		#self.add_subsystem('A_struct', Astruct(), promotes_inputs=['CoG_rotor', 'I_d', 'dtorque_dv', 'dtorque_drotspeed', 'Astr_stiff', 'Astr_damp', 'Astr_ext'], promotes_outputs=['A_struct'])

		self.add_subsystem('B_str_ext', BstrExt(), promotes_inputs=['M_global', 'A_global', 'dthrust_dbldpitch', 'CoG_rotor'], promotes_outputs=['Bstr_ext'])

		self.add_subsystem('B_struct', Bstruct(), promotes_inputs=['Bstr_ext', 'I_d', 'dtorque_dbldpitch'], promotes_outputs=['B_struct'])

		#self.add_subsystem('C_struct', Cstruct(), promotes_outputs=['C_struct'])
		self.add_subsystem('C_struct', Cstruct(), promotes_inputs=['CoG_rotor'], promotes_outputs=['C_struct'])

		self.add_subsystem('A_contrl', Acontrl(), promotes_inputs=['omega_lowpass'], promotes_outputs=['A_contrl'])

		self.add_subsystem('B_contrl', Bcontrl(), promotes_inputs=['omega_lowpass'], promotes_outputs=['B_contrl'])

		self.add_subsystem('C_contrl', Ccontrl(), promotes_inputs=['windspeed_0', 'rotspeed_0', 'k_i', 'k_p', 'gain_corr_factor'], promotes_outputs=['C_contrl'])

		self.add_subsystem('D_contrl', Dcontrl(), promotes_inputs=['k_t'], promotes_outputs=['D_contrl'])

		self.add_subsystem('Bs_Dc_Cs', BsDcCs(), promotes_inputs=['B_struct', 'D_contrl', 'C_struct'], promotes_outputs=['BsDcCs'])

		self.add_subsystem('Bs_Cc', BsCc(), promotes_inputs=['B_struct', 'C_contrl'], promotes_outputs=['BsCc'])

		self.add_subsystem('Bc_Cs', BcCs(), promotes_inputs=['B_contrl', 'C_struct'], promotes_outputs=['BcCs'])

		#self.add_subsystem('A_feedbk', Afeedbk(), promotes_inputs=['A_struct', 'A_contrl', 'BsCc', 'BcCs'], promotes_outputs=['A_feedbk'])

		self.add_subsystem('B_fb_ext', BfbExt(), promotes_inputs=['M_global', 'A_global', 'CoG_rotor', 'dthrust_dv', 'dmoment_dv', 'x_d_towertop'], promotes_outputs=['Bfb_ext'])

		self.add_subsystem('B_feedbk', Bfeedbk(), promotes_inputs=['Bfb_ext', 'I_d', 'dtorque_dv'], promotes_outputs=['B_feedbk'])

		#self.add_subsystem('transfer_function', TransferFunction(freqs=freqs), promotes_inputs=['A_feedbk', 'B_feedbk'], promotes_outputs=['Re_H_feedbk', 'Im_H_feedbk'])