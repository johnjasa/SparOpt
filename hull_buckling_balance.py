import numpy as np

from openmdao.api import Group, BalanceComp

from shell_hull_buckling_group import ShellHullBuckling
from hoop_stress_hull_buckling_group import HoopStressHullBuckling
from mom_inertia_hull_buckling_group import MomInertiaHullBuckling

class HullBalance(Group):

	def setup(self):

		shell_hull_buckling_group = ShellHullBuckling()
		hoop_stress_hull_buckling_group = HoopStressHullBuckling()
		mom_inertia_hull_buckling_group = MomInertiaHullBuckling()

		self.add_subsystem('hull_shell_buckling', shell_hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', \
		'M_moor', 'z_moor', 'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'A_R'], \
		promotes_outputs=['shell_buckling'])

		self.add_subsystem('hull_constr_hoop_stress', hoop_stress_hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', \
		'M_moor', 'z_moor', 'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'A_R'], \
		promotes_outputs=['constr_hoop_stress'])

		self.add_subsystem('hull_constr_mom_inertia_ringstiff', mom_inertia_hull_buckling_group, promotes_inputs=['D_spar_p', 'wt_spar_p', 'Z_spar', 'M_spar', 'M_ball', 'L_ball', 'spar_draft', \
		'M_moor', 'z_moor', 'dthrust_dv', 'dmoment_dv', 't_w_stiff', 't_f_stiff', 'h_stiff', 'b_stiff', 'l_stiff', 'angle_hull', 'f_y', 'A_R'], \
		promotes_outputs=['constr_mom_inertia_ringstiff', 'r_f'])
		
		bal1 = self.add_subsystem('balance_shell_buckling', BalanceComp(), promotes_outputs=['My_shell_buckling'])
		bal1.add_balance('My_shell_buckling', val=-np.ones(10), units='N*m')

		bal2 = self.add_subsystem('balance_constr_hoop_stress', BalanceComp(), promotes_outputs=['My_constr_hoop_stress'])
		bal2.add_balance('My_constr_hoop_stress', val=-np.ones(10), units='N*m')

		bal3 = self.add_subsystem('balance_constr_mom_inertia_ringstiff', BalanceComp(), promotes_outputs=['My_constr_mom_inertia_ringstiff'])
		bal3.add_balance('My_constr_mom_inertia_ringstiff', val=-np.ones(10), units='N*m')

		self.connect('My_shell_buckling', 'hull_shell_buckling.My_hull')
		self.connect('shell_buckling', 'balance_shell_buckling.lhs:My_shell_buckling')

		self.connect('My_constr_hoop_stress', 'hull_constr_hoop_stress.My_hull')
		self.connect('constr_hoop_stress', 'balance_constr_hoop_stress.lhs:My_constr_hoop_stress')

		self.connect('My_constr_mom_inertia_ringstiff', 'hull_constr_mom_inertia_ringstiff.My_hull')
		self.connect('constr_mom_inertia_ringstiff', 'balance_constr_mom_inertia_ringstiff.lhs:My_constr_mom_inertia_ringstiff')