class Problem(parapint.interfaces.DynamicSchwarzInteriorPointInterface):
    def __init__(self,
                 t0=0,
                 delta_t=1,
                 num_finite_elements=90,
                 constant_control_duration=10,
                 time_scale=0.1,
                 num_time_blocks=9,
                 num_partitions=3,
                 overlap = 1):
        assert num_finite_elements % num_time_blocks == 0
        self.t0: int = t0
        self.delta_t: int = delta_t
        self.num_finite_elements: int = num_finite_elements
        self.constant_control_duration: int = constant_control_duration
        self.time_scale: float = time_scale
        self.num_time_blocks: int = num_time_blocks
        self.num_partitions:int = num_partitions
        self.overlap = overlap

        self.tf = self.t0 + self.delta_t*self.num_finite_elements
        super(Problem, self).__init__(start_t=self.t0,end_t=self.tf,num_time_blocks=self.num_time_blocks,comm=MPI.COMM_WORLD)

    def build_model_for_time_block(self, ndx: int,
                                   start_t: float,
                                   end_t: float,
                                   add_init_conditions: bool) -> Tuple[_BlockData,
                                                                       Sequence[_GeneralVarData],
                                                                       Sequence[_GeneralVarData]]:
        assert int(start_t) == start_t
        assert int(end_t) == end_t
        assert end_t == start_t + self.delta_t*(self.num_finite_elements/self.num_time_blocks)
        start_t = int(start_t)
        end_t = int(end_t)
        m = build_time_block(t0=start_t,
                             delta_t=self.delta_t,
                             num_finite_elements=int(self.num_finite_elements/self.num_time_blocks),
                             constant_control_duration=self.constant_control_duration,
                             time_scale=self.time_scale)
        start_states = [m.x[start_t]]
        end_states = [m.x[end_t]]
        return m, start_states, end_states

def main(subproblem_solver_class, subproblem_solver_options, show_plot=True):
    t0: int = 0
    delta_t: int = 1
    num_finite_elements: int = 90
    constant_control_duration: int = 10
    time_scale: float = 0.1
    num_time_blocks: int = 3
    num_partitions: int = 3
    overlap:int = 1
    interface = Problem(t0=t0,
                        delta_t=delta_t,
                        num_finite_elements=num_finite_elements,
                        constant_control_duration=constant_control_duration,
                        time_scale=time_scale,
                        num_time_blocks=num_time_blocks,
                        num_partitions=num_partitions,
                        overlap = overlap)

    linear_solver = parapint.linalg.SchwarzLinearSolver(subproblem_solvers={ndx: subproblem_solver_class(**subproblem_solver_options) for ndx in range(num_partitions)},
    iterator=subproblem_solver_class(**subproblem_solver_options))

    opt = parapint.algorithms.InteriorPointSolver(linear_solver)
    status = opt.solve(interface)
    assert status == parapint.algorithms.InteriorPointStatus.optimal
    interface.load_primals_into_pyomo_model()
