import numpy as np
import casadi
import matplotlib.pyplot as plt
from matplotlib import animation
from plan_dubins import plan_dubins_path
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib as mpl
mpl.rcParams['font.size'] = 14
mpl.rcParams['text.usetex'] = True

def dm_to_array(dm):
        return np.array(dm.full())

def animate(anim_params):
    n_agents = anim_params['n_agents']
    ref_state_list = anim_params['ref_state_list']
    agents_init_state = anim_params['agents_init_state']
    agents_state_list = anim_params['agents_state_list']
    agents_control_list = anim_params['agents_control_list'] 
    num_frames = anim_params['num_frames']
    max_iter = anim_params['max_iter']
    pred_horizon = anim_params['pred_horizon'] 
    save = anim_params['save'] 
    obs_list = anim_params['obs_list']
    

    def create_triangle(state=[0,0,0], h=1.2, w=0.75, update=False):
        x, y, th = state
        triangle = np.array([
            [h, 0   ],
            [0,  w/2],
            [0, -w/2],
            [h, 0   ]
        ]).T
        rotation_matrix = np.array([
            [np.cos(th), -np.sin(th)],
            [np.sin(th),  np.cos(th)]
        ])
        coords = np.array([[x, y]]) + (rotation_matrix @ triangle).T
        if update == True:
            return coords
        else:
            return coords[:3, :]

    # Function to create a gradient-filled circle
    def radial_gradient_circle(ax, center_x, center_y, radius, colormap='viridis'):
        """
        Creates a radial gradient circle.
        """
        # Create a meshgrid for the circle
        x, y = np.meshgrid(np.linspace(center_x - radius, center_x + radius, 100),
                        np.linspace(center_y - radius, center_y + radius, 100))
        # Calculate the distance from the center for each point
        r = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
        # Normalize the distance to be between 0 and 1
        r = np.clip(r, 0, radius) / radius
        # Create a colormap
        cmap = plt.get_cmap(colormap).reversed()
        # Map the distance to the colormap
        colors = cmap(r)
        # Plot the circle
        ax.imshow(colors, extent=[center_x - radius, center_x + radius, center_y - radius, center_y + radius], alpha=0.1)
        # Set aspect to 'equal' to ensure the circle looks circular
        ax.set_aspect('equal')
    
    def plot_circle(ax, x, y, obs_r, color="-b"):
        circle = plt.Circle((x, y), obs_r, color='orange')
        ax.add_patch(circle)


    def init():
        return path_list, horizon_list
    
    def animate(i):
        ax.clear()
        for k in range(n_agents):
            # Plot Track
            plt.plot([0, 10], [5, 5], 'k--', dashes=(5, 5), lw = 4, alpha = 0.1)
            plt.plot([5, 5], [0, 10], 'k--', dashes=(5, 5), lw = 4, alpha = 0.1)
            plt.plot([0, 10], [0, 10], 'k--', dashes=(5, 5), lw = 4, alpha = 0.1)
            plt.plot([10, 0], [0, 10], 'k--', dashes=(5, 5), lw = 4, alpha = 0.1)
            ax.add_patch(Rectangle((-1.0,4.0),12,2,linewidth=1,facecolor='k', alpha=0.03))
            ax.add_patch(Rectangle((4.0,-1.0),2,12,linewidth=1,facecolor='k', alpha=0.03))
            ax.add_patch(Rectangle((0.5,-1.0),16,2,linewidth=1, angle = 45, facecolor='k', alpha=0.03))
            ax.add_patch(Rectangle((0.0,11.5),2,16,linewidth=1,angle=-135, facecolor='k', alpha=0.03))
            
            # Plot State
            x = agents_state_list[k][0, 0, i]
            y = agents_state_list[k][1, 0, i]
            th = agents_state_list[k][2, 0, i]
            current_triangle = create_triangle([x, y, th],h =1.0, w= 0.75)
            current_state = ax.fill(current_triangle[:, 0], current_triangle[:, 1], color = colors[k])
            current_state = current_state[0]
            
            # Plot Horizon
            x_new = agents_state_list[k][0, :, i]
            y_new = agents_state_list[k][1, :, i]
            horizon, = ax.plot(x_new, y_new, 'x-g', alpha=1)

            # Plot Refererence Traj
            x_ref = ref_state_list[k][:,0]
            y_ref = ref_state_list[k][:,1]
            horizon, = ax.plot(x_ref, y_ref, '--', color = colors[k], alpha=1)


            # Plot Obstacles 
            for (ox, oy) in obs_list[i]:
                plot_circle(ax, ox, oy, 0.3)
            
            # Plot Safety
            radial_gradient_circle(ax, x, y, radius=1.0, colormap='Reds')
            if k==31:
                plt.savefig('autotaxiing_case1_mission.png', dpi=300)
            
       
        legend_elements = [Line2D([0], [0], marker='>', color=colors[i], markerfacecolor=colors[i], markersize=15, label=f'Aircraft {i+1}') 
                           for i in range(n_agents)]
        legend_elements +=[Line2D([0], [0], marker='x',color='g', markerfacecolor='g', markersize=15,label='MPC Predicted Path',)]
        legend_elements +=[Line2D([0], [0], marker='o',color='orange', markerfacecolor='orange', markersize=15,label='Unplanned Obstacle',)]


        ax.legend(handles=legend_elements, loc='upper right', ncol=3,  fontsize = 10)   

        ax.set_xlim(-2,12)
        ax.set_ylim(-2,12)
        ax.set_xlabel('x position', fontsize =12)
        ax.set_ylabel('y position', fontsize =12)       

        plt.tight_layout()

        return path, horizon

    # create figure and axes
    fig, ax = plt.subplots(figsize=(6, 6))
    path_list = []
    ref_path_list = []
    horizon_list = []
    colors = ['b', 'r', 'g', 'k']
    for k in range(n_agents):
        path, = ax.plot([], [], 'r', linewidth=2)
        ref_path, = ax.plot([], [], 'b', linewidth=2)
        horizon, = ax.plot([], [], 'x-g', alpha=0.5)
        path_list.append(path)
        ref_path_list.append(ref_path)
        horizon_list.append(horizon)

    sim = animation.FuncAnimation(
        fig=fig,
        func = animate,
        init_func=init,
        frames=num_frames,
        interval=100,
        blit=False,
        repeat=False
    )
    # plt.show()
    # if save == True:
    #     sim.save(anim_params['file_name'], writer='ffmpeg', fps=10)
    return sim


class MPC_CBF_Bicycle:
    def __init__(self, init_state, n_neighbors, dt ,N, a_lim, delta_lim, L,  
                 Q, R, cbf_const, 
                 obstacles= None,  obs_diam = 0.5, robot_diam = 0.5, alpha=0.1):
        
        self.dt = dt # Period
        self.N = N  # Horizon Length 
        self.L = L # Length of the bicycle
        self.n_neighbors = n_neighbors
        self.Q_x = Q[0]
        self.Q_y = Q[1]
        self.Q_theta = Q[2]
        self.Q_v = Q[3]
        self.R_a = R[0]
        self.R_delta = R[1]
        self.n_states = 0
        self.n_controls = 0

        self.a_lim = a_lim
        self.delta_lim = delta_lim

        # Initialized in mpc_setup
        self.solver = None
        self.f = None
        self.states = init_state

        self.robot_diam = robot_diam
        self.obs_diam = obs_diam
        self.cbf_const = cbf_const # Bool flag to enable obstacle avoidance
        self.alpha= alpha # Parameter for scalar class-K function, must be positive
        self.obstacles = obstacles
        self.n_obstacles = len(obstacles)
        # Setup with initialization params
        self.setup()

    ## Utilies used in MPC optimization
    # CBF Implementation
    def h_obs(self, state, obstacle, r):
        return ((obstacle[0] - state[0])**2 + (obstacle[1] - state[1])**2 - r**2)


    def shift_timestep(self, h, time, state, control):
        delta_state = self.f(state, control[:, 0])
        next_state = casadi.DM.full(state + h * delta_state)
        next_time = time + h
        next_control = casadi.horzcat(control[:, 1:],
                                    casadi.reshape(control[:, -1], -1, 1))
        self.states = np.array(next_state)[:,0]
        return next_time, next_state, next_control

    def update_param(self, x0, ref, k, N, nb_states, obstacles):
        p = casadi.vertcat(x0)
        # Reference trajectory as parameter
        for l in range(N):
            if k+l < ref.shape[0]:
                ref_state = ref[k+l, :]
            else:
                ref_state = ref[-1, :]
            xt = casadi.DM([ref_state[0], ref_state[1], ref_state[2], ref_state[3]])
            p = casadi.vertcat(p, xt)
        
        # Neigbouring robots states as parameter
        for i in range(self.n_neighbors):
            nb_state = casadi.DM([nb_states[i,0], nb_states[i,1], nb_states[i,2], nb_states[i,3]])
            p = casadi.vertcat(p, nb_state)

        for i in range(len(obstacles)):
            obs = casadi.DM([obstacles[i][0], obstacles[i][1]])
            p = casadi.vertcat(p, obs)    
        
        return p
    
    def setup(self):
        x = casadi.SX.sym('x')
        y = casadi.SX.sym('y')
        theta = casadi.SX.sym('theta')
        v = casadi.SX.sym('v')

        states = casadi.vertcat(x, y, theta, v)
        self.n_states = states.numel()

        a = casadi.SX.sym('a')
        delta = casadi.SX.sym('delta')
        controls = casadi.vertcat(a, delta)
        self.n_controls = controls.numel()

        X = casadi.SX.sym('X', self.n_states, self.N + 1)
        U = casadi.SX.sym('U', self.n_controls, self.N)

        # Reference trajectory + Neigboring robots's state + Target Bearing
        P = casadi.SX.sym('P', (self.N + 1) * self.n_states + self.n_neighbors*self.n_states + self.n_obstacles*2)
        
        Q = casadi.diagcat(self.Q_x, self.Q_y, self.Q_theta, self.Q_v)
        R = casadi.diagcat(self.R_a, self.R_delta)
        L = self.L
        #rhs = casadi.vertcat(v * casadi.cos(theta), v * casadi.sin(theta), omega)
        rhs = casadi.vertcat(
            v*casadi.cos(theta),
            v*casadi.sin(theta),
            v/L*casadi.tan(delta),
            a
        )    
        self.f = casadi.Function('f', [states, controls], [rhs])
        cost = 0
        g = X[:, 0] - P[:self.n_states]
        for k in range(self.N):
            state = X[:, k]
            control = U[:, k]
            ref = P[(k+1)*self.n_states:(k+2)*self.n_states]
            track_err = (state - ref)
            track_err[2] = casadi.fmod(track_err[2]+np.pi,(2*np.pi)) - np.pi
            track_cost = track_err.T @ Q @ track_err
            ctrl_cost = control.T @ R @ control 
            cost = cost + track_cost + ctrl_cost 

            next_state = X[:, k + 1]
            k_1 = self.f(state, control)
            k_2 = self.f(state + self.dt/2 * k_1, control)
            k_3 = self.f(state + self.dt/2 * k_2, control)
            k_4 = self.f(state + self.dt * k_3, control)
            predicted_state = state + self.dt/6 * (k_1 + 2 * k_2 + 2 * k_3 + k_4)
            g = casadi.vertcat(g, next_state - predicted_state)
        
       
        if self.cbf_const:
            for k in range(self.N):
                state = X[:, k]
                next_state = X[:, k+1]
                for j in range(self.n_neighbors):
                    nb_state = P[(self.N+1)*self.n_states + j*self.n_states: (self.N+1)*self.n_states + (j+1)*self.n_states] 
                    nb_pos = nb_state[:2] 
                    h = self.h_obs(state, (nb_pos[0], nb_pos[1]), self.robot_diam)
                    h_next = self.h_obs(next_state, (nb_pos[0], nb_pos[1]), self.robot_diam)
                    g = casadi.vertcat(g,-(h_next-h + self.alpha*h))
                for j in range(self.n_obstacles):  
                    obs = P[(self.N+1+self.n_neighbors)*self.n_states + j*2: (self.N+1+self.n_neighbors)*self.n_states + (j+1)*2] 
                    h = self.h_obs(state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    h_next = self.h_obs(next_state, obs, (self.robot_diam / 2 + self.obs_diam / 2))
                    g = casadi.vertcat(g,-(h_next-h + self.alpha*h))    
                    

        opt_variables = casadi.vertcat(X.reshape((-1, 1)), U.reshape((-1, 1)))

        nlp_prob = {
            'f': cost,
            'x': opt_variables,
            'g': g,
            'p': P
        }

        opts = {
            'ipopt': {
                'sb': 'yes',
                'max_iter': 10,
                'print_level': 0,
                'acceptable_tol': 1e-3,
                'acceptable_obj_change_tol': 1e-3
            },
            'print_time': 0,
        }
        self.solver = casadi.nlpsol('solver', 'ipopt', nlp_prob, opts)


    def solve(self, X0, u0, ref, idx, nb_states, obstacles):  
        lbx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))
        ubx = casadi.DM.zeros((self.n_states * (self.N + 1) + self.n_controls * self.N, 1))

        lbx[0:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[1:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[2:self.n_states * (self.N + 1):self.n_states] = -casadi.inf
        lbx[3:self.n_states * (self.N + 1):self.n_states] = -casadi.inf

        ubx[0:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[1:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[2:self.n_states * (self.N + 1):self.n_states] = casadi.inf
        ubx[3:self.n_states * (self.N + 1):self.n_states] = casadi.inf

        lbx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.a_lim[0]
        ubx[self.n_states * (self.N + 1):self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.a_lim[1]
        lbx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls *self.N:self.n_controls] = self.delta_lim[0]
        ubx[self.n_states * (self.N + 1) + 1:self.n_states * (self.N + 1) + self.n_controls * self.N:self.n_controls] = self.delta_lim[1]
        
        if self.cbf_const:
            lbg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N) + self.n_neighbors * self.N, 1))
            ubg = casadi.DM.zeros((self.n_states * (self.N + 1) + len(self.obstacles)*(self.N) + self.n_neighbors * self.N, 1))

            lbg[self.n_states * (self.N + 1):] = -casadi.inf
            ubg[self.n_states * (self.N + 1):] = 0
        else:
            lbg = casadi.DM.zeros((self.n_states * (self.N+1)))
            ubg = -casadi.DM.zeros((self.n_states * (self.N+1)))
           
         
        args = {
            'lbg': lbg,
            'ubg': ubg,
            'lbx': lbx,
            'ubx': ubx
        }
        args['p'] = self.update_param(X0[:,0], ref, idx, self.N, nb_states, obstacles)
        args['x0'] = casadi.vertcat(casadi.reshape(X0, self.n_states * (self.N + 1), 1),
                                        casadi.reshape(u0, self.n_controls * self.N, 1))

        sol = self.solver(x0=args['x0'], lbx=args['lbx'], ubx=args['ubx'],
                        lbg=args['lbg'], ubg=args['ubg'], p=args['p'])

        u = casadi.reshape(sol['x'][self.n_states * (self.N + 1):], self.n_controls, self.N)
        X = casadi.reshape(sol['x'][:self.n_states * (self.N + 1)], self.n_states, self.N + 1)
        
        return u, X 
        

def main(args=None):

    # Consider all homogenous agents with identical parameters
    Q_x = 10
    Q_y = 10
    Q_theta = 1
    Q_v = 0.00
    R_a = 0.1
    R_delta = 0.01

    dt = 0.1
    N = 30

    L = 1
    a_lim = [-1, 1]
    delta_lim = [-casadi.pi/4, casadi.pi/4]
    Q = [Q_x, Q_y, Q_theta, Q_v]
    R = [R_a, R_delta]

    n_agents = 4
   

    agents_init_state = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [4.0, 4.0, -np.pi/2, 0.0]])
    agents_goal_state = np.array([[2.0, 2.0, np.pi/4, 0.0], [4.0, 1.0, np.pi/4, 0.0], [4.0, 4.0, np.pi/4, 0.0], [2.0, 4.0, -3*np.pi/4, 0.0]])       

    obs_list = [(1, 2, 1, 0.5), (3, 2, 1, 0.5), (5, 2, 1, 0.5), (1, 3, 1, 0.5), (3, 4, 1, 0.5), (6, 4, 1, 0.5), (1, 5, 4, 2)]


    t0_list = [0 for i in range(n_agents)]
    agents = [MPC_CBF_Bicycle(agents_init_state[i], n_agents-1, dt, N, a_lim, delta_lim, L, Q, R, obstacles= obs_list, cbf_const=True) for i in range(n_agents)]
    state_0_list = [casadi.DM([agents_init_state[i][0], agents_init_state[i][1], agents_init_state[i][2], agents_init_state[i][3]]) for i in range(n_agents)]
    u0_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X0_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]

    u_list = [casadi.DM.zeros((agents[i].n_controls, N)) for i in range(n_agents)]
    X_pred_list = [casadi.repmat(state_0_list[i], 1, N + 1) for i in range(n_agents)]
    agents_state_list = [dm_to_array(X0_list[i]) for i in range(n_agents)]
    agents_control_list = [dm_to_array(u0_list[i][:, 0]) for i in range(n_agents)]
    ref_state_list = [np.array([[agents_goal_state[j][0]], [agents_goal_state[j][1]], [agents_goal_state[j][2]], [agents_goal_state[j][3]]]).T for j in range(n_agents)]
    max_iter = 200

    for t in range(max_iter):
        print(t)
        # Construct a list of neighbor states for all robots
        agent_states = np.array([agents[i].states for i in range(n_agents)])
        for j in range(n_agents):
            neighbor_states = []
            for k,agent_state in enumerate(agent_states):
                if k !=j :
                    neighbor_states.append(agent_state)
            neighbor_states = np.array(neighbor_states)
            u_list[j], X_pred_list[j] = agents[j].solve(X0_list[j], u0_list[j], ref_state_list[j], t, neighbor_states)
        
        for j in range(n_agents):
            agents_state_list[j] = np.dstack((agents_state_list[j], dm_to_array(X_pred_list[j])))
            agents_control_list[j] = np.dstack((agents_control_list[j], dm_to_array(u_list[j][:, 0])))
            t0_list[j], X0_list[j], u0_list[j] = agents[j].shift_timestep(dt, t0_list[j], X_pred_list[j], u_list[j])

    anim_params = {
        'ref_state_list': ref_state_list,
        'agents_init_state':agents_init_state,
        'agents_state_list':agents_state_list,
        'agents_control_list':agents_control_list,
        'obstacles': None,
        'num_frames':max_iter,
        'max_iter':max_iter,
        'pred_horizon':N,
        'save': True,
        'file_name':'collision_avoid.mp4',
        'obs_list': obs_list,

    }
    sim = animate(anim_params)
    
if __name__ == '__main__':
    main()