import numpy as np
import parking_model as pm
import os

class TileCoder:
    def __init__(self, num_tilings=8, x_range=(-13.0, 13.0), y_range=(-2.0, 9.0), alpha_range=(-np.pi, np.pi), x_bins=35, y_bins=20, alpha_bins=20):
        self.num_tilings = num_tilings
        self.x_range = x_range
        self.y_range = y_range
        self.alpha_range = alpha_range
        
        self.x_bins = x_bins
        self.y_bins = y_bins
        self.alpha_bins = alpha_bins
        
        # Calculate tile width for each dimension
        self.x_width = (x_range[1] - x_range[0]) / (x_bins - 1)
        self.y_width = (y_range[1] - y_range[0]) / (y_bins - 1)
        self.alpha_width = (alpha_range[1] - alpha_range[0]) / (alpha_bins - 1)
        
        # Define offsets for each tiling
        self.offsets_x = np.linspace(0, self.x_width, num_tilings, endpoint=False)
        self.offsets_y = np.linspace(0, self.y_width, num_tilings, endpoint=False)
        self.offsets_alpha = np.linspace(0, self.alpha_width, num_tilings, endpoint=False)
        
        self.tiles_per_tiling = x_bins * y_bins * alpha_bins
        self.total_tiles = num_tilings * self.tiles_per_tiling

    def get_tiles(self, state):
        x, y, alpha = state
        
        # Wrap alpha to [-pi, pi]
        alpha = (alpha + np.pi) % (2 * np.pi) - np.pi
        
        active_tiles = []
        for i in range(self.num_tilings):
            ox = x - self.x_range[0] + self.offsets_x[i]
            oy = y - self.y_range[0] + self.offsets_y[i]
            oalpha = alpha - self.alpha_range[0] + self.offsets_alpha[i]
            
            idx_x = int(ox / self.x_width)
            idx_y = int(oy / self.y_width)
            idx_alpha = int(oalpha / self.alpha_width)
            
            idx_x = max(0, min(idx_x, self.x_bins - 1))
            idx_y = max(0, min(idx_y, self.y_bins - 1))
            idx_alpha = max(0, min(idx_alpha, self.alpha_bins - 1))
            
            tiling_idx = (idx_x * self.y_bins * self.alpha_bins) + (idx_y * self.alpha_bins) + idx_alpha
            global_idx = i * self.tiles_per_tiling + tiling_idx
            active_tiles.append(global_idx)
            
        return active_tiles

tc_global = None
w_global = None
actions_global = [
    (0.0, 0.0),
    (-2.0, -np.pi/4),
    (-2.0, 0.0),
    (-2.0, np.pi/4),
    (2.0, -np.pi/4),
    (2.0, 0.0),
    (2.0, np.pi/4)
]

def reward(param_phis, state, if_collision, if_stop):
    value = 0
    x = state[0]
    y = state[1]
    alfa = state[2]
    dist_xy_sq = x*x + (y + 0.35)**2
    alfa_zred = 0
    if param_phis.if_side_parking_place:
        if np.abs(alfa) > np.pi / 2:
            alfa_zred = np.pi - np.abs(alfa)
        else:
            alfa_zred = np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)

    alfa_zred = alfa_zred/(dist_xy_sq+0.5)

    dist_eval = 1/(dist_xy_sq+0.5)-1
    angle_eval = 0.5 - alfa_zred

    # if V==0 reward is calculated based on a distance
    
    if if_collision:
        value = -1
    elif if_stop:
        value = min(dist_eval,angle_eval)
    else:
        value = 0

    return value

def calc_dist_sq(state):
    dx = state[0]
    dy = state[1] + 0.35
    return dx*dx + dy*dy

def is_parked(state, param_phis):
    x, y, alfa = state
    alfa = (alfa + np.pi) % (2 * np.pi) - np.pi
    if param_phis.if_side_parking_place:
        alfa_zred = np.pi - np.abs(alfa) if np.abs(alfa) > np.pi / 2 else np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)
    return (np.abs(x) < 0.8) and (y < 0.50) and (y > -0.7) and (alfa_zred < 0.25)

def choose_action(param_phis, state):
    global tc_global, w_global, actions_global
    if w_global is None:
        tc_global = TileCoder()
        if os.path.exists('weights.npy'):
            w_global = np.load('weights.npy')
        else:
            w_global = np.zeros((tc_global.total_tiles, len(actions_global)))
            
    tiles = tc_global.get_tiles(state)
    q_vals = np.sum(w_global[tiles, :], axis=0)
    
    if not is_parked(state, param_phis):
        q_vals[0] = -9999.0
        
    act_idx = np.argmax(q_vals)
    V, angle = actions_global[act_idx]
    return angle, V

def park_test(param_phis, initial_state):
    pm.park_save("param.txt", param_phis)
    phist = open('history.txt', 'w')
    num_of_initial_states, lparam = initial_state.shape
    avg_sum_of_rewards = 0
    num_of_steps = 0 
    for episode in range(num_of_initial_states):
        # We choose the starting state:
        init_state_no = episode %  num_of_initial_states
        state = initial_state[init_state_no,:]

        step = 0
        if_collision = False
        if_stop = False
        sum_of_rewards_in_episode = 0
        while if_stop == False:
            step = step + 1

            # We determine actions a (angle + direction of motion) in the state state according to the learned strategy:
            angle, V = choose_action(param_phis,state)
            
            # saveing the step of history :
            phist.write("%d %d %.4f %.4f %.4f %.4f %.4f\n" % ((episode + 1),step,state[0],state[1],state[2],angle,V))
            # new state determination:
            new_state, rotation_center, if_collision = pm.model_of_car(state, angle, V, param_phis)

            if (if_collision)|(step >= param_phis.max_number_of_steps)|(V == 0.0):
                if_stop = True

            R = reward(param_phis, new_state, if_collision, if_stop)
            sum_of_rewards_in_episode += R

            state = new_state

        avg_sum_of_rewards = avg_sum_of_rewards + sum_of_rewards_in_episode / num_of_initial_states
        num_of_steps = num_of_steps + step
        print("in %d episode sum of rewards = %g, num of steps = %d" %(episode, sum_of_rewards_in_episode, step))

    print("average sum of rewards in episode = %g" % (avg_sum_of_rewards))
    print("average number of steps = %g" % (num_of_steps/num_of_initial_states))
    phist.close()

def get_potential(param_phis, state):
    x, y, alfa = state
    
    # Target center is (0, -0.35)
    dx = x
    dy = y + 0.35
    dist = np.sqrt(dx*dx + dy*dy)
    
    alfa = (alfa + np.pi) % (2 * np.pi) - np.pi
    if param_phis.if_side_parking_place:
        alfa_zred = np.pi - np.abs(alfa) if np.abs(alfa) > np.pi / 2 else np.abs(alfa)
    else:
        alfa_zred = np.abs(np.abs(alfa) - np.pi / 2)
        
    pot_dist = max(0.0, 1.0 - (dist / 15.0))
    pot_angle = max(0.0, 1.0 - (alfa_zred / (np.pi / 2)))
    
    return 350.0 * pot_dist + 250.0 * pot_dist * pot_angle

# training procedure proper for reinforcement learning with approximation of action values:
def park_train():
    global w_global, tc_global
    liczba_epizodow = 60000
    
    initial_state = np.array([[9.1, 4.6, 0],[6.3, 5.06, 0],[9.6, 3.15, 0],[7.3, 5.75, 0],[10.1, 6.21, 0]],dtype=float)
    num_of_initial_states, lparam = initial_state.shape

    param_phis = pm.GlobalVar()     # phisical parameters of a parking and a car

    tc = TileCoder()
    tc_global = tc
    
    num_actions = len(actions_global)
    if os.path.exists('weights.npy'):
        w = np.load('weights.npy')
        w_best = w.copy()
        best_test_reward = -0.282915
        epsilon = 0.2
        alpha_lr = 0.05
        print("Warm-starting training from weights.npy")
    else:
        w = np.zeros((tc.total_tiles, num_actions))
        w_best = np.zeros_like(w)
        best_test_reward = -999.0
        epsilon = 0.5
        alpha_lr = 0.2

    gamma = 0.99

    for episode in range(liczba_epizodow):
        # Initial state choosing:
        init_state_no = episode %  num_of_initial_states
        state = initial_state[init_state_no, :].copy()

        step = 0
        if_collision = False
        if_stop = False
        
        # Calculate initial potential
        pot = get_potential(param_phis, state)

        while if_stop == False:
            step = step + 1
            tiles = tc.get_tiles(state)

            is_stop_allowed = is_parked(state, param_phis)
            allowed_actions = list(range(num_actions)) if is_stop_allowed else list(range(1, num_actions))

            # We determine actions a
            if np.random.rand() < epsilon:
                act_idx = np.random.choice(allowed_actions)
            else:
                q_vals = np.sum(w[tiles, :], axis=0)
                if not is_stop_allowed:
                    q_vals[0] = -9999.0
                act_idx = np.argmax(q_vals)
                
            V, angle = actions_global[act_idx]

            # determination of a new state:
            new_state, rotation_center, if_collision = pm.model_of_car(state, angle, V, param_phis)

            if (if_collision)|(step >= param_phis.max_number_of_steps)|(V == 0.0):
                if_stop = True

            if if_stop:
                if if_collision:
                    R_orig = -10.0
                    target = R_orig - pot
                else:
                    R_orig = pot  # R_stop matches current potential
                    target = R_orig - pot  # which is 0.0

            else:
                R_orig = -0.05
                new_pot = get_potential(param_phis, new_state)
                F = gamma * new_pot - pot
                R_shaped = R_orig + F
                
                new_tiles = tc.get_tiles(new_state)
                q_next = np.sum(w[new_tiles, :], axis=0)
                
                if not is_parked(new_state, param_phis):
                    q_next[0] = -9999.0
                    
                target = R_shaped + gamma * np.max(q_next)

            q_curr = np.sum(w[tiles, act_idx])
            td_error = target - q_curr
            
            # We update the Q values:
            w[tiles, act_idx] += (alpha_lr / tc.num_tilings) * td_error

            if not if_stop:
                state = new_state
                pot = new_pot

        # Decay epsilon and learning rate
        epsilon = max(0.01, epsilon * 0.9999)
        alpha_lr = max(0.02, alpha_lr * 0.9999)

        # test with generating history to a file and early stopping evaluation:
        if (episode > 0) and (episode % 2000 == 0):
            # Evaluate current performance on the 5 initial states
            avg_rew = 0
            for i in range(num_of_initial_states):
                s = initial_state[i, :].copy()
                st = 0
                stopped = False
                coll = False
                while st < 100 and not stopped:
                    st += 1
                    t = tc.get_tiles(s)
                    q = np.sum(w[t, :], axis=0)
                    
                    if not is_parked(s, param_phis):
                        q[0] = -9999.0
                    a_opt = np.argmax(q)
                    V_opt, ang_opt = actions_global[a_opt]
                    s, _, coll = pm.model_of_car(s, ang_opt, V_opt, param_phis)
                    if coll or V_opt == 0.0:
                        stopped = True
                
                r = reward(param_phis, s, coll, True)
                avg_rew += r
            
            avg_rew /= num_of_initial_states
            print("episode %d, validation reward = %g" % (episode, avg_rew))
            
            # Save the best model
            if avg_rew > best_test_reward:
                best_test_reward = avg_rew
                w_best = w.copy()
                np.save('weights.npy', w_best)
                w_global = w_best

    # Save the final best weights and run test
    if best_test_reward > -999.0:
        w_global = w_best
        np.save('weights.npy', w_best)
    else:
        w_global = w
        np.save('weights.npy', w)
        
    print("Training finished! Best validation reward: %g" % best_test_reward)
    print("Generating history file...")
    park_test(param_phis, initial_state)

park_train()



