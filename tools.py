import numpy as np
import pandas as pd
from scipy.stats import norm
import random
import time
import matplotlib.pyplot as plt
import bspline
import bspline.splinelab as splinelab



def terminal_payoff(ST, K, option_type):
    if option_type == 'call':
        return max(ST-K, 0)
    elif option_type == 'put':
        return max(K-ST, 0)
    else:
        raise ValueError(f'Invalid option type {option_type}')
        

def get_stock_prices(S0, N_MC, T, n_steps, mu, sigma, r):
    dt = T / n_steps
    rand_num = pd.DataFrame(np.random.randn(N_MC,n_steps), index=range(1, N_MC+1), columns=range(1, n_steps+1))
    S = pd.DataFrame([], index=range(1, N_MC+1), columns=range(n_steps+1))
    S[0] = S0
    
    for t in range(1, n_steps+1):
        S[t] = S[t-1] * np.exp((mu - 1/2 * sigma**2) * dt + sigma * np.sqrt(dt) * rand_num[t])
    S = S.astype(float)
    delta_S = S.loc[:,1:n_steps].values - np.exp(r * dt) * S.loc[:,0:n_steps-1]
    delta_S_hat = delta_S - delta_S.mean(axis=0)
    X = - (mu - 1/2 * sigma**2) * np.arange(n_steps+1) * dt + np.log(S)
    return S, delta_S, delta_S_hat, X
    
    
        
def get_basis(X, p=4, ncolloc=12, plot=True):
    ## p=4: number of coefficients - this has 4 coef, therefore is cubic
    
    X_min = np.min(np.min(X))
    X_max = np.max(np.max(X))
    tau = np.linspace(X_min,X_max,ncolloc)  # collocation points - These are the sites to which we would like to interpolate (ie checkpoints)

    # k is a knot vector that adds endpoints repeats as appropriate for a spline of order p
    k = splinelab.aptknt(tau, p) 

    # Spline basis of order p on knots k
    basis = bspline.Bspline(k, p)   
    
    if plot:
        print(f'X.shape {X.shape}')
        print(f'X_min {X_min:.3f}')
        print(f'X_max {X_max:.3f}')
        basis.plot()
    return basis


def get_data_from_splines(X, basis, n_steps, N_MC, num_basis=12):
    ## num_basis =  ncolloc
    data_mat_t = np.zeros((n_steps+1, N_MC, num_basis))
    print('num_basis', num_basis)
    print('dim data_mat_t', data_mat_t.shape)
    t_0 = time.time()
    for i in range(n_steps+1):
        x = X.values[:,i]
        data_mat_t[i,:,:] = np.array([basis(el) for el in x])
    print(f'Computational time {time.time() - t_0:.2f}s\n')
    return data_mat_t



def function_A_vec(t, delta_S_hat, data_mat, reg_param):
    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]
    this_dS = delta_S_hat.loc[:, t].values
    hat_dS2 = (this_dS ** 2).reshape(-1, 1)
    A_mat = np.dot(X_mat.T, X_mat * hat_dS2) + reg_param * np.eye(num_basis_funcs)
    return A_mat.astype(float)


def function_B_vec(t, Pi_hat, delta_S_hat, data_mat):
    tmp = Pi_hat[t+1] * delta_S_hat[t]
    X_mat = data_mat[t, :, :]  # matrix of dimension N_MC x num_basis
    B_vec = np.dot(X_mat.T, tmp)
    return B_vec.astype(float)


def get_pi_and_opt_hedge(
    delta_S_hat, data_mat_t, S, delta_S,
    N_MC, n_steps, K, reg_param, gamma, option_type,
    epsilon=0,
    func_A=function_A_vec, func_B=function_B_vec,terminal_payoff=terminal_payoff,
):
    # portfolio value
    Pi = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi[n_steps] = S[n_steps].apply(lambda x: terminal_payoff(x, K, option_type))
    
    # demeaned portfolio
    Pi_hat = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi_hat[n_steps] = Pi[n_steps] - np.mean(Pi[n_steps])

    # optimal hedge
    a = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    a[n_steps] = 0 ## hedge position should close out at maturity

    for t in range(n_steps-1, -1, -1):
        A_mat = func_A(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = func_B(t, Pi_hat, delta_S_hat, data_mat_t)

        # coefficients for expansions of the optimal action
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a[t] = np.dot(data_mat_t[t,:,:],phi)
        Pi[t] = gamma * (Pi[t+1] - a[t] * delta_S[t] + epsilon * S[t+1] * abs(a[t+1]-a[t]))
        Pi_hat[t] = Pi[t] - np.mean(Pi[t])

    return  a.astype('float'), Pi.astype('float'), Pi_hat.astype('float')



def get_rewards(Pi, a, delta_S, n_steps, risk_lambda, gamma):
    rewards = pd.DataFrame(index=a.index, columns=a.columns)
    rewards.iloc[:,-1] = - risk_lambda * np.var(Pi.iloc[:,-1])
    for t in range(n_steps):
        rewards[t] = gamma * a[t] * delta_S[t] - risk_lambda * np.var(Pi[t])
    return rewards



def function_C_vec(t, data_mat, reg_param):
    X_mat = data_mat[t, :, :]
    num_basis_funcs = X_mat.shape[1]
    C_mat = np.dot(X_mat.T, X_mat) + reg_param * np.eye(num_basis_funcs)
    return C_mat
   
    
def function_D_vec(t, Q, R, data_mat, gamma):
    X_mat = data_mat[t, :, :]
    D_vec = np.dot(X_mat.T, R[t] + gamma * Q[t+1])
    return D_vec


def get_q_function(
    data_mat_t, rewards, Pi, 
    reg_param, gamma, n_steps, risk_lambda,
    func_C=function_C_vec, func_D=function_D_vec,
):
    Q = pd.DataFrame([], index=rewards.index, columns=rewards.columns)
    Q[n_steps] = - Pi[n_steps] - risk_lambda * np.var(Pi[n_steps])

    for t in range(n_steps-1, -1, -1):
        C_mat = func_C(t, data_mat_t, reg_param)
        D_vec = func_D(t, Q, rewards, data_mat_t, gamma)
        omega = np.dot(np.linalg.inv(C_mat), D_vec)
        Q[t] = np.dot(data_mat_t[t,:,:], omega)

    return Q.astype('float')


def get_bsm_price(T, S, K, r, sigma, option_type):
    d1 = (np.log(S/K) + (r + 1/2 * sigma**2) * T) / sigma / np.sqrt(T)
    d2 = d1 - sigma * np.sqrt(T)
    N = norm.cdf
    if option_type == 'call':
        return S * N(d1) - K * np.exp(-r * T) * N(d2)
    elif option_type=='put':
        return K * np.exp(-r * T) * N(-d2) - S * N(-d1)
    else:
        raise ValueError(f"Invalid option type '{option_type}'")

        
def get_bsm_delta(T, S, K, r, sigma, option_type):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    N = norm.cdf
    if option_type == 'call':
        return N(d1)
    elif option_type == 'put':
        return -N(-d1)

    
def get_df_delta(S, K, r, sigma, T, option_type, N_MC, n_steps):
    bsm_delta = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    for t in range(n_steps+1):
        ttm = T-t*T/n_steps
        bsm_delta[t] = get_bsm_delta(ttm, S[t], K, r, sigma, option_type)

    return bsm_delta

    
def get_df_bsm_prices(S, K, r, sigma, T, option_type, N_MC, n_steps):
    bsm_prices = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    for t in range(n_steps+1):
        ttm = T-t*T/n_steps
        bsm_prices[t] = get_bsm_price(ttm, S[t], K, r, sigma, option_type)
    return bsm_prices



def get_period_data(x_data, y_data, option_type, K, data_type, grouped=False, lower_moneyness=0.75, upper_moneyness=1.25):
    delta_curve = pd.DataFrame({
        'x': x_data.values.flatten(),
        'y': y_data.values.flatten()
    }).dropna()
    
    if grouped:
        grouped_mean = delta_curve.groupby(
            pd.cut(
                delta_curve['x'], 
                bins=range(
                   int(delta_curve['x'].min()), 
                   int(delta_curve['x'].max()) + 1, 1
                )
            )
        )['y'].mean().reset_index()
        grouped_mean['x'] = grouped_mean['x'].apply(lambda x: x.left)
    else:
        grouped_mean = delta_curve.copy()
        
    grouped_mean['x'] = grouped_mean['x'].astype(float)
    
    if data_type == 'delta':
        if option_type=='call':
            grouped_mean = grouped_mean[grouped_mean.y >= 0]
        elif option_type=='put':
            grouped_mean = grouped_mean[grouped_mean.y <= 0]

    grouped_mean = grouped_mean[
        (grouped_mean.x >= lower_moneyness*K)
        & (grouped_mean.x <=upper_moneyness*K)
    ]
    return grouped_mean


def get_pnl(S, deltas, n_steps):
    pnl = pd.DataFrame(np.nan, index=S.index, columns=S.columns)
    for t in range(1, n_steps+1):
        pnl[t] = -deltas[t-1] * (S[t] - S[t-1])
    return pnl






def plot_mean_curve(S, K, a, bsm_delta, option_type, data_type, title=None, l1=None, l2=None):
    curve_qlbs = get_period_data(S, a, option_type, K, data_type, grouped=True)
    curve_bsm = get_period_data(S, bsm_delta, option_type, K, data_type, grouped=True)

    f, ax = plt.subplots()

    plt.axvspan(75, K, color='pink' if option_type=='call' else 'lightgreen', alpha=0.2)
    plt.axvspan(125, K, color='pink' if option_type=='put' else 'lightgreen', alpha=0.2)

    plt.scatter(curve_qlbs.x, curve_qlbs.y, label='QLBS' if l1==None else l1, marker='x',)
    plt.scatter(curve_bsm.x, curve_bsm.y, label='BSM' if l2==None else l2, marker='x',)

    plt.text(0.25, 0.99, 'OTM' if option_type=='call' else 'ITM', ha='left', va='top', transform=ax.transAxes)
    plt.text(0.75, 0.99, 'OTM' if option_type=='put' else 'ITM', ha='right', va='top', transform=ax.transAxes)

    plt.axvline(x=K, ls='--', color='black')
    plt.xlabel('Stock Price')
    if data_type == 'option':
        plt.ylabel('Option price')
        plt.title(f'Mean option price of BSM and QLBS Hedging, K={K}')
    else:
        plt.ylabel('Delta')    
        plt.title(f'Mean delta of BSM and QLBS Hedging, K={K}')
    if title:
        plt.title(title)
    plt.legend()
    plt.show()
    
    
def plot_time_curve(S, K, a, bsm_delta, option_type, data_type, ts=[0, 6, 12, 18, 23, 24], lower_moneyness=0.75, upper_moneyness=1.25):
    f, axes = plt.subplots(nrows=3, ncols=2, figsize=(10, 10), constrained_layout=True)
    f.suptitle(f'{option_type} option, K={K}')
    for idx, t in enumerate(ts):
        i = idx // 2
        j = idx % 2

        period_curve_qlbs = get_period_data(S[t], a[t], option_type, K, data_type, grouped=False)
        period_curve_bsm = get_period_data(S[t], bsm_delta[t], option_type, K, data_type, grouped=False)

        axes[i,j].scatter(period_curve_qlbs.x, period_curve_qlbs.y, s=5, alpha=0.5, marker='x',  label='QLBS')
        axes[i,j].scatter(period_curve_bsm.x, period_curve_bsm.y, s=5, alpha=0.5, marker='x',  label='BSM', color='orange')

        axes[i,j].set_title(f'time step {t}')
        axes[i,j].set_xlim([lower_moneyness*K, upper_moneyness*K])
        if data_type == 'option':
            axes[i,j].set_ylim(bottom=0)
            axes[i,j].set_ylabel('Option price')
        else:
            if option_type=='call':
                axes[i,j].set_ylim([-0.1, 1.1])
            else:
                axes[i,j].set_ylim([-1.1, 0.1])
            axes[i,j].set_ylabel('Delta')
        axes[i,j].set_xlabel('Stock Price')
        axes[i,j].legend()
        axes[i,j].axvline(x=K, ls='--', color='black')
        
        
def vary_risk_lambda(
    risk_lambdas, delta_S_hat, data_mat_t, S, delta_S,
    N_MC, n_steps, K, reg_param, gamma, option_type,
):
    actions = {}
    q_tables = {}
    pis = {}
    print('lambda\ttime')
    print('------\t----')
    for risk_lam in risk_lambdas:
        t0 = time.time()
        a, Pi, Pi_hat = get_pi_and_opt_hedge(
            delta_S_hat, data_mat_t, S, delta_S,
            N_MC, n_steps, K, reg_param, gamma, option_type,
        )
        rewards = get_rewards(Pi, a, delta_S, n_steps, risk_lam, gamma)
        Q = get_q_function(
            data_mat_t, rewards, Pi,
            reg_param, gamma, n_steps, risk_lam,
        )
        c_qlbs = -Q        
        print(f'{risk_lam}\t{time.time()-t0:.2f}s')
        
        actions[risk_lam] = a
        q_tables[risk_lam] = c_qlbs
        pis[risk_lam] = Pi
    return actions, q_tables, pis


##############################################################################################
## -----------------------------------------------------------------------------------------

def get_rewards_txn(Pi, a, delta_S, S, n_steps, risk_lambda, gamma, sigma=0, epsilon=0, y=0, kolm=False):
    rewards = pd.DataFrame(index=a.index, columns=a.columns)
    if kolm:
        n = abs(a[n_steps] - a[n_steps-1])
        rewards.iloc[:,-1] = - risk_lambda * np.var(Pi.iloc[:,-1]) - epsilon * S[n_steps] * (n + 0.01*n**2)
        for t in range(n_steps):
            n = abs(a[t] - a[t+1])
            rewards[t] = gamma * a[t] * delta_S[t] - risk_lambda * np.var(Pi[t]) - epsilon * S[t+1] * (n + 0.01*n**2)

    else:
        rewards.iloc[:,-1] = - risk_lambda * np.var(Pi.iloc[:,-1]) - epsilon * S[n_steps] * abs(a[n_steps]-a[n_steps-1]) - y * sigma/np.sqrt(n_steps) * np.sqrt(abs(a[n_steps]-a[n_steps-1]) / 100) * S[n_steps]
        for t in range(n_steps):
            rewards[t] = gamma * a[t] * delta_S[t] - risk_lambda * np.var(Pi[t]) - epsilon * S[t+1] * abs(a[t+1]-a[t]) - y * sigma/np.sqrt(n_steps) * np.sqrt(abs(a[t+1]-a[t]) / 100) * S[t+1]
            
    return rewards


def get_pi_and_opt_hedge_txn(
    delta_S_hat, data_mat_t, S, delta_S,
    N_MC, n_steps, K, reg_param, gamma, option_type,
    sigma=0, epsilon=0, y=0, kolm=False, 
    func_A=function_A_vec, func_B=function_B_vec,terminal_payoff=terminal_payoff,
):
    # portfolio value
    Pi = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi[n_steps] = S[n_steps].apply(lambda x: terminal_payoff(x, K, option_type))
    
    # demeaned portfolio
    Pi_hat = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi_hat[n_steps] = Pi[n_steps] - np.mean(Pi[n_steps])

    # optimal hedge
    a = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    a[n_steps] = 0 ## hedge position should close out at maturity

    for t in range(n_steps-1, -1, -1):
        A_mat = func_A(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = func_B(t, Pi_hat, delta_S_hat, data_mat_t)

        # coefficients for expansions of the optimal action
        phi = np.dot(np.linalg.inv(A_mat), B_vec)

        a[t] = np.dot(data_mat_t[t,:,:],phi)
        
#         Pi[t] = gamma * (Pi[t+1] - a[t] * delta_S[t] + epsilon * (S[t+1] * abs(a[t+1]-a[t])))

        if kolm:
            n = abs(a[t+1]-a[t])
            Pi[t] = gamma * (Pi[t+1] - a[t] * delta_S[t] + epsilon * S[t+1] * (n + 0.01*n**2))
        else:            
            Pi[t] = gamma * (Pi[t+1] - a[t] * delta_S[t] + epsilon * S[t+1] * abs(a[t+1]-a[t]) + y * sigma/np.sqrt(n_steps) * np.sqrt(abs(a[t+1]-a[t]) / 100) * S[t+1])
        Pi_hat[t] = Pi[t] - np.mean(Pi[t])

    return  a.astype('float'), Pi.astype('float'), Pi_hat.astype('float')


def vary_txn_cost(
    epsilons, delta_S_hat, data_mat_t, S, delta_S,
    N_MC, n_steps, K, reg_param, gamma, option_type, print_line=True, kolm=False
):
    actions = {}
    q_tables = {}
    if print_line:
        print('epsilon\ttime')
        print('-------\t----')
    for eps in epsilons:
        t0 = time.time()
        a, Pi, Pi_hat = get_pi_and_opt_hedge_txn(
            delta_S_hat, data_mat_t, S, delta_S,
            N_MC, n_steps, K, reg_param, gamma, option_type, epsilon=eps, kolm=kolm
        )
#         a, Pi, Pi_hat = get_pi_and_opt_hedge(
#             delta_S_hat, data_mat_t, S, delta_S,
#             N_MC, n_steps, K, reg_param, gamma, option_type, epsilon=eps, kolm=kolm
#         )
#         rewards = get_rewards(Pi, a, delta_S, n_steps, risk_lambda=0, gamma=gamma)
        rewards = get_rewards_txn(Pi, a, delta_S, S, n_steps, risk_lambda=0, gamma=gamma, epsilon=eps, kolm=kolm)
        Q = get_q_function(
            data_mat_t, rewards, Pi,
            reg_param, gamma, n_steps, risk_lambda=0,
        )
        c_qlbs = -Q
        if print_line:
            print(f'{eps}\t{time.time()-t0:.2f}s')
        
        actions[eps] = a
        q_tables[eps] = c_qlbs
    return actions, q_tables


def vary_mkt_impact_cost_sqrt(
    ys, delta_S_hat, data_mat_t, S, delta_S,
    N_MC, n_steps, K, reg_param, gamma, option_type, sigma, print_line=True
):
    actions = {}
    q_tables = {}
    if print_line:
        print('y\ttime')
        print('---\t----')
    for y in ys:
        t0 = time.time()
        a, Pi, Pi_hat = get_pi_and_opt_hedge_txn(
            delta_S_hat, data_mat_t, S, delta_S,
            N_MC, n_steps, K, reg_param, gamma, option_type, sigma=sigma, epsilon=0, y=y
        )
#         rewards = get_rewards(Pi, a, delta_S, n_steps, risk_lambda=0, gamma=gamma)
        rewards = get_rewards_txn(Pi, a, delta_S, S, n_steps, risk_lambda=0, gamma=gamma, sigma=sigma, epsilon=0, y=y)
        Q = get_q_function(
            data_mat_t, rewards, Pi,
            reg_param, gamma, n_steps, risk_lambda=0,
        )
        c_qlbs = -Q
        if print_line:
            print(f'{y}\t{time.time()-t0:.2f}s')
        
        actions[y] = a
        q_tables[y] = c_qlbs
    return actions, q_tables


##############################################################################################
## --------------------------------------- FQI -----------------------------------------------
##############################################################################################


def get_perturbated_hedge(
    delta_S_hat, data_mat_t, S, delta_S, a_opt,
    N_MC, n_steps, K, reg_param, gamma, option_type, eta, risk_lambda,
    epsilon=0,
    func_A=function_A_vec, func_B=function_B_vec,terminal_payoff=terminal_payoff,
):
    # portfolio value
    Pi = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi[n_steps] = S[n_steps].apply(lambda x: terminal_payoff(x, K, option_type))
    
    # demeaned portfolio
    Pi_hat = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Pi_hat[n_steps] = Pi[n_steps] - np.mean(Pi[n_steps])

    # optimal hedge
    a = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    a[n_steps] = 0 ## hedge position should close out at maturity
    
    # rewards
    rewards = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    rewards[n_steps] = - risk_lambda * np.var(Pi[n_steps])

    for t in range(n_steps-1, -1, -1):
        ## add noise from uniform dist to optimal action
        a[t] = a_opt[t] * np.random.uniform(1-eta, 1+eta, size=N_MC)
        
        ## compute new portfolio value from the new action
        Pi[t] = gamma * (Pi[t+1] - a[t] * delta_S[t])
        Pi_hat[t] = Pi[t] - np.mean(Pi[t])
        
        ## compute rewards from the new action
        rewards[t] = gamma * a[t] * delta_S[t] - risk_lambda * np.var(Pi[t])              

    return  a.astype('float'), Pi.astype('float'), Pi_hat.astype('float'), rewards.astype('float')


def init_matrices(a, data_mat_t, N_MC, n_steps):
    a_reshaped = a.values.reshape((1, N_MC, n_steps+1))
    a_sq = 0.5 * a_reshaped**2
    a_stacked = np.vstack((np.ones((1, N_MC, n_steps+1)), a_reshaped, a_sq))
    
    ## add new dimensions
    a_expanded = np.expand_dims(a_stacked, axis=1)
    data_mat_expanded = np.expand_dims(data_mat_t.T, axis=0)
    
    ## create matrices
    psi_mat = np.multiply(a_expanded, data_mat_expanded).reshape(-1, N_MC, n_steps+1, order='F')
    s_t_mat = np.sum(np.multiply(np.expand_dims(psi_mat, axis=1), np.expand_dims(psi_mat, axis=0)), axis=2)
    
    return psi_mat, s_t_mat


def get_s_reg_function(t, s_t_mat, reg_param):   
    ## regularisation
    s_mat_reg = s_t_mat[:,:,t] + reg_param * np.eye(s_t_mat.shape[0])
    return s_mat_reg


def get_m_function(t, psi_mat_t, rewards, Q_star, gamma):
    return np.dot(psi_mat_t, rewards[t] + gamma * Q_star[t+1])


def get_fqi(
    psi_mat, s_t_mat, data_mat_t, Pi, rewards, delta_S_hat, Pi_hat, 
    risk_lambda, N_MC, n_steps, reg_param, gamma,
    func_A=function_A_vec, func_B=function_B_vec,
    up_percentile_Q_RL=95, low_percentile_Q_RL=5
):
    ## initialise q table
    q = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    q[n_steps] = -Pi[n_steps] - risk_lambda * np.var(Pi[n_steps])
    
    ## optimal action
    a_opt = np.zeros((N_MC, n_steps+1))
    a_star = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    a_star[n_steps] = 0
    
    Q_star = pd.DataFrame(index=range(1, N_MC+1), columns=range(n_steps+1))
    Q_star[n_steps] = q[n_steps]
    
    ## optimal q function with optimal action
    max_Q_star = np.zeros((N_MC, n_steps+1))
    max_Q_star[:,-1] = q[n_steps].values
    
    num_basis = data_mat_t.shape[2]
        
    ## calculate from final state
    for t in range(n_steps-1, -1, -1):
        S_mat_reg = get_s_reg_function(t, s_t_mat, reg_param)
        M_t = get_m_function(t, psi_mat[:,:,t], rewards, Q_star, gamma)
        W_t = np.dot(np.linalg.inv(S_mat_reg),M_t)
        
        W_mat = W_t.reshape((3, num_basis), order='F')
        Phi_mat = data_mat_t[t,:,:].T
        U_mat = np.dot(W_mat, Phi_mat)
        
        ## compute vectors U_W^0,U_W^1,U_W^2 as rows of matrix U_mat  
        U_W_0 = U_mat[0,:]
        U_W_1 = U_mat[1,:]
        U_W_2 = U_mat[2,:]
        
        A_mat = func_A(t, delta_S_hat, data_mat_t, reg_param)
        B_vec = func_B(t, Pi_hat, delta_S_hat, data_mat_t)
        phi = np.dot(np.linalg.inv(A_mat), B_vec)
        
        a_opt[:,t] = np.dot(data_mat_t[t,:,:], phi)
        a_star[t] = a_opt[:,t]
        
        max_Q_star[:,t] = U_W_0 + a_opt[:,t] * U_W_1 + 0.5 * (a_opt[:,t]**2) * U_W_2
        Q_star[t] = max_Q_star[:,t]
        psi_t = psi_mat[:,:,t].T
        q[t] = np.dot(psi_t, W_t)
        
        # trim outliers for q function
        low_perc_Q_RL, up_perc_Q_RL = np.percentile(q[t],[low_percentile_Q_RL,up_percentile_Q_RL])
        
        flag_lower = q.loc[:,t].values < low_perc_Q_RL
        flag_upper = q.loc[:,t].values > up_perc_Q_RL
        q.loc[flag_lower,t] = low_perc_Q_RL
        q.loc[flag_upper,t] = up_perc_Q_RL
    return q, a_star