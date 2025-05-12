import numpy as np

# --- Adam Optimizer Implementation ---

def initialize_adam_state(params):
    state = {}

    for i, param in enumerate(params):

        state[i] = {
            'm': np.zeros_like(param),
            'v': np.zeros_like(param),
        }
    state['step'] = 0
    return state

def adam_update(params, grads, state, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):

    state['step'] += 1
    t = state['step'] # Get the current time step (1-indexed for bias correction) [5]

    updated_params = []
    updated_state = {} # Dictionary to build the new state


    for i, (param, grad) in enumerate(zip(params, grads)):

        param_state = state[i]


        m_prev = param_state['m']
        v_prev = param_state['v']


        m = beta1 * m_prev + (1 - beta1) * grad
        v = beta2 * v_prev + (1 - beta2) * grad**2 # Use **2 for element-wise squaring in NumPy [5]


        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)


        updated_param = param - lr * m_hat / (np.sqrt(v_hat) + epsilon)

        # Store the updated state for this parameter
        param_state['m'] = m
        param_state['v'] = v
        updated_state[i] = param_state # Add the updated parameter state to the new state dict

    # Store the updated global state (specifically, the step counter)
    updated_state['step'] = t

    # Return the list of updated parameters and the updated state
    return updated_params, updated_state

# --- Adan Optimizer Implementation ---

def initialize_adan_state(params):
    state = {}
    # Initialize moment estimates and previous gradient for each parameter to zeros [4, 14]
    # Initializing to zeros handles the first step (k=1) using the general update rules.
    for i, param in enumerate(params):
         # Using integer index 'i' as key for each parameter's state
        state[i] = {
            'mk': np.zeros_like(param),
            'vk': np.zeros_like(param),
            'nk': np.zeros_like(param),
            'gk_prev': np.zeros_like(param),
        }
    state['step'] = 0
    return state


def adan_update(params, grads, state, lr=0.001, beta1=0.02, beta2=0.08, beta3=0.01, epsilon=1e-8, weight_decay=0.02):
    # Increment the step counter before applying updates
    state['step'] += 1
    k = state['step'] # Get the current time step (1-indexed) [4]

    updated_params = []
    updated_state = {} # Dictionary to build the new state

    # Create copies of current gradients. These will become gk_prev for the next step (k+1).
    current_grads = [np.copy(grad) for grad in grads]

    # Iterate through each parameter and its corresponding gradient
    for i, (param, grad) in enumerate(zip(params, grads)):

        param_state = state[i]


        mk_prev = param_state['mk']
        vk_prev = param_state['vk']
        nk_prev = param_state['nk']
        gk_prev = param_state['gk_prev']


        mk = (1 - beta1) * mk_prev + beta1 * grad


        grad_diff = grad - gk_prev
        vk = (1 - beta2) * vk_prev + beta2 * grad_diff


        term_for_nk = grad + (1 - beta2) * grad_diff
        nk = (1 - beta3) * nk_prev + beta3 * (term_for_nk)**2 # Use **2 for element-wise squaring [4]


        learning_rate_k = lr / (np.sqrt(nk) + epsilon) # Use np.sqrt for element-wise square root [4]


        m_bar_k = mk + (1 - beta2) * vk


        updated_param = (1 - weight_decay * lr) * param - learning_rate_k * m_bar_k

        # Store the updated state values for this parameter
        param_state['mk'] = mk
        param_state['vk'] = vk
        param_state['nk'] = nk

        updated_state[i] = param_state # Add the updated parameter state to the new state dict

    for i, grad in enumerate(current_grads):
         updated_state[i]['gk_prev'] = grad # Store current grad as gk_prev for the next step

    # Store the updated global state (specifically, the step counter)
    updated_state['step'] = k

    # Return the list of updated parameters and the updated state
    return updated_params, updated_state