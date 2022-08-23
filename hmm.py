import numpy as np

class HMM: # 2 STATES
    def __init__(self, number_of_states, A, PI, O, params) -> None:
        self.number_of_states = number_of_states # 2 for easy case: folded & unfolded
        self.A = A # transition probability matrix
        self.PI = PI # initial state distribution -> uniform
        self.O = O # observation sequence (pandas dataFrame) == data collected in lab -> hp: gaussian distribution
        self.params = params # fit parameters of the PDF distribution used
        self.c_1, self.mean_1, self.std_1, self.c_2, self.mean_2, self.std_2 = self.params
        self.par_1 = [self.c_1, self.mean_1, self.std_1]
        self.par_2 = [self.c_2, self.mean_2, self.std_2]
        self.par = [self.par_1, self.par_2]
    
    '''
        There are 3 problems in HMM Δ:
        - evaluation problem: how to calculate the probability P(O|Δ) of the observation sequence, indicating how much
            the HMM Δ parameters affects the sequence O;
        - uncovering problem: how to find the sequence of states X ={x_1, x_2, ....., x_T} so that it is more likely to
            produce the observation sequence O;
        -learning problem: how to adjust parameters of Δ such as initial state distribution PI, transition probability matrix A
            and observation probability matrix B (-> θ = {μ, σ} for continuous observation HMM) 
            so that the quality of HMM Δ is enhanced.
    '''

    def evaluation_problem(self, stampa=True): 
        '''
            Forward-backward algorithm:
                1 <= t <= T
        '''
        
        '''
            Forward Variable
        '''
        
        # 1. Initializing step
        b_1 = self._gaussian(self.par_1, self.O[0])
        b_2 = self._gaussian(self.par_2, self.O[0])
        forward_variable = [[b_1*self.PI[0], b_2*self.PI[1]]]
        # 2. Recurrence step
        for t in range(self.O.size-1):
            f_1, f_2 = 0, 0
            b_1 = self._gaussian(self.par_1, self.O[t+1])
            b_2 = self._gaussian(self.par_2, self.O[t+1])
            # print(f"b_1 e b_2 valogono : {b_1}, {b_2}")
            for j in range(self.number_of_states):
                f_1 += forward_variable[t][j]*self.A[j][0]
                f_2 += forward_variable[t][j]*self.A[j][1]  
                # print(f"f_1 e f_2 valogono : {f_1}, {f_2}")  
            forward_variable.append([f_1*b_1, f_2*b_2])
        # print(forward_variable)
        
        # 3. Evaluation step:
        probability_O_Delta_forw = sum(forward_variable[len(forward_variable)-1])
        if stampa:
            print(f"La probabilità P(O|Δ) = {probability_O_Delta_forw}, ottenuta con il metodo forward-backward sulla forward-variable")

        '''
            Backward Variable
        '''
        
        # 1. Initializing step
        beta = [0.]*self.O.size
        beta[self.O.size-1] = [1., 1.] # this is the T-th element, so the last one
        
        # 2. Recurrence step
        for k in range(self.O.size-1, 0, -1):
            par_sum_1, par_sum_2 = 0, 0
            for j in range(self.number_of_states):
                par_sum_1 += self.A[0][j] * self._gaussian(self.par[j], self.O[k]) * beta[k][j]
                par_sum_2 += self.A[1][j] * self._gaussian(self.par[j], self.O[k]) * beta[k][j]
            beta[k-1] = [par_sum_1, par_sum_2]
       
        # beta = [[1., 1.]]
        # beta.append([1., 1.]) # this is the T-th element, so the last one
        # 2. Recurrence step
        # for k in range(self.O.size):
        #     par_sum_1, par_sum_2 = 0, 0
        #     for j in range(self.number_of_states):
        #         par_sum_1 += self.A[0][j] * self._gaussian(self.par[j], self.O[self.O.size-1-k]) * beta[k][j]
        #         par_sum_2 += self.A[1][j] * self._gaussian(self.par[j], self.O[self.O.size-1-k]) * beta[k][j]
        #     beta.append([par_sum_1, par_sum_2])

        # beta = beta.reverse()
                
        # 3. Evaluation step:
        probability_O_Delta_back = sum([self.PI[i]*self._gaussian(self.par[i], self.O[0])*beta[0][i] for i in range(self.number_of_states)])
        if stampa:
            print(f"La probabilità P(O|Δ) = {probability_O_Delta_back}, ottenuta con il metodo forward-backward sulla backward-variable")

        self.alpha = forward_variable
        self.beta = beta
        return self.alpha, self.beta


    def uncovering_problem(self):
        '''
            Viterbi algorithm
        '''
        delta = [] # joint optimal criterion
        q = [] # backtracking state
        
        # 1. Initialization step:
        temp_1 = self._gaussian(self.par_1, self.O[0])*self.PI[0]
        temp_2 = self._gaussian(self.par_2, self.O[0])*self.PI[1]
        delta.append([temp_1, temp_2]) 
        q.append([0., 0.])
        
        # 2. Recurrence step:
        for t in range(self.O.size-1):
            var_1, var_2 = 0., 0.
            # for j in range(self.number_of_states):
            var_1 = max([delta[t][0]*self.A[0][0], delta[t][1]*self.A[1][0]])
            var_2 = max([delta[t][0]*self.A[0][1], delta[t][1]*self.A[1][1]])
            delta.append([var_1*self._gaussian(self.par_1, self.O[t+1]), var_2*self._gaussian(self.par_2, self.O[t+1])])
            q.append([0 if delta[t][0]*self.A[0][0] > delta[t][1]*self.A[1][0] else 1, 0 if delta[t][0]*self.A[0][1] > delta[t][1]*self.A[1][1] else 1]) # the state that maximizes delta is stored in the backtracking state

        # 3. State sequence backtracking step:
        X = [0]*self.O.size
        # X[self.O.size-1] = max(delta[len(delta)-1])
        X[self.O.size-1] = 0 if delta[len(delta)-1][0] > delta[len(delta)-1][1] else 1
        for k in range(self.O.size-2, 0, -1):
            # X[k] = q[k+1][int(X[k+1])] # here I round up X[t+1] because due to calculation I guess it is not obvious it will be an integer
            X[k] = q[k+1][X[k+1]]
        self.X = X
        return self.X

    
    def learning_problem(self):
        '''
            Expectation Maximization (EM) algortihm === Baum-Welch algorithm
        '''
    
        # E-step:
        xi = [[0, 0, 0, 0]]
        for t in range(1, self.O.size):
            zero = self.alpha[t-1][0]*self.A[0][0]*self._gaussian(self.par_1, self.O[t])*self.beta[t][0]
            zero_one = self.alpha[t-1][0]*self.A[0][1]*self._gaussian(self.par_2, self.O[t])*self.beta[t][1]
            one_zero = self.alpha[t-1][1]*self.A[1][0]*self._gaussian(self.par_1, self.O[t])*self.beta[t][0]
            one = self.alpha[t-1][1]*self.A[1][1]*self._gaussian(self.par_2, self.O[t])*self.beta[t][1]
            xi.append([zero, zero_one, one_zero, one])
        gamma = []
        for t in range(self.O.size):
            gamma.append([self.alpha[t][0]*self.beta[t][0], self.alpha[t][1]*self.beta[t][1]])

        # M-step:     
        
        # New transition probability matrix A       
        num_a_0_0 = sum([xi[k][0] for k in range(len(xi))])
        # den_a_0_0 = sum(sum([xi[k][i] for i in range(self.number_of_states)]) for k in range(len(xi)))
        den_a_0 = sum([xi[k][0] + xi[k][1] for k in range(len(xi))])
        a_0_0 = num_a_0_0/den_a_0

        num_a_0_1 = sum([xi[k][1] for k in range(len(xi))])
        a_0_1 = num_a_0_1/den_a_0

        num_a_1_0 = sum([xi[k][2] for k in range(len(xi))])
        den_a_1 = sum([xi[k][2] + xi[k][3] for k in range(len(xi))])
        a_1_0 = num_a_1_0/den_a_1

        num_a_1_1 = sum([xi[k][3] for k in range(len(xi))])
        a_1_1 = num_a_1_1/den_a_1

        new_matrix = [[a_0_0, a_0_1], [a_1_0, a_1_1]]
        self.A = new_matrix


        # New initial state distribution PI
        pi_0 = gamma[0][0]/sum(gamma[0])
        pi_1 = gamma[0][1]/sum(gamma[0])
        new_pi = [pi_0, pi_1]
        self.PI = new_pi

        return self.A, self.PI


    def iteration(self, MAX_ITERATION = 10000, epsilon = 0.0001, show=True):
        # epsilon this could be use after, or by comparing the Probability P(O|Δ) in the evaluation-problem
        way = 0
        while way < MAX_ITERATION:
            
            self.alpha, self.beta = self.evaluation_problem(stampa=False)
            self.X = self.uncovering_problem()
            self.A, self.Pi = self.learning_problem()

            way += 1
            my_variable = way/MAX_ITERATION*100
            if way%(int(MAX_ITERATION/100)) == 0 and show:
                print(f"Loading {int(my_variable)}%")

        return self.alpha, self.beta, self.X, self.A, self.PI

    
    
    # def _doublegaussian(self, params, x):
    #     # params is a vector of the parameters:
    #     # params = [f_F, sigma_F, w_F, f_U, sigma_U, w_U]
    #     (c1, mu1, sigma1, c2, mu2, sigma2) = params
    #     res =   c1 * np.exp( - (x - mu1)**2.0 / (2.0 * sigma1**2.0) ) \
    #       + c2 * np.exp( - (x - mu2)**2.0 / (2.0 * sigma2**2.0) )
    #     return res

    def _gaussian(self, par, x):
        c, mu, sigma = par
        res = c*np.exp(-(x-mu)**2./(2.*sigma**2.))
        return res