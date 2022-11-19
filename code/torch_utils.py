from collections import OrderedDict
from time import time

import numpy as np
import torch as pt
from matplotlib import pyplot as plt

from utils import printv, tqdm, makedir, os

device = pt.device('cuda' if pt.cuda.is_available() else 'cpu')


def to_device(x, grad=True):
    '''
        Maps x from cpu/numpy to pytorch tensor.
        If grad==True, then the returned vector has x.requires_grad = True
    '''
    if not pt.is_tensor(x):
        x = pt.tensor(x)
    x = x.float().to(device)
    if grad:
        x.requires_grad_()
    return x


def from_device(x):
    '''
        Maps x from pytorch tensor to numpy array.
    '''
    if pt.is_tensor(x):
        return x.detach().cpu().numpy()
    else:
        return x


def str2loss(s):
    '''Converts the name of a loss into a pytorch loss fuction'''
    if s in ('MAE', 'Mean Absolute Error', 'L1'):
        return pt.nn.L1Loss(reduction='mean')
    elif s in ('MSE', 'Mean Squared Error', 'L2^2'):
        return pt.nn.MSELoss(reduction='mean')
    elif s == 'L2':
        class Loss(pt.nn.Module):
            def __init__(self):
                super(Loss, self).__init__()

            def forward(self, x, y):
                criterion = pt.nn.MSELoss(reduction='mean')
                loss = pt.sqrt(criterion(x, y))
                return loss
        return Loss()


def str2activation(s):
    '''Converts string to pytorch activation function'''
    s = s.lower()
    if s == 'relu':
        return pt.nn.ReLU
    elif s == 'tanh':
        return pt.nn.Tanh
    elif s == 'leakyrelu':
        return pt.nn.LeakyReLU
    elif s == 'sigmoid':
        return pt.nn.Sigmoid
    else:
        raise NotImplementedError


class DenseNN(pt.nn.Module):
    def __init__(self, layers, activation='tanh'):
        '''
        Wrapper to create a basic dense neural network
            u = DenseNN(layers, activation='tanh')
        where <layers> is a list of integers and <activation> is either a string or pytorch 
        activation function.


        Attributes of u:
            depth: number of layers - 2 (i.e. hidden layers)
            activation: pytorch activation function
            layers: the pytorch NN
            numel(): the total number of learnable parameters
        '''
        super(DenseNN, self).__init__()

        self.depth = len(layers) - 2  # ignore input and output layers
        self.activation = str2activation(activation)

        # both weights and biases are randomly initialised
        init_w = pt.nn.init.xavier_uniform_
        init_b = lambda x: None  # pt.nn.init.zeros_

        pt_layers = []
        for i in range(len(layers) - 1):
            L = pt.nn.Linear(layers[i], layers[i + 1], bias=True)
            init_w(L.weight); init_b(L.bias)

            pt_layers.append((f'layer_{i}', L))
            pt_layers.append((f'activation_{i}', self.activation()))
        self.layers = pt.nn.Sequential(OrderedDict(pt_layers[:-1])).to(device)

    def forward(self, x): return self.layers(x)

    def numel(self): return sum(p.numel() for p in self.parameters())


class Basic_model:
    def __init__(self, u):
        '''
        Elementary class for optimisation.
            M = Basic_model(u)
        where u is the function with parameters to optimise.

        Attributes of M:
            u: the function to optimise
            history: record of loss through training
                history[i] = (j, t, loss, flag) where j is cumulative number of iterations, t is 
                    cumulative training time in minutes, loss is the corresponding loss, and flag
                    is an integer noting which optimiser was being used
        Methods of M:
            loss(): returns training loss (float)
            predict(X): returns M.u evaluated at X (1D numpy array)
            flag2sr(f): converts the integer flag in M.history to the name (string) of optimiser
            compile(method, **kwargs): sets up the optimiser name <method> with corresponding parameters
            train(iters, ...): runs optimiser for prescribed number of iterations
            save(p): saves M.u and M.history to path <p>
            load(p): loads M.u and M.history from path <p>

        '''
        self.u = u
        self.history = []
        self._iter = self._time = 0

    def loss(self): raise NotImplementedError
    def predict(self, X): raise NotImplementedError

    def optim_flag(self, s=None):
        '''
        Each optimiser has an associated integer 'flag'

        Parameter:
        ----------
            s: str or None
                If string, must be in ['lbfgs', 'sgd', 'adam']
                If None, selects string corresponding to self.optimiser
        Returns:
        --------
            f: int
                Integer flag corresonding to optimiser s
        '''
        if s is None:
            L = [pt.optim.LBFGS, pt.optim.SGD, pt.optim.Adam]
            for i, o in enumerate(L):
                if isinstance(self.optimiser, o):
                    return i
        else:
            return ['lbfgs', 'sgd', 'adam'].index(s.lower())

    def flag2str(self, f):
        '''
        Each optimiser has an associated integer 'flag'

        Parameter:
        ----------
            f: int
                Integer flag
        Returns:
        --------
            s: str
                Returns name of optimiser corresponding to flag <f>
        '''
        return ['LBFGS', 'SGD', 'Adam'][f]

    def record(self, i, t, L, flag, check=True):
        '''
        Appends row (I, T, L, flag) to self.history

        Parameters:
        -----------
            i: int
                Number of iterations in current training loop
            t: float
                Time (in minutes) in current training loop
            L: float
                Loss corresponding to current state
            flag: int
                The flag corresponding to current optimisation algorithm

        The values I and T are cumulative iterations and times including previous training loops 
        for the same model.
        '''
        i = i + self._iter
        t = t + self._time
        if check:
            h = self.history[-1]
            if (h[0] == i):
                return  # don't record duplicates

        self.history.append(
            (i, t, L, flag)
        )

    def compile(self, method, **kwargs):
        '''
        Assigns self.optimiser the optimiser specified by <method> and parameters <kwargs>

        Parameters:
        -----------
            method: str
                One of 'LBFGS', 'SGD', or 'Adam'
            kwargs:
                Extra parameters for particular optimiser, see below.

        The default parameters for each algorithm are:
            LBFGS: dict(lr=1.0, max_iter=1000, max_eval=10000, history_size=100,
                    tolerance_grad=1e-8, tolerance_change=1e-15, line_search_fn='strong_wolfe')
            SGD: dict(lr=1e-3, momentum=0, dampening=0, weight_decay=0, nesterov=False)
            Adam: dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False)
        '''
        method = method.lower()
        if method in ('l-bfgs', 'lbfgs', 'bfgs'):
            params = dict(lr=1.0, max_iter=1000, max_eval=10000, history_size=100,
                          tolerance_grad=1e-8, tolerance_change=1e-15, line_search_fn='strong_wolfe')
            func = pt.optim.LBFGS
        elif method == 'sgd':
            params = dict(lr=1e-3, momentum=0, dampening=0,
                          weight_decay=0, nesterov=False)
            func = pt.optim.SGD
        elif method == 'adam':
            params = dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                          weight_decay=0, amsgrad=False)
            func = pt.optim.Adam
        else:
            raise NotImplementedError
        params.update(kwargs)
        self.optimiser = func(self.u.parameters(), **params)

    def train(self, iters, threshold=0, verbosity=pt.inf, print_freq=1e3):
        '''
        Runs optimisation algorithm with additional parameters

        Parameters:
        -----------
            iters: int
                Maximum number of iterations to run
            threshold: float, default is 0
                If the current loss <= threshold, then algorithm terminates early
            verbosity: int or infinity, default is infinity
                Dictates how many details are printed:
                    verbosity=0: nothing is printed
                    verbosity=1: live progress bar and final loss are printed
                    verbosity>1: loss is printed at every recorded iteration
            print_freq: int, default is 1e3
                Dictates how often loss is recorded and/or printed.
                Default only records every 1000th iteration
                Note that for LBFGS print_freq should be >> history_size of algorithm
        '''
        opt = self.optimiser
        L = [float(self.loss())] * 2

        def closure():
            loss = self.loss()
            opt.zero_grad()
            loss.backward()
            L[1] = float(loss)
            return loss

        print_freq = int(print_freq)
        LBFGS = isinstance(opt, pt.optim.LBFGS)
        if LBFGS:
            opt.param_groups[0]['n_iter'] = 0

        flag = self.optim_flag()
        self.record(0, 0, L[0], flag, check=False)

        # if verbosity==1 then use a tqdm progress bar
        if verbosity == 1:
            pbar, last_i = tqdm(total=iters), 0
            pbar.set_description_str('Training')
        else:
            pbar = None

        printv('Iteration: %5d, Loss: %.3e, Time: %4.1fm' % (0, L[0], self._time), verbosity=verbosity, importance=2)
        i, I, tic = 0, print_freq, time()
        try:
            while i < iters and L[1] > threshold:
                if LBFGS:
                    opt.param_groups[0]['max_iter'] = I - i
                opt.step(closure)
                if LBFGS:
                    i = opt.state[opt._params[0]]['n_iter']
                    if i < I:  # algorithm terminated before maxiter
                        break
                else:
                    i += 1
                if i >= I:
                    t = (time() - tic) / 60
                    printv('Iteration: %5d, Loss: %.3e, Time: %4.1fm' %
                           (i, L[1], t), verbosity=verbosity, importance=2)
                    I = min(I + print_freq, iters)
                    self.record(i, t, L[1], flag, check=True)
                if pbar is not None:
                    pbar.update(i - last_i)
                    pbar.set_postfix_str('loss=%.2e' % L[1])
                    last_i = i

        except KeyboardInterrupt:
            printv('Terminating training early', verbosity=verbosity, importance=2)
            i = opt.state[opt._params[0]]['n_iter'] if LBFGS else i
        finally:
            if pbar is not None:
                pbar.refresh()
                pbar.close()

        L[1] = float(self.loss())
        self._iter += i
        t = (time() - tic) / 60
        self._time += t
        printv('Starting loss: %.3e, End loss: %.3e, Iterations: %5d, Time: %4.1fm' %
               (L[0], L[1], i, t), verbosity=verbosity, importance=1)

        self.record(0, 0, L[1], flag, check=True)

    def multi_train(self, optimise, load=None):
        '''
        Run training of model with multiple successive parameter choices.

        Parameters:
        -----------
            optimise: tuple of tuples (i, s, kw1, kw2), see below
            load: str, tuple of str, or None (default is None)
                If not None, represents a path where to load optimised model from.
                This method will only check if the file exists, not if different <optimise> 
                parameters were used in the previous reconstruction.
                If a tuple of strings is provided, os.path.join(*load) is used.
                If there is no model currently saved at the provided path, the current model will
                be saved there after training is complete.

            i: int, iters parameter from Basic_model.train (maximum number of iterations)
            s: str, method parameter Basic_model.compile (either LBFGS, Adam or SGD)
            kw1: dict, default {}, keyword arguments for Basic_model.train
            kw2: dict, default {}, keyword arguments for Basic_model.compile


        '''
        # Decide where to save if necessary

        if load is None:
            load = False
        else:
            load = (load,) if type(load) is str else load
            load = os.path.join(*load)
            makedir(load)

        if load and os.path.exists(load):
            self.load(load)
            return

        if type(optimise[0]) in (int, float):
            optimise = (optimise,)
        assert all(len(p) > 1 for p in optimise)
        optimise = (tuple(p) + ({},) * (4 - len(p)) for p in optimise)  # append default {} values
        for (i, s, kw1, kw2) in optimise:
            if i == 0:
                continue
            self.compile(s, **kw2)
            self.train(iters=int(i), **kw1)

        if load:
            self.save(load)

    def save(self, p):
        '''Saves model parameters and history to path <p>'''
        D = {'u': self.u.state_dict(),
             'history': self.history}
        pt.save(D, p)

    def load(self, p):
        '''Loads model parameters and history to path <p>'''
        D = pt.load(p)
        self.u.load_state_dict(D['u'])
        self.history = D['history']
        self._iter, self._time = self.history[-1][:2]


def plotHistory(model, fig=None, threshold=0):
    '''
    Plots history from training of <model>.

    Parameters:
    -----------
        model: instance of Basic_model
            This function plots model.history
        fig: matplotlib figure handle or None
            The figure on which to plot, default is a new figure with figsize=(10,4)
        threshold: float or tuple of floats
            Plots each threshold as a horizontal line, but will ignore it if the minimum loss does 
            not reach it. Default is 0, which is probably never plotted.
    '''
    threshold = (threshold,) if not hasattr(threshold, '__iter__') else threshold
    fig = plt.figure(figsize=(10, 4)) if fig is None else fig
    ax = fig.subplots(1, 2, sharey=True)

    tmp = np.array(model.history)
    for i in range(2):
        for f in np.unique(tmp[:, 3]):
            where = (tmp[:, 3] != f)
            ax[i].plot(np.ma.masked_where(where, tmp[:, i]),
                       np.ma.masked_where(where, tmp[:, 2]),
                       label=model.flag2str(int(f)))
        first = True
        for thresh in threshold:
            if tmp[:, 2].min() < thresh:
                ax[i].axhline(thresh, color='r', label='Early stopping criterion' if first else None)
                first = False
        ax[i].set_xlabel('time (minutes)' if i else 'iterations')
        ax[i].set_yscale('log');
        ax[i].set_title('Convergence of energy, in ' + ('time' if i else 'iterations'))
    ax[0].legend(loc='upper right')
    plt.tight_layout()
