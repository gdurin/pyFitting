#!/usr/bin/env 

"""bestFit is a simple python script to perform data fitting 
using nonlinear least-squares minimization.
"""
import sys, os
import locale
import argparse
import scipy
from scipy.optimize import leastsq
import scipy.special as special
import numpy as np
from numpy import pi
#import matplotlib as mpl
#try:
#    mpl.use('Qt4Agg')
#except:
#    raise

import matplotlib.pyplot as plt
import numexpr as ne
from time import time
import re
from getAnalyticalDerivatives import getDiff
#from scitools.StringFunction import StringFunction

def getColor():
    colors = 'brgcmb'*4
    for i in colors:
        yield i

def getSymbol():
    symbols = "ov^<>12sp*h+D"*3
    for i in symbols:
        yield i

def genExpr2Scipy(op, function):
    r"""
    Insert the proper scipy.* module to handle math functions

    Parameters:
    ----------------
    op : string
        math function to be replace by the scipy equivalent
    function : string
        the theoretical function

    Returns:
    -----------
    function : string
        the theoretical function with the proper scipy method

    Example:
    ------------
    >>> f = "sin(x/3.)"
    >>> print genExpr2Scipy("sin", f)
    scipy.sin(x/3.)

    Notes:
    --------
    Both usual functions and special ones are considered
    """
    try:
        if op in dir(np):
            sub = "np."
            function = function.replace(op, sub+op)
        elif op in dir(special):
            op_occurrences = [q for q in dir(special) if op in q]
            op_occurrences_in_function = [q for q in op_occurrences if q in function]
            if len(op_occurrences_in_function) > 1:
                for q in op_occurrences_in_function:
                    string_to_search = r'\b'+q
                    function = re.sub(string_to_search, 'special.'+q, function) 
            else:
                sub = "special."
                function = function.replace(op, sub+op)
        return function
    except:
        print(("Function %s not defined in numpy" % op))
        return None

class Theory:
    r"""
    Defines the theoretical function to fit the data with (the model)
    """
    def __init__(self, xName, function, paramsNames, dFunc=False):
        self.xName = xName
        self.parameters = paramsNames
        paramsNamesList = paramsNames.split(",")
        self.fz = function
        self.fzOriginal = function
        self.checkFunction = True 
        # Calculate the analytical derivatives
        # Return None if not available
        self.dFunc = dFunc
        if dFunc:
            self.dFunc = getDiff(xName, function, paramsNamesList)
            try: 
                # Then try to compile them  to be reused by NumExpr
                self.dFuncCompiled = list(map(ne.NumExpr, self.dFunc))
            except TypeError:
                print("Warning:  one or more functions are undefined in NumExpr")
                self.dFuncCompiled = None

    def Y(self, x, params):
        # Check if there is only a parameter
        if len(params) == 1:
            params = params[0]
        exec("%s = params" % self.parameters)
        exec("%s = x " % self.xName)
        # Check if the function needs to be changed with scipy.functions
        if self.checkFunction:
            while self.checkFunction:
                try:
                    exec("f = %s" % (self.fz))
                    self.checkFunction = False
                except NameError as inst:
                    op = inst.name.split("\'")[0]
                    function = genExpr2Scipy(op, self.fz)
                    if function:
                        self.fz = function
                    else:
                        raise ValueError("Function %s not found" % op)
        return eval(self.fz)
        #print(self.parameters, self.xName, self.fz)
        #return f

    def jacobian(self, x, params):
        """
        Calculus of the jacobian with analytical derivatives
        """
        jb = []
        checkDerivative = True
        exec("%s = params" % self.parameters)
        exec("%s = x" % self.xName)
        if self.dFuncCompiled:
            for q in self.dFuncCompiled:
                values = list(map(eval, q.input_names))
                jb.append(q(*values))
        else:
            for i, q in enumerate(self.dFunc):
                while checkDerivative:
                    try:
                        exec("deriv = %s" % q)
                        self.dFunc[i] = q
                        checkDerivative = False
                    except NameError as inst:
                        op = inst.message.split("'")[1]
                        q = genExpr2Scipy(op, q)
                jb.append(deriv)
                checkDerivative = True
        return np.array(jb)

class DataCurve:
    def __init__(self, input_data, cols, dataRange=None, data_logY=False):
        # Check if there is a file to load data from
        if type(input_data) is str:
            if os.path.isfile(input_data):
                print("File %s exists" % input_data)
                self.fileName = input_data
                data = np.loadtxt(input_data)
                self.X, self.Y, self.Yerror = self.get_data(data, cols)
                if data_logY:
                    print("Y Data in log scale")
                    self.Y = np.log10(self.Y)
            else:
                print("Error with data, file %s not found" % input_data)
                sys.exit()
        # or data are passed here directly in the variable input_data
        else:
            #print("Assumuming data passed here")
            #print input_data
            self.fileName = None
            self.X, self.Y, self.Yerror = self.get_data(input_data, cols)
        if dataRange is not None:
            i0,i1 = self.select_data(self.X, dataRange)
            self.X = self.X[i0:i1]
            self.Y = self.Y[i0:i1]
            if self.Yerror is not None:
                self.Yerror = self.Yerr[i0:i1]

    def get_data(self, data, cols):
        print(data.shape)
        x = data[:, cols[0]]
        y = data[:, cols[1]]
        if len(cols) > 2:
            yerr = data[:, cols[2]]
        else:
            yerr = None
        return x, y, yerr

    def select_data(self, x, dataRange):
        rngType, = list(dataRange.keys())
        if rngType == 'indx':
            i0, i1 = dataRange['indx']
            if i0 == 'min':
                i0 = 0
            if i1 == 'max':
                i1 = None
        elif rngType == 'vals':
            xmin, xmax = dataRange['vals']
            if xmin == 'min':
                i0 = 0.
            else:
                i0 = np.argwhere(x>xmin)[0][0]
            if xmax == 'max':
                i1 = None
            else:
                i1 = np.argwhere(x>xmax)[0][0]
        return (i0, i1)

    def len(self):
        return len(self.X)

class Model():
    r"""Link data to theory, and provides all the methods
    to calculate the residual, the jacobian and the cost
    """
    def __init__(self, dataAndFunction, cols, dataRange, variables, parNames, \
                 linlog='lin', sigma=None, dFunc=False, data_logY=False):
        data, func = dataAndFunction
        if type(func) is list:
            func = func[0]
        self.data = DataCurve(data, cols, dataRange, data_logY)
        self.theory = Theory(variables[0], func, parNames, dFunc)
        self.dFunc = self.theory.dFunc
        self.linlog = linlog
        self.sigma = self.data.Yerror

    def residual(self, params):
        """Calculate residual for fitting"""
        self.residuals = np.array([])
        if self.sigma is None:  
            sigma = 1.
        else:
            sigma = self.sigma
        P = self.theory.Y(self.data.X, params)
        if self.linlog == 'lin':
            res = (P - self.data.Y)/sigma
        elif self.linlog == 'log':
            #res = (P*scipy.log10(P) - self.data.Y*scipy.log10(self.data.Y))/sigma
            res = (scipy.log10(P) - scipy.log10(self.data.Y))/sigma
            #print res
        self.residuals = np.concatenate((self.residuals, res))
        return self.residuals

    def jacobian(self, params):
        jac = self.theory.jacobian(self.data.X, params)
        if self.sigma is not None:
            jac = jac/self.sigma
        return jac

class CompositeModel():
    """Join the models
    """
    def __init__(self, models, parNames):
        self.models = models
        self.parStr = parNames.split(",")
        # Check if the model have the error in the data
        # and use analytical derivatives
        self.isSigma = None
        self.isAnalyticalDerivs = False
        for model in models:
            if model.sigma is not None:
                self.isSigma = True
            if model.dFunc:
                self.isAnalyticalDerivs = True

    def residual(self, params):
        res = np.array([])
        for model in self.models:
            res = np.concatenate((res, model.residual(params)))
        return res

    def cost(self, params):
        res = self.residual(params)
        cst = np.dot(res,res)
        # Standard error of the regression
        lenData = sum([model.data.len() for model in self.models])
        ser = (cst/(lenData-len(params)))**0.5
        return cst, ser

    def jacobian(self, params):
        for i, model in enumerate(self.models):
            if i == 0:
                jac = model.jacobian(params)
            else:
                jac = np.concatenate((jac, model.jacobian(params)), axis=1)
        return jac
    
def doBestFit(compositeModel, params0, maxfev=None, factor=None):
    if not maxfev:
        maxfev = 500*(len(params0)+1)
    if not factor:
        factor = 100
    residual = compositeModel.residual
    if compositeModel.isAnalyticalDerivs:
        jacobian = compositeModel.jacobian
        full_output = leastsq(residual, params0,\
                            maxfev=maxfev, Dfun=jacobian, col_deriv=True, \
                            factor=factor, full_output=1)
    else:
        full_output = leastsq(residual, params0, maxfev=maxfev, \
                            factor=factor, full_output=1)
    return full_output

def plotBestFit(compositeModel, params0, isPlot='lin',
                errorbar=None, data_logY=False):
    nStars = 80
    print("="*nStars)
    t0 = time()
    printOut = []
    table = []
    table.append(['parameter', 'value', 'st. error', 't-statistic'])
    print("Initial parameters = ", params0)
    initCost = compositeModel.cost(params0)
    printOut.append(initCost)
    print('initial cost = %.10e (StD: %.10e)' % compositeModel.cost(params0))
    full_output = doBestFit(compositeModel, params0)
    params, covmatrix, infodict, mesg, ier = full_output
    costValue, costStdDev = compositeModel.cost(params)
    print('optimized cost = %.10e (StD: %.10e)' % (costValue, costStdDev))
    printOut.append(costValue)
    #if compositeModel.isAnalyticalDerivs:
        #jcb = jacobian(params)
    # # The method of calculating the covariance matrix as
        #analyCovMatrix = scipy.matrix(scipy.dot(jcb, jcb.T)).I
        #print analyCovMatrix
        #print covmatrix
    # is not valid in some cases. A general solution is to make the QR 
    # decomposition, as done by the routine
    if covmatrix is None: # fitting not converging
        for i in range(len(params)):
            stOut = compositeModel.parStr[i], '\t', params[i]
            print(compositeModel.parStr[i], '\t', params[i])
            printOut.append(stOut)
    else:
        for i in range(len(params)):
            if compositeModel.isSigma and errorbar=="e": 
                # This is the case of weigthed least-square
                # with error bar
                stDevParams = covmatrix[i,i]**0.5
            else:
                stDevParams = covmatrix[i,i]**0.5*costStdDev
            par = params[i]
            table.append([compositeModel.parStr[i], par, stDevParams, par/stDevParams])
            stOut = compositeModel.parStr[i], '\t', params[i], '+-', stDevParams
            printOut.append(stOut)

    print(("="*nStars))
    pprint_table(table)
    print(("="*nStars))        
    print("Done in %d iterations" % infodict['nfev'])
    print(mesg)
    print(("="*nStars))
    # Chi2 test
    # n. of degree of freedom
    lenData = sum([model.data.len() for model in compositeModel.models])
    print("n. of data = %d" % lenData)
    dof = lenData - len(params)
    print("degree of freedom = %d" % (dof))
    print("X^2_rel = %f" % (costValue/dof))
    #pValue = 1. - scipy.special.gammainc(dof/2., costValue/2.)
    #print "pValue = %f (statistically significant if < 0.05)" % (pValue)
    ts = round(time() - t0, 3)
    print("*** Time elapsed:", ts)
    if isPlot:
        # Prepare the plot
        nModels = len(compositeModel.models)
        fig = plt.figure()
        fig.set_size_inches(7*nModels,6,forward=True)
        getCol = getColor()
        getSyb = getSymbol()
        kFig = 0
        for model in compositeModel.models:
            kFig += 1
            ax = fig.add_subplot(1, nModels, kFig)
            X = model.data.X
            Y = model.data.Y
            X1 = np.linspace(X[0], X[-1], 300)
            calculatedData= model.theory.Y(X1, params)
            color = next(getCol)
            style = next(getSyb) + color
            color = next(getCol)
            labelData = model.data.fileName
            labelTheory = model.theory.fzOriginal
            if isPlot == "lin":
                plt.plot(X, Y, style, label=labelData)
                plt.plot(X1, calculatedData, color, label=labelTheory)
            elif isPlot == 'creep':
                mu = params[1]
                if data_logY:
                    plt.plot(X**-mu, Y, style, label=labelData)
                    plt.plot(X1**-mu, calculatedData, color, label=labelTheory)
                else:
                    plt.semilogy(X**-mu, Y, style, label=labelData)
                    plt.semilogy(X1**-mu, calculatedData, color, label=labelTheory)
            else: 
                plt.loglog(X, Y, style, label=labelData)
                plt.loglog(X1, calculatedData, color, label=labelTheory)
            if model.sigma is not None:
                plt.errorbar(X, Y, yerr=model.sigma, fmt="")
            plt.draw()
            plt.legend(loc=0)
            if isPlot == 'creep':
                plt.xlabel(r"$H^{-\mu}$", size=20)
            else:
                plt.xlabel(model.theory.xName, size=20)

        plt.show()
    # Alternative fitting
    #full_output = scipy.optimize.curve_fit(func,data.X,data.Y,params0,None)
    #print "Alternative fitting"
    #print full_output
        #fig2 = plt.figure(2)
        #plt.semilogx(data.X, data.Y-theory.Y(data.X,params),'-ro')
        #plt.draw()
        #plt.show()
    return full_output

def format_num(num):
    """Format a number according to given places.
    Adds commas, etc. Will truncate floats into ints!"""
    try:
        inum = int(num)
        return locale.format_string("%.5e", (0, inum), True)
    except (ValueError, TypeError):
        return str(num)

def get_max_width(table, index):
    """Get the maximum width of the given column index"""
    return max([len(format_num(row[index])) for row in table])    

def pprint_table(table, out=sys.stdout):
    """Prints out a table of data, padded for alignment
    @param out: Output stream (file-like object)
    @param table: The table to print. A list of lists.
    Each row must have the same number of columns. """
    print(table)
    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print(row[0].ljust(col_paddings[0] + 1), end=' ', file=out)
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            col = row[i]
            print(f"{col}", end=' ', file=out)
        print(file=out)
        
def split_range(rng):
    if ":" not in rng:
        return None, None
    m, M = rng.split(":")
    if m == "" or m == "None" or m=='min':
        rngMin = 'min'
    else:
        rngMin = float(m)
    if M == "" or M == "None" or M=='max':
        rngMax = 'max'
    else:
        rngMax = float(M)
    return rngMin, rngMax  

def main(args=None):
    print(args)
    if not args:
        parser = argparse.ArgumentParser(description='Best fit of data using least-square minimization')
        parser.add_argument('-f','--filename', metavar='filename', nargs='+', required=True,
                            help='Filename(s) of the input data')
        parser.add_argument('-t','--theory', metavar='theory', nargs='+', required=True,
                            help='Theoretical function(s)')
        parser.add_argument('-p','--params', metavar='params', required=True,  nargs='+',
                            help='Parameter(s) name(s), i.e. -p a b c')
        parser.add_argument('-i','--initvals', metavar='initvals', required=True, type=float, nargs='+',
                            help='Initial values of the parameter(s), i.e. -i 1 2. 3.')
        parser.add_argument('-v', '--var', metavar='var', default='x', nargs='+',
                            help='Name(s) of the independent variable(s), default: x')
        parser.add_argument('-c','--cols', metavar='cols', default=[0, 1],  type=int, nargs='+',
                            help='Columns of the file to load the data, default: 0 1 a third col \
                            is used as error bars')
        parser.add_argument('-w','--weight', action='store_true',
                            help='Use the 3rd column to weight data')
        parser.add_argument('-e','--errbar', action='store_true',
                            help='Use the 3rd column as the true error bar of the data')    
        parser.add_argument('-rIndx', '--Irange', metavar='Irange', default=None,
                                help='Range of the data (as index of rows)')
        parser.add_argument('-rVals', '--Vrange', metavar='Vrange', default=None,
                            help='Select the range of the data values (has priority over Index range)')
        parser.add_argument('-d', '--deriv', action='store_true',
                            help='Use Analytical Derivatives')
        parser.add_argument('-s','--sigma', metavar='sigma', type=float, default=None,
                            help='Estimation of the error in the data (as a constant value)')
        parser.add_argument('--held', metavar='heldParams', nargs='+', default = None,
                            help='Held one or more parameters, i.e. a=3 b=4')
        parser.add_argument('--lin', action='store_true',   
                            help='Use data in linear mode (default)')
        parser.add_argument('--log', action='store_true',
                            help='Use data in log mode (best for log-log data)')
        parser.add_argument('--noplot', action='store_true',
                            help=r"Don't show the plot output")
        parser.add_argument('--logplot', action='store_true',
                            help='Use log-log axis to plot data (default if --log)')
        parser.add_argument('--creep', action='store_true',
                            help='Use x-axis as x**-mu to plot data')
        parser.add_argument('--data_logY', action='store_true',
                            help='Use the log of Y data as input')
        args = parser.parse_args()
        print(args)
    else:
        pass
        print("Passing data: ", args.filename)
        print(args.theory)
    #
    # Analyze input
    #
    fileNames = args.filename
    cols =  args.cols
    xVariables = args.var
    functions = args.theory
    if len(functions) != len(xVariables):
        xVariables *= len(functions)
    parNames = ",".join(args.params)
    params0 = tuple(args.initvals)
    dFunc = args.deriv
    valsRange = args.Vrange
    indxRange = args.Irange
    if args.held:
        heldParams = {}
        for p in args.held:
            [par,val] = p.split("=")
            heldParams[par] = float(val)
    else:
        heldParams = None


    if valsRange is None and indxRange is None:
        dataRange = None
    else:
        dataRange = {}
    #print dataRange
    if not valsRange and indxRange:
        dataRange['indx'] = split_range(indxRange)
    elif valsRange:
        dataRange['vals'] = split_range(valsRange)

    linlog = "lin"
    isPlot = "lin"
    data_logY = False
    if args.log:
        linlog = "log"
        isPlot = 'log'
    if args.noplot:
        isPlot = False
    if args.logplot:
        isPlot = 'log'
    if args.creep:
        isPlot = 'creep'
    if args.data_logY:
        data_logY = True
    # Deal with error bar and weight
    sigma = args.sigma
    if args.weight and args.errbar:
        print("Warning: use the 3rd col as error bar")
    elif args.weight:
        errorbar = "w"
    elif args.errbar:
        errorbar = "e"
    else:
        errorbar = None

    dataAndFunction = list(zip(fileNames, functions))
    models = []
    nmodels = len(fileNames)
    if len(xVariables) != nmodels:
        xVariables = nmodels*xVariables
        #print xVariables
    for i in range(nmodels):
        model = Model(dataAndFunction[i], cols, dataRange, xVariables[i], parNames, \
                      linlog=linlog, dFunc=dFunc, data_logY=data_logY)
        models.append(model)
        if model.sigma is None and sigma is not None:
            model.sigma = sigma

    composite_model = CompositeModel(models, parNames)
    params = plotBestFit(composite_model, params0, isPlot, errorbar, data_logY)
    return params

if __name__ == "__main__":
    plt.ioff()
    main()
