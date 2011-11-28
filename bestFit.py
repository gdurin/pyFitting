#!/usr/bin/env python

"""bestFit is a simple python script to perform data fitting 
using nonlinear least-squares minimization.
"""
import sys
import locale
import argparse
import scipy
from scipy.optimize.minpack import leastsq
import scipy.special as special
import numpy as np
from numpy import pi
import matplotlib as mpl
try:
    mpl.use('Qt4Agg')
except:
    raise

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
    Both usual function and special ones are considered
    """
    try:
        if op in dir(scipy):
            sub = "scipy."
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
        print("Function %s not defined in scipy" % op)
        return None

class Theory:
    r"""
    Defines the theoretical function to fit the data with (the model)
    """
    def __init__(self, xName, function, paramsNames, heldParams = None, dFunc=False):
        self.xName = xName
        self.parameters = paramsNames
        paramsNamesList = paramsNames.split(",")
        self.fz = function
        self.fzOriginal = function
        self.checkFunction = True 
        self.heldParams = heldParams
        # Calculate the analytical derivatives
        # Return None if not available
        self.dFunc = dFunc
        if dFunc:
            self.dFunc = getDiff(xName, function, paramsNamesList)
            try: 
                # Then try to compile them  to be reused by NumExpr
                self.dFuncCompiled = map(ne.NumExpr, self.dFunc)
            except TypeError:
                print("Warning:  one or more functions are undefined in NumExpr")
                self.dFuncCompiled = None

    def Y(self, x, params):
        exec "%s = params" % self.parameters
        if self.heldParams:
            for par in self.heldParams:
                exec "%s = %s" % (par, str(self.heldParams[par]))
        # Check if the function needs to be changed with scipy.functions
        if self.checkFunction:
            while self.checkFunction:
                try:
                    exec "f = %s" % (self.fz)
                    self.checkFunction = False
                except NameError as inst:
                    op = inst.message.split("'")[1]
                    function = genExpr2Scipy(op, self.fz)
                    if function:
                        self.fz = function
                    else:
                        raise ValueError("Function %s not found" % op)
        else:
            exec "f = %s" % (self.fz)
        return f

    def jacobian(self, x, parameterValues):
        """
        Calculus of the jacobian with analytical derivatives
        """
        jb = []
        checkDerivative = True
        exec "%s = parameterValues" % self.parameters
        if self.heldParams:
            for par in self.heldParams:
                exec "%s = %s" % (par, str(self.heldParams[par]))
        exec "%s = x" % self.xName
        if self.dFuncCompiled:
            for q in self.dFuncCompiled:
                values = map(eval, q.input_names)
                jb.append(q(*values))
        else:
            for i, q in enumerate(self.dFunc):
                while checkDerivative:
                    try:
                        exec "deriv = %s" % q
                        self.dFunc[i] = q
                        checkDerivative = False
                    except NameError as inst:
                        op = inst.message.split("'")[1]
                        q = genExpr2Scipy(op, q)
                jb.append(deriv)
                checkDerivative = True
        return scipy.array(jb)

class DataCurve:
    def __init__(self, fileName, cols, dataRange=(0,None)):
        data = scipy.loadtxt(fileName)
        self.fileName = fileName
        i0,i1 = dataRange
        self.X = data[:,cols[0]][i0:i1]
        self.Y = data[:,cols[1]][i0:i1]
        if len(cols) > 2:
            self.Yerror = data[:,cols[2]][i0:i1]
        else:
            self.Yerror = None
        
    def len(self):
        return len(self.X)

class Model():
    r"""Link data to theory, and provides all the methods
    to calculate the residual, the jacobian and the cost
    """
    def __init__(self, dataAndFunction, cols, dataRange, variables, parNames, \
                 heldParams=None, linlog='lin', sigma=None, dFunc=False):
        fileName, func = dataAndFunction
        self.data = DataCurve(fileName, cols, dataRange)
        self.theory = Theory(variables[0], func, parNames, heldParams, dFunc)
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
            res = (scipy.log10(P) - scipy.log10(self.data.Y))/sigma
        self.residuals = np.concatenate((self.residuals, res))
        return self.residuals

    def jacobian(self, params):
        jac = self.theory.jacobian(self.data.X, params)
        if self.sigma is not None:
            jac = jac/self.sigma
        return jac

class CompositeModel():
    r"""Join the models
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
        res = scipy.array([])
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
    
def plotBestFitT(compositeModel, params0, isPlot='lin'):
    nStars = 80
    print("="*nStars)
    t0 = time()
    printOut = []
    table = []
    table.append(['parameter', 'value', 'st. error', 't-statistic'])
    print "Initial parameters = ", params0
    initCost = compositeModel.cost(params0)
    printOut.append(initCost)
    print 'initial cost = %.10e (StD: %.10e)' % compositeModel.cost(params0)
    maxfev = 250*(len(params0)+1)
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
    params, covmatrix, infodict, mesg, ier = full_output
    costValue, costStdDev = compositeModel.cost(params)
    print 'optimized cost = %.10e (StD: %.10e)' % (costValue, costStdDev)
    printOut.append(costValue)
    #if compositeModel.isAnalyticalDerivs:
        #jcb = jacobian(params)
    ## The method of calculating the covariance matrix as
        #analyCovMatrix = scipy.matrix(scipy.dot(jcb, jcb.T)).I
        #print analyCovMatrix
        #print covmatrix
    # is not valid in some cases. A general solution is to make the QR 
    # decomposition, as done by the routine
    if covmatrix is None: # fitting not converging
        for i in range(len(params)):
            stOut = compositeModel.parStr[i], '\t', params[i]
            print compositeModel.parStr[i], '\t', params[i]
            printOut.append(stOut)
    else:
        for i in range(len(params)):
            if compositeModel.isSigma: # This is the case of weigthed least-square
                stDevParams = covmatrix[i,i]**0.5
            else:
                stDevParams = covmatrix[i,i]**0.5*costStdDev
            par = params[i]
            table.append([compositeModel.parStr[i], par, stDevParams, par/stDevParams])
            #if abs(params[i]) > 1e5:
                #print "%s = %.8e +- %.8e" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            #else:
                #print "%s = %.8f +- %.8f" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            stOut = compositeModel.parStr[i], '\t', params[i], '+-', stDevParams

            printOut.append(stOut)    
    print("="*nStars)
    pprint_table(table)
    print("="*nStars)        
    print "Done in %d iterations" % infodict['nfev']
    print mesg
    print("="*nStars)
    # Chi2 test
    # n. of degree of freedom
    lenData = sum([model.data.len() for model in compositeModel.models])
    print "n. of data = %d" % lenData
    dof = lenData - len(params)
    print "degree of freedom = %d" % (dof)
    print "X^2_rel = %f" % (costValue/dof)
    #pValue = 1. - scipy.special.gammainc(dof/2., costValue/2.)
    #print "pValue = %f (statistically significant if < 0.05)" % (pValue)
    ts = round(time() - t0, 3)
    print "*** Time elapsed:", ts
    if isPlot:
        getCol = getColor()
        getSyb = getSymbol()
        for model in compositeModel.models:
            X = model.data.X
            Y = model.data.Y
            X1 = scipy.linspace(X[0], X[-1], 300)
            calculatedData= model.theory.Y(X1, params)
            color = getCol.next()
            style = getSyb.next() + color
            labelData = model.data.fileName
            labelTheory = model.theory.fzOriginal
            if isPlot == "lin":
                plt.plot(X, Y, style, label=labelData)
                plt.plot(X1, calculatedData, color, label=labelTheory)
            else: 
                plt.loglog(X, Y, style, label=labelData)    
                plt.loglog(X1, calculatedData, color, label=labelTheory)    
            if model.sigma is not None:
                plt.errorbar(X, Y, model.sigma, fmt=None)
            plt.draw()
        #plt.legend(loc=0)
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
        return locale.format("%.5f", (0, inum), True)
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

    col_paddings = []

    for i in range(len(table[0])):
        col_paddings.append(get_max_width(table, i))

    for row in table:
        # left col
        print >> out, row[0].ljust(col_paddings[0] + 1),
        # rest of the cols
        for i in range(1, len(row)):
            col = format_num(row[i]).rjust(col_paddings[i] + 2)
            print >> out, col,
        print >> out

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Best fit of data using least-square minimization')
    parser.add_argument('-f','--filename', metavar='filename', nargs='+', required=True,
                       help='Filename(s) of the input data')
    parser.add_argument('-t','--theory', metavar='theory', nargs='+', required=True,
                        help='Theoretical function(s)')
    parser.add_argument('-p','--params', metavar='params', required=True,  nargs='+',
                        help='Parameter(s) name(s), i.e. -p a b c')
    parser.add_argument('-i','--initvals', metavar='initvals', required=True, type=float, nargs='+',
                        help='Initial values of the parameter(s), i.e. -i 1 2. 3.')
    parser.add_argument('-v', '--var', metavar='var', default='x y', nargs='+',
                        help='Variable(s) names, default: x y')
    parser.add_argument('-c','--cols', metavar='cols', default=[0, 1],  type=int, nargs='+',
                       help='Columns of the file to load the data, default: 0 1')
    parser.add_argument('-r', '--drange', metavar='range', default='0:None',
                        help='Range of the data (as index of rows)')
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
    
    
    args = parser.parse_args()
    fileNames = args.filename
    cols =  args.cols
    variables = args.var
    functions = args.theory
    parNames = ",".join(args.params)
    params0 = tuple(args.initvals)
    if args.held:
        heldParams = {}
        for p in args.held:
            [par,val] = p.split("=")
            heldParams[par] = float(val)
    else:
        heldParams = None
        
    dFunc = args.deriv
    
    dataRange = args.drange
    m, M = dataRange.split(":")
    if m == "":
        dataRangeMin = 0
    else:
        dataRangeMin = int(m)
    if M == "" or M == 'None':
        dataRangeMax = None
    else:
        dataRangeMax = int(M)
    dataRange = dataRangeMin, dataRangeMax
    
    linlog = "lin"
    isPlot = "lin"
    if args.log:
        linlog = "log"
        isPlot = 'log'
    if args.noplot:
        isPlot = False
    if args.logplot:
        isPlot = 'log'
    sigma = args.sigma

    dataAndFunction = zip(fileNames, functions)
    models = []
    nmodels = len(fileNames)
    for i in range(nmodels):
        model = Model(dataAndFunction[i], cols, dataRange, variables, parNames, \
                      heldParams=heldParams, linlog=linlog, dFunc=dFunc)
        models.append(model)
        if model.sigma is None and sigma is not None:
            model.sigma = sigma

    composite_model = CompositeModel(models, parNames)
    params = plotBestFitT(composite_model, params0, isPlot)

    
if __name__ == "__main__":
    plt.ioff()
    main()
