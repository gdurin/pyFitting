#!/usr/bin/env python

"""bestFit is a simple python script to perform data fitting 
using nonlinear least-squares minimization.
"""
import sys
import locale
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
    def __init__(self, xName, function, paramsNames, analyDeriv=None):
        self.xName = xName
        self.parameters = paramsNames
        self.fz = function
        self.fzOriginal = function
        self.checkFunction = True 
        self.analyDeriv = analyDeriv
        paramsNamesList = paramsNames.split(",")
        # Calculate the analytical derivatives
        # Return None if not available
        self.analyDeriv = getDiff(xName, function, paramsNamesList)
        try: 
            # Then try to compile them  to be reused by NumExpr
            self.derivsCompiled = map(ne.NumExpr, self.analyDeriv)
        except TypeError:
            print("Warning:  one or more functions are undefined in NumExpr")
            self.derivsCompiled = None

    def Y(self, x, params):
        exec "%s = params" % self.parameters
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
        exec "%s = x" % self.xName
        if self.derivsCompiled:
            for q in self.derivsCompiled:
                values = map(eval, q.input_names)
                jb.append(q(*values))
        else:
            for i, q in enumerate(self.analyDeriv):
                #print(q)
                while checkDerivative:
                    try:
                        exec "deriv = %s" % q
                        self.analyDeriv[i] = q
                        checkDerivative = False
                    except NameError as inst:
                        op = inst.message.split("'")[1]
                        q = genExpr2Scipy(op, q)
                        #print("Name error", q)
                jb.append(deriv)
                checkDerivative = True
            print self.analyDeriv
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
        self.typeColors = 'b'
        self.typePoints = 'o'

    def len(self):
        return len(self.X)

class Model():
    r"""Link data to theory, and provides all the methods
    to calculate the residual and the cost
    """
    def __init__(self,dataAndFunction,cols,dataRange, variables, parNames, linlog='lin', sigma=None, analyticalDerivs=False):
        fileName, func = dataAndFunction
        self.data = DataCurve(fileName,cols,dataRange)
        self.theory = Theory(variables[0], func,parNames,analyticalDerivs)
        self.linlog = linlog
        self.sigma = self.data.Yerror

    def residual(self, params):
        """Calculate residual for fitting"""
        self.residuals = np.array([])
        if self.sigma is None:  sigma = 1.
        P = self.theory.Y(self.data.X, params)
        if self.linlog=='lin':
            res = (P - self.data.Y)/sigma
        elif self.linlog=='log':
            #print scipy.log10(P), scipy.log10(data.Y)
            res = (scipy.log10(P) - scipy.log10(self.data.Y))/sigma
        self.residuals = np.concatenate((self.residuals, res))
        return self.residuals

    def jacobian(self, params):
        jb = self.theory.jacobian(self.data.X, params)
        if self.sigma is not None:
            jb = jb/self.sigma
        return jb

class CompositeModel():
    r"""Join the models
    """
    def __init__(self,models,parNames):
        self.models = models
        self.parStr = parNames.split(",")

    def residual(self, params):
        res = scipy.array([])
        for model in self.models:
            res = np.concatenate((model.residual(params), res))
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
                J = model.jacobian(params)
            else:
                J = np.concatenate((J, model.jacobian(params)),1)
        return J
    
def plotBestFitT(compositeModel, params0, linlog='lin',sigma=None, analyticalDerivs=False, isPlot='lin'):
    nStars = 80
    print("="*nStars)
    t0 = time()
    printOut = []
    table = []
    table.append(['parameter', 'value', 'st. error', 't-statistics'])
    print "Initial parameters = ", params0
    initCost = compositeModel.cost(params0)
    printOut.append(initCost)
    print 'initial cost = %.10e (StD: %.10e)' % compositeModel.cost(params0)
    maxfev = 500*(len(params0)+1)
    residual = compositeModel.residual
    jacobian = compositeModel.jacobian
    if analyticalDerivs:
        full_output = leastsq(residual,params0,
                              maxfev=maxfev, Dfun=jacobian, col_deriv=True, full_output=1)
    else:
        full_output = leastsq(residual,params0, maxfev=maxfev, full_output=1)
    params, covmatrix, infodict, mesg, ier = full_output
    costValue, costStdDev = compositeModel.cost(params)
    print 'optimized cost = %.10e (StD: %.10e)' % (costValue, costStdDev)
    printOut.append(costValue)
    jcb = compositeModel.jacobian(params)
    # The method of calculating the covariance matrix as
    # analyCovMatrix = scipy.matrix(scipy.dot(jcb, jcb.T)).I
    # is not valid in some cases. A general solution is to make the QR 
    # decomposition, as done by the routine
    if covmatrix is None: # fitting not converging
        for i in range(len(params)):
            stOut = compositeModel.parStr[i], '\t', params[i]
            print compositeModel.parStr[i], '\t', params[i]
            printOut.append(stOut)
    else:
        for i in range(len(params)):
            if not sigma == None: # This is the case of weigthed least-square
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
    pValue = 1. - scipy.special.gammainc(dof/2., costValue/2.)
    print "X^2_rel = %f" % (costValue/dof)
    print "pValue = %f (statistically significant if < 0.05)" % (pValue)
    ts = round(time() - t0, 3)
    print "*** Time elapsed:", ts
    if isPlot:
        getCol = getColor()
        getSyb = getSymbol()
        for model in compositeModel.models:
            calculatedData= model.theory.Y(model.data.X,params)
            X = model.data.X
            Y = model.data.Y
            color = getCol.next()
            style = getSyb.next() + color
            labelData = model.data.fileName
            labelTheory = model.theory.fzOriginal
            if isPlot == "lin":
                plt.plot(X,Y,style,label=labelData)
                plt.plot(X,calculatedData,color,label=labelTheory)
            else: 
                plt.loglog(X,Y,style,label=labelData)    
                plt.loglog(X,calculatedData,color,label=labelTheory)    
            if sigma is not None:
                plt.errorbar(X,Y, sigma,fmt=None)
            plt.draw()
        plt.legend(loc=0)
        plt.show()
    # Alternative fitting
    #full_output = scipy.optimize.curve_fit(func,data.X,data.Y,params0,None)
    #print "Alternative fitting"
    #print full_output
        #fig2 = plt.figure(2)
        #plt.semilogx(data.X, data.Y-theory.Y(data.X,params),'-ro')
        #plt.draw()
        #plt.show()
    return params


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
    # default values of input parameters:
    linlog = "lin"
    isPlot = 'lin'
    dataRangeMin = 0
    dataRangeMax = None
    dataRange = dataRangeMin, dataRangeMax
    cols = 0,1
    parNames = "a,b"
    variables = "x","y"
    func = "a+b*x"
    sigma = None
    analyticalDerivs = False
    helpString = """

    Usage summary: bestFit [OPTIONS]

    OPTIONS:
    -f, --filename   Filename of the data in form of columns
    -c, --cols          Columns to get the data (defaults: 0,1); a third number is used for errors' column
    -v, --vars         Variables (defaults: x,y)
    -r, --range       Range of the data to consider (i.e. 0:4; 0:-1 takes all)
    -p, --fitpars      Fitting Parameters names (separated by comas)
    -i, --initvals      Initial values of the parameters (separated by comas)
    -t, --theory       Theoretical function to best fit the data (between "...")
    -s, --sigma       Estimation of the error in the data (as a constant value)
    -d, --derivs      Use analytical derivatives
    --lin                 Use data in linear mode (default)    
    --log                Use data il log mode (best for log-log data)
    --noplot           Don't show the plot output
    --logplot          Use log-log axis to plot data (default if --log)

    EXAMPLE:
    bestfit -f mydata.dat -c 0,2 -r 10:-1 -v x,y -p a,b -i 1,1. -t "a+b*x"
    """
    failString = "Failed: Not enough input filenames specified"


    if len(sys.argv) == 1:
        print failString
        print helpString
        sys.exit()

    if "-f" not in sys.argv and "--filename" not in sys.argv:
        print failString
        print helpString
        sys.exit()

    # read variables from the command line, one by one:
    while len(sys.argv) > 1:
        option = sys.argv[1]
        del sys.argv[1]
        if option == '-f' or option == "--filename":
            fileNames = sys.argv[1].split(",")
            del sys.argv[1]
        elif option == '-c' or option == "--cols":
            cols = sys.argv[1].split(",")
            cols = [int(i) for i in cols]
            del sys.argv[1]
        elif option == '-d' or option == '--deriv':
            analyticalDerivs = True
        elif option == '-v' or option == "--vars":
            variables = tuple(sys.argv[1].split(","))
            del sys.argv[1]
        elif option == '-t' or option == "--theory":
            functions = sys.argv[1].split(",")
            del sys.argv[1]
        elif option == '-p' or option == "--fitpars":
            parNames = sys.argv[1]
            del sys.argv[1]
        elif option == '-i' or option == "--initvals":
            pOut = []
            for p in sys.argv[1].split(","):
                pOut.append(float(p))
            params0 = tuple(pOut)
            del sys.argv[1]
        elif option=='--log':
            linlog = "log"
            isPlot = 'log'
        elif option=='--lin':
            linlog = "lin"
        elif option=='-s':
            sigma = float(sys.argv[1])
            del sys.argv[1]
        elif option=='--noplot' :
            isPlot = False
        elif option=='--logplot':
            isPlot = 'log'
        elif option == '-r' or option == "--range":
            dataRange = sys.argv[1]
            if ":" in dataRange:
                m, M = dataRange.split(":")
                if m == "":
                    dataRangeMin = 0
                else:
                    dataRangeMin = int(m)
                if M == "":
                    dataRangeMax = None
                else:
                    dataRangeMax = int(M)
                dataRange = dataRangeMin, dataRangeMax
            else:
                print "Error in setting the data range: use min:max"
                sys.exit()

    dataAndFunction = zip(fileNames, functions)
    models = []
    nmodels = len(fileNames)
    for i in range(nmodels):
        model = Model(dataAndFunction[i],cols,dataRange, variables, parNames, linlog, analyticalDerivs)
        models.append(model)
    
    composite_model = CompositeModel(models, parNames)
   #if not theory.analyDeriv:
        #analyticalDerivs = False
    params = plotBestFitT(composite_model,params0,isPlot)

    
if __name__ == "__main__":
    plt.ioff()
    main()
