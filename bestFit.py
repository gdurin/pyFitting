#!/usr/bin/env python

import sys
import locale
import scipy
import numpy as np
from numpy import pi
import matplotlib.pylab as pylab
import scipy.optimize
import scipy.special
from scipy.optimize.minpack import leastsq
import numexpr as ne
from time import time
from getAnalyticalDerivatives import getDiff
from matplotlib import rc
rc('font', **{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
#operators = ["exp","log", "log10","cos","sin","tan","tanh"]



class Theory:
    def __init__(self, xName, function, paramsNames, params0, analyDeriv):
        self.xName = xName
        self.parameters = paramsNames
        self.initialParams = params0
        self.parStr = self.parameters.split(",")
        self.fz = function
        self.checkFunction = True 
        if analyDeriv:
            paramsNamesList = paramsNames.split(",")
            # Calculate the analytical derivatives
            self.derivs = getDiff(xName, function, paramsNamesList)
            # Then compiled them  to be reused by NumExpr
            self.derivsCompiled = map(ne.NumExpr, self.derivs)
            
    def Y(self, x, params):
        exec "%s = params" % self.parameters
        # Check if the function needs to be changed with np.functions
        if self.checkFunction:
            while self.checkFunction:
                try:
                    exec "f = %s" % (self.fz)
                    self.checkFunction = False
                except NameError as inst:
                    op = inst.message.split("'")[1]
                    self.fz = self.fz.replace(op, "np."+op)
        else:
            exec "f = %s" % (self.fz)
        return f

    def jacobian(self, x, parameterValues, sigma=None):
        """
        Calculus of the jacobian with analytical derivatives
        """
        if sigma == None: 
            sigma = 1.
        jb = []
        exec "%s = parameterValues" % self.parameters
        exec "%s = x" % self.xName
        # Prepare the dictionary of all the variables in the derivatives
        for q in self.derivsCompiled:
            values = map(eval, q.input_names)
            jb.append(q(*values))
        jb = scipy.array(jb)/sigma

        #jb = scipy.array(map(ne.evaluate,self.derivs))/sigma
        return jb
    
class DataCurve:
    def __init__(self, fileName, cols, dataRange=(0,None)):
        data = scipy.loadtxt(fileName)
        i0,i1 = dataRange
        self.X = data[:,cols[0]][i0:i1]
        self.Y = data[:,cols[1]][i0:i1]
        if len(cols) > 2:
            self.Yerror = 2.*data[:,cols[2]][i0:i1]
        else:
            self.Yerror = None
        self.typeColors = 'b'
        self.typePoints = 'o'

    def len(self):
        return len(self.X)

    
def residual(params, theory, data, linlog, sigma=None, logResidual=False):
    """Calculate residual for fitting"""
    residuals = np.array([])
    if sigma is None:  sigma = 1.
    P = theory.Y(data.X,params)
    if not logResidual:
        res = (P - data.Y)/sigma
    else:
        res = (scipy.log10(P)-scipy.log10(data.Y))/sigma
    residuals = np.concatenate((residuals,res))
    return residuals

def cost(params, theory, data, linlog, sigma):
    res = residual(params,theory,data,linlog,sigma)
    cst = np.dot(res,res)
    # Standard error of the regression 
    # see parameter SER in 
    # http://en.wikipedia.org/wiki/Numerical_methods_for_linear_least_squares
    # under: Parameter errors, correlation and confidence limits
    #
    ser = (cst/(data.len()-len(params)))**0.5
    return cst, ser

def func(params, data, theory):
    return theory.Y(data.X, params)

def jacobian(params, theory, data, linlog,sigma):
    return theory.jacobian(data.X, params,sigma)

def plotBestFitT(theory, data, linlog, sigma, analyticalDerivs=False, noplot=False):
    t0 = time()
    printOut = []
    table = []
    table.append(['parameter', 'value', 'st. error', 't-statistics'])
    params0 = theory.initialParams
    print "Initial parameters = ", params0
    initCost = cost(params0, theory, data,linlog,sigma)
    printOut.append(initCost)
    print 'initial cost = %.10e (StD: %.10e)' % cost(params0, theory, data,linlog,sigma)
    if analyticalDerivs:
        full_output = leastsq(residual,params0,args=(theory,data,linlog,sigma),\
                              Dfun=jacobian, col_deriv=True, full_output=1)
    else:
        full_output = leastsq(residual,params0,\
                              args=(theory,data,linlog,sigma), full_output=1)
    params = full_output[0]
    costValue, costStdDev = cost(params, theory, data, sigma=sigma, linlog='lin')
    print 'optimized cost = %.10e (StD: %.10e)' % (costValue, costStdDev)
    printOut.append(costValue)
    covarianceMatrix = full_output[1]
    # Check che value of the cov. matrix
    #jcb = jacobian(params, theory, data, linlog,sigma)
    #covMatrix = scipy.matrix(scipy.dot(jcb, jcb.T)).I
    #print covarianceMatrix
    #print "================="
    #print covMatrix
    #print "================="
    #print covarianceMatrix/covMatrix
    if covarianceMatrix is None:
        for i in range(len(params)):
            stOut = theory.parStr[i], '\t', params[i]
            print theory.parStr[i], '\t', params[i]
            printOut.append(stOut)
    else:
        for i in range(len(params)):
            if sigma: # This is the case of weigthed least-square
                stDevParams = scipy.sqrt(covarianceMatrix[i,i])
            else:
                stDevParams = scipy.sqrt(covarianceMatrix[i,i])*costStdDev
            par = params[i]
            table.append([theory.parStr[i], par, stDevParams, par/stDevParams])
            #if abs(params[i]) > 1e5:
                #print "%s = %.8e +- %.8e" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            #else:
                #print "%s = %.8f +- %.8f" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            stOut = theory.parStr[i], '\t', params[i], '+-', scipy.sqrt(covarianceMatrix[i,i])
            printOut.append(stOut)    
    print "===================="
    pprint_table(table)
    print "===================="
    # Chi2 test
    # n. of degree of freedom
    print "Done in %d iteractions" % full_output[2]['nfev']
    print "n. of data = %d" % data.len()
    dof = data.len() - len(params)
    print "degree of freedom = %d" % (dof)
    pValue = 1.-scipy.special.gammainc(dof/2., costValue/2.)
    print "X^2_rel = %f" % (costValue/dof)
    print "pValue = %f (statistically significant if < 0.05)" % (pValue)
    ts = round(time() - t0, 3)
    print "*** Time elapsed:", ts
    if not noplot:
        P = theory.Y(data.X,params)
        if linlog == "lin":
            pylab.plot(data.X,data.Y, 'bo',data.X,P,'r')
        else: 
            pylab.loglog(data.X,data.Y, 'bo',data.X,P,'r')
        if sigma is not None:
            pylab.errorbar(data.X,data.Y, sigma)
        pylab.show()
    # Alternative fitting
    #full_output = scipy.optimize.curve_fit(func,data.X,data.Y,params0,None)
    #print "Alternative fitting"
    #print full_output

    return


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
    dataRangeMin = 0
    dataRangeMax = None
    dataRange = dataRangeMin, dataRangeMax
    cols = 0,1
    parNames = "a,b"
    func = "a+b*x"
    sigma = None
    helpString = """
    bestFit v.0.1.0
    july 26 - 2011

    Usage summary: bestFit [OPTIONS]

    OPTIONS:
    -f, --filename     Filename of the data in form of columns
    -c, --cols         Columns to get the data (defaults: 0,1); a third number is used for errors' column
    -v, --vars         Variables (defaults: x,y)
    -r, --range        Range of the data to consider (i.e. 0:4; 0:-1 takes all)
    -p, --fitpars      Fitting Parameters names (separated by comas)
    -i, --initvals     Initial values of the parameters (separated by comas)
    -t, --theory       Theoerical function to best fit the data
    -s, --sigma        Estimation of the error in the data
    -d, --derivs    Use analytical derivatives
    --lin              Use data in linear mode (default)
    --log              Use data il log mode (best for log-log data)
    --noplot         Don't show the plot output

    EXAMPLE
    bestfit -f mydata.dat -c 0,2 -r 10:-1 -v x,y -p a,b -i 1,1. -t a+b*x
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

    variables = "x","y"
    analyticalDerivs = False
    noPlot = False
    # read variables from the command line, one by one:
    while len(sys.argv) > 1:
        option = sys.argv[1]
        del sys.argv[1]
        if option == '-f' or option == "--filename":
            fileName = sys.argv[1]
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
            func = sys.argv[1]
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
        elif '-log' in option :
            linlog = "log"
        elif  '-lin' in option:
            linlog = "lin"        
        elif '-s' in option:
            sigma = float(sys.argv[1])
            del sys.argv[1]
        elif '--noplot':
            noPlot = True
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
                
    data = DataCurve(fileName,cols,dataRange)
    theory = Theory(variables[0], func, parNames, params0,analyticalDerivs)
    if data.Yerror is not None:
        sigma = data.Yerror
    plotBestFitT(theory,data,linlog,sigma, analyticalDerivs,noPlot)

    
if __name__ == "__main__":
    main()
