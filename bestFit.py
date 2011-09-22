#!/usr/bin/env python

import sys
import locale
import scipy as sp
from scipy.optimize.minpack import leastsq
import scipy.special as special
import numpy as np
from numpy import pi
import matplotlib as mpl
import matplotlib.pyplot as plt
import numexpr as ne
from time import time
import re
from getAnalyticalDerivatives import getDiff


def genExpr2Scipy(op, function):
    """
    Insert the proper scipy.* module to handle math functions
    """
    try:
        if op in dir(sp):
            sub = "sp."
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
    def __init__(self, xName, function, paramsNames, params0, analyDeriv=None):
        self.xName = xName
        self.parameters = paramsNames
        self.initialParams = params0
        self.parStr = self.parameters.split(",")
        self.fz = function
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
        # Check if the function needs to be changed with sp.functions
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

    def jacobian(self, x, parameterValues, sigma=None):
        """
        Calculus of the jacobian with analytical derivatives
        """
        if sigma == None: 
            sigma = 1.
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
        return sp.array(jb)/sigma

class DataCurve:
    def __init__(self, fileName, cols, dataRange=(0,None)):
        data = sp.loadtxt(fileName)
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


def residual(params, theory, data, linlog, sigma=None):
    """Calculate residual for fitting"""
    residuals = np.array([])
    if sigma is None:  sigma = 1.
    P = theory.Y(data.X, params)
    if linlog=='lin':
        res = (P - data.Y)/sigma
    elif linlog=='log':
        #print sp.log10(P), sp.log10(data.Y)
        res = (sp.log10(P) - sp.log10(data.Y))/sigma
    residuals = np.concatenate((residuals, res))
    return residuals

def cost(params, theory, data, linlog, sigma):
    res = residual(params, theory, data, linlog, sigma)
    cst = np.dot(res,res)
    # Standard error of the regression
    ser = (cst/(data.len()-len(params)))**0.5
    return cst, ser

def func(params, data, theory):
    return theory.Y(data.X, params)

def jacobian(params, theory, data, linlog,sigma):
    return theory.jacobian(data.X, params,sigma)

def plotBestFitT(theory, data, linlog, sigma=None, analyticalDerivs=False, isPlot='lin'):
    nStars = 80
    print("="*nStars)
    t0 = time()
    printOut = []
    table = []
    table.append(['parameter', 'value', 'st. error', 't-statistics'])
    params0 = theory.initialParams
    print "Initial parameters = ", params0
    initCost = cost(params0, theory,data,linlog,sigma)
    printOut.append(initCost)
    print 'initial cost = %.10e (StD: %.10e)' % cost(params0, theory,data,linlog,sigma)
    maxfev = 500*(len(params0)+1)
    if analyticalDerivs:
        full_output = leastsq(residual,params0,args=(theory,data,linlog,sigma),\
                              maxfev=maxfev, Dfun=jacobian, col_deriv=True, full_output=1)
    else:
        full_output = leastsq(residual,params0,\
                              args=(theory,data,linlog,sigma), maxfev=maxfev, full_output=1)
    params, covmatrix, infodict, mesg, ier = full_output
    costValue, costStdDev = cost(params,theory,data,linlog,sigma)
    print 'optimized cost = %.10e (StD: %.10e)' % (costValue, costStdDev)
    printOut.append(costValue)
    jcb = jacobian(params, theory, data, linlog,sigma)
    # The method of calculating the covariance matrix as
    # analyCovMatrix = sp.matrix(sp.dot(jcb, jcb.T)).I
    # is not valid in some cases. A general solution is to make the QR 
    # decomposition, as done by the routine
    if covmatrix is None: # fitting not converging
        for i in range(len(params)):
            stOut = theory.parStr[i], '\t', params[i]
            print theory.parStr[i], '\t', params[i]
            printOut.append(stOut)
    else:
        for i in range(len(params)):
            if not sigma == None: # This is the case of weigthed least-square
                stDevParams = covmatrix[i,i]**0.5
            else:
                stDevParams = covmatrix[i,i]**0.5*costStdDev
            par = params[i]
            table.append([theory.parStr[i], par, stDevParams, par/stDevParams])
            #if abs(params[i]) > 1e5:
                #print "%s = %.8e +- %.8e" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            #else:
                #print "%s = %.8f +- %.8f" % (theory.parStr[i].ljust(5), params[i], stDevParams)
            stOut = theory.parStr[i], '\t', params[i], '+-', stDevParams

            printOut.append(stOut)    
    print("="*nStars)
    pprint_table(table)
    print("="*nStars)        
    print "Done in %d iterations" % infodict['nfev']
    print mesg
    print("="*nStars)
    # Chi2 test
    # n. of degree of freedom
    print "n. of data = %d" % data.len()
    dof = data.len() - len(params)
    print "degree of freedom = %d" % (dof)
    pValue = 1. - sp.special.gammainc(dof/2., costValue/2.)
    print "X^2_rel = %f" % (costValue/dof)
    print "pValue = %f (statistically significant if < 0.05)" % (pValue)
    ts = round(time() - t0, 3)
    print "*** Time elapsed:", ts
    if isPlot:
        calculatedData= theory.Y(data.X,params)
        if isPlot == "lin":
            plt.plot(data.X,data.Y,'bo',data.X,calculatedData,'r')
        else: 
            plt.loglog(data.X,data.Y,'bo',data.X,calculatedData,'r')    
        if sigma is not None:
            plt.errorbar(data.X,data.Y, sigma,fmt=None)
        plt.title(data.fileName)
        #plt.show()
    # Alternative fitting
    #full_output = sp.optimize.curve_fit(func,data.X,data.Y,params0,None)
    #print "Alternative fitting"
    #print full_output
        fig2 = plt.figure(2)
        plt.semilogx(data.X, data.Y-theory.Y(data.X,params),'-ro')
        plt.draw()
        plt.show()
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
    bestFit v.0.1.3
    august 19 - 2011

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
    --logplot          Use log-log to plot data (default if --log)

    EXAMPLE
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

    data = DataCurve(fileName,cols,dataRange)
    theory = Theory(variables[0], func,parNames, params0,analyticalDerivs)
    if data.Yerror is not None:
        sigma = data.Yerror
    if not theory.analyDeriv:
        analyticalDerivs = False
    params = plotBestFitT(theory,data,linlog,sigma,analyticalDerivs,isPlot)

    
if __name__ == "__main__":
    plt.ioff()
    main()
