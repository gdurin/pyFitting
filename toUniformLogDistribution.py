#!/usr/bin/env python
"""
calculate a distribution uniform in log scale from raw data
"""
import sys, os
import scipy
import matplotlib.pyplot as plt
import glob

N_POINTS_PER_DECADE = 8
PERCENTAGE_ADJUSTMENT_FIRST_BIN = 0.02

def makeSymbolsAndColors():
    """
    Yield a Symbol and Color
    """
    # YJC: right now we have 128 different values, need to have more colors
    # which means we need to fix the 
    # 13 symbols, 7 colors: numbers should be relatively prime
    # 143 maximum different
    pointSymbolTypes = ['o','^','v','<','>','s','p','+','x','d','h','*','.']
    pointColorTypes = ['b','g','r','c','m','k','y']
    # Replicate to make enough symbols for different data types
    symbolList = len(pointColorTypes) * pointSymbolTypes
    colorList = len(pointSymbolTypes) * pointColorTypes
    for s,c in zip(symbolList, colorList):
        yield s+c

if __name__ == "__main__":
    if len(sys.argv)<2:
        print """
        Usage:  python toUniformLogDistribution filename
        or
        python toUniformLogDistribution "*.dat"
        """
        sys.exit()
    sc = makeSymbolsAndColors()
    #print sys.argv
    arg1 = sys.argv[1]
    print arg1
    if "*" in arg1:
        #print arg1
        fileNames  = glob.glob(arg1)
        #print fileNames
    else:
        fileNames = arg1
    for filename in fileNames:
        print "Analysing...", filename
        data = scipy.loadtxt(filename)
        # Get rid of zeros
        data = [d for d in data if d!=0]
        lxMin, lxMax= scipy.log10(min(data)), scipy.log10(max(data))
        print 10**lxMin, 10**lxMax
        # Set the lower limit        
        lxMin *= 1. - PERCENTAGE_ADJUSTMENT_FIRST_BIN
        lxMax *= 1. + PERCENTAGE_ADJUSTMENT_FIRST_BIN
        decades = scipy.int16(lxMax-lxMin)+1
        # construct the edges uniform in log scale
        nPoints = decades*8
        edges = scipy.logspace(lxMin, lxMax, nPoints)
        print edges
        h, edg = scipy.histogram(data, edges, normed=False)
        centers = 10**((scipy.log10(edges[1:])+scipy.log10(edges[:-1]))/2.)
        bins = edges[1:]-edges[:-1]
        # Check if any zero occurs
        if 0.0 in h:
            print "Zero value occurs"
        # Check if h is 1 somewhere
        bool1 = h!=1.
        bool0 = h!=0.
        bools = bool0 + bool1
        h = scipy.compress(bools, h)
        centers = scipy.compress(bools, centers)
        bins = scipy.compress(bools, bins)
        hist = h/bins
        hist = hist/sum(h)
        # We suppose the errorbar of the hist value is given by binomial distribution
        hTotal = sum(h)
        error = (h*(1.-h/hTotal))**0.5/(bins*hTotal)

        color = sc.next()
        plt.loglog(centers, hist, color, label=filename)
        plt.errorbar(centers, hist,error)
        plt.draw()
        
        # Save the files
        f, ext = os.path.splitext(filename)
        fOut = f+".bnd"
        F = scipy.savetxt(fOut,zip(centers,hist, error))
        # Check if normalization is correct
        X = centers
        lgX = scipy.log10(X)
        D = lgX[1] - lgX[0]
        b = 10**(lgX+D/2.) - 10**(lgX-D/2.)
        print sum(b*hist)
    #plt.legend(loc=0)
    plt.show()
