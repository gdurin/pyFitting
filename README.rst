bestFit: a command-line tool for least-square fitting
=====================================================

A simple python script to perform data fitting using nonlinear least-squares minimization. 

http://emma.inrim.it:8080/gdurin/software/bestfit

The routine is based on the scipy.optimize.leastsq method (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq), which is a wrapper around MINPACK’s lmdif and lmder algorithms.

This is a command line tool, to keep everything easy and fast to use. 

0. Download and installation
----------------------------

The easiest way to install bestFit is::
   
    $ git clone git://github.com/gdurin/pyFitting.git

It requires the following packages::

python 2.7.x (because of the print("Print this") statement. It can be easily change to the 2.6.x version print "Print this"

numpy and scipy  (see http://www.scipy.org)

matplotlib (1.0.x - see http://matplotlib.sourceforge.net/ )

numexpr (see http://code.google.com/p/)

If not present, install them with easy_install, i.e. easy_install numexpr (under root)

1. How to use it
----------------

Make the bestFit.py executable (under Linux: chmod +x bestFit.py) and run it without any option. The output is the following::

    Failed: Not enough input filenames specified

    bestFit v.0.1.2
    august 18 - 2011

    Usage summary: bestFit [OPTIONS]

    OPTIONS:
    -f, --filename   Filename of the data in form of columns
    -c, --cols       Columns to get the data (defaults: 0,1); a third number is used for errors' column
    -v, --vars       Variables (defaults: x,y)
    -r, --range      Range of the data to consider (i.e. 0:4; 0:-1 takes all)
    -p, --fitpars    Fitting Parameters names (separated by comas)
    -i, --initvals   Initial values of the parameters (separated by comas)
    -t, --theory     Theoretical function to best fit the data (between "...")
    -s, --sigma      Estimation of the error in the data (as a constant value)
    -d, --derivs     Use analytical derivatives
    --lin            Use data in linear mode (default)
    --log            Use data il log mode (best for log-log data)
    --noplot         Don't show the plot output

    EXAMPLE
    bestfit -f mydata.dat -c 0,2 -r 10:-1 -v x,y -p a,b -i 1,1. -t "a+b*x"


2. Test the script
------------------

To be sure that the script is working correctly, try one of the test fits included.

For instance, try to fit the eckerle4 data (see: http://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml for details). 
This is a case considered of high difficulty.

On the command line copy the following line:: 

   $ ./bestFit.py -f test/eckerle4/data.dat -p b1,b2,b3 -t "b1/b2*exp(-(x-b3)**2/(2.*b2**2))" -i 1.,10.,500. -c 1,0 -d

[Hint: make a soft link to bestFit.py in your local bin, such as
ln -s yourDirectory/bestFit.py /usr/local/bin/bestFit
and then move to the test/ecklerle4 directory so to simply use as:
bestFit -t data.dat ....]

The results should be similar to my output::

    >>> Initial parameters =  (1.0, 10.0, 500.0)
    >>> initial cost = 7.2230265030e-01 (StD: 1.5023966794e-01)
    >>> optimized cost = 1.4635887487e-03 (StD: 6.7629245447e-03)

    >>> parameter            value         st. error    t-statistics
    >>> b1            1.5543826681   0.0154080427126   100.881253842
    >>> b2           4.08883191419   0.0468029694065   87.3626602338
    >>> b3           451.541218624   0.0468004675198   9648.22025407
    >>> Done in 18 iterations
    >>> Both actual and predicted relative reductions in the sum of squares are at most 0.000000

    >>> n. of data = 35
    >>> degree of freedom = 32
    >>> X^2_rel = 0.000046
    >>> pValue = 1.000000 (statistically significant if < 0.05)

In this run we have used the analytical derivatives with the "-d" option. Try now not to use it, so::
 
    $ ./bestFit.py -f test/eckerle4/data.dat -p b1,b2,b3 -t "b1/b2*exp(-(x-b3)**2/(2.*b2**2))" -i 1.,10.,500. -c 1,0 

    >>> Initial parameters =  (1.0, 10.0, 500.0)
    >>> initial cost = 7.2230265030e-01 (StD: 1.5023966794e-01)
    >>> optimized cost = 1.4635887487e-03 (StD: 6.7629245447e-03)
    >>> parameter            value         st. error    t-statistics
    >>> b1           1.55438266849   0.0154080427799   100.881253426
    >>> b2           4.08883191593   0.0468029705094   87.3626582123
    >>> b3           451.541218624   0.0468004675655   9648.22024464
    >>> Done in 63 iterations
    >>> Both actual and predicted relative reductions in the sum of squares are at most 0.000000
    >>> n. of data = 35
    >>> degree of freedom = 32
    >>> X^2_rel = 0.000046
    >>> pValue = 1.000000 (statistically significant if < 0.05)

If it is similar, your are done!
