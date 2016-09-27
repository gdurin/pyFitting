bestFit: a command-line tool for least-square fitting
=====================================================

A simple python script to perform data fitting using nonlinear least-squares minimization. 

The routine is based on the scipy.optimize.leastsq method (see http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.leastsq.html#scipy.optimize.leastsq), which is a wrapper around MINPACK’s lmdif and lmder algorithms.

This is a command line tool, to keep everything easy and fast to use. 

0. Download and installation
----------------------------

The easiest way to install bestFit is::
   
    $ git clone git://github.com/gdurin/pyFitting.git

It requires the following packages

python 2.7.x (because of the print("Print this") statement. It can be easily change to the 2.6.x version print "Print this"

`numpy and scipy <http://www.scipy.org>`_ 

`matplotlib <http://matplotlib.sourceforge.net/>`_ (1.0.x)

`numexpr <http://code.google.com/p/numexpr/>`_

If not present, install them with easy_install, i.e. easy_install numexpr (under root)

1. How to use it
----------------
[New in 0.2.3]
From version 0.2.3 there is a new method to insert information on the command line. The main differences are related to the input of data 
filenames, and of the parameters and its initial values. No more commas are need to separate the variables and the values.

For instance, fitting two gaussian with a common parameter can be done as:

   $ bestFit -f data1.dat data2.dat -t "a*exp(-((x-x01)/sigma)**2)" "b*exp(-((x-x02)/sigma)**2)" -p a b x01 x02 sigma -i 1 1 1 0 1. -d

Make the bestFit.py executable (under Linux: chmod +x bestFit.py) and run it with the -h option. The output is the following::

  usage: bestFit [-h] -f filename [filename ...] -t theory [theory ...] -p
               params [params ...] -i initvals [initvals ...]
               [-v var [var ...]] [-c cols [cols ...]] [-r range] [-d]
               [-s sigma] [--held heldParams [heldParams ...]] [--lin] [--log]
               [--noplot] [--logplot]

  Best fit of data using least-square minimization

  optional arguments:
  -h, --help            show this help message and exit
  -f filename [filename ...], --filename filename [filename ...]
                        Filename(s) of the input data
  -t theory [theory ...], --theory theory [theory ...]
                        Theoretical function(s)
  -p params [params ...], --params params [params ...]
                        Parameter(s) name(s), i.e. -p a b c
  -i initvals [initvals ...], --initvals initvals [initvals ...]
                        Initial values of the parameter(s), i.e. -i 1 2. 3.
  -v var [var ...], --var var [var ...]
                        Name(s) of the independent variable(s), default: x
  -c cols [cols ...], --cols cols [cols ...]
                        Columns of the file to load the data, default: 0 1
  -r range, --drange range
                        Range of the data (as index of rows)
  -d, --deriv           Use Analytical Derivatives
  -s sigma, --sigma sigma
                        Estimation of the error in the data (as a constant
                        value)
  --held heldParams [heldParams ...]
                        Held one or more parameters, i.e. a=3 b=4
  --lin                 Use data in linear mode (default)
  --log                 Use data in log mode (best for log-log data) 
  --noplot              Don't show the plot output
  --logplot             Use log-log axis to plot data (default if --log)
  --creep               Use x-axis as x**-mu to plot data
  --data_logY           Use the log of Y data as input

NOTE: --held parameter NOT WORKING YET (as of version 0.2.3)
 
NOTE: The use of --creep and --data_logY (from version 0.4.0) is still experimental, and must be fixed properly.
Using data_logY the Y data are replaced by log10(Y), so the fitting function has to be changed as well.
This is usefull for fitting data for creep:
Using v = v0 * exp(-(Hd/H)**mu), with v0, Hd and mu as fitting parameters can be difficult. Using data_logY implies to
use lv0 - (Hd/H)**mu, with lv0 = log10(v0).

2. Test the script
------------------

To be sure that the script is working correctly, try one of the test fits included.

For instance, try to fit the eckerle4 data (see: http://www.itl.nist.gov/div898/strd/nls/data/eckerle4.shtml for details). 
This is a case considered of high difficulty.

Now copy the following line (see the change from version 0.2.3 - no commas between parameters and initial values):: 

   $ ./bestFit.py -f test/eckerle4/data.dat -p b1 b2 b3 -t "b1/b2*exp(-(x-b3)**2/(2.*b2**2))" -i 1. 10. 500. -c 1 0 -d

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
 
    $ bestFit.py -f test/eckerle4/data.dat -p b1 b2 b3 -t "b1/b2*exp(-(x-b3)**2/(2.*b2**2))" -i 1. 10. 500. -c 1 0 

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

Try also the test under 2vars:

    $ bestFit -f data1.txt data2.txt -t "A*x**alpha+b1" "A/2*z**(1-b1)" -i 8 .5 .2 -p A alpha b1 -v x z

showing a large figure with two plots, and something like:


    >>> Initial parameters =  (8.0, 0.5, 0.2)
    >>> initial cost = 2.6730748489e+02 (StD: 1.7528547648e+00)
    >>> optimized cost = 2.3804426372e-23 (StD: 5.2308134762e-13)

    >>> parameter    value           st. error         t-statistic
    >>> A              8.4   1.09368924024e-13    7.6804266614e+13
    >>> alpha         0.43   7.38776456836e-15   5.82043453093e+13
    >>> b1             0.3   7.30383553018e-15   4.10743093489e+13
    >>> Done in 17 iterations
    >>> The relative error between two consecutive iterates is at most 0.000000
    >>> n. of data = 90
    >>> degree of freedom = 87
    >>> X^2_rel = 0.000000
    >>> *** Time elapsed: 0.006

