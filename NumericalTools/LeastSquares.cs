using System;

namespace NumericalTools.AdvancedFitting
{
    /// <summary>
    /// Performs a general nonlinear least-squares fit.
    /// </summary>
    public class LeastSquares
    {
        /// <summary>
        /// Chi-Square function.
        /// </summary>
        protected class Chi2F : NumericalTools.Minimization.ITargetFunction
        {
            protected NumericalTools.Minimization.ITargetFunction m_F;

            protected int m_FitPars;

            double[][] m_Indep;

            double[] m_Dep;

            double[] m_DepErr;

            int m_Cases;

            int m_Vars;

            public Chi2F(NumericalTools.Minimization.ITargetFunction f, int fitpars, double[][] indep, double[] dep, double[] deperr)
            {
                m_F = f;
                m_FitPars = fitpars;
                m_Indep = indep;
                m_Dep = dep;
                m_DepErr = deperr;
                m_Cases = indep.Length;
                if (m_Cases != dep.Length || m_Cases != deperr.Length) throw new Exception("Data size is inconsistent: " + m_Cases + " for indep.var., " + m_Dep.Length + " for dep.var. and " + m_DepErr.Length + " for dep.var.err.");
                m_Vars = f.CountParams - m_FitPars;

                m_Start = new double[m_FitPars];
                int i;
                for (i = 0; i < m_FitPars; i++) m_Start[i] = m_F.Start[i];
            }

            #region ITargetFunction Members

            public int CountParams
            {
                get { return m_FitPars; }
            }

            public double Evaluate(params double[] x)
            {
                int i, j;
                double x2 = 0.0;
                double d;
                double[] xx = new double[m_F.CountParams];
                if (m_FitPars != x.Length) throw new Exception("Incorrect number of fit parameters.");
                for (j = 0; j < m_FitPars; j++) xx[j] = x[j];
                for (i = 0; i < m_Cases; i++)
                {
                    d = 0.0;
                    for (j = 0; j < m_Indep[i].Length; j++)
                    {
                        if (m_Indep[i].Length != m_Vars) throw new Exception("Data size is inconsistent: expected " + m_Vars + " variables, found " + m_Indep.Length + ".");
                        xx[m_FitPars + j] = m_Indep[i][j];                        
                    }
                    d = (m_F.Evaluate(xx) - m_Dep[i]) / m_DepErr[i];
                    x2 += d * d;
                }
                return x2;
            }

            public double RangeMin(int i)
            {
                return m_F.RangeMin(i);
            }

            public double RangeMax(int i)
            {
                return m_F.RangeMax(i);
            }

            public NumericalTools.Minimization.ITargetFunction Derive(int i)
            {
                return new Chi2D(m_F, m_F.Derive(i), m_FitPars, m_Indep, m_Dep, m_DepErr);
            }

            public bool StopMinimization(double fval, double fchange, double xchange)
            {
                return m_F.StopMinimization(fval, fchange, xchange);
            }

            protected double[] m_Start;

            public double[] Start
            {
                get
                {
                    return m_Start;
                }
            }

            public void SetStart(double[] ns)
            {
                m_Start = ns;
            }

            #endregion

        }

        /// <summary>
        /// Chi-Square partial derivative.
        /// </summary>
        protected class Chi2D : NumericalTools.Minimization.ITargetFunction
        {
            protected NumericalTools.Minimization.ITargetFunction m_F;

            protected NumericalTools.Minimization.ITargetFunction m_D;

            protected int m_FitPars;

            double[][] m_Indep;

            double[] m_Dep;

            double[] m_DepErr;

            int m_Cases;

            int m_Vars;

            public Chi2D(NumericalTools.Minimization.ITargetFunction f, NumericalTools.Minimization.ITargetFunction fd, int fitpars, double[][] indep, double[] dep, double[] deperr)
            {
                m_F = f;
                m_D = fd;
                m_FitPars = fitpars;
                m_Indep = indep;
                m_Dep = dep;
                m_DepErr = deperr;
                m_Cases = indep.Length;
                if (m_Cases != dep.Length || m_Cases != deperr.Length) throw new Exception("Data size is inconsistent: " + m_Cases + " for indep.var., " + m_Dep.Length + " for dep.var. and " + m_DepErr.Length + " for dep.var.err.");
                m_Vars = f.CountParams - m_FitPars;
            }

            #region ITargetFunction Members

            public int CountParams
            {
                get { return m_FitPars; }
            }

            public double Evaluate(params double[] x)
            {
                int i, j;
                double x2 = 0.0;
                double d;
                double[] xx = new double[m_F.CountParams];
                if (m_FitPars != x.Length) throw new Exception("Incorrect number of fit parameters.");
                for (j = 0; j < m_FitPars; j++) xx[j] = x[j];
                for (i = 0; i < m_Cases; i++)
                {
                    d = 0.0;
                    for (j = 0; j < m_Indep[i].Length; j++)
                    {
                        if (m_Indep[i].Length != m_Vars) throw new Exception("Data size is inconsistent: expected " + m_Vars + " variables, found " + m_Indep.Length + ".");
                        xx[m_FitPars + j] = m_Indep[i][j];
                    }
                    d = 2.0 * (m_F.Evaluate(xx) - m_Dep[i]) / (m_DepErr[i] * m_DepErr[i]) * m_D.Evaluate(xx);
                    x2 += d;
                }
                return x2;
            }

            public double RangeMin(int i)
            {
                return m_F.RangeMin(i);
            }

            public double RangeMax(int i)
            {
                return m_F.RangeMax(i);
            }

            public NumericalTools.Minimization.ITargetFunction Derive(int i)
            {
                return new Chi2H(m_D, m_F.Derive(i), m_FitPars, m_Indep, m_Dep, m_DepErr);
            }

            public bool StopMinimization(double fval, double fchange, double xchange)
            {
                return m_F.StopMinimization(fval, fchange, xchange);
            }

            public double[] Start
            {
                get
                {
                    double[] xs = m_F.Start;
                    double[] s = new double[m_FitPars];
                    int i;
                    for (i = 0; i < m_FitPars; i++) s[i] = xs[i];
                    return s;
                }
            }

            #endregion
        }

        /// <summary>
        /// Chi-Square partial derivative.
        /// </summary>
        protected class Chi2H : NumericalTools.Minimization.ITargetFunction
        {
            protected NumericalTools.Minimization.ITargetFunction m_D1;

            protected NumericalTools.Minimization.ITargetFunction m_D2;

            protected int m_FitPars;

            double[][] m_Indep;

            double[] m_Dep;

            double[] m_DepErr;

            int m_Cases;

            int m_Vars;

            public Chi2H(NumericalTools.Minimization.ITargetFunction d1, NumericalTools.Minimization.ITargetFunction d2, int fitpars, double[][] indep, double[] dep, double[] deperr)
            {
                m_D1 = d1;
                m_D2 = d2;
                m_FitPars = fitpars;
                m_Indep = indep;
                m_Dep = dep;
                m_DepErr = deperr;
                m_Cases = indep.Length;
                if (m_Cases != dep.Length || m_Cases != deperr.Length) throw new Exception("Data size is inconsistent: " + m_Cases + " for indep.var., " + m_Dep.Length + " for dep.var. and " + m_DepErr.Length + " for dep.var.err.");
                m_Vars = d1.CountParams - m_FitPars;
            }

            #region ITargetFunction Members

            public int CountParams
            {
                get { return m_FitPars; }
            }

            public double Evaluate(params double[] x)
            {
                int i, j;
                double x2 = 0.0;
                double d;
                double[] xx = new double[m_D1.CountParams];
                if (m_FitPars != x.Length) throw new Exception("Incorrect number of fit parameters.");
                for (j = 0; j < m_FitPars; j++) xx[j] = x[j];
                for (i = 0; i < m_Cases; i++)
                {
                    d = 0.0;
                    for (j = 0; j < m_Indep[i].Length; j++)
                    {
                        if (m_Indep[i].Length != m_Vars) throw new Exception("Data size is inconsistent: expected " + m_Vars + " variables, found " + m_Indep.Length + ".");
                        xx[m_FitPars + j] = m_Indep[i][j];
                    }
                    d = (m_D1.Evaluate(xx) * m_D2.Evaluate(xx)) / (m_DepErr[i] * m_DepErr[i]);
                    x2 += d;
                }
                return x2;
            }

            public double RangeMin(int i)
            {
                return m_D1.RangeMin(i);
            }

            public double RangeMax(int i)
            {
                return m_D1.RangeMax(i);
            }

            public NumericalTools.Minimization.ITargetFunction Derive(int i)
            {
                throw new Exception("The method or operation is not implemented.");
            }

            public bool StopMinimization(double fval, double fchange, double xchange)
            {
                return m_D1.StopMinimization(fval, fchange, xchange);
            }

            public double[] Start
            {
                get
                {
                    double[] xs = m_D1.Start;
                    double[] s = new double[m_FitPars];
                    int i;
                    for (i = 0; i < m_FitPars; i++) s[i] = xs[i];
                    return s;
                }
            }

            #endregion

        }



        /// <summary>
        /// Fits the data to the specified function. The function is meant to have <b>p</b>+<b>v</b> parameters, the first <b>p</b> of which are to be fitted,
        /// whereas the remaining <b>v</b> are assumed to be independent variables, whose values are picked from the lists for the independent variables.
        /// </summary>
        /// <param name="f">the function to be fitted. First derivatives w.r.t. the fit parameters are needed.</param>
        /// <param name="fitparameters">the number of parameters to be fitted. They must be the first parameters to be passed to the function.</param>
        /// <param name="indep">the list of values for the independent variables.</param>
        /// <param name="dep">the list of values for the dependent variable.</param>
        /// <param name="deperr">the list of errors for the dependent variable.</param>
        /// <param name="maxiterations">maximum number of iterations to find the minimum.</param>
        /// <returns>the parameters of the fit.</returns>
        public double[] Fit(NumericalTools.Minimization.ITargetFunction f, int fitparameters, double[][] indep, double[] dep, double[] deperr, int maxiterations)
        {
            m_DegreesOfFreedom = dep.Length - fitparameters;
            if (m_DegreesOfFreedom < 0) throw new NumericalTools.Minimization.MinimizationException("Degrees of freedom = " + m_DegreesOfFreedom + ". Aborting.");
            NumericalTools.Minimization.NewtonMinimizer MA = new NumericalTools.Minimization.NewtonMinimizer();
            MA.Logger = m_TW;
            Chi2F chi2 = new Chi2F(f, fitparameters, indep, dep, deperr);
            MA.FindMinimum(chi2, maxiterations);
            m_EstimatedVariance = MA.Value / m_DegreesOfFreedom;
            m_BestFit = MA.Point;
            Minimization.ITargetFunction[] g = new NumericalTools.Minimization.ITargetFunction[fitparameters];
            double[,] hessian = new double[fitparameters, fitparameters];            
            int i, j;
            for (i = 0; i < fitparameters; i++)
            {
                g[i] = chi2.Derive(i);
                for (j = 0; j < fitparameters; j++)
                    hessian[i, j] = g[i].Derive(j).Evaluate(m_BestFit);
            }
            m_CorrelationMatrix = new double[fitparameters, fitparameters];
            double[][] c = new Cholesky(hessian, 0.0).Inverse(0.0);
            for (i = 0; i < fitparameters; i++)
            {
                for (j = 0; j < i; j++) m_CorrelationMatrix[j, i] = m_CorrelationMatrix[i, j] = c[i][j] / (Math.Sqrt(c[i][i]) * Math.Sqrt(c[j][j]));
                m_CorrelationMatrix[i, j] = 1.0;                
            }
            m_StandardErrors = new double[fitparameters];
            for (i = 0; i < fitparameters; i++)
                m_StandardErrors[i] = Math.Sqrt(m_EstimatedVariance * c[i][i]);
            return m_BestFit;
        }



        /// <summary>
        /// Fits the data to the specified function. The function is meant to have <b>p</b>+<b>v</b> parameters, the first <b>p</b> of which are to be fitted,
        /// whereas the remaining <b>v</b> are assumed to be independent variables, whose values are picked from the lists for the independent variables.
        /// </summary>
        /// <param name="f">the function to be fitted. First derivatives w.r.t. the fit parameters are needed.</param>
        /// <param name="fitparameters">the number of parameters to be fitted. They must be the first parameters to be passed to the function.</param>
        /// <param name="indep">the list of values for the independent variables.</param>
        /// <param name="indeperr">the list of errors for the independent variable.</param>
        /// <param name="dep">the list of values for the dependent variable.</param>
        /// <param name="deperr">the list of errors for the dependent variable.</param>
        /// <param name="maxiterations">maximum number of iterations to find the minimum.</param>
        /// <returns>the parameters of the fit.</returns>
        /// <remarks>The method of effective variance is used to take errors on the independent variables into account. </remarks>
        public double[] Fit(NumericalTools.Minimization.ITargetFunction f, int fitparameters, double[][] indep, double[] dep, double [][] indeperr, double[] deperr, int maxiterations)
        {
            m_DegreesOfFreedom = dep.Length - fitparameters;
            if (m_DegreesOfFreedom < 0) throw new NumericalTools.Minimization.MinimizationException("Degrees of freedom = " + m_DegreesOfFreedom + ". Aborting.");
            NumericalTools.Minimization.NewtonMinimizer MA = new NumericalTools.Minimization.NewtonMinimizer();
            MA.Logger = m_TW;
            NumericalTools.Minimization.ITargetFunction[] f_d = new NumericalTools.Minimization.ITargetFunction[f.CountParams - fitparameters];
            int i, j;
            for (i = 0; i < f_d.Length; i++) f_d[i] = f.Derive(i + fitparameters);
            double [] c_deperr = new double[deperr.Length];
            double [] xp = new double[f.CountParams];
            double [] xfp = new double[fitparameters];
            double dfx;
            for (i = 0; i < f_d.Length; i++) xfp[i] = f.Start[i];
            int maxouteriter = maxiterations;
            if (maxouteriter <= 0) maxouteriter = -1;
            double f0, f1, dxchange;
            do
            {
                if (m_TW != null) m_TW.WriteLine("Starting with derivative guess - remaining iterations: " + maxouteriter);
                for (i = 0; i < f_d.Length; i++) xp[i] = xfp[i];
                for (i = 0; i < c_deperr.Length; i++)
                {
                    for (j = 0; j < f_d.Length; j++)                    
                        xp[j + fitparameters] = indep[i][j];
                    c_deperr[i] = deperr[i] * deperr[i];
                    for (j = 0; j < f_d.Length; j++)
                    {
                        dfx = f_d[j].Evaluate(xp) * indeperr[i][j];
                        c_deperr[i] += dfx * dfx;
                    }
                    c_deperr[i] = Math.Sqrt(c_deperr[i]);
                }
                Chi2F chi2 = new Chi2F(f, fitparameters, indep, dep, c_deperr);
                chi2.SetStart(xfp);
                f0 = chi2.Evaluate(xfp);
                MA.FindMinimum(chi2, maxiterations);
                m_EstimatedVariance = MA.Value / m_DegreesOfFreedom;
                m_BestFit = MA.Point;
                Minimization.ITargetFunction[] g = new NumericalTools.Minimization.ITargetFunction[fitparameters];
                double[,] hessian = new double[fitparameters, fitparameters];                
                for (i = 0; i < fitparameters; i++)
                {
                    g[i] = chi2.Derive(i);
                    for (j = 0; j < fitparameters; j++)
                        hessian[i, j] = g[i].Derive(j).Evaluate(m_BestFit);
                }
                m_CorrelationMatrix = new double[fitparameters, fitparameters];
                double[][] c = new Cholesky(hessian, 0.0).Inverse(0.0);
                for (i = 0; i < fitparameters; i++)
                {
                    for (j = 0; j < i; j++) m_CorrelationMatrix[j, i] = m_CorrelationMatrix[i, j] = c[i][j] / (Math.Sqrt(c[i][i]) * Math.Sqrt(c[j][j]));
                    m_CorrelationMatrix[i, j] = 1.0;
                }
                m_StandardErrors = new double[fitparameters];
                for (i = 0; i < fitparameters; i++)
                    m_StandardErrors[i] = Math.Sqrt(m_EstimatedVariance * c[i][i]);
                dxchange = 0.0;
                for (i = 0; i < f_d.Length; i++) dxchange += (xfp[i] - m_BestFit[i]) * (xfp[i] - m_BestFit[i]);
                f1 = chi2.Evaluate(m_BestFit);
                for (i = 0; i < xfp.Length; i++) xfp[i] = m_BestFit[i];
                if (m_TW != null) m_TW.WriteLine("End with derivative guess - remaining iterations: " + maxouteriter);
                if (--maxouteriter < 0) maxouteriter = -1;                
            }
            while (maxouteriter != 0 && f.StopMinimization(f1, f0, dxchange) == false);
            return m_BestFit;
        }

        /// <summary>
        /// Property backer for <c>DegreesOfFreedom</c>.
        /// </summary>
        protected int m_DegreesOfFreedom;

        /// <summary>
        /// Degrees of freedom.
        /// </summary>
        public int DegreesOfFreedom { get { return m_DegreesOfFreedom; } }

        /// <summary>
        /// Property backer for <c>EstimatedVariance</c>.
        /// </summary>
        protected double m_EstimatedVariance;

        /// <summary>
        /// Estimated variance of the fit.
        /// </summary>
        public double EstimatedVariance { get { return m_EstimatedVariance; } }

        /// <summary>
        /// Property backer for <c>BestFit</c>.
        /// </summary>
        protected double [] m_BestFit;

        /// <summary>
        /// Best fit values for the parameters.
        /// </summary>
        public double [] BestFit { get { return m_BestFit; } }

        /// <summary>
        /// Property backer for <c>StandardErrors</c>.
        /// </summary>
        protected double [] m_StandardErrors;

        /// <summary>
        /// Standard errors for the fit parameters.
        /// </summary>
        public double [] StandardErrors { get { return m_StandardErrors; } }

        /// <summary>
        /// Property backer for <c>CorrelationMatrix</c>.
        /// </summary>
        protected double [,] m_CorrelationMatrix;

        /// <summary>
        /// Correlation matrix for the fit parameters.
        /// </summary>
        public double [,] CorrelationMatrix { get { return m_CorrelationMatrix; } }

        /// <summary>
        /// Property backer for Logger.
        /// </summary>
        protected System.IO.TextWriter m_TW;

        /// <summary>
        /// Text stream to log the fit progress to. Set to <c>null</c> to disable logging.
        /// </summary>
        public System.IO.TextWriter Logger
        {
            set { m_TW = value; }
        }
    
        }
}