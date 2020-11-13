using System;

namespace NumericalTools.Minimization
{
    /// <summary>
    /// Newton's algorithm.
    /// </summary>
    public class NewtonMinimizer : IMinimizer
    {
        #region IMinimizer Members
        /// <summary>
        /// Finds a local minimum for the function using Newton's algorithm.
        /// </summary>
        /// <param name="f">the function to be minimized.</param>
        /// <param name="maxiterations">the maximum number of iterations to try. Set to a non-positive number to allow infinite iterations.</param>
        /// <returns>the location of the local minimum.</returns>
        /// <remarks>The complete Hessian must be available through function derivation.</remarks>
        public double[] FindMinimum(ITargetFunction f, int maxiterations)
        {
            int par = f.CountParams;
            ITargetFunction[] grad = new ITargetFunction[par];
            ITargetFunction[,] hessian = new ITargetFunction[par, par];
            int i, j;
            double [] xstart = f.Start;
            double [] x = new double[par];
            double [] xn = new double[par];
            double [] dx = new double[par];
            double dxnorm;
            double [] mins = new double[par];
            double [] maxs = new double[par];
            for (i = 0; i < par; i++)
            {
                x[i] = xstart[i];
                mins[i] = f.RangeMin(i);
                maxs[i] = f.RangeMax(i);
                if (x[i] < mins[i] || x[i] > maxs[i]) throw new MinimizationException("Starting point is out of bounds.");
            }
            double f1 = f.Evaluate(x);
            double f2 = f1;
            double[] g = new double[par];
            double[,] h = new double[par, par];
            double[][] hi = null;
            for (i = 0; i < par; i++)
            {
                try
                {
                    grad[i] = f.Derive(i);
                }
                catch (Exception ex)
                {
                    throw new MinimizationException("The partial derivative #" + i + " is not available.\r\n" + ex.ToString());
                }
                for (j = i; j < par; j++)
                {
                    try
                    {
                        hessian[i, j] = grad[i].Derive(j);
                    }
                    catch (Exception ex)
                    {
                        throw new MinimizationException("The second partial derivative #" + i + "," + j + " is not available.\r\n" + ex.ToString());
                    }
                    hessian[j, i] = hessian[i, j];
                }
            }
            do
            {
                f1 = f2;
                for (i = 0; i < par; i++)
                {
                    g[i] = grad[i].Evaluate(x);
                    for (j = i; j < par; j++)
                        h[j, i] = h[i, j] = hessian[i, j].Evaluate(x);
                }
                if (m_TW != null)
                {
                    m_TW.WriteLine("Remaining iterations: " + maxiterations);
                    m_TW.Write("X =");
                    for (i = 0; i < par; i++)
                        m_TW.Write(" " + x[i]);
                    m_TW.WriteLine();
                    m_TW.WriteLine("F = " + f1);
                    m_TW.Write("G = ");
                    for (i = 0; i < par; i++)
                        m_TW.Write(" " + g[i]);
                    m_TW.WriteLine();
                    m_TW.WriteLine("H = ");
                    for (i = 0; i < par; i++)
                    {
                        for (j = 0; j < par; j++)
                            m_TW.Write(" " + h[i, j]);
                        m_TW.WriteLine();
                    }
                }
                try
                {
                    hi = new Cholesky(h, 0.0).Inverse(0.0);
                }
                catch (Exception ex)
                {
                    string xs = "";
                    for (i = 0; i < par; i++) xs += " " + x[i];
                    throw new MinimizationException("Hessian is not positive-definite at" + xs + ".\r\n" + ex.ToString());
                }
                for (i = 0; i < par; i++)
                {
                    dx[i] = 0.0;
                    for (j = 0; j < par; j++)
                        dx[i] -= hi[Math.Max(i, j)][Math.Min(i, j)] * g[j];
                    dx[i] *= 0.5;
                }                
                while (true)
                {
                    for (i = 0; i < par; i++)
                    {
                        xn[i] = x[i] + dx[i];
                        if (xn[i] < mins[i]) xn[i] = mins[i];
                        else if (xn[i] > maxs[i]) xn[i] = maxs[i];
                    }
                    if (f.Evaluate(xn) <= f1) break;
                    else for (i = 0; i < par; i++) dx[i] /= 2;
                }
                for (i = 0; i < par; i++) dx[i] = xn[i] - x[i];                
                dxnorm = 0.0;
                for (i = 0; i < par; i++)
                {
                    x[i] = xn[i];
                    dxnorm += dx[i] * dx[i];
                }
                dxnorm = Math.Sqrt(dxnorm);
                f2 = f.Evaluate(xn);
                if (m_TW != null)
                {
                    m_TW.Write("DX =");
                    for (i = 0; i < par; i++)
                        m_TW.Write(" " + dx[i]);
                    m_TW.WriteLine();
                    m_TW.WriteLine("DXNORM = " + dxnorm + " FNEW = " + f2 + " FCHANGE = " + (f2 - f1));
                }
            }
            while (f.StopMinimization(f2, f2 - f1, dxnorm) == false && ((maxiterations <= 0) ? -1 : --maxiterations) != 0);

            m_Value = f2;
            m_Point = x;
            return m_Point;
        }
        /// Property backer for <c>Value</c>.
        protected double m_Value;
        /// <summary>
        /// The value of the function at its minimum.
        /// </summary>
        public double Value
        {
            get { return m_Value; }
        }
        /// <summary>
        /// Property backer for <c>Point</c>.
        /// </summary>
        protected double[] m_Point;
        /// <summary>
        /// The location of the minimum.
        /// </summary>
        public double[] Point
        {
            get { return m_Point; }
        }
        /// <summary>
        /// Property backer for <c>Logger</c>.
        /// </summary>
        protected System.IO.TextWriter m_TW = null;
        /// <summary>
        /// Logging stream the algorithm progress. Set to <c>null</c> to disable logging.
        /// </summary>
        public System.IO.TextWriter Logger { set { m_TW = value; } }

        #endregion
    }
}