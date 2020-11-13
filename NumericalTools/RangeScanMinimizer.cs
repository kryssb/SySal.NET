using System;

namespace NumericalTools.Minimization
{
    /// <summary>
    /// Minimizing algorithm that performs a range scan.
    /// </summary>
    public class RangeScanMinimizer : IMinimizer
    {
        /// <summary>
        /// Property backer for <c>GridSteps</c>.
        /// </summary>
        protected double[] m_GridSteps;
    
        /// <summary>
        /// Grid steps to use.
        /// </summary>
        public double[] GridSteps
        {
            get { return m_GridSteps; }
            set { m_GridSteps = value; }
        }

        #region IMinimizer Members

        /// <summary>
        /// Finds the minimum of the function.
        /// </summary>
        /// <param name="f">the function to be minimized.</param>
        /// <param name="maxiterations">ignored.</param>
        /// <returns>the minimum of the function.</returns>
        /// <remarks>The function domain must be bounded on all parameters.</remarks>
        public double[] FindMinimum(ITargetFunction f, int maxiterations)
        {
            int i, j;
            int par = f.CountParams;
            double [] mins = new double[par];
            double [] maxs = new double[par];
            int [] steps = new int[par];
            double [][] gridpoints = new double [par][];
            for (i = 0; i < par; i++)
            {
                mins[i] = f.RangeMin(i);
                maxs[i] = f.RangeMax(i);
                if (double.IsInfinity(mins[i]) || double.IsInfinity(maxs[i])) throw new MinimizationException("Function domain is unbounded on parameter #" + i + ".");
                if (maxs[i] < mins[i]) throw new MinimizationException("Null domain on parameter #" + i + ".");
                steps[i] = (int)(1 + Math.Floor((maxs[i] - mins[i]) / GridSteps[i]));                
                gridpoints[i] = new double[steps[i]];
                for (j = 0; j < steps[i]; j++)
                    gridpoints[i][j] = Math.Min(mins[i] + j * GridSteps[i], maxs[i]);
            }
            double fmin = f.Evaluate(mins);
            int[] iiib = new int[par];
            int[] iiis = new int[par];
            double v;
            double[] x = new double[par];
            if (m_TW != null) m_TW.WriteLine("Start RangeScan");
            while (true)
            {
                for (i = 0; i < par && ++iiis[i] == steps[i]; i++) iiis[i] = 0;
                if (i == par) break;
                for (i = 0; i < par; i++) x[i] = gridpoints[i][iiis[i]];
                v = f.Evaluate(x);
                if (m_TW != null)
                {                    
                    for (i = 0; i < par; i++)
                        m_TW.Write(x[i] + " ");
                    m_TW.WriteLine(v);
                }                
                if (v < fmin)
                {
                    fmin = v;
                    for (i = 0; i < par; i++) iiib[i] = iiis[i];
                }
            }
            for (i = 0; i < par; i++)
                x[i] = gridpoints[i][iiib[i]];
            m_Point = x;
            m_Value = fmin;
            if (m_TW != null)
            {
                m_TW.WriteLine("Minimum");
                for (i = 0; i < par; i++)
                    m_TW.Write(x[i] + " ");
                m_TW.WriteLine(fmin);

                m_TW.WriteLine("End RangeScan");
            }
            return x;
        }
        /// Property backer for <c>Value</c>.
        protected double m_Value;
        /// <summary>
        /// The minimum value of the function.
        /// </summary>
        public double Value
        {
            get { throw new Exception("The method or operation is not implemented."); }
        }
        /// <summary>
        /// Property backer for <c>Point</c>.
        /// </summary>
        protected double[] m_Point;
        /// <summary>
        /// The point where the minimum is located, approximated on grid points.
        /// </summary>
        public double[] Point
        {
            get { throw new Exception("The method or operation is not implemented."); }
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