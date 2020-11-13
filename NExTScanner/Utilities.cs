using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    class Utilities
    {
        public static NumericalTools.ComputationResult SafeMeanAverage(double[] x, double fraction, out double mean, out double min, out double max, out double err)
        {
            System.Collections.ArrayList a = new System.Collections.ArrayList(x);
            if (a.Count <= 1)
            {
                mean = min = max = err = 0.0;
                return NumericalTools.ComputationResult.SingularityEncountered;
            }
            a.Sort();
            int minf = Math.Min(a.Count - 1, (int)Math.Round((1.0 - fraction) * 0.5 + a.Count));
            int maxf = Math.Min(a.Count - 1, (int)Math.Round((1.0 + fraction) * 0.5 + a.Count));            
            a = a.GetRange(minf, maxf - minf);
            if (minf > maxf) minf = maxf;
            min = (double)a[0];
            max = (double)a[a.Count - 1];
            double d = 0.0;
            foreach (double dx in a)
                d += dx;
            mean = d / a.Count;
            err = (max - min) * 0.5 / Math.Sqrt(a.Count);
            return NumericalTools.ComputationResult.OK;
        }
    }
}
