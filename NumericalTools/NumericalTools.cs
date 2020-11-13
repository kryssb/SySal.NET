using System;

namespace NumericalTools
{
	/// <summary>
	/// The generator.
	/// </summary>
	public class RandomGenerator
	{
		/// <summary>
		/// A single instance of a random generator that is used by all procedures.
		/// </summary>
		public static System.Random RND = new System.Random();
	};

	/// <summary>
	/// Result of a computation.
	/// </summary>
	public enum ComputationResult : int 
	{ 
		/// <summary>
		/// Computation OK.
		/// </summary>
		OK = 0, 
		/// <summary>
		/// Input was invalid.
		/// </summary>
		InvalidInput = 1, 
		/// <summary>
		/// A singularity has been encountered.
		/// </summary>
		SingularityEncountered = 2 
	}

	/// <summary>
	/// Options for rounding.
	/// </summary>
	public enum RoundOption : int 
	{ 
		/// <summary>
		/// Round to the greatest non-greater integer.
		/// </summary>
		FloorRound = -1, 
		/// <summary>
		/// Round to the closest integer.
		/// </summary>
		MathematicalRound = 0, 
		/// <summary>
		/// Round to the smallest non-smaller integer.
		/// </summary>
		CeilingRound = 1
	}

	/// <summary>
	/// Resources for various fitting procedures.
	/// </summary>
	public class Fitting
	{
		/// <summary>
		/// The Fitting class cannot be instantiated: it supports only static methods.
		/// </summary>
		protected Fitting()
		{
			//
			// TODO: Add constructor logic here
			//
		}

		private static double[,] GJSolutionMatrix;

		/// <summary>
		/// Rounds a number according to the specified rounding policy, using a specified number of decimal places.
		/// </summary>
		/// <param name="m">the number to be rounded.</param>
		/// <param name="j">the number of decimal places.</param>
		/// <param name="ropt">the rounding option.</param>
		/// <returns>the rounded number.</returns>
		public static double ExtendedRound(double m, int j, RoundOption ropt)
		{
			double n, k=1, l;
			int i=0;

			n=Math.Abs(m);
			if(n==0) return 0;
			//Caso maggiore di 1
			if (n < k) 
			{
				do
				{
					i++;
					k=k*0.1;
				} while (n<k);
				if (ropt==RoundOption.MathematicalRound)
					return (m<0?-1:1)*Math.Round(n,i+j-1);
				else 
				{
					for(int h=0; h<j-1; h++) k=k*0.1;
					return Math.Round((m<0?-1:1)*Math.Round(n,i+j-1)+(ropt==RoundOption.FloorRound?-1:1)*Math.Round(k,i+j-1),i+j);
				};
			} 
				//Caso minore di 1
			else
			{
				k=k*10;
				while (n>k)
				{
					i++;
					k=k*10;
				};
				if (i==j)
				{
					l=k;
					for(int h=0; h<j; h++) l=l*0.1;
					if (ropt==RoundOption.MathematicalRound)
						return (m<0?-1:1)*(int)(n/Math.Pow(10,i-j+1))*l; 
					else
					{
						return (m<0?-1:1)*(int)(n/Math.Pow(10,i-j+1))*l +(ropt==RoundOption.FloorRound?-1:1)*l;
					};
				} 
				else if (i>j)
				{
					l=k;
					for(int h=0; h<j; h++) l=l*0.1;
					if (ropt==RoundOption.MathematicalRound)
						return (m<0?-1:1)*(int)(n/Math.Pow(10,i-j+1))*l;
					else
					{
						return (m<0?-1:1)*(int)(n/Math.Pow(10,i-j+1))*l + (ropt==RoundOption.FloorRound?-1:1)*l;
					};
				} 
				else
				{
					if (ropt==RoundOption.MathematicalRound)
						return (m<0?-1:1)*Math.Round(n,j-i-1);
					else 
					{
						for(int h=0; h<j; h++) k=k*0.1;
						return Math.Round((m<0?-1:1)*Math.Round(n,j-i-1)+(ropt==RoundOption.FloorRound?-1:1)*Math.Round(k,j-i-1),j-i);
					};
				};

			};

		}
		
		/// <summary>
		/// Linear fitting without error specification.
		/// </summary>
		/// <param name="x">the values for the independent variable.</param>
		/// <param name="y">the values for the dependent variable.</param>
		/// <param name="a">the first-order coefficient of the fit line.</param>
		/// <param name="b">the zeroth-order coefficient of the fit line.</param>
		/// <param name="range">the range of the dependent variable.</param>
		/// <param name="erry">the error on the dependent variable.</param>
		/// <param name="erra">the error on the first-order fit coefficient.</param>
		/// <param name="errb">the error on the zeroth-order fit coefficient.</param>
		/// <param name="ccor">the correlation estimator.</param>
		/// <returns>a flag reflecting the outcome of the computation procedure.</returns>
		public static ComputationResult LinearFitSE(double [] x, double [] y, ref double a, ref double b, 
			ref double range, ref double erry , ref double erra,
			ref double errb, ref double ccor)
		{
			int i,n;
			double Tw=0, Tx=0, Ty=0, Txy=0, Tx2=0, Ty2=0, DTxy=0, DTx2=0, DTy2=0;
			double Min, Max, Den;
			n = x.GetLength(0);

			if (y.GetLength(0)!= n || n < 2)
			{
				return ComputationResult.InvalidInput;
			}
			else if(n==2)
			{
				if ((Den = x[1] - x[0]) == 0) return ComputationResult.SingularityEncountered;
				a = (y[1]-y[0]) / Den;
				b = y[0]-(a*x[0]);
				Max = (x[1]>x[0]?x[1]:x[0]);
				Min = (x[1]==Max?x[0]:x[1]);
				range = Max - Min;
				return ComputationResult.OK;
			}
			else
			{
				for (i = 0; i<n; i++)
				{
					Tx += x[i];
					Ty += y[i];
					Txy +=  x[i] * y[i];
					Tx2 +=  x[i] * x[i];
					Ty2 += y[i] * y[i];
				};

				Den = n * Tx2 - (Tx * Tx);
				if (Den == 0.0) return ComputationResult.SingularityEncountered;
				a = ((n * Txy) - (Tx * Ty)) / Den;
				b = ((Tx2 * Ty) - (Tx * Txy)) / Den;
				Max = x[1];
				Min = x[1];
				for (i = 0;i< n;i++)
				{
					if (x[i] > Max) Max = x[i];
					if (x[i] < Min) Min = x[i];
					Tw += (a * x[i] + b - y[i]) * (a * x[i] + b - y[i]);
					DTxy += (Tx / n - x[i]) * (Ty / n - y[i]);
					DTx2 += (Tx / n - x[i]) * (Tx / n -  x[i]);
					DTy2 += (Ty / n - y[i]) * (Ty / n - y[i]);
				};
			
				erry = Math.Sqrt(Tw / n);
				errb = erry * Math.Sqrt(Tx2 / Den);
				erra = erry * Math.Sqrt(n / Den);
				range = Max - Min;
				ccor = 0;
				if (DTx2 != 0 && DTy2 != 0) ccor = DTxy / Math.Sqrt(DTx2 * DTy2);
				return ComputationResult.OK;
			}
		
		}
		
		
		/// <summary>
		/// Linear fitting with error specification, not computing errors on the fit results.
		/// </summary>
		/// <param name="x">the values for the independent variable.</param>
		/// <param name="y">the values for the dependent variable.</param>
		/// <param name="sy">the values of the errors for the dependent variables.</param>
		/// <param name="a">the first-order coefficient of the fit line.</param>
		/// <param name="b">the zeroth-order coefficient of the fit line.</param>
		/// <param name="range">the range of the dependent variable.</param>
		/// <returns>a flag reflecting the outcome of the computation procedure.</returns>
		public static ComputationResult LinearFitDE(double [] x, 
			double [] y, double [] sy, ref double a , ref double b, ref double range)
		{
			int i,n;
			double Tx =0, Ty=0, Txy=0, Tx2=0, Ty2=0, Tds=0; 
			double Min, Max, Den;
			n=x.GetLength(0);
			if (y.GetLength(0) != n || sy.GetLength(0) != n || n < 2 ) return ComputationResult.InvalidInput;
			for(i = 0;i< n;i++)
				if (sy[i] == 0 ) return ComputationResult.InvalidInput;
			Max = x[1];
			Min = x[1];
			for (i = 0; i< n; i++)
			{
				if (x[i] > Max) Max = x[i];
				if (x[i] < Min) Min = x[i];
				Tds += 1 / (sy[i] * sy[i]);
				Tx += x[i] / (sy[i] * sy[i]);
				Ty += y[i] / (sy[i] * sy[i]);
				Txy += (x[i] * y[i]) / (sy[i] * sy[i]);
				Tx2 += (x[i] * x[i]) / (sy[i] * sy[i]);
				Ty2 += (y[i] * y[i]) / (sy[i] * sy[i]);
			}; 
			range = Max - Min;
			Den = Tds * Tx2 - (Tx * Tx);
			if (Den == 0) return ComputationResult.SingularityEncountered;
			a = ((Tds * Txy) - (Tx * Ty)) / Den;
			b = ((Tx2 * Ty) - (Tx * Txy)) / Den;
			
			return ComputationResult.OK;
		}

        /// <summary>
        /// Selects data in a density peak.
        /// </summary>
        /// <param name="x">the coordinates (n rows of m components per each point)</param>
        /// <param name="binsize">a vector with m components containing the cell size for each dimension. If set to <c>null</c>, automatic sizes are used.</param>
        /// <param name="peakquantile">a number between 0.0 and 1.0 specifying which fraction of all density cells the peak should contain. If negative, this parameter is ignored.</param>
        /// <param name="peakentriesquantile">a number between 0.0 and 1.0 specifying which fraction of all entries the peak should contain. This parameter applies if <c>peakdensityquantile</c> is negative.</param>
        /// <returns>the id's of the elements in the peak.</returns>
        public static int[] PeakDataSel(double[,] x, double [] binsize, double peakquantile, double peakentriesquantile)
        {
            bool autosize = false;
            if (binsize == null)
            {
                binsize = new double[x.GetLength(1)];
                autosize = true;
            }
            else if (binsize.Length != x.GetLength(1)) throw new Exception("The number of components of the size vector must be equal to the number of components of all vectors.");
            int n = 0;
            if ((n = x.GetLength(0)) <= 0) return new int[0];
            double[] min = new double[binsize.Length];
            double[] max = new double[binsize.Length];
            int d;            
            for (d = 0; d < binsize.Length; d++)
                min[d] = max[d] = x[0, d];
            int i, j;
            for (i = 1; i < n; i++)
                for (d = 0; d < binsize.Length; d++)
                {
                    if (x[i, d] < min[d]) min[d] = x[i, d];
                    else if (x[i, d] > max[d]) max[d] = x[i, d];
                }
            for (d = 0; d < binsize.Length; d++)
                if (max[d] == min[d])
                {
                    min[d] -= 0.5;
                    max[d] += 0.5;
                }
            int[] cellcounts = new int[binsize.Length];
            int allcells = 1;
            for (d = 0; d < binsize.Length; d++)
            {
                if (autosize) binsize[d] = (max[d] - min[d]) / Math.Sqrt(x.GetLength(0));
                cellcounts[d] = (int)Math.Ceiling((max[d] - min[d]) / binsize[d]);
                allcells *= cellcounts[d];
            }
            System.Collections.ArrayList[] cells = new System.Collections.ArrayList[allcells];
            int[] ij = new int[binsize.Length];
            for (i = 0; i < cells.Length; i++) cells[i] = new System.Collections.ArrayList();
            for (i = 0; i < n; i++)
            {
                for (d = 0; d < binsize.Length; d++)
                    ij[d] = (int)((x[i, d] - min[d]) / binsize[d]);
                int ci = 0;
                for (d = 0; d < binsize.Length; d++)
                    ci = ci * cellcounts[d] + ij[d];
                cells[ci].Add(i);
            }            
            System.Collections.ArrayList zero = new System.Collections.ArrayList();
            System.Collections.ArrayList outa = new System.Collections.ArrayList();
            if (peakquantile >= 0.0)
            {
                System.Collections.ArrayList sort = new System.Collections.ArrayList();
                sort.AddRange(cells);
                sort.Sort(ReverseArraySorter.Sorter);
                int quant = (int)(allcells * peakquantile);
                foreach (System.Collections.ArrayList a in sort)
                {
                    outa.AddRange(a);
                    quant--;
                    if (quant <= 0) break;
                }
            }
            else
            {
                System.Collections.ArrayList sort = new System.Collections.ArrayList();
                sort.AddRange(cells);
                sort.Sort(ReverseArraySorter.Sorter);
                int quant = (int)(n * peakentriesquantile);
                foreach (System.Collections.ArrayList a in sort)
                {
                    outa.AddRange(a);
                    quant -= a.Count;
                    if (quant <= 0) break;
                }
            }
            return (int[])outa.ToArray(typeof(int));
        }

        class ReverseArraySorter : System.Collections.IComparer
        {

            #region IComparer Members

            public int Compare(object x, object y)
            {
                return ((System.Collections.ArrayList)y).Count - ((System.Collections.ArrayList)x).Count;
            }

            #endregion

            public static ReverseArraySorter Sorter = new ReverseArraySorter();
        }

		/// <summary>
		/// Finds general statistics on a distribution.
		/// </summary>
		/// <param name="X">the values.</param>
		/// <param name="Max">the maximum value.</param>
		/// <param name="Min">the minimum value.</param>
		/// <param name="Avg">the average value.</param>
		/// <param name="RMS">the standard deviation.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult FindStatistics( double[] X, ref double Max, 
			ref double Min,  ref double Avg,  ref double RMS )
		{
			int i, N;
			double Sum, Sum2;
			N = X.GetLength(0);
			Max=0; Min=0; 
			Avg=0; RMS=0; 
			if (N < 2) 
			{
				return NumericalTools.ComputationResult.InvalidInput;
			};		
			Max = X[0]; Min = Max;
			Sum = Max; Sum2 = Max * Max;
			for (i = 1; i< N; i++)
			{
				Sum = Sum + X[i];
				Sum2 = Sum2 + X[i] * X[i];
				if (X[i] > Max) Max = X[i];
				if (X[i] < Min) Min = X[i];
			}; 
			try
			{
				Avg = Sum / N;
				RMS = System.Math.Sqrt((N / (N - 1)) * ((Sum2 / N) - (Avg * Avg)));
			}
			catch
			{
				return NumericalTools.ComputationResult.SingularityEncountered;
			};
			return NumericalTools.ComputationResult.OK;

		}

        /// <summary>
        /// Computes quantiles of a distribution.
        /// </summary>
        /// <param name="X">the values of the variable.</param>
        /// <param name="q">the list of quantiles to be computed.</param>
        /// <returns>the computed quantiles.</returns>
        public static double [] Quantiles(double [] X, double [] q)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList(X);
            ar.Sort();
            double [] qs = new double[q.Length];
            int i;
            for (i = 0; i < q.Length; i++)
                if (q[i] < 0.0 || q[i] > 1.0) throw new Exception("Quantiles should be in the range 0 - 1.");
                else qs[i] = (double)ar[(int)Math.Round(X.Length * q[i])];
            return qs;
        }

        /// <summary>
        /// Computes the range of a variable.
        /// </summary>
        /// <param name="X">the values of the variable.</param>
        /// <returns>the range.</returns>
        public static double Range(double[] X)
		{
			int N;
			N = X.GetLength(0);
 
			if (N < 2) return 0;
			return Maximum(X) - Minimum(X);

		}

		/// <summary>
		/// Computes the maximum of a variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the maximum value.</returns>
		public static double Maximum( double[] X )
		{
			int i, N;
			N = X.GetLength(0);
 
			if (N < 2) return 0;
			double Max = X[0];
			for (i = 1; i< N; i++) if (X[i] > Max) Max = X[i];

			return Max;

		}

		/// <summary>
		/// Computes the minimum of a variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the minimum value.</returns>
		public static double Minimum( double[] X )
		{
			int i, N;
			N = X.GetLength(0);
 
			if (N < 2) return 0;
			double Min = X[0];
			for (i = 1; i< N; i++) if (X[i] < Min) Min = X[i];

			return Min;

		}

		/// <summary>
		/// Computes the average of a variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the average value.</returns>
		public static double Average( double[] X )
		{
			int i, N;
			double Sum=0;
			N = X.GetLength(0);

			if (N < 1) return 0;
			for (i = 0; i< N; i++) Sum = Sum + X[i];

			try
			{
				return Sum / N;
			}
			catch
			{
				return 0;
			};

		}

		/// <summary>
		/// Computes the average of a bivariate variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the average value.</returns>
		public static double Average( double[,] X )
		{
			int i, j, N0, N1;
			double Sum=0;
			N0 = X.GetLength(0);
			N1 = X.GetLength(1);

			if (N0 < 1 && N1 < 1) return 0;
			for (i = 0; i< N0; i++) for (j = 0; j< N1; j++) Sum = Sum + X[i,j];

			try
			{
				return Sum / (N0*N1);
			}
			catch
			{
				return 0;
			};

		}

		/// <summary>
		/// Computes the standard deviation of a variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the standard deviation.</returns>
		public static double RMS( double[] X )
		{
			int i, N;
			double Sum=0, Sum2=0;
			N = X.GetLength(0);

			if (N < 2) return 0;
			for (i = 0; i< N; i++)
			{
				Sum = Sum + X[i];
				Sum2 = Sum2 + X[i] * X[i];
			}; 

			try
			{
				return System.Math.Sqrt((N / (N - 1)) * ((Sum2 / N) - ((Sum / N) * (Sum / N))));
			}
			catch
			{
				return 0;
			};

		}

		/// <summary>
		/// Computes the standard deviation of a bivariate variable.
		/// </summary>
		/// <param name="X">the values of the variable.</param>
		/// <returns>the standard deviation.</returns>
		public static double RMS( double[,] X )
		{
			int i, j, N0, N1, N;
			double Sum=0, Sum2=0;
			N0 = X.GetLength(0);
			N1 = X.GetLength(1);
			N = N0*N1;

			if (N0 < 2 && N1 < 2) return 0;
			for (i = 0; i< N0; i++)
				for (j = 0; j< N1; j++)
				{
					Sum = Sum + X[i,j];
					Sum2 = Sum2 + X[i,j] * X[i,j];
				}; 

			try
			{
				return System.Math.Sqrt((N / (N - 1)) * ((Sum2 / N) - ((Sum / N) * (Sum / N))));
			}
			catch
			{
				return 0;
			};

		}

		/// <summary>
		/// Prepares a custom distribution.
		/// </summary>
		/// <param name="Vec">the data values (e.g. measured flight lengths).</param>
		/// <param name="HistoOpt">if set to 2, the domain of the distribution coincides with the minimum and maximum values; if set to 1, the domain is enlarged by half Delta up and down.</param>
		/// <param name="Delta">the bin size.</param>
		/// <param name="N_Cat">the number of categories.</param>
		/// <param name="X_Mean">the central values of the categories.</param>
		/// <param name="Y_Vec">the cumulative values per category.</param>
		/// <param name="N_Y_Vec">the normalized values per category.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_Custom_Distribution(double[] Vec, short HistoOpt, 
			double Delta, int N_Cat, 
			out double[] X_Mean , out double[] Y_Vec,
			out double[] N_Y_Vec)
		{
			int k, i;
			int N;
			double Sup, Inf;
			X_Mean = null;
			Y_Vec = null;
			N_Y_Vec= null;

			//controlla la consistenza dell'input
			if ((HistoOpt < 1) || (HistoOpt > 2) || (HistoOpt == 1 && Delta <= 0) ||
				(HistoOpt == 2 && N_Cat <= 0))
			{
				return ComputationResult.InvalidInput;
			};
			N = Vec.GetLength(0);
			Sup = Vec[0]; Inf = Sup;
			for(i = 1; i< N;i++)
			{
				if (Vec[i] > Sup) Sup = Vec[i];
				if (Vec[i] < Inf) Inf = Vec[i];
			};
			
			//Determina i bins
			if (HistoOpt == 2) 
			{
				Delta = (Sup - Inf) / N_Cat;
			}
			else if (HistoOpt == 1)
			{
				N_Cat = 0;
				while(Inf - Delta/2 + N_Cat*Delta< Sup+Delta/2)
				{
					N_Cat++;
				};
			};
		
			X_Mean = new double[N_Cat];
			Y_Vec = new double[N_Cat];
			N_Y_Vec= new double[N_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			X_Mean[0] = Inf - (Delta/2);
			for (i = 1 ; i< N_Cat; i++)
			{
				X_Mean[i] = X_Mean[i - 1] + Delta;
			}; 

			//Determina le entries di ogni bin
			for(i = 0; i< N; i++)
			{
				for (k = 0; k< N_Cat; k++)
				{
					if ((k == N_Cat -1 && Vec[i] <= (Sup + (Delta/2)) && Vec[i] >= X_Mean[k]) 
						||(k < N_Cat -1 && Vec[i] < X_Mean[k + 1] && Vec[i] >= X_Mean[k]))
					{
						Y_Vec[k]++;
						break;
					};
				}; 
			}; 
		
			//Normalizza
			for(k = 0; k< N_Cat; k++)
			{
				N_Y_Vec[k] = Y_Vec[k] / N;
			}; 
			
			//Determina i valori medi delle x_Mean
			X_Mean[0] = Inf;
			for(i = 1; i< N_Cat; i++)
			{
				X_Mean[i] = X_Mean[i-1] + Delta;
			}; 
		
			return ComputationResult.OK;
		}
	
		/// <summary>
		/// Prepares a custom distribution.
		/// </summary>
		/// <param name="Vec">the data values (e.g. measured flight lengths).</param>
		/// <param name="Delta">the bin size.</param>
		/// <param name="X_Mean">the central values of the categories.</param>
		/// <param name="Y_Vec">the cumulative counts per category.</param>
		/// <param name="N_Y_Vec">the normalized counts per category.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_IntCounts_Distribution(int[] Vec,  
			int Delta, 
			out int[] X_Mean , out int[] Y_Vec,
			out double[] N_Y_Vec)
		{
			int k, i, N_Cat;
			int N;
			int Sup, Inf;
			X_Mean = null;
			Y_Vec = null;
			N_Y_Vec= null;

			//controlla la consistenza dell'input
			if  (Delta <= 0)
			{
				return ComputationResult.InvalidInput;
			};
			N = Vec.GetLength(0);
			Sup = Vec[0]; Inf = Sup;
			for(i = 1; i< N;i++)
			{
				if (Vec[i] > Sup) Sup = Vec[i];
				if (Vec[i] < Inf) Inf = Vec[i];
			};
			
			//Determina i bins
			N_Cat = 0;
			while(Inf + N_Cat*Delta < Sup)
			{
				N_Cat++;
			};
		
			X_Mean = new int[N_Cat];
			Y_Vec = new int[N_Cat];
			N_Y_Vec= new double[N_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			X_Mean[0] = Inf;
			for (i = 1 ; i< N_Cat; i++)
			{
				X_Mean[i] = X_Mean[i - 1] + Delta;
			}; 

			//Determina le entries di ogni bin
			for(i = 0; i< N; i++)
			{
				for (k = 0; k< N_Cat; k++)
				{
					if ((k == N_Cat -1 && Vec[i] <= Sup  && Vec[i] >= X_Mean[k]) 
						||(k < N_Cat -1 && Vec[i] < X_Mean[k + 1] && Vec[i] >= X_Mean[k]))
					{
						Y_Vec[k]++;
						break;
					};
				}; 
			}; 
		
			//Normalizza
			for(k = 0; k< N_Cat; k++)
			{
				N_Y_Vec[k] = Y_Vec[k] / N;
			}; 
			
			return ComputationResult.OK;
		}
	
		/// <summary>
		/// Computes a group scatter dataset.
		/// </summary>
		/// <param name="X">the set of independent values.</param>
		/// <param name="Y">the set of dependent values.</param>
		/// <param name="SetInterval">if set to 1, the number of intervals is imposed, and the bin size is computed; vice-versa if it is 2.</param>
		/// <param name="dx">the bin size.</param>
		/// <param name="Nint">the number of intervals.</param>
		/// <param name="SetX">can be 1, 2, 3; 1 = the central value of the bins, 2 = the lower value of the bins, 3 = the upper value of the bins.</param>
		/// <param name="Xout">the bins.</param>
		/// <param name="Yout">the average Y per bin.</param>
		/// <param name="SYout">the errors on the average Y per bin.</param>
		/// <param name="Entries">the count of the entries per each bin.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult GroupScatter(double[] X, double[] Y, 
			short SetInterval, double dx, int Nint, 
			short SetX, out double[] Xout, out double[] Yout,
			out double[] SYout, out int[] Entries) 
		{
			int i, j, N; 
			double Max=0, Min=0, Dummy=0;
			double [] LLim, ULim, Yout2;

			Xout = null;
			Yout = null;
			SYout = null;
			Entries = null;

			N = X.GetLength(0);

			//ScatterToLaw = 0
			if (N < 2 || SetInterval > 2 || SetInterval < 1 || SetX > 3 || SetX < 1)
				return ComputationResult.InvalidInput;
			
			FindStatistics(X, ref Max, ref Min, ref Dummy, ref Dummy);
			if (Max == Min)
				return ComputationResult.InvalidInput;

			//Settaggio Intervalli
			if (SetInterval == 1)
				Nint = (int)((Max - Min) / dx) + 1;
			else 
				dx = (Max - Min) / Nint;

			LLim = new double[Nint];
			ULim = new double[Nint];
			Xout = new double[Nint];
			Yout = new double[Nint];
			Yout2 = new double[Nint];
			SYout = new double[Nint];
			Entries = new int[Nint];
		
			//Prepara i bordi
			for (j = 0; j<  Nint;j++)
			{
				LLim[j] = Min + (j - 1) * dx;
				ULim[j] = LLim[j] + dx;
				if (SetX == 1)
					Xout[j] = LLim[j] + (dx / 2);
				else if (SetX == 2) 
					Xout[j] = LLim[j];
				
				if (j == Nint) ULim[j] = Max;
			};

			//Riempie Yout e SYout
			for( i = 0; i< N; i++)
				for(j = 0; j< Nint;j++)
					if ((j < Nint && X[i] < ULim[j] && X[i] >= LLim[j]) 
						|| (j == Nint && X[i] <= ULim[j] && X[i] >= LLim[j]))
					{
						Yout[j] = Yout[j] + Y[i];
						Yout2[j] = Yout2[j] + Y[i] * Y[i];
						Entries[j]++;
						if (SetX == 3 ) Xout[j] = Xout[j] + X[i];
						break;
					};

			for (j = 0; j< Nint;j++)
			{
				if (Entries[j] > 0) Yout[j] = Yout[j] / Entries[j];
				if (Entries[j] > 2) 
				{
					SYout[j] = System.Math.Sqrt((Yout2[j] / Entries[j]) - (Yout[j] * Yout[j]));
					if (SetX == 3) Xout[j] = Xout[j] / Entries[j];
				}
				else
				{
					SYout[j] = 0;
					if (SetX == 3) Xout[j] = LLim[j] + (dx / 2);
				};
			};
			return ComputationResult.OK;

		}

		/// <summary>
		/// Prepares a bivariate distribution.
		/// </summary>
		/// <param name="X">the set of X values.</param>
		/// <param name="Y">the set of Y values.</param>
		/// <param name="DeltaX">the X bin.</param>
		/// <param name="DeltaY">the Y bin.</param>
		/// <param name="X_Mean">the values of the reference points for the X bins.</param>
		/// <param name="Y_Mean">the values of the reference points for the Y bins.</param>
		/// <param name="Z_Vec">the set of Z values.</param>
		/// <param name="N_Z_Vec">the count of the Z values.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_2DCustom_Distribution(double[] X, double[] Y, 
			double DeltaX, double DeltaY,
			out double[] X_Mean, out double[] Y_Mean,
			out double[,] Z_Vec, out double[,] N_Z_Vec)
		{
			int  k, i, h, N;
			double SupX, InfX, SupY, InfY;
			int NX_Cat, NY_Cat;

			X_Mean=null; Y_Mean=null;
			Z_Vec=null; N_Z_Vec=null;

			//Prepare_2DCustom_Distribution = 0
			//controlla la consistenza dell'input
			if (DeltaX <= 0 || DeltaY <= 0)
				return ComputationResult.InvalidInput;


			//Trova Max e Min
			N = X.GetLength(0);
			SupX = X[0]; InfX = SupX;
			SupY = Y[0]; InfY = SupY;
			for (i = 1; i< N; i++)
			{
				if (X[i] > SupX) SupX = X[i];
				if (X[i] < InfX) InfX = X[i];
				if (Y[i] > SupY) SupY = Y[i];
				if (Y[i] < InfY) InfY = Y[i];
			};

			//Determina i bins
			NX_Cat = (int)((SupX - InfX) / DeltaX) + 1;
			NY_Cat = (int)((SupY - InfY) / DeltaY) + 1;

			double[] tX_Mean= new double[NX_Cat + 1];
			double[] tY_Mean= new double[NY_Cat + 1];
			Z_Vec= new double[NX_Cat, NY_Cat];
			N_Z_Vec= new double[NX_Cat, NY_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			tX_Mean[0] = InfX;
			tY_Mean[0] = InfY;
			for (i = 1 ; i< NX_Cat + 1; i++)
			{
				tX_Mean[i] = tX_Mean[i - 1] + DeltaX;
				if (i == NX_Cat + 1) tX_Mean[i] = SupX;
			};
			for (i = 1; i< NY_Cat + 1;i++)
			{
				tY_Mean[i] = tY_Mean[i - 1] + DeltaY;
				if (i == NY_Cat + 1) tY_Mean[i] = SupY;
			};

			//Determina le entries di ogni bin
			for (i = 0; i < N; i++)
			{
				k = (int)((X[i] - InfX) / DeltaX);
				if (k < 0 || k >= NX_Cat) continue;
				h = (int)((Y[i] - InfY) / DeltaY);
				if (h < 0 || h >= NY_Cat) continue;
				Z_Vec[k, h]++;
			}
			/*
						for (i = 0; i<  N; i++)
							for (k = 0; k< NX_Cat; k++)
								if (k == NX_Cat && X[i] <= tX_Mean[k + 1] && 
									X[i] >= tX_Mean[k] || k < NX_Cat && X[i] < tX_Mean[k + 1]
									&& X[i] >= tX_Mean[k])
								{	
									for(h = 0;h< NY_Cat;h++)
										if (h == NY_Cat && Y[i] <= tY_Mean[h + 1] && 
											Y[i] >= tY_Mean[h] || h < NY_Cat && Y[i] < tY_Mean[h + 1] &&
											Y[i] >= tY_Mean[h])
										{
											Z_Vec[k, h] = Z_Vec[k, h] + 1;
											break;
										};
									break;
								};
			*/			
			//Normalizza
			for(k = 0; k< NX_Cat; k++)
				for(h = 0; h< NY_Cat; h++)
					N_Z_Vec[k, h] = Z_Vec[k, h] / N;
			
			//Determina i valori medi delle x_Mean
			X_Mean = new double[NX_Cat];
			for(i = 0; i< NX_Cat; i++)
				X_Mean[i] = tX_Mean[i] + (DeltaX / 2);

			Y_Mean = new double[NY_Cat];
			for(i = 0; i< NY_Cat; i++)
				Y_Mean[i] = tY_Mean[i] + (DeltaY / 2);

			return ComputationResult.OK;

		}

		/// <summary>
		/// Prepares a bivariate distribution with cuts.
		/// </summary>
		/// <param name="X">the set of X values.</param>
		/// <param name="Y">the set of Y values.</param>
		/// <param name="DeltaX">the X bin.</param>
		/// <param name="DeltaY">the Y bin.</param>
		/// <param name="InfX">the lower boundary of X.</param>
		/// <param name="SupX">the upper boundary of X.</param>
		/// <param name="InfY">the lower boundary of Y.</param>
		/// <param name="SupY">the upper boundary of Y.</param>
		/// <param name="X_Mean">the values of the reference points for the X bins.</param>
		/// <param name="Y_Mean">the values of the reference points for the Y bins.</param>
		/// <param name="Z_Vec">the set of Z values.</param>
		/// <param name="N_Z_Vec">the count of the Z values.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_2DCustom_Distribution(double[] X, double[] Y, 
			double DeltaX, double DeltaY,
			double InfX, double SupX, 
			double InfY, double SupY, 
			out double[] X_Mean, out double[] Y_Mean,
			out double[,] Z_Vec, out double[,] N_Z_Vec)
		{
			int  k, i, h, N, IncludedN;
			int NX_Cat, NY_Cat;

			X_Mean=null; Y_Mean=null;
			Z_Vec=null; N_Z_Vec=null;

			//Prepare_2DCustom_Distribution = 0
			//controlla la consistenza dell'input
			if (DeltaX <= 0 || DeltaY <= 0)
				return ComputationResult.InvalidInput;


			//Trova Max e Min
			N = X.GetLength(0);
			IncludedN = 0;

			//Determina i bins
			InfX = Math.Floor(InfX / DeltaX) * DeltaX;
			SupX = Math.Ceiling(SupX / DeltaX) * DeltaX;
			InfY = Math.Floor(InfY / DeltaY) * DeltaY;
			SupY = Math.Ceiling(SupY / DeltaY) * DeltaY;
			NX_Cat = (int)((SupX - InfX) / DeltaX) + 1;
			NY_Cat = (int)((SupY - InfY) / DeltaY) + 1;

			double[] tX_Mean= new double[NX_Cat + 1];
			double[] tY_Mean= new double[NY_Cat + 1];
			Z_Vec= new double[NX_Cat, NY_Cat];
			N_Z_Vec= new double[NX_Cat, NY_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			tX_Mean[0] = InfX;
			tY_Mean[0] = InfY;
			for (i = 1 ; i< NX_Cat + 1; i++)
			{
				tX_Mean[i] = tX_Mean[i - 1] + DeltaX;
				if (i == NX_Cat + 1) tX_Mean[i] = SupX;
			};
			for (i = 1; i< NY_Cat + 1;i++)
			{
				tY_Mean[i] = tY_Mean[i - 1] + DeltaY;
				if (i == NY_Cat + 1) tY_Mean[i] = SupY;
			};

			//Determina le entries di ogni bin
			for (i = 0; i < N; i++)
			{
				k = (int)((X[i] - InfX) / DeltaX);
				if (k < 0 || k >= NX_Cat) continue;
				h = (int)((Y[i] - InfY) / DeltaY);
				if (h < 0 || h >= NY_Cat) continue;
				IncludedN++;
				Z_Vec[k, h]++;
			}
			
			//Normalizza
			if (IncludedN > 0)
				for(k = 0; k< NX_Cat; k++)
					for(h = 0; h< NY_Cat; h++)
						N_Z_Vec[k, h] = Z_Vec[k, h] / IncludedN;
			
			//Determina i valori medi delle x_Mean
			X_Mean = new double[NX_Cat];
			for(i = 0; i< NX_Cat; i++)
				X_Mean[i] = tX_Mean[i] + (DeltaX / 2);

			Y_Mean = new double[NY_Cat];
			for(i = 0; i< NY_Cat; i++)
				Y_Mean[i] = tY_Mean[i] + (DeltaY / 2);

			return ComputationResult.OK;

		}

		/// <summary>
		/// Prepares a bivariate distribution of Z values.
		/// </summary>
		/// <param name="X">the set of X values.</param>
		/// <param name="Y">the set of Y values.</param>
		/// <param name="Z">the set of Z values.</param>
		/// <param name="DeltaX">the X bin.</param>
		/// <param name="DeltaY">the Y bin.</param>
		/// <param name="X_Mean">the values of the reference points for the X bins.</param>
		/// <param name="Y_Mean">the values of the reference points for the Y bins.</param>		
		/// <param name="Mean_Z_Vec">Z averages per bin.</param>
		/// <param name="RMS_Z_Vec">Z standard deviations per bin.</param>
		/// <param name="Entries">The number of entries per bin.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_2DCustom_Distribution_ZVal(double[] X, double[] Y, double[] Z,
			double DeltaX, double DeltaY,
			out double[] X_Mean, out double[] Y_Mean,
			out double[,] Mean_Z_Vec, out double[,] RMS_Z_Vec,
			out int [,] Entries)
		{
			int k, i, h, N;
			double SupX, InfX, SupY, InfY;
			int NX_Cat, NY_Cat;
			double[,] tmpZ2;
			
			X_Mean= null;
			Y_Mean= null;
			Mean_Z_Vec= null;
			RMS_Z_Vec= null;
			tmpZ2= null;
			Entries= null;

			//controlla la consistenza dell'input
			if (DeltaX <= 0 || DeltaY <= 0)	  return ComputationResult.InvalidInput;

			//Trova Max e Min
			N = X.GetLength(0);
			SupX = X[0] ; InfX = SupX;
			SupY = Y[0] ; InfY = SupY;
			for(i = 1; i< N; i++)
			{
				if(X[i] > SupX) SupX = X[i];
				if(X[i] < InfX) InfX = X[i];
				if(Y[i] > SupY) SupY = Y[i];
				if(Y[i] < InfY) InfY = Y[i];
			};

			//Determina i bins
			NX_Cat = (int)((SupX - InfX) / DeltaX) + 1;
			NY_Cat = (int)((SupY - InfY) / DeltaY) + 1;

			double[] tX_Mean= new double[NX_Cat + 1];
			double[] tY_Mean= new double[NY_Cat + 1];
			Mean_Z_Vec= new double[NX_Cat, NY_Cat];
			RMS_Z_Vec= new double[NX_Cat, NY_Cat];
			tmpZ2= new double[NX_Cat, NY_Cat];
			Entries= new int[NX_Cat, NY_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			tX_Mean[0] = InfX;
			tY_Mean[0] = InfY;
			for(i = 1 ; i< NX_Cat + 1;i++)
			{
				tX_Mean[i] = tX_Mean[i - 1] + DeltaX;
				if (i == NX_Cat + 1) tX_Mean[i] = SupX;
			};
			for(i = 1; i< NY_Cat + 1;i++)
			{	
				tY_Mean[i] = tY_Mean[i - 1] + DeltaY;
				if (i == NY_Cat + 1) tY_Mean[i] = SupY;
			};

			//Determina le entries di ogni bin
			for (i = 0; i < N; i++)
			{
				k = (int)((X[i] - InfX) / DeltaX);
				if (k < 0 || k >= NX_Cat) continue;
				h = (int)((Y[i] - InfY) / DeltaY);
				if (h < 0 || h >= NY_Cat) continue;
				Mean_Z_Vec[k, h] += Z[i];
				tmpZ2[k, h] += Z[i] * Z[i];
				Entries[k, h]++;
			}

			//Normalizza
			for(k = 0; k< NX_Cat;k++)
				for(h = 0; h< NY_Cat;h++)
					if (Entries[k, h] > 0 )
					{		
						Mean_Z_Vec[k, h] = Mean_Z_Vec[k, h] / Entries[k, h];
						if (Entries[k, h] > 2) RMS_Z_Vec[k, h] = System.Math.Sqrt((Entries[k, h] / (Entries[k, h] - 1)) * ((tmpZ2[k, h] / Entries[k, h]) - (Mean_Z_Vec[k, h] * Mean_Z_Vec[k, h])));
					};

			//Determina i valori medi delle x_Mean
			X_Mean = new double[NX_Cat];
			for(i = 0; i< NX_Cat;i++)
				X_Mean[i] = tX_Mean[i] + (DeltaX / 2);


			Y_Mean= new double[NY_Cat];
			for(i = 0;i< NY_Cat;i++)
				Y_Mean[i] = tY_Mean[i] + (DeltaY / 2);

			return ComputationResult.OK;

		}


		/// <summary>
		/// Prepares a bivariate distribution of Z values with cuts.
		/// </summary>
		/// <param name="X">the set of X values.</param>
		/// <param name="Y">the set of Y values.</param>
		/// <param name="Z">the set of Z values.</param>
		/// <param name="DeltaX">the X bin.</param>
		/// <param name="DeltaY">the Y bin.</param>
		/// <param name="InfX">the lower bound of X.</param>
		/// <param name="SupX">the upper bound of X.</param>
		/// <param name="InfY">the lower bound of Y.</param>
		/// <param name="SupY">the upper bound of Y.</param>
		/// <param name="X_Mean">the values of the reference points for the X bins.</param>
		/// <param name="Y_Mean">the values of the reference points for the Y bins.</param>		
		/// <param name="Mean_Z_Vec">Z averages per bin.</param>
		/// <param name="RMS_Z_Vec">Z standard deviations per bin.</param>
		/// <param name="Entries">The number of entries per bin.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Prepare_2DCustom_Distribution_ZVal(double[] X, double[] Y, double[] Z,
			double DeltaX, double DeltaY,
			double InfX, double SupX,
			double InfY, double SupY,
			out double[] X_Mean, out double[] Y_Mean,
			out double[,] Mean_Z_Vec, out double[,] RMS_Z_Vec,
			out int [,] Entries)
		{
			int k, i, h, N, IncludedN;
			int NX_Cat, NY_Cat;
			double[,] tmpZ2;
			
			X_Mean= null;
			Y_Mean= null;
			Mean_Z_Vec= null;
			RMS_Z_Vec= null;
			tmpZ2= null;
			Entries= null;

			//controlla la consistenza dell'input
			if (DeltaX <= 0 || DeltaY <= 0)	  return ComputationResult.InvalidInput;

			//Trova Max e Min
			N = X.GetLength(0);
			IncludedN = 0;

			//Determina i bins
			InfX = Math.Floor(InfX / DeltaX) * DeltaX;
			SupX = Math.Ceiling(SupX / DeltaX) * DeltaX;
			InfY = Math.Floor(InfY / DeltaY) * DeltaY;
			SupY = Math.Ceiling(SupY / DeltaY) * DeltaY;
			NX_Cat = (int)((SupX - InfX) / DeltaX) + 1;
			NY_Cat = (int)((SupY - InfY) / DeltaY) + 1;

			double[] tX_Mean= new double[NX_Cat + 1];
			double[] tY_Mean= new double[NY_Cat + 1];
			Mean_Z_Vec= new double[NX_Cat, NY_Cat];
			RMS_Z_Vec= new double[NX_Cat, NY_Cat];
			tmpZ2= new double[NX_Cat, NY_Cat];
			Entries= new int[NX_Cat, NY_Cat];

			//Gli x_Mean prima di assumere i loro valori definitivi
			//fungono da bordi degli intervalli
			tX_Mean[0] = InfX;
			tY_Mean[0] = InfY;
			for(i = 1 ; i< NX_Cat + 1;i++)
			{
				tX_Mean[i] = tX_Mean[i - 1] + DeltaX;
				if (i == NX_Cat + 1) tX_Mean[i] = SupX;
			};
			for(i = 1; i< NY_Cat + 1;i++)
			{	
				tY_Mean[i] = tY_Mean[i - 1] + DeltaY;
				if (i == NY_Cat + 1) tY_Mean[i] = SupY;
			};

			//Determina le entries di ogni bin
			for (i = 0; i < N; i++)
			{
				k = (int)((X[i] - InfX) / DeltaX);
				if (k < 0 || k >= NX_Cat) continue;
				h = (int)((Y[i] - InfY) / DeltaY);
				if (h < 0 || h >= NY_Cat) continue;
				IncludedN++;
				Mean_Z_Vec[k, h] += Z[i];
				tmpZ2[k, h] += Z[i] * Z[i];
				Entries[k, h]++;
			}

			//Normalizza
			if (IncludedN > 0)
				for(k = 0; k< NX_Cat;k++)
					for(h = 0; h< NY_Cat;h++)
						if (Entries[k, h] > 0 )
						{		
							Mean_Z_Vec[k, h] = Mean_Z_Vec[k, h] / Entries[k, h];
							if (Entries[k, h] > 2) RMS_Z_Vec[k, h] = System.Math.Sqrt((Entries[k, h] / (Entries[k, h] - 1)) * ((tmpZ2[k, h] / Entries[k, h]) - (Mean_Z_Vec[k, h] * Mean_Z_Vec[k, h])));
						};

			//Determina i valori medi delle x_Mean
			X_Mean = new double[NX_Cat];
			for(i = 0; i< NX_Cat;i++)
				X_Mean[i] = tX_Mean[i] + (DeltaX / 2);


			Y_Mean= new double[NY_Cat];
			for(i = 0;i< NY_Cat;i++)
				Y_Mean[i] = tY_Mean[i] + (DeltaY / 2);

			return ComputationResult.OK;

		}

		/// <summary>
		/// Forward elimination procedure for linear system-solving.
		/// Currently limited to square matrices as large as 2..15 components per row/column.
		/// </summary>
		/// <param name="inA">input matrix.</param>
		/// <param name="outA">output matrix.</param>
		/// <returns>the outcome of the computation.</returns>
		private static ComputationResult ForwardElimination(double[,] inA, out double[,] outA)
		{
			int k, i, N, j;
			double Qt;
			N = inA.GetLength(0);
			outA= null;
			//controlla la consistenza dell'input
			if (N < 2 || N > 16) return NumericalTools.ComputationResult.InvalidInput;
			try
			{
				outA = (double [,])inA.Clone();

				for (k = 0; k < N - 1; k++)
				{
					//A questo punto il pivoting
					PivotingAlgorithm(outA, k);
					for(i = k + 1; i < N; i++)
					{
						Qt = outA[i, k] / outA[k, k];
						for (j = k + 1 ; j< N + 1; j++)
						{
							outA[i, j] = outA[i, j] - Qt * outA[k, j];
						};
					};
					for(i = k + 1; i< N; i++)
					{
						outA[i, k] = 0;
					};
				};
			}
			catch
			{
				return NumericalTools.ComputationResult.SingularityEncountered;
			};
			return NumericalTools.ComputationResult.OK;

		}

		//Method called by SolveLinearSystem
		private static ComputationResult BackwardSubstitution(double[,] inA, ref double[] outX)
		{
			int i , N, j , nx;
			double Sum;
			N = inA.GetLength(0);

			//controlla la consistenza dell'input
			if(N < 2 || N > 16) return NumericalTools.ComputationResult.InvalidInput;
			
			try
			{
				outX= new double[N];
				//				outX[N] = inA[N, N + 1] / inA[N, N];
				outX[N-1] = inA[N-1, N ] / inA[N-1, N-1];
				//				for (nx = 0; nx< N-1 ; nx++)
				for (nx = 0; nx< N ; nx++)
				{
					Sum = 0;
					i = N-1 - nx;
					for(j = i + 1; j<  N; j++)
					{
						Sum = Sum + inA[i, j] * outX[j];
					};
					outX[i] = (inA[i, N ] - Sum) / inA[i, i];
				};
			}
			catch
			{
				return NumericalTools.ComputationResult.SingularityEncountered;
			};
			return NumericalTools.ComputationResult.OK;
		}

		//Method called by ForwardElimination
		private static ComputationResult PivotingAlgorithm(double[,] mA, int Pivot)
		{
			int r,c, j, MaxIndex;
			double Max2, Max, TE;
			r = mA.GetLength(0);
			c = mA.GetLength(1);
			//controlla la consistenza dell'input

			if(r < 2 || r > 15) return NumericalTools.ComputationResult.InvalidInput;

			Max = System.Math.Abs(mA[Pivot, Pivot]);
			Max2 = Max;
			MaxIndex=0;
			for(j = Pivot + 1; j< r; j++)
			{
				if (System.Math.Abs(mA[j, Pivot]) > Max2)
				{
					Max2 = System.Math.Abs(mA[j, Pivot]);
					MaxIndex = j;
				};
			};
			if(Max2 > Max)
			{
				for(j = Pivot; j< c; j++)
				{
					TE = mA[Pivot, j];
					mA[Pivot, j] = mA[MaxIndex, j];
					mA[MaxIndex, j] = TE;
				};
			};
			return NumericalTools.ComputationResult.OK;
		}

		/// <summary>
		/// Solves a Linear System.
		/// </summary>
		/// <param name="inA">matrix of coefficients.</param>
		/// <param name="outX">vector of data values.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult SolveLinearSystem(double[,] inA, ref double[] outX)
		{
			
			double[,] outA;
			ComputationResult Chk1;
			// se metto ref allora outA = null;
			Chk1=ForwardElimination(inA, out outA);
			if ( Chk1 == ComputationResult.OK)
			{
				Chk1 =BackwardSubstitution(outA, ref outX);
				if(Chk1==ComputationResult.OK) 
					return ComputationResult.OK;
				else
					return  Chk1;
			}
			else
				return  Chk1;
		}

		internal static ComputationResult GaussJordanAlgorithm(double[,] inA, out double[,] outA)
		{
			int k, i, r, c, j;
			double Qt;
			r = inA.GetLength(0);
			c = inA.GetLength(1);
			outA= null;
			//controlla la consistenza dell'input
			if (r < 2 || r > 16) return NumericalTools.ComputationResult.InvalidInput;
			try
			{
				outA = (double [,])inA.Clone();

				for (k = 0; k < r; k++)
				{
					//A questo punto il pivoting
					PivotingAlgorithm(outA, k);
					double serv = outA[k, k];
					for(j = 0; j < c; j++)
					{
						outA[k, j] = outA[k, j] / serv;
					};
					for (i = 0 ; i< r; i++)
					{
						if(i!=k)
						{
							double serv2 = outA[i, k];
							for (j = 0 ; j< c; j++)
							{
								outA[i, j] = outA[i, j] - serv2 * outA[k, j];
							};
						};
					};
				};
			}
			catch
			{
				return NumericalTools.ComputationResult.SingularityEncountered;
			};
			return NumericalTools.ComputationResult.OK;

		}

		/// <summary>
		/// Solves a linear system with the Gauss-Jordan algorithm.
		/// </summary>
		/// <param name="inA">matrix of coefficients.</param>
		/// <param name="outX">vector of data values.</param>
		/// <param name="StoreInverseMatrix">if true, the matrix of coefficients receives the inverse matrix on return.</param>
		/// <returns>the outcome of the computation.</returns>
		/// <returns></returns>
		public static ComputationResult SolveLinearSystemGJ(double[,] inA, ref double[] outX, bool StoreInverseMatrix)
		{
			
			double[,] outA;
			ComputationResult Chk1;
			// se metto ref allora outA = null;
			Chk1=GaussJordanAlgorithm(inA, out outA);
			if ( Chk1 == ComputationResult.OK)
			{
				int n = outA.GetLength(1);
				int m = outA.GetLength(0);

				for (int i=0; i<m; i++) outX[i] =  outA[i,n-1];
				if(StoreInverseMatrix)
				{
					//GJSolutionMatrix = new double[m,m];
					double[,] tmpa  = new double[m,m];
					for(int i = 0; i<m; i++) for(int j = 0; j<m; j++) tmpa[i,j]=inA[i,j];
					
					GJSolutionMatrix = Matrices.InvertMatrix(tmpa);
				}
				return  Chk1;
			}
			else
				return  Chk1;
		}

		/// <summary>
		/// Computes a new solution for a new vector of data, using a stored solution for Gauss-Jordan elimination.
		/// </summary>
		/// <param name="inX">the new vector of data.</param>
		/// <param name="Sol">the new solution.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult NewSolutionGJ(double[] inX, out double[] Sol)
		{
			Sol = Matrices.Product(GJSolutionMatrix, inX);
			return NumericalTools.ComputationResult.OK;
		}

		/// <summary>
		/// Computes an affine transformation + longitudinal zoom.
		/// </summary>
		/// <param name="inDX">input vector of X displacements.</param>
		/// <param name="inDY">input vector of Y displacements.</param>
		/// <param name="inX">input vector of X positions.</param>
		/// <param name="inY">input vector of Y positions.</param>
		/// <param name="inSX">input vector of X slopes.</param>
		/// <param name="inSY">input vector of Y slopes.</param>
		/// <param name="outPar">output parameters.
		/// <list type="table">
		/// <listheader><term>Parameter number</term><description>Meaning</description></listheader>
		/// <item><term>0</term><description>AXX</description></item>
		/// <item><term>1</term><description>AXY</description></item>
		/// <item><term>2</term><description>AYX</description></item>
		/// <item><term>3</term><description>AYY</description></item>
		/// <item><term>4</term><description>TX</description></item>
		/// <item><term>5</term><description>TY</description></item>
		/// <item><term>6</term><description>TZ</description></item>
		/// </list>
		/// </param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Affine_Focusing(double[] inDX, double[] inDY, 
			double[] inX, double[] inY, double[] inSX, double[] inSY,
			ref double[] outPar)
		{
			int i, N; 
			double[,] Mat;
			N = inX.GetLength(0);
			if (N < 7 || N!= inY.GetLength(0) || N!= inDX.GetLength(0) || 
				N!= inDY.GetLength(0) || N!= inSX.GetLength(0) || N!= inSY.GetLength(0)) return NumericalTools.ComputationResult.InvalidInput;

			Mat = new double[7, 8];
		
			for(i = 0;i< N; i++)
			{
				
				Mat[0,0] += inX[i]*inX[i];
				Mat[0,1] += inX[i]*inY[i];
				Mat[1,1] += inY[i]*inY[i];
				Mat[0,4] += inX[i];
				Mat[1,4] += inY[i];
				Mat[0,6] += inX[i]*inSX[i];
				Mat[1,6] += inY[i]*inSX[i];
				Mat[2,6] += inX[i]*inSY[i];
				Mat[3,6] += inY[i]*inSY[i];
				Mat[4,6] += inSX[i];
				Mat[5,6] += inSY[i];
				Mat[6,6] += inSX[i] * inSX[i] + inSY[i] * inSY[i];							
				Mat[0,7] += inDX[i] * inX[i];
				Mat[1,7] += inDX[i] * inY[i];
				Mat[2,7] += inDY[i] * inX[i];
				Mat[3,7] += inDY[i] * inY[i];
				Mat[4,7] += inDX[i];
				Mat[5,7] += inDY[i];
				Mat[6,7] += inDX[i] * inSX[i] + inDY[i] * inSY[i];
			};
			Mat[4,4] = N;
			Mat[5,5] = N;

			Mat[2,2] = Mat[0,0];
			Mat[1,0] = Mat[0,1];
			Mat[3,3] = Mat[1,1];
			Mat[2,3] = Mat[0,1];
			Mat[3,2] = Mat[0,1];
			Mat[4,0] = Mat[0,4];
			Mat[4,1] = Mat[1,4];
			Mat[5,2] = Mat[0,4];
			Mat[5,3] = Mat[1,4];
			Mat[2,5] = Mat[0,4];
			Mat[3,5] = Mat[1,4];
			for(i=0; i<6; i++) Mat[6,i]=Mat[i,6];

			SolveLinearSystem(Mat, ref outPar);

			return NumericalTools.ComputationResult.OK;

		}

		/// <summary>
		/// Computes an affine transformation.
		/// </summary>
		/// <param name="inDX">input vector of X displacements.</param>
		/// <param name="inDY">input vector of Y displacements.</param>
		/// <param name="inX">input vector of X positions.</param>
		/// <param name="inY">input vector of Y positions.</param>
		/// <param name="inSX">input vector of X slopes.</param>
		/// <param name="inSY">input vector of Y slopes.</param>
		/// <param name="outPar">output parameters.
		/// <list type="table">
		/// <listheader><term>Parameter number</term><description>Meaning</description></listheader>
		/// <item><term>0</term><description>AXX</description></item>
		/// <item><term>1</term><description>AXY</description></item>
		/// <item><term>2</term><description>AYX</description></item>
		/// <item><term>3</term><description>AYY</description></item>
		/// <item><term>4</term><description>TX</description></item>
		/// <item><term>5</term><description>TY</description></item>
		/// </list>
		/// </param>
		/// <remarks>The zoom translation (along Z) is set to 0.</remarks>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Affine(double[] inDX, double[] inDY, 
			double[] inX, double[] inY, /*double[] inSX, double[] inSY,*/
			ref double[] outPar)
		{
			int i, N; 
			double[,] Mat;
			N = inX.GetLength(0);
			if (N < 3 || N!= inY.GetLength(0) || N!= inDX.GetLength(0) || 
				N!= inDY.GetLength(0) /*|| N!= inSX.GetLength(0) || N!= inSY.GetLength(0)*/) return NumericalTools.ComputationResult.InvalidInput;

			Mat = new double[6, 7];
			double[] t_outpar = new double[6];
		
			for(i = 0;i< N; i++)
			{
				
				Mat[0,0] += inX[i]*inX[i];
				Mat[0,1] += inX[i]*inY[i];
				Mat[1,1] += inY[i]*inY[i];
				Mat[0,4] += inX[i];
				Mat[1,4] += inY[i];
				/*				Mat[0,6] += inX[i]*inSX[i];
								Mat[1,6] += inY[i]*inSX[i];
								Mat[2,6] += inX[i]*inSY[i];
								Mat[3,6] += inY[i]*inSY[i];
								Mat[4,6] += inSX[i];
								Mat[5,6] += inSY[i];
								Mat[6,6] += inSX[i] * inSX[i] + inSY[i] * inSY[i];							
				*/				Mat[0,6] += inDX[i] * inX[i];
				Mat[1,6] += inDX[i] * inY[i];
				Mat[2,6] += inDY[i] * inX[i];
				Mat[3,6] += inDY[i] * inY[i];
				Mat[4,6] += inDX[i];
				Mat[5,6] += inDY[i];
				//				Mat[6,7] += inDX[i] * inSX[i] + inDY[i] * inSY[i];
			};
			Mat[4,4] = N;
			Mat[5,5] = N;

			Mat[2,2] = Mat[0,0];
			Mat[1,0] = Mat[0,1];
			Mat[3,3] = Mat[1,1];
			Mat[2,3] = Mat[0,1];
			Mat[3,2] = Mat[0,1];
			Mat[4,0] = Mat[0,4];
			Mat[4,1] = Mat[1,4];
			Mat[5,2] = Mat[0,4];
			Mat[5,3] = Mat[1,4];
			Mat[2,5] = Mat[0,4];
			Mat[3,5] = Mat[1,4];

			SolveLinearSystem(Mat, ref t_outpar);
			
			for(i = 0; i < 6; i++) outPar[i]=t_outpar[i];
			//outPar[6]=0;

			return NumericalTools.ComputationResult.OK;

		}

		/// <summary>
		/// Computes an affine transformation + longitudinal zoom.
		/// </summary>
		/// <param name="inDX">input vector of X displacements.</param>
		/// <param name="inDY">input vector of Y displacements.</param>
		/// <param name="inX">input vector of X positions.</param>
		/// <param name="inY">input vector of Y positions.</param>
		/// <param name="inSX">input vector of X slopes.</param>
		/// <param name="inSY">input vector of Y slopes.</param>
		/// <param name="outPar">output parameters.
		/// <list type="table">
		/// <listheader><term>Parameter number</term><description>Meaning</description></listheader>
		/// <item><term>0</term><description>AXX</description></item>
		/// <item><term>1</term><description>AXY</description></item>
		/// <item><term>2</term><description>AYX</description></item>
		/// <item><term>3</term><description>AYY</description></item>
		/// <item><term>4</term><description>TX</description></item>
		/// <item><term>5</term><description>TY</description></item>
		/// <item><term>6</term><description>TZ</description></item>
		/// </list>
		/// </param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Affine_Focusing(float[] inDX, float[] inDY, 
			float[] inX, float[] inY, float[] inSX, float[] inSY,
			ref float[] outPar)
		{
			int i, n;
			double[] tinDX, tinDY, tinX, tinY, tinSX, tinSY;
			double[] toutPar = new double[7];

			n = inX.GetLength(0);
			if (n < 7 || n!= inY.GetLength(0) || n!= inDX.GetLength(0) || 
				n!= inDY.GetLength(0) || n!= inSX.GetLength(0) || n!= inSY.GetLength(0)) return NumericalTools.ComputationResult.InvalidInput;

			tinDX = new double[n];
			tinDY = new double[n];
			tinX = new double[n];
			tinY = new double[n];
			tinSX = new double[n];
			tinSY = new double[n];

			for(i = 0;i< n; i++)
			{
				tinDX[i] = (double)inDX[i];
				tinDY[i] = (double)inDY[i];
				tinX[i] = (double)inX[i];
				tinY[i] = (double)inY[i];
				tinSX[i] = (double)inSX[i];
				tinSY[i] = (double)inSY[i];
			};				
			ComputationResult cr = Affine_Focusing(tinDX, tinDY, tinX, tinY, tinSX, tinSY, ref toutPar);
			for(i = 0;i< 7; i++) outPar[i]=(float)toutPar[i];
			return cr;

		}

		/// <summary>
		/// Computes a polynomial fit.
		/// </summary>
		/// <param name="inX">values of the independent variable.</param>
		/// <param name="inY">values of the dependent variable.</param>
		/// <param name="Degree">the degree of the polynomial.</param>
		/// <param name="outA">the output coefficients, in ascending order.</param>
		/// <param name="CCorr">the correlation coefficient.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult PolynomialFit(double[] inX, double[] inY, short Degree,
			ref double[] outA, ref double CCorr)
		{
			int i, k, N, l, j; 
			double[,] Mat;
			if (Degree < 2 || Degree > 15) return NumericalTools.ComputationResult.InvalidInput;
			N = inX.GetLength(0);

			Mat = new double[Degree + 1, Degree + 2];
		
			//for(i = 1;i<= Degree + 1; i++)
			for(i = 0;i< Degree + 1; i++)
			{
				//for(j = 1; j<= Degree + 1; j++)
				for(j = 0; j< Degree + 1; j++)
				{
					k = i + j;
					//k = i + j - 2;
					//for(l = 1 ; l<= N;l++)
					for(l = 0 ; l< N;l++)
					{
						Mat[i, j] = Mat[i, j] + System.Math.Pow(inX[l],k);
					};
				};
				//for(l = 1; l<= N;l++)
				for(l = 0 ; l< N;l++)
				{
					Mat[i, Degree + 1] = Mat[i, Degree + 1] + inY[l] * System.Math.Pow(inX[l] ,i);
					//Mat[i, Degree + 2] = Mat[i, Degree + 2] + inY[l] * System.Math.Pow(inX[l] ,(i - 1));
				};
			};

			SolveLinearSystem(Mat, ref outA);

			double Delta, Sr, St, dum;
			Sr=0; dum=0; St=0;
			FindStatistics(inY,  ref dum,  ref dum,  ref dum,  ref St);
			St = (St *St) * (N - 1); // era/ N

			//for(i = 1; i<= N; i++)
			for(i = 0; i< N; i++)
			{
				Delta = inY[i];
				//for(j = 1; j<= Degree + 1; j++)
				for(j = 0; j< Degree + 1; j++)
				{
					//Delta = Delta - (Math.Pow(inX[i], (j - 1))) * outA[j];
					Delta = Delta - (Math.Pow(inX[i], j)) * outA[j];
				};
				Sr = Sr + Delta * Delta;
			};
			CCorr = Math.Sqrt(1 - Sr / St);

			return NumericalTools.ComputationResult.OK;

		}

		//Method called by LM_GaussianRegression
		private static ComputationResult ComputeAlphaAndBeta(double[] inPoints, 
			double[] inVals, double[] vectorPars, double fLambda, 
			out double[,] matrixAlpha, out double[] vectorBeta) 
		{
			int iLen; 
			int i, j;
			ComputationResult chk1, chk2;
			chk1=ComputationResult.OK;
			chk2=chk1;

			//			try
			//			{

			//Vector pars is composed of Norm, Mean and Sigma
			iLen = vectorPars.GetLength(0);

			//Levenberg-Marquardt method: 
			//Increase absolute value of elements on diagonal;
			//Calculate -Gradient;
			chk1 = GaussianHessian(inPoints, inVals, vectorPars,  out matrixAlpha);
			chk2 = GaussianGradient(inPoints, inVals, vectorPars, out vectorBeta);
			for(i = 0; i< iLen; i++)
			{
				for(j = 0; j< iLen; j++)
				{
					matrixAlpha[i, j] = 0.5 * matrixAlpha[i, j];
					if(i == j) matrixAlpha[i, j] = matrixAlpha[i, j] * (1.0 + fLambda);
				};
				vectorBeta[i] = -1 * 0.5 * vectorBeta[i];
			};
			//			}
			//			catch
			//			{
			if(chk1!= ComputationResult.OK) 
			{
				return chk1;
			}
			else if(chk2!= ComputationResult.OK)
			{
				return chk2;
				//				return ComputationResult.SingularityEncountered;
				//			};
			}
			else
			{
				return chk1;
			};
		}

		//Method called by LM_GaussianRegression
		private static ComputationResult ComputeInverseAlphaAndBeta(double[] inPoints, 
			double[] inVals, double[] vectorPars, double fLambda, 
			out double[,] matrixAlpha, out double[] vectorBeta) 
		{
			int iLen; 
			int i, j;
			ComputationResult chk1, chk2;
			chk1=ComputationResult.OK;
			chk2=chk1;

			//			try
			//			{

			//Vector pars is composed of Norm, Mean and Sigma
			iLen = vectorPars.GetLength(0);

			//Levenberg-Marquardt method: 
			//Increase absolute value of elements on diagonal;
			//Calculate -Gradient;
			chk1 = InverseGaussianHessian(inPoints, inVals, vectorPars,  out matrixAlpha);
			chk2 = InverseGaussianGradient(inPoints, inVals, vectorPars, out vectorBeta);
			for(i = 0; i< iLen; i++)
			{
				for(j = 0; j< iLen; j++)
				{
					matrixAlpha[i, j] = 0.5 * matrixAlpha[i, j];
					if(i == j) matrixAlpha[i, j] = matrixAlpha[i, j] * (1.0 + fLambda);
				};
				vectorBeta[i] = -1 * 0.5 * vectorBeta[i];
			};
			//			}
			//			catch
			//			{
			if(chk1!= ComputationResult.OK) 
			{
				return chk1;
			}
			else if(chk2!= ComputationResult.OK)
			{
				return chk2;
				//				return ComputationResult.SingularityEncountered;
				//			};
			}
			else
			{
				return chk1;
			};
		}
		
		//Method called by LM_GaussianRegression
		private static ComputationResult GaussianGradient(double[] inPoints,
			double[] inVals, double[] vectorPars, out double[] GradientVector)
		{
			int i, CountOfPoints, nGr;
			double g, dx, dy;
			double[] TmpGrad = new double[3];
			GradientVector = new double[3];
			double Sig_2, Sig_3;
			CountOfPoints = inPoints.GetLength(0);
			nGr = GradientVector.GetLength(0);
			Sig_2 = vectorPars[1] * vectorPars[1];
			Sig_3 = Sig_2 * vectorPars[1];
			for(i = 0; i< CountOfPoints; i++)
			{
				g = vectorPars[2] * Math.Exp(-Math.Pow((inPoints[i] - vectorPars[0]), 2) / (2 * Sig_2));
				dx = (inPoints[i] - vectorPars[0]);
				dy = inVals[i] - g;
				TmpGrad[0] = TmpGrad[0] + (g * dx / Sig_2) * dy;
				TmpGrad[1] = TmpGrad[1] + (g * dx * dx / Sig_3) * dy;
				TmpGrad[2] = TmpGrad[2] + (g / vectorPars[2]) * dy;
			};
			for(i = 0;  i<nGr; i++)
			{
				GradientVector[i] = -2 * TmpGrad[i];
			};
			return ComputationResult.OK;
		}

		//Method called by LM_GaussianRegression
		private static ComputationResult InverseGaussianGradient(double[] inPoints,
			double[] inVals, double[] vectorPars, out double[] GradientVector)
		{
			int i, CountOfPoints, nGr;
			double g, dx, dy;
			double[] TmpGrad = new double[3];
			GradientVector = new double[3];
			double Sig0_2, Sig_2, Sig_3;
			CountOfPoints = inPoints.GetLength(0);
			nGr = GradientVector.GetLength(0);
			Sig0_2 = vectorPars[0] * vectorPars[0];
			Sig_2 = vectorPars[1] * vectorPars[1];
			Sig_3 = Sig_2 * vectorPars[1];
			for(i = 0; i< CountOfPoints; i++)
			{
				g = (vectorPars[2]/(inPoints[i]*inPoints[i]))* Math.Exp(-Math.Pow((1/inPoints[i] - 1/vectorPars[0]), 2) / (Sig_2));
				dx = (1/inPoints[i] - 1/vectorPars[0]);
				dy = inVals[i] - g;
				TmpGrad[0] = TmpGrad[0] + (-2 * g * dx / (Sig_2 * Sig0_2)) * dy;
				TmpGrad[1] = TmpGrad[1] + (g * 2 * dx * dx / Sig_3) * dy;
				TmpGrad[2] = TmpGrad[2] + (g / vectorPars[2]) * dy;
			};
			for(i = 0;  i<nGr; i++)
			{
				GradientVector[i] = -2 * TmpGrad[i];
			};
			return ComputationResult.OK;
		}

		//Method called by LM_GaussianRegression
		private static ComputationResult GaussianHessian(double[] inPoints,
			double[] inVals, double[] vectorPars, out double[,] HessianMatrix)
		{
			short i, j; 
			int nPt, nMat;
			double g, dx_2, dx, dx_3, Sig_2, Sig_3 ;
			double[,] TmpHess= new double[3, 3];
			HessianMatrix= new double[3, 3];
			Sig_2 = vectorPars[1] * vectorPars[1];
			Sig_3 = Sig_2 * vectorPars[1];
			nPt = inPoints.GetLength(0);
			nMat =HessianMatrix.GetLength(0);

			for (i=0; i<nPt; i++)
			{
				g = vectorPars[2] * Math.Exp(-Math.Pow((inPoints[i] - vectorPars[0]), 2) / (2 * Sig_2));
				dx = (inPoints[i] - vectorPars[0]);
				dx_2 = dx * dx;
				dx_3 = dx * dx_2;
				TmpHess[0, 0] = TmpHess[0, 0] + Math.Pow((g * dx / Sig_2), 2);
				TmpHess[0, 1] = TmpHess[0, 1] + (g * dx / Sig_2) * (g * dx_2 / Sig_3);
				TmpHess[0, 2] = TmpHess[0, 2] + (g * dx / Sig_2) * (g / vectorPars[2]);
				TmpHess[1, 1] = TmpHess[1, 1] + Math.Pow((g * dx_2 / Sig_3) , 2);
				TmpHess[1, 2] = TmpHess[1, 2] + (g * dx_2 / Sig_3) * (g / vectorPars[2]);
				TmpHess[2, 2] = TmpHess[2, 2] + Math.Pow((g / vectorPars[2]) , 2);
			};
			
			for (i=0; i<nMat; i++)
			{
				for (j=i; j<nMat; j++)
				{
					HessianMatrix[i,j] = 2 * TmpHess[i,j];
				};
			};
			HessianMatrix[1, 0] = HessianMatrix[0, 1];
			HessianMatrix[2, 0] = HessianMatrix[0, 2];
			HessianMatrix[2, 1] = HessianMatrix[1, 2];

			return ComputationResult.OK;

		}

		//Method called by LM_GaussianRegression
		private static ComputationResult InverseGaussianHessian(double[] inPoints,
			double[] inVals, double[] vectorPars, out double[,] HessianMatrix)
		{
			short i, j; 
			int nPt, nMat;
			double g, dx_2, dx, dx_3, Sig_2, Sig0_2, Sig_3 ;
			double[,] TmpHess= new double[3, 3];
			HessianMatrix= new double[3, 3];
			Sig0_2 = vectorPars[0] * vectorPars[0];
			Sig_2 = vectorPars[1] * vectorPars[1];
			Sig_3 = Sig_2 * vectorPars[1];
			nPt = inPoints.GetLength(0);
			nMat =HessianMatrix.GetLength(0);

			for (i=0; i<nPt; i++)
			{
				g = (vectorPars[2]/(inPoints[i]*inPoints[i]))* Math.Exp(-Math.Pow((1/inPoints[i] - 1/vectorPars[0]), 2) / (Sig_2));
				dx = (1/inPoints[i] - 1/vectorPars[0]);
				dx_2 = dx * dx;
				dx_3 = dx * dx_2;

				//Cambiare questi 6
				TmpHess[0, 0] = TmpHess[0, 0] + Math.Pow((-2 * g * dx / (Sig_2 * Sig0_2)) , 2);
				TmpHess[0, 1] = TmpHess[0, 1] + (-2 * g * dx / (Sig_2 * Sig0_2))  * (g * 2 * dx * dx / Sig_3);
				TmpHess[0, 2] = TmpHess[0, 2] + (-2 * g * dx / (Sig_2 * Sig0_2))  * (g / vectorPars[2]);
				TmpHess[1, 1] = TmpHess[1, 1] + Math.Pow((g * 2 * dx * dx / Sig_3) , 2);
				TmpHess[1, 2] = TmpHess[1, 2] + (g * 2 * dx * dx / Sig_3) * (g / vectorPars[2]);
				TmpHess[2, 2] = TmpHess[2, 2] + Math.Pow((g / vectorPars[2]) , 2);
			};
			
			for (i=0; i<nMat; i++)
			{
				for (j=i; j<nMat; j++)
				{
					HessianMatrix[i,j] = 2 * TmpHess[i,j];
				};
			};
			HessianMatrix[1, 0] = HessianMatrix[0, 1];
			HessianMatrix[2, 0] = HessianMatrix[0, 2];
			HessianMatrix[2, 1] = HessianMatrix[1, 2];

			return ComputationResult.OK;

		}

		//Method called by LM_GaussianRegression
		private static ComputationResult ComputeChiSquared(double[] inPoints, 
			double[] inVals, double[] vectorPars, out double ChiSq, out double GaussSum)
		{
			int i, N;
			double inGauss, tmpN; 
			N = inPoints.GetLength(0);
			ChiSq = 0;
			GaussSum = 0;
			//Since errors Sigma_y are almost the same for each entry
			//and since we are looking for a minimum value of chisquared
			//we can obmit them
			for(i = 0; i< N; i++)
			{
				tmpN = vectorPars[2];
				inGauss = tmpN * Math.Exp(-Math.Pow((inPoints[i] - vectorPars[0]), 2) / (2 * Math.Pow(vectorPars[1],2)));
				GaussSum = GaussSum + inGauss;
				ChiSq = ChiSq + (inVals[i] - inGauss) *(inVals[i] - inGauss);
			};
			return ComputationResult.OK;
		}
		
		//Method called by LM_InverseGaussianRegression
		private static ComputationResult ComputeInverseChiSquared(double[] inPoints, 
			double[] inVals, double[] vectorPars, out double ChiSq, out double GaussSum)
		{
			int i, N;
			double inGauss, tmpN; 
			N = inPoints.GetLength(0);
			ChiSq = 0;
			GaussSum = 0;
			//Since errors Sigma_y are almost the same for each entry
			//and since we are looking for a minimum value of chisquared
			//we can obmit them
			for(i = 0; i< N; i++)
			{
				tmpN = vectorPars[2];
				inGauss = (tmpN/(inPoints[i]*inPoints[i]))* Math.Exp(-Math.Pow((1/inPoints[i] - 1/vectorPars[0]), 2) / (Math.Pow(vectorPars[1],2)));
				GaussSum = GaussSum + inGauss;
				ChiSq = ChiSq + (inVals[i] - inGauss) *(inVals[i] - inGauss);
			};
			return ComputationResult.OK;
		}
		

		/// <summary>
		/// Computes the Gaussian fit of a set of values.
		/// </summary>
		/// <param name="X">the values for which mean and sigma are sought.</param>
		/// <param name="BinSize">the size of bins.</param>
		/// <param name="MinChi2Change">if the difference in chi<sup>2</sup> between two iterations is smaller than this, iteration stops.</param>
		/// <param name="MaxUnsuccessIter">maximum number of unsuccessful iterations.</param>
		/// <param name="MaxIters">maximum number of iterations.</param>
		/// <param name="MaxAttempts">if the first attempt fails, other attempts are performed starting from random points, until the fit converges or this number of attempts is exceeded.</param>
		/// <param name="Parameters">output parameters:
		/// <list type="table">
		/// <listheader><term>Parameter Number</term><description>Meaning</description></listheader>
		/// <item><term>0</term><description>Mu</description></item>
		/// <item><term>1</term><description>Sigma</description></item>
		/// <item><term>2</term><description>Normalization</description></item>
		/// </list>
		/// </param>
		/// <param name="Iter">iterations per attempt.</param>
		/// <param name="Chi2Fit">chi<sup>2</sup> of the fit.</param>
		/// <param name="FitResult">the result of the fit.
		/// <list type="table">
		/// <listheader><term>Fit Code</term><description>Meaning</description></listheader>
		/// <item><term>1</term><description>Fit OK</description></item>
		/// <item><term>2</term><description>Maximum unsuccessful iterations exceeded</description></item>
		/// <item><term>3</term><description>Fit OK with stationary chi<sup>2</sup></description></item>
		/// <item><term>4</term><description>Maximum iterations exceeded</description></item>
		/// <item><term>5</term><description>Fit converged to an anomalous result</description></item>
		/// </list>
		/// </param>
		public static void LM_GaussianRegression(double[] X, double BinSize,
			double MinChi2Change, int MaxUnsuccessIter, int MaxIters, int MaxAttempts, 
			out double[] Parameters, out int[] Iter, out double Chi2Fit, out short FitResult)
		{
			int iUnsuccIters, iCountOfParams, j, i, N,iters , k;
			double Chi2, NewChi2,LambdaStart, Lambda, LambdaStep;
			double MinX=0,MaxX=0,AccuFactor=0, TraslaX=0, GSum;
			ComputationResult CheckOk;
			double[] vectorCurrParams;
			double[] VectorB;
			double[,] MatrixA;
			double[,] LSMatrixA;
			double[,] CollectionParams;
			double[] vectorDeltaParams;
			double[] inPoints;
			double[] inVals;
			double[] tmpX;
			double[] Y;
			
			//Preliminary operations
			Parameters = new double[3];
			Iter = new int[2];
			Chi2Fit=0;
			iCountOfParams = Parameters.GetLength(0);
			vectorDeltaParams = new double[iCountOfParams];
			vectorCurrParams = new double[iCountOfParams];
			CollectionParams = new double[2, 10];
			VectorB = new double[iCountOfParams];
			MatrixA = new double[iCountOfParams, iCountOfParams];
			LSMatrixA = new double[iCountOfParams, iCountOfParams + 1];

			iters=0;
			Chi2=0;
			FitResult=0;
			LambdaStart = 0.001;
			LambdaStep = 10.0;
			N = X.GetLength(0);
			tmpX = new double[N];
			//   Initial values
			for(k = 1; k<= MaxAttempts; k++)
			{
				tmpX = (double[])X.Clone();
				iters = 0 ; iUnsuccIters = 0;
				FindStatistics(tmpX, ref MaxX, ref MinX, ref TraslaX, ref AccuFactor);

				//Always normalize to Gaussian having <X> = 0 and St_Dev = 1
				for(j = 0; j< N; j++)
				{
					tmpX[j] = (tmpX[j] - TraslaX) / AccuFactor;
				};
				vectorCurrParams[0] = 0;
				vectorCurrParams[1] = 1;
				Prepare_Custom_Distribution(tmpX, 1, BinSize / AccuFactor, 0, out inPoints, out Y, out inVals);
				
				//If the first attempt fails
				if(k >= 2 && k <= MaxAttempts / 4) 
				{
					vectorCurrParams[0] = vectorCurrParams[0] + 0.1 * RandomGenerator.RND.NextDouble();
					vectorCurrParams[1] = vectorCurrParams[1] + 0.1 * RandomGenerator.RND.NextDouble() * vectorCurrParams[1];
				}
				else if(k > MaxAttempts / 4 && k <= MaxAttempts / 2)
				{
					vectorCurrParams[0] = vectorCurrParams[0] + 0.1 * RandomGenerator.RND.NextDouble();
					vectorCurrParams[1] = vectorCurrParams[1] - 0.1 * RandomGenerator.RND.NextDouble() * vectorCurrParams[1];
				}
				else if( k > MaxAttempts / 2 && k <= 3 * MaxAttempts / 4 )
				{
					vectorCurrParams[0] = vectorCurrParams[0] - 0.1 * RandomGenerator.RND.NextDouble();
					vectorCurrParams[1] = vectorCurrParams[1] + 0.1 * RandomGenerator.RND.NextDouble() * vectorCurrParams[1];
				}
				else if( k > 3 * MaxAttempts / 4 && k <= MaxAttempts)
				{
					vectorCurrParams[0] = vectorCurrParams[0] - 0.1 * RandomGenerator.RND.NextDouble();
					vectorCurrParams[1] = vectorCurrParams[1] - 0.1 * RandomGenerator.RND.NextDouble() * vectorCurrParams[1];
				};																																							  
				vectorCurrParams[2] = 1 / (Math.Sqrt(2 * Math.PI) * vectorCurrParams[1]);

				//   Init iterative algorithm
				Lambda = LambdaStart;
				ComputeChiSquared(inPoints, inVals, vectorCurrParams, out Chi2, out GSum);

				while(iters < MaxIters || iUnsuccIters < MaxUnsuccessIter)
				{	
					iters++;
					ComputeAlphaAndBeta(inPoints, inVals, vectorCurrParams, Lambda, out MatrixA, out VectorB);

					for(i = 0;i<  iCountOfParams;i++)
					{
						for(j = 0; j< iCountOfParams; j++)
							LSMatrixA[i, j] = MatrixA[i, j];

						LSMatrixA[i, j] = VectorB[i];
					};

					CheckOk = SolveLinearSystem(LSMatrixA, ref vectorDeltaParams);

					// We will soon need CurrParams + DeltaParams, so use DeltaParams to temporarily
					// store the value, while preserving CurrParams in case of unsuccessful iteration
					if (CheckOk == ComputationResult.OK)
					{
						for( i = 0; i< iCountOfParams; i++)
							vectorDeltaParams[i] = vectorCurrParams[i] + vectorDeltaParams[i];

						ComputeChiSquared(inPoints, inVals, vectorDeltaParams, out NewChi2, out GSum);
					
						if(NewChi2 > Chi2)
						{
							//If fNewChi2 is worse the temporary vectorCurrParams
							//still remains the better set of parameters and vectorDeltaParams
							//will be lost
							iUnsuccIters ++;
							if (iUnsuccIters >= MaxUnsuccessIter) 
							{
								FitResult = 2;
								break;
							}
							else
								Lambda = Lambda * LambdaStep;
						}
						else
						{
							iUnsuccIters = 0;
							if ((Chi2 - NewChi2) < MinChi2Change)
							{
								//This Case is successful
								if(Chi2 != NewChi2)
									FitResult = 1;
								else if(Chi2 == NewChi2)
									FitResult = 3;
								break;
							}
							else
							{
								Lambda = Lambda / LambdaStep;
								Chi2 = NewChi2;
								//If fNewChi2 is better the temporary vectorDeltaParams
								//is stored into the vectorCurrParams
								vectorCurrParams = (double[])vectorDeltaParams.Clone();
							};
						};
						if(iters > MaxIters)
						{
							FitResult = 4;
							break;
						};
					}
					else
					{
						iUnsuccIters ++;
						if (iUnsuccIters >= MaxUnsuccessIter) 
						{
							FitResult = 2;
							break;
						}
						else
							Lambda = Lambda * LambdaStep;

					};
					
				};
				
				if (FitResult == 1) 
				{
					if (Math.Abs(vectorCurrParams[0]) < 5 && Math.Abs(vectorCurrParams[1] - 1) < 5 
						&& Math.Abs((vectorCurrParams[1] - 1) / vectorCurrParams[1]) < 5 && vectorCurrParams[2]<1)
					{
						//Sometimes it converges to -sigma so we take abs(sigma)

						vectorCurrParams[0] = vectorCurrParams[0]* AccuFactor + TraslaX;
						vectorCurrParams[1] = Math.Abs(vectorCurrParams[1]) * AccuFactor;
						break;
					}
					else
						FitResult = 5;
				};

			};

			if(FitResult != 1)
			{ 
				FindStatistics(X,  ref MaxX,  ref MinX,  ref vectorCurrParams[0],  ref vectorCurrParams[1]);
				vectorCurrParams[2] = 1 / (Math.Sqrt(2 * Math.PI) * vectorCurrParams[1]);
			}
			else
			{
				Iter[0] = k;
				Iter[1] = iters;
				Chi2Fit = Chi2;
			};
			for(i = 0; i<iCountOfParams ; i++)
				Parameters[i] = vectorCurrParams[i];
		}


		/// <summary>
		/// Computes the Inverse Gaussian fit of a set of values.
		/// </summary>
		/// <param name="X">the values for which mean and sigma are sought.</param>
		/// <param name="BinSize">the size of bins.</param>
		/// <param name="MinChi2Change">if the difference in chi<sup>2</sup> between two iterations is smaller than this, iteration stops.</param>
		/// <param name="MaxUnsuccessIter">maximum number of unsuccessful iterations.</param>
		/// <param name="MaxIters">maximum number of iterations.</param>
		/// <param name="MaxAttempts">if the first attempt fails, other attempts are performed starting from random points, until the fit converges or this number of attempts is exceeded.</param>
		/// <param name="Parameters">output parameters:
		/// <list type="table">
		/// <listheader><term>Parameter Number</term><description>Meaning</description></listheader>
		/// <item><term>0</term><description>Mu</description></item>
		/// <item><term>1</term><description>Sigma</description></item>
		/// <item><term>2</term><description>Normalization</description></item>
		/// </list>
		/// </param>
		/// <param name="Iter">iterations per attempt.</param>
		/// <param name="Chi2Fit">chi<sup>2</sup> of the fit.</param>
		/// <param name="FitResult">the result of the fit.
		/// <list type="table">
		/// <listheader><term>Fit Code</term><description>Meaning</description></listheader>
		/// <item><term>1</term><description>Fit OK</description></item>
		/// <item><term>2</term><description>Maximum unsuccessful iterations exceeded</description></item>
		/// <item><term>3</term><description>Fit OK with stationary chi<sup>2</sup></description></item>
		/// <item><term>4</term><description>Maximum iterations exceeded</description></item>
		/// </list>
		/// </param>		
		public static void LM_InverseGaussianRegression(double[] X, double BinSize,
			double MinChi2Change, int MaxUnsuccessIter, int MaxIters, int MaxAttempts, 
			out double[] Parameters, out int[] Iter, out double Chi2Fit, out short FitResult)
		{
			int iUnsuccIters, iCountOfParams, j, i, N,iters , k=0;
			double Chi2, NewChi2,LambdaStart, Lambda, LambdaStep;
			double MinX=0,MaxX=0; //,AccuFactor=0, TraslaX=0, 
			double GSum;
			ComputationResult CheckOk;
			double[] vectorCurrParams;
			double[] VectorB;
			double[,] MatrixA;
			double[,] LSMatrixA;
			double[,] CollectionParams;
			double[] vectorDeltaParams;
			double[] inPoints;
			double[] inVals;
			double[] tmpX;
			double[] Y;
			
			//Preliminary operations
			Parameters = new double[3];
			Iter = new int[2];
			Chi2Fit=0;
			iCountOfParams = Parameters.GetLength(0);
			vectorDeltaParams = new double[iCountOfParams];
			vectorCurrParams = new double[iCountOfParams];
			CollectionParams = new double[2, 10];
			VectorB = new double[iCountOfParams];
			MatrixA = new double[iCountOfParams, iCountOfParams];
			LSMatrixA = new double[iCountOfParams, iCountOfParams + 1];

			iters=0;
			Chi2=0;
			FitResult=0;
			LambdaStart = 0.001;
			LambdaStep = 10.0;
			N = X.GetLength(0);
			tmpX = new double[N];

			tmpX = (double[])X.Clone();
			iters = 0 ; iUnsuccIters = 0;

			FindStatistics(tmpX, ref MaxX, ref MinX, ref vectorCurrParams[0], ref vectorCurrParams[1]);
			Prepare_Custom_Distribution(tmpX, 1, BinSize, 0, out inPoints, out Y, out inVals);

			vectorCurrParams[1]/=vectorCurrParams[0]*vectorCurrParams[0];
			vectorCurrParams[2] = 1 / (Math.Sqrt(2 * Math.PI) * vectorCurrParams[1]);
			
			//   Init iterative algorithm
			Lambda = LambdaStart;
			ComputeInverseChiSquared(inPoints, inVals, vectorCurrParams, out Chi2, out GSum);

			while(iters < MaxIters || iUnsuccIters < MaxUnsuccessIter)
			{	
				iters++;
				ComputeInverseAlphaAndBeta(inPoints, inVals, vectorCurrParams, Lambda, out MatrixA, out VectorB);

				for(i = 0;i<  iCountOfParams;i++)
				{
					for(j = 0; j< iCountOfParams; j++)
						LSMatrixA[i, j] = MatrixA[i, j];

					LSMatrixA[i, j] = VectorB[i];
				};

				CheckOk = SolveLinearSystem(LSMatrixA, ref vectorDeltaParams);

				// We will soon need CurrParams + DeltaParams, so use DeltaParams to temporarily
				// store the value, while preserving CurrParams in case of unsuccessful iteration
				if (CheckOk == ComputationResult.OK)
				{
					for( i = 0; i< iCountOfParams; i++)
						vectorDeltaParams[i] = vectorCurrParams[i] + vectorDeltaParams[i];

					ComputeInverseChiSquared(inPoints, inVals, vectorDeltaParams, out NewChi2, out GSum);
					
					if(NewChi2 > Chi2)
					{
						//If fNewChi2 is worse the temporary vectorCurrParams
						//still remains the better set of parameters and vectorDeltaParams
						//will be lost
						iUnsuccIters ++;
						if (iUnsuccIters >= MaxUnsuccessIter) 
						{
							FitResult = 2;
							break;
						}
						else
							Lambda = Lambda * LambdaStep;
					}
					else
					{
						iUnsuccIters = 0;
						if ((Chi2 - NewChi2) < MinChi2Change)
						{
							//This Case is successful
							if(Chi2 != NewChi2)
								FitResult = 1;
							else if(Chi2 == NewChi2)
								FitResult = 3;
							break;
						}
						else
						{
							Lambda = Lambda / LambdaStep;
							Chi2 = NewChi2;
							//If fNewChi2 is better the temporary vectorDeltaParams
							//is stored into the vectorCurrParams
							vectorCurrParams = (double[])vectorDeltaParams.Clone();
						};
					};
					if(iters > MaxIters)
					{
						FitResult = 4;
						break;
					};
				}
				else
				{
					iUnsuccIters ++;
					if (iUnsuccIters >= MaxUnsuccessIter) 
					{
						FitResult = 2;
						break;
					}
					else
						Lambda = Lambda * LambdaStep;

				};
					
			};
				
			if(FitResult != 1)
			{ 
				FindStatistics(X,  ref MaxX,  ref MinX,  ref vectorCurrParams[0],  ref vectorCurrParams[1]);
				vectorCurrParams[2] = 1 / (Math.Sqrt(2 * Math.PI) * vectorCurrParams[1]);
			}
			else
			{
				Iter[0] = k;
				Iter[1] = iters;
				Chi2Fit = Chi2;
			};
			for(i = 0; i<iCountOfParams ; i++)
				Parameters[i] = vectorCurrParams[i];
		}
		
		/// <summary>
		/// Performs multiple linear regression of up to 15 variables.
		/// </summary>
		/// <param name="inX">values of independent vectors, stored by rows</param>
		/// <param name="inZ">values of dependent variable</param>
		/// <param name="outA">vector of regression results; the value at index 0 is the constant</param>
		/// <param name="CCorr">correlation coefficient</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult MultipleLinearRegression(double[,] inX, double[] inZ,
			ref double[] outA, ref double CCorr)
		{
			int i, N, l, j, h;
			double[,] Mat;
			int Degree;

			Degree = inX.GetLength(0); //è il numero di variabili indipendenti
			N = inX.GetLength(1); //è il numero di entries

			if(Degree > 15 || N <= Degree || inZ.GetLength(0) != N) return ComputationResult.InvalidInput;

			Mat= new double[Degree + 1, Degree + 2];
			double tmpMolt;

			for(i = 0; i< Degree + 1; i++)
				for(j = 0; j< Degree + 1;j++)
					for(l = 0; l< N;l++)
					{
						tmpMolt = 1;
						//in vb andava da 1 a Degree
						for(h = 0;h< Degree;h++)
							if(i > 0)
							{
								if(i != j && (h == j - 1 || h == i - 1))
									tmpMolt *= inX[h, l];
								else if(i == j && h == i - 1)
									tmpMolt *= inX[h, l] *inX[h, l];
							}	
							else
							{
								if(i != j && h == j - 1) tmpMolt *= inX[h, l];
							};

						Mat[i, j] += tmpMolt;
					};

			for(i = 0; i< Degree + 1;i++)
				for(l = 0; l< N;l++)
				{
					tmpMolt = inZ[l];
					for(j = 0; j< Degree; j++)
						if(j == i - 1) tmpMolt *=  inX[j, l];

					Mat[i, Degree + 1] += tmpMolt;
				};

			ComputationResult Chk = SolveLinearSystem(Mat, ref outA);
			if(Chk != ComputationResult.OK ) return ComputationResult.SingularityEncountered;

			double Delta, Sr=0, St=0;
			double dum=0;
			Chk = FindStatistics(inZ, ref dum, ref dum, ref dum, ref St);

			St = (St *St) * (N - 1); // N
			if(St == 0 || Chk != ComputationResult.OK ) return ComputationResult.SingularityEncountered;

			for(i = 0; i< N; i++)
			{
				Delta = inZ[i] - outA[0];
				for(j = 0;j< Degree;j++) Delta -= inX[j, i] * outA[j + 1];
				Sr += Delta *Delta;
			};

			CCorr = System.Math.Sqrt(1 - Sr / St);

			return ComputationResult.OK;

		}			


		/// <summary>
		/// Polynomial fit.
		/// </summary>
		/// <param name="inX">values of the independent variable</param>
		/// <param name="inY">values of the dependent variable</param>
		/// <param name="Degree">the degree of the polynomial</param>
		/// <param name="outA">the output coefficients, in order of ascending degree</param>
		/// <param name="CCorr">the correlation coefficient</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult PolynomialFit(double[] inX, double[] inY,
			short Degree, double[] outA, double CCorr)
			
		{
			int  i, k, j, N, l;
			double[,] Mat;

			//PolynomialFit = 0
			if (Degree < 2 || Degree > 15)
			{
				return ComputationResult.InvalidInput;
			};
			N = inX.GetLength(0);
			//(Degree + 1, Degree + 2)
			Mat = new double[Degree+1, Degree+2];
			for (i = 0; i< Degree + 1; i++)
			{
				for (j = 1 ; j< Degree + 1;j++)
				{
					k = i + j - 2;
					for (l = 0; l< N; l++) Mat[i, j] = Mat[i, j] + Math.Pow(inX[l], k);
				};
				for (l = 0; l< N; l++) Mat[i, Degree + 1] = Mat[i, Degree + 1] + inY[l] * Math.Pow(inX[l], (i - 1));
			};

			SolveLinearSystem(Mat, ref outA);
			double Delta, Sr=0, St=0 , dum=0; 
			FindStatistics(inY, ref dum, ref dum, ref dum, ref St);
			St = (St * St) * (N - 1);

			for (i = 0; i< N; i++)
			{
				Delta = inY[i];
				for(j = 1; j< Degree + 1; j++)
				{
					Delta = Delta - Math.Pow(inX[i] , (j - 1)) * outA[j];
				};
				Sr = Sr + Delta *Delta;
			};
			CCorr = Math.Sqrt(1 - Sr / St);
			return ComputationResult.OK;

		}
	
		/// <summary>
		/// Implements rejection by Chauvenet's criterion
		/// </summary>
		/// <param name="VecIn">input values</param>
		/// <param name="VecOut">output values (i.e., survivors)</param>
		/// <param name="MeanOut">mean value</param>
		/// <param name="RMSOut">standard deviation</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult RejectAllByChauvenet(double[] VecIn, out double[] VecOut, double MeanOut, double RMSOut)
		{
			int k, i,n;
			double Sup, Mean=0, dum=0, RMS=0, Alfa, Pout;
			int RejectedIndex;
			bool[] Indices;
			double[] tmpVecOut;
			ComputationResult CheckIn;
			double[] A = new double[8];
			A[0] = 0.09398; A[1] = 77.41144;
			A[2] = 10.27237; A[3] = -30.34994;
			A[4] = 13.26922; A[5] = -2.66993;
			A[6] = 0.26047; A[7] = -0.00985;
			VecOut = new double[0];
			MeanOut = 0; RMSOut = 0;
			n = VecIn.GetLength(0);
			//controlla la consistenza dell'input
			if (n < 3) return NumericalTools.ComputationResult.InvalidInput;
			Indices= new bool[n];
			int RejectedCounts =0;
			tmpVecOut= new double[n-RejectedCounts];

			CheckIn = FindStatistics(VecIn,  ref dum, ref dum, ref Mean, ref RMS);
			if (CheckIn != ComputationResult.OK) return NumericalTools.ComputationResult.SingularityEncountered;

			RejectedIndex = -1;

			//Trova Max e Candidato
			Sup = RMS;
			for (i = 0 ; i< n;i++)
				if (System.Math.Abs(VecIn[i] - Mean) > Sup && !Indices[i])
				{
					Sup = System.Math.Abs(VecIn[i] - Mean);
					RejectedIndex = i;
				};

			//effettua il rigetto se necessario
			if (RMS != 0)
			{
				Alfa = Sup / RMS;
				if (Alfa < 5)
				{
					Pout = 100;
					for (i = 0 ; i< 8; i++)	Pout = Pout - A[i] * Math.Pow(Alfa, i);
					if (n * Pout > 50) RejectedIndex = -1;
				};
			};

			while (RejectedIndex != -1)
			{
				Indices[RejectedIndex] = true;
				RejectedCounts++;
				tmpVecOut= new double[n-RejectedCounts];
				k = 0;
				for (i = 0; i< n; i++)
					if (!Indices[i]) 
					{
						tmpVecOut[k] = VecIn[i];
						k++;
					};

				CheckIn = FindStatistics(tmpVecOut, ref dum, ref dum, ref Mean, ref RMS);
				if(CheckIn != ComputationResult.OK) return NumericalTools.ComputationResult.SingularityEncountered;

				//Trova Max e Candidato
				k = 0;
				Sup = RMS;
				for (i =0 ; i< n;i++)
				{
					if (!Indices[i]) k++; 
					if (Math.Abs(VecIn[i] - Mean) > Sup && !Indices[i])
					{ 
						Sup = Math.Abs(VecIn[i] - Mean);
						RejectedIndex = i;
					};
				};
				//effettua il rigetto se necessario
				if (RMS != 0) 
				{
					Alfa = Sup / RMS;
					if (Alfa < 5) 
					{
						Pout = 100;
						for (i = 0 ; i< 8; i++)	Pout = Pout - A[i] * Math.Pow(Alfa, i);
						if (k * Pout > 50) RejectedIndex = -1;
					};
				}
				else
				{
					break;
				};
			};

			//lo rifa nel caso non ha tolto gnente
			k = 0;
			VecOut= new double[n-RejectedCounts];
			for (i = 0; i<n; i++)
				if (!Indices[i])
				{
					VecOut[k] = VecIn[i];
					k++;
				};

			//tmpVecOut= new double[k];
			//for (i=0; i<k ; i++) VecOut[i]=tmpVecOut[i];

			CheckIn = FindStatistics(VecOut,  ref dum, ref dum, ref MeanOut, ref RMSOut);
			if(CheckIn != ComputationResult.OK) return NumericalTools.ComputationResult.SingularityEncountered;
			return NumericalTools.ComputationResult.OK;
		}
	}

	/// <summary>
	/// Implements several procedures useful in random distribution generation
	/// </summary>
	public class MonteCarlo
	{
		/// <summary>
		/// Prevents instantiation of this class. Only static methods are supported.
		/// </summary>
		protected MonteCarlo() {}

		/// <summary>
		/// Generates a Gaussian random number.
		/// </summary>
		/// <param name="M">the value of the Mu parameter for the Gaussian distribution.</param>
		/// <param name="S">the value of the Sigma parameter for the Gaussian distribution.</param>
		/// <param name="Number">the generated number.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Gaussian_Rnd_Number(double M, double S, 
			ref double Number)
		{
			double r1, r2, g, hwb, Loga2;
			if(S == 0) return ComputationResult.InvalidInput;
			Loga2 = 0.693147180559945;
			r1 = RandomGenerator.RND.NextDouble();
			r2 = RandomGenerator.RND.NextDouble();
			hwb = 2 * S * System.Math.Sqrt(2 * Loga2);
			while (r2 == 0)
			{
				r2 = RandomGenerator.RND.NextDouble();
			};
			g = Math.Sqrt(-2 * Math.Log(r2)) * Math.Cos(2 * Math.PI * r1);
			g = g * hwb / (2 * Math.Sqrt(2 * Loga2));
			Number = g + M;
			return ComputationResult.OK;
		}

		/// <summary>
		/// Generates a Gaussian random number.
		/// </summary>
		/// <param name="M">the value of the Mu parameter for the Gaussian distribution.</param>
		/// <param name="S">the value of the Sigma parameter for the Gaussian distribution.</param>
		/// <returns>the generated number.</returns>
		public static double Gaussian_Rnd_Number(double M, double S)
		{
			double r1, r2, g, hwb, Loga2 = 0.693147180559945;

			if(S == 0) return 0;
			r1 = RandomGenerator.RND.NextDouble();
			r2 = RandomGenerator.RND.NextDouble();
			hwb = 2 * S * System.Math.Sqrt(2 * Loga2);
			while (r2 == 0) r2 = RandomGenerator.RND.NextDouble();
			g = Math.Sqrt(-2 * Math.Log(r2)) * Math.Cos(2 * Math.PI * r1);
			g = g * hwb / (2 * Math.Sqrt(2 * Loga2));
			return g + M;
		}

		/// <summary>
		/// Generates a uniformly distributed random number between the specified extents.
		/// </summary>
		/// <param name="Inf">the lower bound of the distribution.</param>
		/// <param name="Sup">the upper bound of the distribution.</param>
		/// <param name="Value">the generated value.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult Flat_Rnd_Number(double Inf, double Sup, 
			ref double Value) 
		{

			if (Inf > Sup) return ComputationResult.InvalidInput;

			Value = (Sup - Inf) * RandomGenerator.RND.NextDouble() + Inf;
			return ComputationResult.OK;
		
		}

		/// <summary>
		/// Generates a uniformly distributed random number between the specified extents.
		/// </summary>
		/// <param name="Inf">the lower bound of the distribution.</param>
		/// <param name="Sup">the upper bound of the distribution.</param>		
		/// <param name="Value">the generated value.</param>
		/// <returns>the generated value.</returns>
		public static double Flat_Rnd_Number(double Inf, double Sup) 
		{
			if (Inf > Sup) return 0;
			return  (Sup - Inf) * RandomGenerator.RND.NextDouble() + Inf;
		}

		/// <summary>
		/// Generates a Poissonian random number
		/// </summary>
		/// <param name="Mu">the average of the Poisson distribution.</param>
		/// <param name="Number">the generated number.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult Poisson_Rnd_Number(double Mu, int Number)
		{
			double Normal, NormalU, NormalD;
			double NumberTry, Poisson;
			int i;
			ulong FactNum=0;
			bool Check;

			Factorial((int)Mu, ref FactNum);
			Normal = Math.Exp(-(int)Mu) * (Math.Pow((int)Mu, (int)Mu) / FactNum);
			Factorial((int)Mu + 1, ref FactNum);
			NormalU = Math.Exp(-((int)Mu + 1)) * (Math.Pow((int)Mu + 1, (int)Mu + 1) / FactNum);
			Factorial((int)Mu - 1, ref FactNum);
			NormalD = Math.Exp(-((int)Mu - 1)) * (Math.Pow((int)Mu - 1, (int)Mu - 1) / FactNum);

			if (NormalU > Normal) Normal = NormalU;
			if (NormalD > Normal) Normal = NormalD;
			Check = false;
			while (!Check)
			{
				Number = (int)(5 * Mu * RandomGenerator.RND.NextDouble());
				NumberTry = RandomGenerator.RND.NextDouble()  * Normal;

				Poisson = 1;
				for(i = 0; i<  Number;i++) Poisson = Poisson * (Mu / i);
			
				Poisson = Math.Exp(-Mu) * Poisson;
				//        Poisson = Exp(-Mu) * ((Mu ^ Nu) / Factorial(Nu))
				if ((Poisson - NumberTry) >= 0) Check = true;
			};
			return ComputationResult.OK;
		}

		/// <summary>
		/// Generates a Binomial random number
		/// </summary>
		/// <param name="P">the average of the Binomial distribution.</param>
		/// <param name="N">the number of trials.</param>
		/// <param name="Number">the generated number.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult Binomial_Rnd_Number(double P, int N, ref int Number)
		{
			double[] Normal;
			double tmpNumber, BinNum=0;
			int i;

			if(P > 1 && N < 1) return ComputationResult.InvalidInput;
			Normal= new double[N+1];
			tmpNumber = RandomGenerator.RND.NextDouble();
			Normal[0] = Math.Pow((1 - P), N);
		
			if(tmpNumber < Normal[0])
				Number = 0;
			else
			{
				for(i = 1; i<=  N;i++)
				{
					Binomial_Coeff(N, i, ref BinNum);
					Normal[i] = (BinNum * Math.Pow(P,i) * (Math.Pow((1 - P), (N - i)))) + Normal[i - 1];
					if (tmpNumber < Normal[i])
					{
						Number = i;
						break;
					};
				};	
			}
			return ComputationResult.OK;
		}

		/// <summary>
		/// Computes the factorial of a number.
		/// </summary>
		/// <param name="N">the number whose factorial is sought.</param>
		/// <param name="Number">the value of the factorial.</param>
		/// <returns>the outcome of the computation.</returns>
		public static ComputationResult Factorial(int N, ref ulong Number)
		{
			if(N< 0) return ComputationResult.InvalidInput;
			Number=1;
			for(int i=1; i<=N; i++) Number=Number*(ulong)i;
			return ComputationResult.OK ;
		}

		/// <summary>
		/// Computes the binomial coefficient.
		/// </summary>
		/// <param name="N">the N (upper) parameter.</param>
		/// <param name="V">the V (lower) parameter.</param>
		/// <param name="Number">the value of the binomial coefficient.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult Binomial_Coeff(int N, int V, ref double Number)
		{

			if (N < 0 || V < 0 || N < V) return ComputationResult.InvalidInput;
			Number=1;
			for (int i = 1 ; i<= V; i++) Number=Number* ((N - V + i) / i);
			return ComputationResult.OK ;
		}
		
		/// <summary>
		/// The behaviour of this method is unknown.
		/// </summary>
		internal static ComputationResult Corr_Gaussian_Rnd_Number(double M_Syst, double N_Syst,
			double Corr_M, double Corr_S,
			double X, ref double Number)
		{
			double r1, r2;
			double g, hwb;
			double Loga2= 0.693147180559945;

			if (Corr_S == 0) return ComputationResult.InvalidInput;

			r1 = RandomGenerator.RND.NextDouble();
			r2 = RandomGenerator.RND.NextDouble();
			hwb = 2 * Corr_S * System.Math.Sqrt(2 * Loga2);

			while(r2 == 0) r2 = RandomGenerator.RND.NextDouble();

			g = Math.Sqrt(-2 * Math.Log(r2)) * Math.Cos(2 * Math.PI * r1);
			g = g * hwb / (2 * Math.Sqrt(2 * Loga2));

			Number = g + Corr_M + M_Syst * X + N_Syst;
			return ComputationResult.OK ;
		}

		/// <summary>
		/// The behaviour of this method is unknown.
		/// </summary>
		internal static ComputationResult Var_Gaussian_Rnd_Number(double M_Syst, double N_Syst,
			double Var_M, double Var_S,
			double X, ref double Number)
		{
			double r1, r2;
			double g, hwb;
			double Loga2 = 0.693147180559945;

			if(Var_S == 0) return ComputationResult.InvalidInput;

			Var_S = M_Syst * X + N_Syst;

			r1 = RandomGenerator.RND.NextDouble();
			r2 = RandomGenerator.RND.NextDouble();
			hwb = 2 * Var_S * Math.Sqrt(2 * Loga2);

			while (r2 == 0) r2 = RandomGenerator.RND.NextDouble();

			g = Math.Sqrt(-2 * Math.Log(r2)) * Math.Cos(2 * Math.PI * r1);
			g = g * hwb / (2 * System.Math.Sqrt(2 * Loga2));
			Number = g + Var_M;
			return ComputationResult.OK ;
		}

		/// <summary>
		/// The behaviour of this method is unknown.
		/// </summary>
		internal static ComputationResult Linear_Rnd_Number(double M, double N,
			double Sup, double Inf, 
			ref double Number)
		{
			double LSup, LInf;
			double CTerm, ATerm, BTerm;

			if(Sup < Inf) return ComputationResult.InvalidInput;

			if(M * Sup + N < 0 || M * Inf + N < 0)
			{
				if(M < 0)
				{
					LSup = -N / M;
					LInf = LSup - (Sup - Inf);
				}
				else
				{
					LInf = -N / M;
					LSup = LInf + (Sup - Inf);
				};
			} 
			else 
			{
				LSup = Sup;
				LInf = Inf;
			};

			CTerm = -(M * LInf * LInf / 2) - (N * LInf) - RandomGenerator.RND.NextDouble() * (M * (LSup * LSup - LInf * LInf) / 2 + N * (LSup - LInf));
			ATerm = M / 2;
			BTerm = N;
			Number = (-BTerm + System.Math.Sqrt(BTerm * BTerm - 4 * ATerm * CTerm)) / (2 * ATerm);

			Number = Number + (Sup - LSup);
			return ComputationResult.OK;
		}

		/// <summary>
		/// The behaviour of this method is unknown.
		/// </summary>
		public static ComputationResult Custom_Rnd_Number(double[] X_Mean, double[] Y_Vec,
			int GenAlgor, ref double Number)
		{
			// Dichiarazioni
			int i;
			double Inf=0, Delta, Sup=0;
			double Norm, Sum=0;
			int SelI=0;
			double N, M, Maximum, Xp;
			int N_Cat ;
			double LSup=0, LInf=0, tmp;
			ComputationResult Chk= ComputationResult.OK;
			// Ricava gli estremi

			N_Cat = X_Mean.GetLength(0);

			if (Y_Vec.GetLength(0) != N_Cat || GenAlgor < 1 || GenAlgor > 3) return ComputationResult.InvalidInput;

			Delta = X_Mean[1] - X_Mean[0];
			Inf = X_Mean[0] - Delta / 2;
			Delta = X_Mean[N_Cat-1] - X_Mean[N_Cat - 2];
			Sup = X_Mean[N_Cat-1] + Delta / 2;

			// Trova il massimo dei y_vec
			Maximum = Y_Vec[0];
			Norm = Maximum;
			for(i = 1; i< N_Cat;i++)
			{
				Norm = Norm + Y_Vec[i];
				if (Y_Vec[i] >= Maximum) Maximum = Y_Vec[i];
			};
			if(Norm == 0) return ComputationResult.SingularityEncountered;
			// Generates the number
			Xp = RandomGenerator.RND.NextDouble();
		
			//Select the Interval Containing the Rnd x
			for(i = 0; i< N_Cat;i++)
			{
				Sum = Sum + (Y_Vec[i] / Norm);
				if(Xp < Sum)
				{
					SelI = i;
					Number = X_Mean[i];
					break;
				};
			};
		
			//On Error GoTo CptErr

			if(GenAlgor == 1)
			{
				tmp = RandomGenerator.RND.NextDouble();
				if(tmp < 0.5 && SelI > 0 && SelI < N_Cat-1)
				{
					M = (Y_Vec[SelI] - Y_Vec[SelI - 1]) / (X_Mean[SelI] - X_Mean[SelI - 1]);
					N = -M * X_Mean[SelI] + Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI], X_Mean[SelI - 1], ref Number);
				} 
				else if( tmp < 0.5 && SelI == 0)
				{
					M = 0;
					N = Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI + 1], X_Mean[SelI], ref Number);
				}
				else if(tmp < 0.5 && SelI == N_Cat-1)
				{
					M = 0;
					N = Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI], X_Mean[SelI-1],ref Number);
				}
				else if(tmp >= 0.5 && SelI < N_Cat -1 && SelI > 0)
				{
					M = (Y_Vec[SelI] - Y_Vec[SelI-1]) / (X_Mean[SelI] - X_Mean[SelI-1]);
					N = -M * X_Mean[SelI] + Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI], X_Mean[SelI-1],ref Number);
				}
				else if(tmp >= 0.5 && SelI == 0)
				{
					M = 0;
					N = Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI+1], X_Mean[SelI],ref Number);
				}
				else if(tmp >= 0.5 && SelI == N_Cat)
				{
					M = 0;
					N = Y_Vec[SelI];
					Chk = Linear_Rnd_Number(M, N, X_Mean[SelI], X_Mean[SelI-1],ref Number);
				};
			}
			else if(GenAlgor == 2)
			{
				if(SelI > 0 && SelI < N_Cat-1)
				{
					LSup = X_Mean[SelI] + (X_Mean[SelI + 1] - X_Mean[SelI]) / 2;
					LInf = X_Mean[SelI] - (X_Mean[SelI] - X_Mean[SelI - 1]) / 2;
				}
				else if(SelI == 0)
				{
					LSup = X_Mean[SelI] + (X_Mean[SelI + 1] - X_Mean[SelI]) / 2;
					LInf = X_Mean[SelI];
				}
				else if(SelI == N_Cat-1)
				{
					LSup = X_Mean[SelI];
					LInf = X_Mean[SelI] - (X_Mean[SelI] - X_Mean[SelI - 1]) / 2;
				};
				Number = (LSup - LInf) * RandomGenerator.RND.NextDouble() + LInf;
			
			};

			if(Chk != ComputationResult.OK) return ComputationResult.SingularityEncountered;
			return ComputationResult.OK;
		}
	}

	/// <summary>
	/// Coordinate transformations.
	/// </summary>
	public class Transformation
	{
		/// <summary>
		/// Prevents instantiation of this class. Only static methods are supported.
		/// </summary>
		protected Transformation() {}

		/// <summary>
		/// Generates discrete 2D shells centered on (0,0).
		/// </summary>
		/// <param name="Radius">the maximum radius.</param>
		/// <returns>the shells in order of distance from (0,0)</returns>
		public static double[,] ConcentricRelativeCoordinates(double Radius) 
		{
			if(Radius<0) throw new Exception("Radius must be >=0");
			int n=0,i,j;
			double ij2;
			double Radius2 = Radius*Radius;
			System.Collections.ArrayList arrx = new System.Collections.ArrayList();
			System.Collections.ArrayList arry = new System.Collections.ArrayList();
			System.Collections.ArrayList arri = new System.Collections.ArrayList();
			for(i=0; i<Radius; i++)
			{
				for(j=0; j<Radius; j++)
				{
					ij2=i*i+j*j;
					if(ij2<= Radius2)
					{
						
						arrx.Add(i); arry.Add(j); arri.Add(ij2);
						n++;
						if(i!=0)
						{
							arrx.Add(-i); arry.Add(j); arri.Add(ij2);
							n++;
							if(j!=0)
							{
								arrx.Add(-i); arry.Add(-j); arri.Add(ij2);
								n++;
							};
						};
						if(j!=0)
						{
							arrx.Add(i); arry.Add(-j); arri.Add(ij2);
							n++;
						};
						
					}
				}
			}
			int[] x = (int[])arrx.ToArray(typeof(int));
			int[] y = (int[])arry.ToArray(typeof(int));
			double[] ind = (double[])arri.ToArray(typeof(double));
			int[] sind;
			AscSortVector(ind, out sind);
			double[,] arr = new double[n,3];
			for(i=0; i<n; i++)
			{
				arr[i,0] = x[sind[i]];
				arr[i,1] = y[sind[i]];
				arr[i,2] = ind[sind[i]];
			}
			return arr;
		}

		/// <summary>
		/// Rotates a 2D vector in the plane.
		/// </summary>
		/// <param name="Xi">input vector (must be 2-component vector).</param>
		/// <param name="Phi">the rotation angle.</param>
		/// <param name="Xf">output 2-component vector.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult RotationInPlane(double[] Xi, double Phi, out double[] Xf) 
		{
			int n;

			n = Xi.GetLength(0);
			if (n != 2)
			{
				Xf = null;
				return ComputationResult.InvalidInput;
			};
			Xf = new double[n];

			Xf[1] = Math.Cos(Phi) * Xi[1] + Math.Sin(Phi) * Xi[2];
			Xf[2] = -Math.Sin(Phi) * Xi[1] + Math.Cos(Phi) * Xi[2];
			return ComputationResult.OK;
		}
		
		/// <summary>
		/// Computes the phase of an (X,Y) vector.
		/// </summary>
		/// <param name="X">X component.</param>
		/// <param name="Y">Y component.</param>
		/// <returns>the phase.</returns>
		public static double FindPhi(double X, double Y)
		{
			return Math.Atan2(Y, X);
		}

		/// <summary>
		/// Performs a rotation in space.
		/// </summary>
		/// <param name="Xi">X component of the vector to be rotated.</param>
		/// <param name="Yi">Y component of the vector to be rotated.</param>
		/// <param name="Zi">Z component of the vector to be rotated.</param>
		/// <param name="OrX">X component of rotation center.</param>
		/// <param name="OrY">Y component of rotation center.</param>
		/// <param name="OrZ">Z component of rotation center.</param>
		/// <param name="AX">X component of rotation axis.</param>
		/// <param name="AY">Y component of rotation axis.</param>
		/// <param name="AZ">Z component of rotation axis.</param>
		/// <param name="Xo">output X component of the rotated vector.</param>
		/// <param name="Yo">output Y component of the rotated vector.</param>
		/// <param name="Zo">output Z component of the rotated vector.</param>
		/// <returns>the outcome of the computation</returns>
		public static ComputationResult RotationInSpace(double Xi, double Yi , double Zi, 
			double OrX, double OrY, double OrZ, 
			double AX, double AY, double AZ,
			ref double Xo, ref double Yo, ref double Zo)
		{
																																																								   
			//x e y coord trasverse z coord long
			//Rotazione totale
			double Ty, Tz, Tx;
			double DX, DY, DZ;

			Tx = Xi - OrX;
			Ty = Yi - OrY;
			Tz = Zi - OrZ;
			DX = (Math.Cos(AY) * Math.Cos(AZ)) * Tx + (Math.Cos(AY) * Math.Sin(AZ)) * Ty + Math.Sin(AY) * Tz;
			DY = -((Math.Sin(AX) * Math.Sin(AY) * Math.Cos(AZ)) + (Math.Cos(AX) * Math.Sin(AZ))) * Tx - ((Math.Sin(AX) * Math.Sin(AY) * Math.Sin(AZ)) - (Math.Cos(AX) * Math.Cos(AZ))) * Ty + (Math.Sin(AX) * Math.Cos(AY)) * Tz;
			DZ = -((Math.Cos(AX) * Math.Sin(AY) * Math.Cos(AZ)) + (Math.Sin(AX) * Math.Sin(AZ))) * Tx -  ((Math.Cos(AX) * Math.Sin(AY) * Math.Sin(AZ)) + (Math.Sin(AX) * Math.Cos(AZ))) * Ty + (Math.Cos(AX) * Math.Cos(AY)) * Tz;

			Xo = DX + OrX;
			Yo = DY + OrY;
			Zo = DZ + OrZ;
			return ComputationResult.OK;
		}

		//CRISTIANO: SONO ARRIVATO FINO A QUI.

		public static double[] AscSortVector(double[] V, out int[] sInd) 
		{
			int i , n , l , k, j;
			double[] sV;
			
			n = V.GetLength(0);
			if (n == 1) 
			{
				sInd = new int[1];
				sInd[0]=0;
				return V;
			};
			if (n == 0) 
			{	
				sInd=null;
				return null;
			};
    
			sV = new double[n];	
			sInd = new int[n];
    
			sV[0] = V[0];
			sInd[0] = 0;
			l = 0;
			for(i = 1; i<n; i++)
			{
				++l;
				for(k = 0; k< l; k++)
				{
					if(V[i] < sV[k] && k < l-1)
					{
						for(j = l; j > k; j--)
						{
							sV[j] = sV[j - 1];
							sInd[j] = sInd[j - 1];
						};
						sV[k] = V[i];
						sInd[k] = i;
						break;
					}
					else if( k == l-1)
					{
						sV[k+1] = V[i];
						sInd[k+1] = i;
						break;
					};
				};
			};
			return sV;
		}

		public static double[] AscSortVector(double[] V) 
		{
			int[] t;
			return AscSortVector(V,out t);
		}
		
		public static int[] AscSortVector(int[] V, out int[] sInd) 
		{
			double[] vi;
			int i, n;
			
			n = V.GetLength(0);
			vi = new double[n];
			for(i=0; i<n;i++) vi[i] = (double)V[i];

			double[] vout = AscSortVector(vi,out sInd);

			int[] viout = new int[n];
			for(i=0; i<n;i++) viout[i] = Convert.ToInt32(vout[i]);

			return viout;
		}

		public static int[] AscSortVector(int[] V) 
		{
			int[] t;
			return AscSortVector(V,out t);
		}


		public static ComputationResult FindAffine(double[] YFix, double[] ZFix, double[] YMov, double[] ZMov, 
			ref double Ayy, ref double Ayz, ref double Azy, ref double Azz, ref double Y0, ref double Z0)
		{
			int i, n;
			double Sz=0, SY=0, Slz=0, Sly=0, Szz=0, Syy=0, Syz=0, Syly=0;
			double Sylz=0, Slyz=0, Szlz=0, Sp0=0, Sp1=0, Sq0=0, Sq1=0;
			double Sar=0, Sbr=0, Scr=0, Sdr=0, Det=0, Sa=0, Sb=0, Sc=0, Sd=0;
			n = YFix.GetLength(0);
    
			if (YMov.GetLength(0) != n || ZFix.GetLength(0) !=n || ZMov.GetLength(0) != n || n<3)
			{
				Ayy = 1;
				Ayz = 0;
				Azy = 0;
				Azz = 1;
				Y0 = 0;
				Z0 = 0;
				return ComputationResult.InvalidInput;
			};

			for (i=0; i<n; i++)
			{
				Sz += ZMov[i];
				SY += YMov[i];
				Slz += ZFix[i];
				Sly += YFix[i];
				Szz += ZMov[i] * ZMov[i];
				Syy += YMov[i] * YMov[i];
				Syz += ZMov[i] * YMov[i];
				Syly += YFix[i] * YFix[i];
				Sylz += YFix[i] * ZFix[i];
				Slyz += YFix[i] * ZFix[i];
				Szlz += ZFix[i] * ZFix[i];
			};

			Sz /=  n;
			SY /= n;
			Slz /=  n;
			Sly /=  n;

			Szz /=  n;
			Syy /=  n; 
			Syz /= n;
			Szlz /= n;
			Sylz /= n;
			Slyz /= n;
			Syly /= n;
			Sp0 = Syly - SY * Sly;
			Sp1 = Slyz - Sly * Sz;
			Sq0 = Sylz - SY * Slz;
			Sq1 = Szlz - Sz * Slz;
			Sar = Syy - SY * SY;
			Sbr = Syz - SY * Sz;
			Scr = Sbr;
			Sdr = Szz - Sz * Sz;
			Det = Sar * Sdr - Sbr * Scr;
			
			if(Det == 0)
			{
				Ayy = 1;
				Ayz = 0;
				Azy = 0;
				Azz = 1;
				Y0 = 0;
				Z0 = 0;
				return ComputationResult.SingularityEncountered;
			};    
			Sa = Sdr / Det;
			Sb = -Sbr / Det;
			Sc = -Scr / Det;
			Sd = Sar / Det;

			Ayy = Sa * Sp0 + Sb * Sp1;
			Ayz = Sc * Sp0 + Sd * Sp1;
			Azy = Sa * Sq0 + Sb * Sq1;
			Azz = Sc * Sq0 + Sd * Sq1;
			Y0 = Sly - Ayy * SY - Ayz * Sz;
			Z0 = Slz - Azy * SY - Azz * Sz;
			return ComputationResult.OK;
		}

		public static ComputationResult FindHomAffine(double[] X, double[] Y, double[] Xc, double[] Yc, 
			out double[,] Aff)
		{
			int i , n;
			double Sx=0, SY=0, Scx=0, Scy=0;
			double Sxy=0, Sx2=0, Sy2=0, Scxx=0;
			double Scyx=0, Scxy=0, Scyy=0, Delta;
			
			Aff = new double[2,2];
			n = X.GetLength(0);

			if (Y.GetLength(0) != n || Xc.GetLength(0) != n || Yc.GetLength(0) != n || n == 1) return ComputationResult.InvalidInput;
    
			//Calcola una ammuina di somme
			for(i = 0; i<n; i++)
			{
				Sx += X[i];
				SY += Y[i];
				Scx += Xc[i];
				Scy += Yc[i];
				Sxy += X[i] * Y[i];
				Sx2 += X[i] * X[i];
				Sy2 += Y[i] * Y[i];
				Scxx += Xc[i] * X[i];
				Scyx += Yc[i] * X[i];
				Scxy += Xc[i] * Y[i];
				Scyy += Yc[i] * Y[i];
			};
    
			//e vaaai coi calcoli finali
			Delta = n * (Sx2 + Sy2) - (Sx *Sx) - (SY *SY);
			if (Delta == 0) return ComputationResult.SingularityEncountered;
	    
			//Vanno verificate le posizioni nelle matrici
			Aff[0,0] = (n * (Scxx + Scyy) - (Sx * Scx) - (SY * Scy)) / Delta;
			Aff[0,1] = (n * (Scyx - Scxy) + (SY * Scx) - (Sx * Scy)) / Delta;
			Aff[1,0] = (Scx - (Aff[0,0] * Sx) + (Aff[0,1] * SY)) / n;
			Aff[1,1] = (Scy - (Aff[0,1] * Sx) - (Aff[0,0] * SY)) / n;
			return ComputationResult.OK;
		}

		public static ComputationResult Affine_Focusing(double[] X, double[] Y, double[] SX, double[] SY, 
			double[] Aff, ref double[] Xc, ref double[] Yc)
		{
			int i , n;
			
			n = Aff.GetLength(0);
			if (n != 7) return ComputationResult.InvalidInput;

			n = X.GetLength(0);
			if (Y.GetLength(0) != n || Xc.GetLength(0) != n 
				|| Yc.GetLength(0) != n || SX.GetLength(0) != n || SY.GetLength(0) != n ) return ComputationResult.InvalidInput;
    
			//Calcola una ammuina di somme
			for(i = 0; i<n; i++)
			{
				Xc[i]= Aff[0]*X[i] + Aff[1]*Y[i] + Aff[4] + Aff[6]*SX[i];
				Yc[i]= Aff[2]*X[i] + Aff[3]*Y[i] + Aff[5] + Aff[6]*SY[i];
			};

			return ComputationResult.OK;
		}

		public static ComputationResult Affine_Focusing(double[] SX, double[] SY, 
			double[] Aff, ref double[] Xc, ref double[] Yc)
		{
			int i , n;
			double tx, ty;
			n = Aff.GetLength(0);
			if (n != 7) return ComputationResult.InvalidInput;

			n = Xc.GetLength(0);
			if (Yc.GetLength(0) != n || SX.GetLength(0) != n || SY.GetLength(0) != n ) return ComputationResult.InvalidInput;
    
			//Calcola una ammuina di somme
			for(i = 0; i<n; i++)
			{
				tx=Xc[i];
				ty=Yc[i];
				Xc[i]= Aff[0]*tx + Aff[1]*ty + Aff[4] + Aff[6]*SX[i];
				Yc[i]= Aff[2]*tx + Aff[3]*ty + Aff[5] + Aff[6]*SY[i];
			};

			return ComputationResult.OK;
		}

		public static bool CatchDifferentIndices(int NewIndex, ref int[] Index)
		{
			int i ,n;
			int[] tmpind;

			bool IndexAdded = true;

			if(Index==null)
			{
				Index = new int[1];
				Index[0]=NewIndex;
			}
			else
			{
				n=Index.GetLength(0);
				for (i=0; i<n;i++)
					if(Index[i]== NewIndex)
					{
						IndexAdded = false;
						break;
					};

				if(IndexAdded)
				{
					tmpind=new int[n];
					tmpind = (int[])Index.Clone();
					Index = new int[n+1];
					for (i=0; i<n;i++) Index[i]=tmpind[i];
					Index[n]=NewIndex;
				};

			};
			return IndexAdded;

		}


	}
	public class Matrices
	{
		static public double[,] Product(double[,] A, double[,] B)
		{
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			int rb = B.GetLength(0);
			int cb = B.GetLength(1);
			if (ca!=rb) return null;
			double[,] C = new double[ra, cb];
			int ia, ja, jb;
			for (ia=0; ia<ra; ia++)
				for(jb=0; jb<cb; jb++)
					for (ja=0; ja<ca; ja++)
					{
						C[ia,jb] += A[ia,ja]*B[ja,jb];
					}
			return C;
		}

		static public double[] Product(double[] A, double[,] B)
		{
			int ca = A.Length;
			int rb = B.GetLength(0);
			int cb = B.GetLength(1);
			if (ca!=rb) return null;
			double[] C = new double[cb];
			int ja, jb;
			for(jb=0; jb<cb; jb++)
				for (ja=0; ja<ca; ja++)
				{
					C[jb] += A[ja]*B[ja,jb];
				}
			return C;
		}

		static public double[] Product(double[,] B, double[] A)
		{
			int ra = A.Length;
			int rb = B.GetLength(0);
			int cb = B.GetLength(1);
			if (ra!=cb) return null;
			double[] C = new double[rb];
			int ib, ia;
			for(ib=0; ib<rb; ib++)
				for (ia=0; ia<ra; ia++)
				{
					C[ib] += B[ib,ia]*A[ia];
				}
			return C;
		}

		static public double ScalarProduct(double[] A, double[] B)
		{
			int ca = A.Length;
			int rb = B.Length;

			if (ca!=rb) return 0;
			double C=0;
			int ib;
			for(ib=0; ib<rb; ib++)
				C += A[ib]*B[ib];
			return C;
		}

		static public double[,] Product(double[] A, double[] B)
		{
			int la = A.Length;
			int lb = B.Length;

			double[,] C = new double[la,lb];
			int ib,ia;
			for(ia=0; ia<la; ia++)
				for(ib=0; ib<lb; ib++)
					C[ia,ib] = A[ia]*B[ib];
			return C;
		}


		static public double[,] Sum(double[,] A, double[,] B)
		{
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			if (ca!=B.GetLength(1) || B.GetLength(0)!=ra) return null;
			double[,] C = new double[ra, ca];
			int ia, ja;
			for (ia=0; ia<ra; ia++)
				for(ja=0; ja<ca; ja++)
				{
					C[ia,ja] = A[ia,ja] + B[ia,ja];
				}
			return C;
		}

		static public double[] Sum(double[] A, double[] B)
		{
			int la = A.Length;
			if (la!=B.Length) return null;
			double[] C = new double[la];
			int ia;
			for (ia=0; ia<la; ia++)
			{
				C[ia] = A[ia] + B[ia];
			}
			return C;
		}

		static public double[,] Subtract(double[,] A, double[,] B)
		{
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			if (ca!=B.GetLength(1) || B.GetLength(0)!=ra) return null;
			double[,] C = new double[ra, ca];
			int ia, ja;
			for (ia=0; ia<ra; ia++)
				for(ja=0; ja<ca; ja++)
				{
					C[ia,ja] = A[ia,ja] - B[ia,ja];
				}
			return C;
		}

		static public double[] Subtract(double[] A, double[] B)
		{
			int la = A.Length;
			if (la!=B.Length) return null;
			double[] C = new double[la];
			int ia;
			for (ia=0; ia<la; ia++)
			{
				C[ia] = A[ia] - B[ia];
			}
			return C;
		}

		static public double[,] ExternalProduct(double a, double[,] A)
		{
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			if (A==null) return null;
			double[,] C = new double[ra, ca];
			int ia, ja;
			for (ia=0; ia<ra; ia++)
				for(ja=0; ja<ca; ja++)
				{
					C[ia,ja] = a*A[ia,ja];
				}
			return C;
		}

		static public double[] ExternalProduct(double a, double[] A)
		{
			if (A==null) return null;
			int la = A.Length;
			double[] C = new double[la];
			int ia;
			for (ia=0; ia<la; ia++)
			{
				C[ia] = a*A[ia];
			}
			return C;
		}

		static public double[,] Identity(int n)
		{

			double[,] C = new double[n, n];
			int i;
			for (i=0; i<n; i++) C[i,i]=1;
			return C;
		}

		static public ComputationResult Traspose(double[,] A, out double[,] C)
		{
			if (A==null)
			{
				C = null;
				return ComputationResult.InvalidInput;
			}
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			C = new double[ca, ra];
			int i, j;
			for (i=0; i<ca; i++)
				for (j=0; j<ra; j++)
					C[i,j]= A[j,i];
			return ComputationResult.OK;
		}

		static public double[,] Transpose(double[,] A)
		{
			if (A==null)return null;

			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			double[,] C = new double[ca, ra];
			int i, j;
			for (i=0; i<ca; i++)
				for (j=0; j<ra; j++)
					C[i,j]= A[j,i];
			return C;
		}

		static public double[,] TransposeSelf(double[,] A)
		{
			int ca = A.GetLength(1);
			int ra = A.GetLength(0);
			int i, j;
			for (i=0; i<ca; i++)
				for (j=i+1; j<ra; j++)
				{
					double tt = A[i,j];
					A[i,j]= A[j,i];
					A[j,i]=tt;
				}
			return A;
		}

		public static ComputationResult InvertMatrix(double[,] inA, ref double[,] outA)
		{
			
			ComputationResult Chk1;
			int n = inA.GetLength(0);
			if(n < 2 || n > 15 || n!=inA.GetLength(1)) return NumericalTools.ComputationResult.InvalidInput;
			double[,] GJA = new double[n,2*n];
			double[,] oGJA = new double[n,2*n];
			int i,j;
			for(i=0; i<n; i++)
				for(j=0; j<n; j++)
					GJA[i,j]=inA[i,j];
			for(i=0; i<n; i++)
				GJA[i,i+n]=1;
			
			Chk1=Fitting.GaussJordanAlgorithm(GJA, out oGJA);
			if ( Chk1 == ComputationResult.OK)
			{
				for(i=0; i<n; i++)
					for(j=n; j<2*n; j++)
						outA[i,j-n]=oGJA[i,j];
				return  Chk1;
			}
			else
				return  Chk1;

		}

		public static double[,] InvertMatrix(double[,] inA)
		{
			
			ComputationResult Chk1;
			int n = inA.GetLength(0);
			if(n < 2 || n > 15 || n!=inA.GetLength(1)) return null;
			double[,] GJA = new double[n,2*n];
			double[,] oGJA = new double[n,2*n];
			int i,j;
			for(i=0; i<n; i++)
				for(j=0; j<n; j++)
					GJA[i,j]=inA[i,j];
			for(i=0; i<n; i++)
				GJA[i,i+n]=1;
			
			Chk1=Fitting.GaussJordanAlgorithm(GJA, out oGJA);
			double[,] outA = new double[n,n];
			if ( Chk1 == ComputationResult.OK)
			{
				for(i=0; i<n; i++)
					for(j=n; j<2*n; j++)
						outA[i,j-n]=oGJA[i,j];
				return  outA;
			}
			else
				return  null;

		}

	
	}
}
	