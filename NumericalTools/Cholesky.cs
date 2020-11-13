using System;
using System.Collections.Generic;
using System.Text;

namespace NumericalTools
{
    /// <summary>
    /// Gets a symmetric matrix and decomposes it using Cholesky algorithm, providing also methods for matrix inversion and linear system solving.
    /// <remarks>The Cholesky algorithm requires the symmetric matrix to be positive-definite. An exception will be thrown if the matrix is found to be semi-definite.</remarks>
    /// </summary>
    public class Cholesky
    {
        /// <summary>
        /// Exception found during execution of Cholesky decomposition.
        /// </summary>
        public class Exception : System.Exception
        {
            /// <summary>
            /// Builds a new exception using the specified error message.
            /// </summary>
            /// <param name="text">the message that explains the exception.</param>
            public Exception(string text) : base(text) { }
        }

        /// <summary>
        /// Builds a new instance of the Cholesky class using the lower-triangular part of the specified matrix, and the specified chopping for numbers.
        /// <remarks>The matrix is not checked for symmetry.</remarks>
        /// </summary>
        /// <param name="matrix">the matrix to be decomposed.</param>
        /// <param name="chopepsilon">must be a positive number. Matrix elements whose modulus are below this number will be chopped to <c>0.0</c>. Use <c>0.0</c> if there are no stringent requirements on chopping.</param>
        public Cholesky(double[,] matrix, double chopepsilon) 
        {
            double[][] m = new double[matrix.GetLength(0)][];
            int i, j;
            for (i = 0; i < m.Length; i++)
            {
                double [] c = (m[i] = new double[i + 1]);
                for (j = 0; j <= i; j++)
                    c[j] = matrix[i, j];
            }
            m_LowerDec = LowerDecomp(m, chopepsilon);        
        }

        /// <summary>
        /// Builds a new instance of the Cholesky class using the a jagged array understood to represent the lower-triangular part of the specified matrix, and the specified chopping for numbers.        
        /// </summary>
        /// <param name="matrix">the matrix to be decomposed.</param>
        /// <param name="chopepsilon">must be a positive number. Matrix elements whose modulus are below this number will be chopped to <c>0.0</c>. Use <c>0.0</c> if there are no stringent requirements on chopping.</param>
        public Cholesky(double[][] matrix, double chopepsilon) 
        {
            m_LowerDec = LowerDecomp(matrix, chopepsilon);        
        }

        /// <summary>
        /// Builds a new instance of the Cholesky class using the result of a previous Cholesky decomposition. Useful for situations where only back-substitution is needed.
        /// <remarks>The class does not clone the matrix passed as a parameter. Therefore, any modification to the matrix will be reflected to the related Cholesky object.</remarks>
        /// </summary>
        /// <param name="ldecomp">the lower-triangular decomposed matrix.</param>
        public Cholesky(double[][] ldecomp) 
        {
            m_LowerDec = ldecomp;        
        }

        /// <summary>
        /// The result of the decomposition, in the form of a jagged array representing the lower triangular matrix.
        /// <remarks>The matrix is not a clone of the internal one. Therefore, any modification applied to the matrix will affect the Cholesky object.</remarks>
        /// </summary>
        public double[][] LDecomp 
        {
            get { return m_LowerDec; } 
        }

        /// <summary>
        /// The determinant of the matrix.
        /// </summary>
        public double Determinant 
        { 
            get 
            {
                double a = m_LowerDec[0][0] * m_LowerDec[0][0];
                int i;
                for (i = 1; i < m_LowerDec.Length; i++)
                    a *= m_LowerDec[i][i] * m_LowerDec[i][i];
                return a;
            } 
        }

        /// <summary>
        /// Solves the linear system that has the specified array as the data vector.
        /// </summary>
        /// <param name="datavect">the vector of data for the linear system.</param>
        /// <returns>the solution of the system.</returns>
        public double[] Solve(double[] datavect) 
        {
            if (datavect.Length != m_LowerDec.Length) throw new Cholesky.Exception("The size of the data vector must match the size of the coefficient matrix.");            
            return BackSubstitution(m_LowerDec, BackSubstitution(m_LowerDec, datavect, true), false);
        }

        /// <summary>
        /// Builds inverse of the original matrix, using the specified chopping.
        /// </summary>
        /// <param name="chopepsilon">the chopping precision. Put to <c>0.0</c> if there is no special requirement on precision chopping.</param>
        /// <returns>the inverse of the original matrix.</returns>
        public double[][] Inverse(double chopepsilon)
        {
            if (chopepsilon < 0.0) throw new Cholesky.Exception("Chopping limit cannot be negative.");
            double[][] m;
            int i, j, n;
            n = m_LowerDec.Length;
            double[] y = new double[n];
            m = new double[n][];
            for (i = 0; i < n; i++)
                m[i] = new double[i + 1];
            for (i = 0; i < n; i++)
            {
                y[i] = 1.0;
                double[] s = Solve(y);
                for (j = i; j < n; j++)
                    if (Math.Abs(m[j][i] = s[j]) < chopepsilon) m[j][i] = 0.0;
                y[i] = 0.0;
            }
            return m;
        }

        /// <summary>
        /// Implements back-substitution on a data vector.
        /// </summary>
        /// <param name="y">the data vector.</param>
        /// <param name="matrix">a jagged array with the elements of the triangular matrix.</param>
        /// <param name="islower"><c>true</c> if the matrix is to be understood as a lower-triangular matrix, <c>false</c> if it is upper-triangular.</param>
        /// <returns>the solution of the back-substitution algorithm.</returns>
        protected double[] BackSubstitution(double[][] matrix, double[] y, bool islower)
        {
            int i, j, n;
            if (y.Length != (n = matrix.Length)) throw new Cholesky.Exception("Data vector and triangular matrix must have the same size.");
            double[] x = new double[n];
            double a;
            double[] c;
            if (islower)
            {
                for (i = 0; i < n; i++)
                {
                    a = y[i];
                    c = matrix[i];
                    for (j = 0; j < i; j++)
                        a -= x[j] * c[j];
                    x[i] = a / c[i];
                }
            }
            else
            {
                for (i = n - 1; i >= 0; i--)
                {
                    a = y[i];
                    for (j = i + 1; j < n; j++)
                        a -= x[j] * matrix[j][i];
                    x[i] = a / matrix[i][i];
                }
            }
            return x;
        }

        /// <summary>
        /// The decomposed matrix (lower-triangular).
        /// </summary>
        protected double[][] m_LowerDec;

        /// <summary>
        /// Performs the Cholesky decomposition on the specified matrix, using the specified chopping.
        /// </summary>
        /// <param name="matrix">the matrix to be decomposed (lower-triangular).</param>
        /// <param name="chop">the chopping precision.</param>
        /// <returns>the result of the decomposition.</returns>
        protected double[][] LowerDecomp(double[][] matrix, double chop)
        {
            if (chop < 0.0) throw new Cholesky.Exception("Chopping limit cannot be negative.");
            int n = matrix.Length;
            int i, j, k;
            double[][] ld = new double[n][];
            double h;
            for (i = 0; i < n; i++)
            {
                double[] c = (ld[i] = new double[i + 1]);
                double[] a = matrix[i];
                for (j = 0; j < i; j++)
                {
                    h = a[j];
                    double[] d = ld[j];
                    for (k = 0; k < j; k++)
                        h -= c[k] * d[k];
                    if (Math.Abs(c[j] = h / d[j]) < chop) c[j] = 0.0;
                }
                h = a[i];
                for (k = 0; k < i; k++)
                    h -= c[k] * c[k];
                if (h <= chop) throw new Cholesky.Exception("Matrix is semi-definite.");
                c[i] = Math.Sqrt(h);
            }
            return ld;
        }        
    }
}
