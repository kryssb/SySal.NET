using System;

namespace NumericalTools.Minimization
{
    /// <summary>
    /// Generic function to be minimized.
    /// </summary>
    public interface ITargetFunction
    {
        /// <summary>
        /// Retrieves the number of parameters of the function.
        /// </summary>
        int CountParams { get; }
        /// <summary>
        /// Evaluates the function at the specified point.
        /// </summary>
        /// <param name="x">the point where the function should be evaluated.</param>
        /// <returns>the value of the function.</returns>
        double Evaluate(params double [] x);
        /// <summary>
        /// The minimum value for the range of the i-th parameter. Can be <c>Double.NegativeInfinity</c> if the range is unbounded.
        /// </summary>
        /// <param name="i">the number of parameter whose range is sought.</param>
        /// <returns>the minimum allowed value for the i-th parameter.</returns>
        double RangeMin(int i);
        /// <summary>
        /// The maximum value for the range of the i-th parameter. Can be <c>Double.PositiveInfinity</c> if the range is unbounded.
        /// </summary>
        /// <param name="i">the number of parameter whose range is sought.</param>
        /// <returns>the maximum allowed value for the i-th parameter.</returns>
        double RangeMax(int i);
        /// <summary>
        /// Yields the partial derivative w.r.t. the i-th parameter.
        /// </summary>
        /// <param name="i">the number of the parameter whose corresponding partial derivative is needed.</param>
        /// <returns>the partial derivative or <c>null</c> if derivatives are not known (or not supported).</returns>
        ITargetFunction Derive(int i);
        /// <summary>
        /// Minimization stopping criterion.
        /// </summary>
        /// <param name="fval">the value of the function.</param>
        /// <param name="fchange">the change in the value of the function.</param>
        /// <param name="xchange">the change in the parameter vector.</param>        
        /// <returns><c>true</c> if the minimization process can be stopped.</returns>
        bool StopMinimization(double fval, double fchange, double xchange);
        /// <summary>
        /// The starting point of the minimization procedure.
        /// </summary>
        double[] Start { get; }
    }

    /// <summary>
    /// Generic exception occurred during minimization.
    /// </summary>
    public class MinimizationException : Exception
    {
        /// <summary>
        /// Builds a new exception.
        /// </summary>
        /// <param name="s">the exception text.</param>
        public MinimizationException(string s) : base(s) { }
    }

    /// <summary>
    /// Generic minimizing algorithm.
    /// </summary>
    public interface IMinimizer
    {
        /// <summary>
        /// Finds the minimum of the specified function. The behaviour depends on the implementation.
        /// </summary>
        /// <param name="f">the function to be minimized.</param>
        /// <param name="maxiterations">the maximum number of iterations to try. Set to a non-positive number to allow infinite iterations.</param>
        /// <returns>the value of the parameter vector that minimizes the function.</returns>
        /// <remarks>A local minimum might be returned instead of a global one. A <see cref="MinimizationException"/> might be thrown.</remarks>
        double[] FindMinimum(ITargetFunction f, int maxiterations);
        /// <summary>
        /// Value of the function at its minimum.
        /// </summary>
        double Value { get; }
        /// <summary>
        /// The point where the minimum is located.
        /// </summary>
        double[] Point { get; }
        /// <summary>
        /// Logging stream the algorithm progress. Set to <c>null</c> to disable logging.
        /// </summary>
        System.IO.TextWriter Logger { set; }
    }
}