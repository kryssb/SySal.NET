using System;
using System.Collections;
using System.Runtime.Serialization;

namespace NumericalTools
{
	/// <summary>
	/// Function parameter. It holds the name and the value of the parameter.
	/// </summary>
	public class Parameter : IComparable
	{
		public readonly string iName;
		public double iValue;

		public Parameter(string name) { iName = name; iValue = 0.0; }
		public int CompareTo(object p)
		{
			return iName.ToLower().CompareTo(((Parameter)p).iName.ToLower());
		}
	}

	
	namespace MathNode
	{
		/// <summary>
		/// Generic node of an evaluation tree.
		/// </summary>
		public abstract class Generic
		{
			/// <summary>
			/// The branches of this node.
			/// </summary>
			public Generic [] Branches;
			/// <summary>
			/// Evaluates the node.
			/// </summary>
			/// <returns>the value of the node.</returns>
			public abstract double Evaluate();
			/// <summary>
			/// Partial derivative of the node with respect to one parameter.
			/// </summary>
			/// <param name="dparam">the name of the parameter with respect to which the node must be derived.</param>
			/// <returns>a new node that is the partial derivative of the current node.</returns>
			public abstract Generic Derivative(string dparam);
		}

		#region No parameter

		/// <summary>
		/// Generic zero-parameter function
		/// </summary>
		public abstract class ZeroParameter : Generic
		{
			/// <summary>
			/// The number of parameters.
			/// </summary>
			/// <returns>zero.</returns>
			public static int ParameterCount() { return 0; }
		}

		/// <summary>
		/// Constant node.
		/// </summary>
		class Const : ZeroParameter
		{
			private double Value;
			public Const(double c) { Value = c; Branches = null; }
			public override double Evaluate() { return Value; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				return new Const(0.0);
			}

		}

		/// <summary>
		/// Parameter node.
		/// </summary>
		class Parameter : ZeroParameter
		{
			private NumericalTools.Parameter Param;
			public string ParameterName { get { return (string)(Param.iName.Clone()); } }
			public Parameter(NumericalTools.Parameter param) { Param = param; }
			public override double Evaluate() { return Param.iValue; }
			public static string Name() { return "Parameter"; }
			public override Generic Derivative(string dparam)
			{
				if (String.Compare(dparam, Param.iName, true) == 0) return new Const(1.0);
				else return new Const(0.0);
			}

		}

		/// <summary>
		/// E number.
		/// </summary>
		class E : ZeroParameter
		{
			public E() { Branches = null; }
			public override double Evaluate() { return Math.E; }
			public static string Name() { return "E"; }
			public override Generic Derivative(string dparam)
			{
				return new Const(0.0);
			}
		}

		/// <summary>
		/// PI number.
		/// </summary>
		class PI : ZeroParameter
		{
			public PI() { Branches = null; }
			public override double Evaluate() { return Math.PI; }
			public static string Name() { return "PI"; }
			public override Generic Derivative(string dparam)
			{
				return new Const(0.0);
			}
		}

		/// <summary>
		/// Random number.
		/// </summary>
		class Rnd : ZeroParameter
		{
			static System.Random RndGen = new System.Random();
			public Rnd() { Branches = null; }
			public override double Evaluate() { return RndGen.NextDouble(); }
			public static string Name() { return "Rnd"; }
			public override Generic Derivative(string dparam)
			{
				return new Const(0.0);
			}
		}
		#endregion

		#region One parameter

		/// <summary>
		/// Generic one-parameter function
		/// </summary>
		abstract class OneParameter : Generic
		{
			public static int ParameterCount() { return 1; }
		}

		/// <summary>
		/// Converts an expression to a boolean value (0 or 1).
		/// </summary>
		class Bool : OneParameter
		{
			public Bool(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return ((Branches[0].Evaluate() != 0.0) ? 1.0 : 0.0); }
			public static string Name() { return "Bool"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Boolean expressions cannot be derived.");
			}
		}

		/// <summary>
		/// Negates a boolean expression (0 or 1).
		/// </summary>
		class Not : OneParameter
		{
			public Not(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return ((Branches[0].Evaluate() == 0.0) ? 1.0 : 0.0); }
			public static string Name() { return "Not"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Boolean expressions cannot be derived.");
			}
		}

		/// <summary>
		/// Unary minus.
		/// </summary>
		class Minus : OneParameter
		{
			public Minus(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return -Branches[0].Evaluate(); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const)) return new Const(-d.Evaluate());
				return new Minus(d);
			}
		}

		/// <summary>
		/// Sign.
		/// </summary>
		class Sign : OneParameter
		{
			public Sign(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Sign(Branches[0].Evaluate()); }
			public static string Name() { return "Sign"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("The Sign function cannot be derived.");
			}
		}

		/// <summary>
		/// Absolute value.
		/// </summary>
		class Abs : OneParameter
		{
			public Abs(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Abs(Branches[0].Evaluate()); }
			public static string Name() { return "Abs"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Add(new Mul(new Greater(Branches[0], new Const(0.0)), d), new Mul(new Less(Branches[0], new Const(0.0)), new Minus(d)));
			}
		}

		/// <summary>
		/// Arcsine.
		/// </summary>
		class Asin : OneParameter
		{
			public Asin(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Asin(Branches[0].Evaluate()); }
			public static string Name() { return "Asin"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, new Sqrt(new Sub(new Const(1.0), new Pow(Branches[0], new Const(2.0)))));
			}
		}

		/// <summary>
		/// Sine.
		/// </summary>
		class Sin : OneParameter
		{
			public Sin(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Sin(Branches[0].Evaluate()); }
			public static string Name() { return "Sin"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Mul(new Cos(Branches[0]), d);
			}

		}

		/// <summary>
		/// Hyperbolic sine.
		/// </summary>
		class Sinh : OneParameter
		{
			public Sinh(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Sinh(Branches[0].Evaluate()); }
			public static string Name() { return "Sinh"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Mul(new Cosh(Branches[0]), d);
			}
		}

		/// <summary>
		/// Arccosine.
		/// </summary>
		class Acos : OneParameter
		{
			public Acos(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Acos(Branches[0].Evaluate()); }
			public static string Name() { return "Acos"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Minus(new Div(d, new Sqrt(new Sub(new Const(1.0), new Pow(Branches[0], new Const(2.0))))));
			}
		}

		/// <summary>
		/// Cosine.
		/// </summary>
		class Cos : OneParameter
		{
			public Cos(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Cos(Branches[0].Evaluate()); }
			public static string Name() { return "Cos"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Minus(new Mul(d, new Sin(Branches[0])));
			}

		}

		/// <summary>
		/// Hyperbolic cosine.
		/// </summary>
		class Cosh : OneParameter
		{
			public Cosh(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Cosh(Branches[0].Evaluate()); }
			public static string Name() { return "Cosh"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Mul(new Sinh(d), Branches[0].Derivative(dparam));
			}
		}

		/// <summary>
		/// Arctangent.
		/// </summary>
		class Atan : OneParameter
		{
			public Atan(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Atan(Branches[0].Evaluate()); }
			public static string Name() { return "Atan"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, new Add(new Const(1.0), new Pow(Branches[0], new Const(2.0))));
			}

		}

		/// <summary>
		/// Tangent.
		/// </summary>
		class Tan : OneParameter
		{
			public Tan(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Tan(Branches[0].Evaluate()); }
			public static string Name() { return "Tan"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, new Pow(new Cos(Branches[0]), new Const(2.0)));
			}

		}

		/// <summary>
		/// Hyperbolic tangent.
		/// </summary>
		class Tanh : OneParameter
		{
			public Tanh(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Tanh(Branches[0].Evaluate()); }
			public static string Name() { return "Tanh"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, new Pow(new Cosh(Branches[0]), new Const(2.0)));
			}

		}

		/// <summary>
		/// Exponential.
		/// </summary>
		class Exp : OneParameter
		{
			public Exp(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Exp(Branches[0].Evaluate()); }
			public static string Name() { return "Exp"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Mul(d, new Exp(Branches[0]));
			}

		}

		/// <summary>
		/// Natural logarithm.
		/// </summary>
		class Log : OneParameter
		{
			public Log(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Log(Branches[0].Evaluate()); }
			public static string Name() { return "Log"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, Branches[0]);
			}

		}

		/// <summary>
		/// Base-10 logarithm.
		/// </summary>
		class Log10 : OneParameter
		{
			public Log10(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Log10(Branches[0].Evaluate()); }
			public static string Name() { return "Log10"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Mul(new Div(d, Branches[0]), new Const(1.0 / Math.Log(10.0)));
			}

		}

		/// <summary>
		/// Rounds to the nearest integer.
		/// </summary>
		class Round : OneParameter
		{
			public Round(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Round(Branches[0].Evaluate()); }
			public static string Name() { return "Round"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes using Round cannot be derived.");
			}
		}

		/// <summary>
		/// The integer part.
		/// </summary>
		class Int : OneParameter
		{
			public Int(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Floor(Branches[0].Evaluate()); }
			public static string Name() { return "Int"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes using Int cannot be derived.");
			}
		}

		/// <summary>
		/// The fractional part.
		/// </summary>
		class Fract : OneParameter
		{
			public Fract(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { double a = Branches[0].Evaluate(); return a - Math.Floor(a); }
			public static string Name() { return "Fract"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes using Fract cannot be derived.");
			}
		}

		/// <summary>
		/// The square root.
		/// </summary>
		class Sqrt : OneParameter
		{
			public Sqrt(Generic g) { Branches = new Generic[1]; Branches[0] = g; }
			public override double Evaluate() { return Math.Sqrt(Branches[0].Evaluate()); }
			public static string Name() { return "Sqrt"; }
			public override Generic Derivative(string dparam)
			{
				if (Branches[0].GetType() == typeof(Const)) return new Const(0.0);
				Generic d = Branches[0].Derivative(dparam);
				if (d.GetType() == typeof(Const) && d.Evaluate() == 0.0) return new Const(0.0);
				return new Div(d, new Mul(new Const(2.0), new Sqrt(Branches[0])));
			}

		}

		#endregion

		#region Two parameters

		/// <summary>
		/// Generic two-parameter function
		/// </summary>
		abstract class TwoParameter : Generic
		{
			public static int ParameterCount() { return 2; }
		}

		/// <summary>
		/// Addition.
		/// </summary>
		class Add : TwoParameter
		{
			public Add(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Branches[0].Evaluate() + Branches[1].Evaluate(); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				Generic l = Branches[0].Derivative(dparam);
				Generic r = Branches[1].Derivative(dparam);
				if (l.GetType() == typeof(Const) && r.GetType() == typeof(Const)) return new Const(l.Evaluate() + r.Evaluate());
				return new Add(l, r);
			}

		}
		
		/// <summary>
		/// Subtraction.
		/// </summary>
		class Sub : TwoParameter
		{
			public Sub(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Branches[0].Evaluate() - Branches[1].Evaluate(); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				Generic l = Branches[0].Derivative(dparam);
				Generic r = Branches[1].Derivative(dparam);
				if (l.GetType() == typeof(Const) && r.GetType() == typeof(Const)) return new Const(l.Evaluate() - r.Evaluate());
				return new Sub(l, r);
			}

		}

		/// <summary>
		/// Multiplication.
		/// </summary>
		class Mul : TwoParameter
		{
			public Mul(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Branches[0].Evaluate() * Branches[1].Evaluate(); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				Generic lh = Branches[0].Derivative(dparam);
				Generic rh = Branches[1].Derivative(dparam);
				if (lh.GetType() == typeof(Const) && lh.Evaluate() == 0.0)
				{				
					if (Branches[0].GetType() == typeof(Const) && rh.GetType() == typeof(Const)) return new Const(Branches[0].Evaluate() * rh.Evaluate());
					return new Mul(Branches[0], rh);
				}				
				if (rh.GetType() == typeof(Const) && rh.Evaluate() == 0.0)
				{					
					if (Branches[1].GetType() == typeof(Const) && lh.GetType() == typeof(Const)) return new Const(Branches[1].Evaluate() * lh.Evaluate());
					return new Mul(Branches[1], lh);
				}				
				return new Add(new Mul(lh, Branches[1]), new Mul(rh, Branches[0]));
			}
		}

		/// <summary>
		/// Division.
		/// </summary>
		class Div : TwoParameter
		{
			public Div(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Branches[0].Evaluate() / Branches[1].Evaluate(); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				Generic lh = Branches[0].Derivative(dparam);
				Generic rh = Branches[1].Derivative(dparam);
				if (lh.GetType() == typeof(Const) && lh.Evaluate() == 0.0)
				{				
					return new Div(new Minus(new Mul(rh, Branches[0])), new Pow(Branches[1], new Const(2.0)));
				}								
				if (rh.GetType() == typeof(Const) && rh.Evaluate() == 0.0)
				{				
					return new Div(lh, Branches[1]);
				}								
				return new Div(new Sub(new Mul(lh, Branches[1]), new Mul(rh, Branches[0])), new Pow(Branches[1], new Const(2.0)));
			}

		}

		/// <summary>
		/// Equality.
		/// </summary>
		class Eq : TwoParameter
		{
			public Eq(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() == Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Less-than.
		/// </summary>
		class Less : TwoParameter
		{
			public Less(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() < Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Less-than or equal.
		/// </summary>
		class LessEq : TwoParameter
		{
			public LessEq(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() <= Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Greater-than.
		/// </summary>
		class Greater : TwoParameter
		{
			public Greater(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() > Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Greater-than or equal.
		/// </summary>
		class GreaterEq : TwoParameter
		{
			public GreaterEq(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() >= Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Not equal.
		/// </summary>
		class NotEq : TwoParameter
		{
			public NotEq(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return (Branches[0].Evaluate() != Branches[1].Evaluate()) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// And.
		/// </summary>
		class And : TwoParameter
		{
			public And(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return ((Branches[0].Evaluate() != 0.0) && (Branches[1].Evaluate() != 0.0)) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Or.
		/// </summary>
		class Or : TwoParameter
		{
			public Or(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return ((Branches[0].Evaluate() != 0.0) || (Branches[1].Evaluate() != 0.0)) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Xor.
		/// </summary>
		class Xor : TwoParameter
		{
			public Xor(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return ((Branches[0].Evaluate() != 0.0) ^ (Branches[1].Evaluate() != 0.0)) ? 1.0 : 0.0; }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Max.
		/// </summary>
		class Max : TwoParameter
		{
			public Max(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Math.Max(Branches[0].Evaluate(), Branches[1].Evaluate()); }
			public static string Name() { return "Max"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Min.
		/// </summary>
		class Min : TwoParameter
		{
			public Min(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Math.Min(Branches[0].Evaluate(), Branches[1].Evaluate()); }
			public static string Name() { return "Min"; }
			public override Generic Derivative(string dparam)
			{
				throw new Exception("Nodes containing boolean expression cannot be derived.");
			}
		}

		/// <summary>
		/// Pow.
		/// </summary>
		class Pow : TwoParameter
		{
			public Pow(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Math.Pow(Branches[0].Evaluate(), Branches[1].Evaluate()); }
			public static string Name() { return null; }
			public override Generic Derivative(string dparam)
			{
				return new Exp(new Mul(Branches[1], new Log(Branches[0]))).Derivative(dparam);
			}
		}

		/// <summary>
		/// AtanXY.
		/// </summary>
		class AtanXY : TwoParameter
		{
			public AtanXY(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() { return Math.Atan2(Branches[1].Evaluate(), Branches[0].Evaluate()); }
			public static string Name() { return "AtanXY"; }
			public override Generic Derivative(string dparam)
			{
				return new Div(Branches[1], Branches[0]).Derivative(dparam);
			}

		}

		/// <summary>
		/// Hypot.
		/// </summary>
		class Hypot : TwoParameter
		{
			public Hypot(Generic left, Generic right) { Branches = new Generic[2]; Branches[0] = left; Branches[1] = right; }
			public override double Evaluate() 
			{ 
				double a = Branches[0].Evaluate();
				double b = Branches[1].Evaluate();
				return Math.Sqrt(a * a + b * b); 
			}
			public static string Name() { return "Hypot"; }
			public override Generic Derivative(string dparam)
			{
				return new Div(new Add(new Mul(Branches[0].Derivative(dparam), Branches[0]), new Mul(Branches[1].Derivative(dparam), Branches[1])), new Sqrt(new Add(new Pow(Branches[0], new Const(2.0)), new Pow(Branches[1], new Const(2.0)))));
			}

		}
		#endregion
	}


	/// <summary>
	/// A function with several user-defined parameters.
	/// </summary>
	public class Function
	{

		#region Internals

		/// <summary>
		/// Function parameters.
		/// </summary>
		internal Parameter [] Parameters;

		/// <summary>
		/// The top node in the evaluation tree.
		/// </summary>
		protected MathNode.Generic Tree = null;

		/// <summary>
		/// List of well-known functions
		/// </summary>
		protected static System.Type [] KnownFunctions =
		{
			typeof(MathNode.Abs),
			typeof(MathNode.Acos),
			typeof(MathNode.Asin),
			typeof(MathNode.Atan),
			typeof(MathNode.AtanXY),
			typeof(MathNode.Bool),
			typeof(MathNode.Cos),
			typeof(MathNode.Cosh),
			typeof(MathNode.E),
			typeof(MathNode.Exp),
			typeof(MathNode.Fract),
			typeof(MathNode.Hypot),
			typeof(MathNode.Int),
			typeof(MathNode.Log),
			typeof(MathNode.Log10),
			typeof(MathNode.Max),
			typeof(MathNode.Min),
			typeof(MathNode.Or),
			typeof(MathNode.PI),
			typeof(MathNode.Rnd),
			typeof(MathNode.Round),
			typeof(MathNode.Sign),
			typeof(MathNode.Sin),
			typeof(MathNode.Sinh),
			typeof(MathNode.Sqrt),
			typeof(MathNode.Tan),
			typeof(MathNode.Tanh),
			typeof(MathNode.Xor)
		};
		#endregion

		/// <summary>
		/// Public representation of a well-known function.
		/// </summary>
		public struct DisplayKnownFunction
		{
			public string Name;
			public int Parameters;
		}

		/// <summary>
		/// Null constructor.
		/// </summary>
		protected Function() {}

		/// <summary>
		/// The list of well-known functions with their number of parameters.
		/// </summary>
		public static DisplayKnownFunction [] KnownFunctionList
		{
			get
			{
				DisplayKnownFunction [] dl = new DisplayKnownFunction[KnownFunctions.Length];
				int i;
				for (i = 0; i < dl.Length; i++)
				{
					MathNode.Generic g = (MathNode.Generic)System.Activator.CreateInstance(KnownFunctions[i]);
					dl[i].Name = (string)(KnownFunctions[i].GetMethod("Name").Invoke(null, null));
					dl[i].Parameters = (int)(KnownFunctions[i].BaseType.GetMethod("ParameterCount").Invoke(null, null));
				}
				return dl;
			}
		}


		/// <summary>
		/// The list of parameters to be supplied.
		/// </summary>
		public string [] ParameterList
		{
			get
			{
				string [] sl = new string[Parameters.Length];
				int i; 
				for (i = 0; i < Parameters.Length; i++)
					sl[i] = Parameters[i].iName;
				return sl;
			}
		}


		/// <summary>
		/// Accesses the parameters using their name.
		/// </summary>
		public double this[string parname]
		{
			get 
			{ 
				string pn = parname.ToLower();
				foreach (Parameter p in Parameters)
					if (p.iName.ToLower() == pn)
						return p.iValue;
				throw new IndexOutOfRangeException("No parameter with name " + parname + ".");			  
			}

			set
			{
				string pn = parname.ToLower();
				foreach (Parameter p in Parameters)
					if (p.iName.ToLower() == pn)
					{
						p.iValue = value;
						return;
					}
				throw new IndexOutOfRangeException("No parameter with name " + parname + ".");			  
			}
		}


		/// <summary>
		/// Accesses the parameters using their position in the parameter list.
		/// </summary>
		public double this[int n]
		{
			get 
			{ 
				return Parameters[n].iValue;
			}

			set
			{
				Parameters[n].iValue = value;
			}
		}


		/// <summary>
		/// Internal name of the function.
		/// </summary>
		protected string iName;

		/// <summary>
		/// The function name.
		/// </summary>
		public string Name
		{
			get { return ((string)iName.Clone()); }
		}


		/// <summary>
		/// Evaluates the function.
		/// </summary>
		/// <param name="paramvalues">the values of the parameters.</param>
		/// <returns>the value of the function.</returns>
		public double Evaluate(double [] paramvalues)
		{
			if (paramvalues.Length != Parameters.Length) throw new WrongParameterValueListException(Parameters.Length, paramvalues.Length);
			int i;
			for (i = 0; i < paramvalues.Length; i++)
				Parameters[i].iValue = paramvalues[i];
			return Tree.Evaluate();
		}

		/// <summary>
		/// Evaluates the function using the current values of the parameters.
		/// </summary>
		/// <returns>the value of the function.</returns>
		public double Evaluate()
		{
			return Tree.Evaluate();
		}

		/// <summary>
		/// Finds parameters in a node and its descendants.
		/// </summary>
		/// <param name="paramcoll">the list of parameters to be updated.</param>
		/// <param name="m">the node.</param>
		protected static void FindParameters(System.Collections.ArrayList paramcoll, NumericalTools.MathNode.Generic m)
		{
			if (m.GetType() == typeof(Parameter))
			{
				string pname = ((NumericalTools.MathNode.Parameter)m).ParameterName;
				foreach (Parameter p in paramcoll)
					if (String.Compare(p.iName, pname, true) == 0)
						return;
				paramcoll.Add(pname);
			}
			else
			{
				foreach (NumericalTools.MathNode.Generic g in m.Branches)
					FindParameters(paramcoll, g);
			}
		}

		/// <summary>
		/// Builds a function from the specified node.
		/// </summary>
		/// <param name="name">the function name.</param>
		/// <param name="tree">the evaluation tree to initialize the function.</param>
		/// <param name="paramlist">if null, the parameter list is extracted from the tree MathNodes; otherwise, the list is simply validated against the tree (i.e. a parameter that does not appear in the tree may appear in the list).</param>
		protected Function(string name, NumericalTools.MathNode.Generic tree, Parameter [] paramlist)
		{
			this.iName = name;
			this.Tree = tree;
			System.Collections.ArrayList paramcoll = new ArrayList();
			FindParameters(paramcoll, tree);
			if (paramlist == null)
			{
				this.Parameters = (Parameter [])paramcoll.ToArray(typeof(Parameter));
			}
			else
			{
				this.Parameters = paramlist;
				foreach (Parameter p in paramlist)
				{
					int i;
					for (i = 0; i < paramcoll.Count; i++)
						if (String.Compare(((Parameter)paramcoll[i]).iName, p.iName, true) == 0)
							break;
					if (i == paramcoll.Count) throw new Exception("Unknown parameter " + p.iName + ".");
				}
			}
		}

		/// <summary>
		/// Builds the partial derivative of the function with respect to a specified parameter.
		/// </summary>
		/// <param name="dparam">the name of the parameter with respect to which the derivative is to be executed. The parameter must be in the function parameter list.</param>
		/// <returns>the partial derivative of the function with respect to the specified parameter. Notice that even if one or more parameters disappear in the derivation, the parameter list will be inherited from the primitive function.</returns>
		public virtual Function Derive(string dparam)
		{
			return new Function("D " + ((this.iName == null) ? "Unnamed" : this.iName) + "/" + dparam, Tree.Derivative(dparam), this.Parameters);
		}
    }

	/// <summary>
	/// The base class for mathematical expression exceptions
	/// </summary>
	[Serializable]
	public class MathExpressionException : System.Exception, ISerializable
	{
		/// <summary>
		/// Builds a new math expression exception.
		/// </summary>
		public MathExpressionException() : base("An unknown mathematical expression exception occurred.") {}
		/// <summary>
		/// Builds a new math expression exception with a specified message.
		/// </summary>
		/// <param name="message">the exception message.</param>
		public MathExpressionException(string message) : base(message) {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public void GetObjectData(SerializationInfo info, StreamingContext context) {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="c"></param>
		public MathExpressionException (SerializationInfo info, StreamingContext c) {}
	}

	/// <summary>
	/// The base class for mathematical expression exceptions
	/// </summary>
	[Serializable]
	public class SyntaxException : MathExpressionException, ISerializable
	{
		/// <summary>
		/// Builds a new syntax exception.
		/// </summary>
		public SyntaxException() : base("A syntax error occurred while parsing the expression string.") {}
		/// <summary>
		/// Initializes a new syntax exception with the indication of the string position that generated the error.
		/// </summary>
		/// <param name="i"></param>
		public SyntaxException(int i) : base("A syntax error occurred at character position " + i.ToString() + " while parsing the expression string.") {}
		/// <summary>
		/// Initializes a new syntax exception with an error message.
		/// </summary>
		/// <param name="message"></param>
		public SyntaxException(string message) : base(message) {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public void GetObjectData(SerializationInfo info, StreamingContext context) {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="c"></param>
		public SyntaxException (SerializationInfo info, StreamingContext c) {}
	}

	/// <summary>
	/// Wrong parameter list exception.
	/// </summary>
	[Serializable]
	public class WrongParameterValueListException : MathExpressionException, ISerializable
	{
		/// <summary>
		/// Builds a new WrongParameterValueList exception.
		/// </summary>
		public WrongParameterValueListException() : base("The list of parameter values does not match the function requirements.") {}
		/// <summary>
		/// Initializes a new WrongParameterValueList with the indication of the first faulty parameter.
		/// </summary>
		/// <param name="i">the zero-based index of the faulty parameter.</param>
		public WrongParameterValueListException(int i) : base("The parameter #" + i.ToString() + ", in the parameter list is wrong or missing") {}
		/// <summary>
		/// Initializes a new WrongParameterValueList with the indication of the first faulty parameter and its name.
		/// </summary>
		/// <param name="i">the zero-based index of the faulty parameter.</param>
		/// <param name="n">the name of the faulty parameter.</param>
		public WrongParameterValueListException(int i, string n) : base("The parameter #" + i.ToString() + ", name = '" + n + "' in the parameter list is wrong or missing") {}
		/// <summary>
		/// Initializes a new WrongParameterValueList with the indication of the first faulty parameter.
		/// </summary>
		/// <param name="needed">the name of the parameter that was expected.</param>
		/// <param name="supplied">the name of the parameter that has been supplied.</param>
		public WrongParameterValueListException(int needed, int supplied) : base("The number of expected parameters is " + needed.ToString() + " and the number of parameters supplied is " + supplied.ToString() + ".") {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="context"></param>
		public void GetObjectData(SerializationInfo info, StreamingContext context) {}
		/// <summary>
		/// Implements ISerializable.
		/// </summary>
		/// <param name="info"></param>
		/// <param name="c"></param>
		public WrongParameterValueListException (SerializationInfo info, StreamingContext c) {}
	}

	/// <summary>
	/// A function built from a string expression by a C-style parser.
	/// </summary>
	public class CStyleParsedFunction : Function
	{
		/// <summary>
		/// Parses an expression in C style and constructs the function, adding a default name.
		/// </summary>
		/// <param name="exprtext"></param>
		public CStyleParsedFunction(string exprtext) 
		{
			InternalConstruct(exprtext, "_unnamed_C_style_parsed_function_");
		}


		/// <summary>
		/// Parses an expression in C style and constructs the function, adding a user-specified name.
		/// </summary>
		/// <param name="exprtext"></param>
		/// <param name="name"></param>
		public CStyleParsedFunction(string exprtext, string name) 
		{
			InternalConstruct(exprtext, name);
		}

		/// <summary>
		/// Parses the expression string in C style and creates the execution tree.
		/// </summary>
		/// <param name="exprtext"></param>
		/// <param name="name"></param>
		protected void InternalConstruct(string exprtext, string name)
		{
			iName = name;

			string x = (string)exprtext.Clone();
			ManageOperators(ref x);
			ManageSpaces(ref x);
			string [] tokens = x.Split(' ');

			ArrayList tParameters = new ArrayList();
			Tree = Parse(tokens, 0, tokens.Length, ref tParameters);
			tParameters.Sort();
			Parameters = (Parameter [])tParameters.ToArray(typeof(Parameter));
		}


		/// <summary>
		/// Removes useless spaces
		/// </summary>
		/// <param name="s"></param>
		protected void ManageSpaces(ref string s)
		{
			string n;
			s = s.Replace("\t", " ").Replace("\n", " ").Trim();
			do
			{
				n = s;
				s = n.Replace("  ", " ");
			}
			while (s.Length < n.Length);
		}


		/// <summary>
		/// Prepares operators for tokenization.
		/// </summary>
		/// <param name="s"></param>
		protected void ManageOperators(ref string s)
		{
			if (s.IndexOf(']') >= 0) throw new SyntaxException("Character ']' is not accepted.");
			s = s.Replace(">=", "]");
			if (s.IndexOf('[') >= 0) throw new SyntaxException("Character '[' is not accepted.");
			s = s.Replace("<=", "[");
			if (s.IndexOf('?') >= 0) throw new SyntaxException("Character '?' is not accepted.");
			s = s.Replace("!=", "?");
			if (s.IndexOf(':') >= 0) throw new SyntaxException("Character '?' is not accepted.");
			s = s.Replace("==", ":");
			if (s.IndexOf('@') >= 0) throw new SyntaxException("Character '@' is not accepted.");
			s = s.Replace("&&", "@");
			if (s.IndexOf('#') >= 0) throw new SyntaxException("Character '#' is not accepted.");
			s = s.Replace("||", "#");
			if (s.IndexOf('$') >= 0) throw new SyntaxException("Character '$' is not accepted.");
			s = s.Replace("^^", "$");
			
			s = s.Replace("+", " + ");
			s = s.Replace("-", " - ");
			s = s.Replace("*", " * ");
			s = s.Replace("/", " / ");
			s = s.Replace("^", " ^ ");
			s = s.Replace(">", " > ");
			s = s.Replace("<", " < ");
			s = s.Replace("!", " ! ");
			s = s.Replace("(", " ( ");
			s = s.Replace(")", " ) ");
			s = s.Replace(",", " , ");

			s = s.Replace("?", " != ");
			s = s.Replace(":", " == ");
			s = s.Replace("[", " <= ");
			s = s.Replace("]", " >= ");
			s = s.Replace("@", " && ");
			s = s.Replace("#", " || ");
			s = s.Replace("$", " ^^ ");
		}


		/// <summary>
		/// Two-operand operators
		/// </summary>
		protected struct Operator2
		{
			/// <summary>
			/// Character representation of the operator.
			/// </summary>
			public string Chars;
			/// <summary>
			/// Prioriy Level of the operator.
			/// </summary>
			public PriorityLevel Priority;
			/// <summary>
			/// Type of the operator.
			/// </summary>
			public System.Type OpType;

			/// <summary>
			/// Builds a new 2-argument operator.
			/// </summary>
			/// <param name="chars">the character representation of the operator.</param>
			/// <param name="priority">the priority level of the operator.</param>
			/// <param name="optype">the operator type.</param>
			public Operator2(string chars, PriorityLevel priority, System.Type optype)
			{
				Chars = chars;
				Priority = priority;
				OpType = optype;
			}

			/// <summary>
			/// Operator priority levels.
			/// </summary>
			public enum PriorityLevel 
			{ 
				/// <summary>
				/// Or-Xor priority level.
				/// </summary>
				OrXor = 0, 
				/// <summary>
				/// And priority level.
				/// </summary>
				And = 1,
				/// <summary>
				/// Relation operator priority level.
				/// </summary>
				Relational = 2, 
				/// <summary>
				/// Additive operation priority level.
				/// </summary>
				AddSub = 3, 
				/// <summary>
				/// Multiplicative operation priority level.
				/// </summary>
				MulDiv = 4, 
				/// <summary>
				/// Power operation priority level.
				/// </summary>
				Pow = 5, 
				/// <summary>
				/// Unknown priority level.
				/// </summary>
				Unknown = 1000 
			}
		}


		/// <summary>
		/// Known operators
		/// </summary>
		protected static Operator2 [] KnownOperators =
		{
			new Operator2("+", Operator2.PriorityLevel.AddSub, typeof(MathNode.Add)), 
			new Operator2("-", Operator2.PriorityLevel.AddSub, typeof(MathNode.Sub)),
			new Operator2("*", Operator2.PriorityLevel.MulDiv, typeof(MathNode.Mul)),
			new Operator2("/", Operator2.PriorityLevel.MulDiv, typeof(MathNode.Div)),
			new Operator2("^", Operator2.PriorityLevel.Pow, typeof(MathNode.Pow)),
			new Operator2("==", Operator2.PriorityLevel.Relational, typeof(MathNode.Eq)),
			new Operator2("<=", Operator2.PriorityLevel.Relational, typeof(MathNode.LessEq)),
			new Operator2("<", Operator2.PriorityLevel.Relational, typeof(MathNode.Less)),
			new Operator2(">=", Operator2.PriorityLevel.Relational, typeof(MathNode.GreaterEq)),
			new Operator2(">", Operator2.PriorityLevel.Relational, typeof(MathNode.Greater)),
			new Operator2("!=", Operator2.PriorityLevel.Relational, typeof(MathNode.NotEq)),
			new Operator2("&&", Operator2.PriorityLevel.And, typeof(MathNode.And)),
			new Operator2("||", Operator2.PriorityLevel.OrXor, typeof(MathNode.Or)),
			new Operator2("^^", Operator2.PriorityLevel.OrXor, typeof(MathNode.Xor))
		};

		private bool ParseConstant(string [] tokens, int begin, int length, out double d)
		{
			try
			{
				string s = "";
				int i;
				for (i = begin; i < (begin + length); i++) s += tokens[i];
                d = Convert.ToDouble(s, System.Globalization.CultureInfo.InvariantCulture);
				return true;
			}
			catch (Exception) 
			{
				d = 0.0;			
			}
			return false;
		}

		private MathNode.Generic Parse(string [] tokens, int begin, int length, ref ArrayList tParameters)
		{
			int i, j;

			if (tokens[begin] == "(" && tokens[begin + length - 1] == ")") 
			{
				int bracelevel = 1;
				for (i = begin + 1; i < (begin + length - 1); i++)
					if (tokens[i] == "(") bracelevel++;
					else if (tokens[i] == ")") 
						if (--bracelevel == 0) break;
				if (i == (begin + length - 1)) return Parse(tokens, begin + 1, length - 2, ref tParameters);
			}
			
			#region Search for two operands operators
			{
				double dummy;
				int lastopindex = -1;
				int lastop = -1;
				Operator2.PriorityLevel lastpriority = Operator2.PriorityLevel.Unknown;

				for (i = begin; i < (begin + length); i++)
				{
					if (tokens[i] == "(")
					{
						int bracelevel = 1;
						while (++i < (begin + length))
						{
							if (tokens[i] == "(") bracelevel++;
							else if (tokens[i] == ")") 
							{
								if (--bracelevel == 0) break;
							}
						}
						if (bracelevel > 0) throw new SyntaxException("Unmatched parenthesis");
						continue;
					}
					for (j = 0; (j < KnownOperators.Length) && (KnownOperators[j].Chars != tokens[i]); j++);
					if (j == KnownOperators.Length) continue;
					if (KnownOperators[j].Priority <= lastpriority) 
					{
						if (KnownOperators[j].Chars == "+" || KnownOperators[j].Chars == "-" )
							if (i == begin || ParseConstant(tokens, i - 1, 3, out dummy)) continue;
						lastpriority = KnownOperators[j].Priority;
						lastop = j;
						lastopindex = i;
					}
				}
				if (lastopindex >= begin)
				{
					return (MathNode.TwoParameter)System.Activator.CreateInstance(KnownOperators[lastop].OpType, new MathNode.Generic [] {Parse(tokens, begin, lastopindex - begin, ref tParameters), Parse(tokens, lastopindex + 1, begin + length - 1 - lastopindex, ref tParameters)});
				}
			}
			#endregion

			#region Search for known functions
			{
				string s = tokens[begin].ToLower();
				string n = "";
				for (j = 0; j < KnownFunctions.Length; j++)
				{
					n = (string)KnownFunctions[j].GetMethod("Name").Invoke(null, null);
					if (n == null) continue;
					if (s == n.ToLower()) break;
				}
				if (j < KnownFunctions.Length)
				{
					Type t = KnownFunctions[j];
					int paramnum = (int)t.BaseType.GetMethod("ParameterCount").Invoke(null, null);
					if (paramnum == 0)
					{
						if (length > 1) throw new SyntaxException("No tokens expected after " + n + ".");
						return (MathNode.Generic)System.Activator.CreateInstance(t);
					}
					if (tokens[begin + 1] != "(" || tokens[begin + length - 1] != ")") throw new SyntaxException("Function arguments must be enclosed in parentheses.");
					ArrayList arglist = new ArrayList();
					j = begin + 2;
					int jend;
					do
					{
						jend = j;
						int tempbracelevel = 0;
						while (jend < (begin + length - 1) && (tokens[jend] != "," || tempbracelevel > 0)) 
						{
							if (tokens[jend] == "(") tempbracelevel++;
							else if (tokens[jend] == ")") tempbracelevel--;
							jend++;
						}
						if (jend == j) throw new SyntaxException("Functions cannot have null arguments");
						arglist.Add(Parse(tokens, j, jend - j, ref tParameters));
						j = jend + 1;
					} while (j < (begin + length - 1));
					if (arglist.Count != paramnum) throw new SyntaxException("Wrong number of arguments in function.");
					MathNode.Generic [] constrargs = (MathNode.Generic [])arglist.ToArray(typeof(MathNode.Generic));
					return (MathNode.Generic)System.Activator.CreateInstance(t, constrargs);
				}
			}
			#endregion

			#region Search for parameter name
			if (length == 1)
			{
				char ch = tokens[begin][0];
				if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
				{
					for (j = 1; j < tokens[begin].Length; j++)
					{
						ch = tokens[begin][j];
						if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || (ch == '_')) continue;
						break;
					}
					if (j != tokens[begin].Length) throw new SyntaxException("Parameter name is not well-formed.");
					{
						string np = tokens[begin].ToLower();
						foreach (Parameter p in tParameters)
							if (np == p.iName.ToLower())
								return new MathNode.Parameter(p);
						Parameter q = new Parameter(tokens[begin]);
						tParameters.Add(q);
						return new MathNode.Parameter(q);
					}
				}
			}
			#endregion

			#region	Search for constants
			{
				double d;
				if (ParseConstant(tokens, begin, length, out d))
					return new MathNode.Const(d);
			}
			#endregion

			#region Search for unary operators
			{	
				switch (tokens[begin])
				{
					case "-":	return new MathNode.Minus(Parse(tokens, begin + 1, length - 1, ref tParameters));
					case "!":	return new MathNode.Not(Parse(tokens, begin + 1, length - 1, ref tParameters));
				}
			}
			#endregion

			throw new SyntaxException("Unknown token type");
		}
	}


	/// <summary>
	/// A function built from a string expression by a BASIC-style parser.
	/// </summary>
	public class BASICStyleParsedFunction : Function
	{
		/// <summary>
		/// Parses an expression in BASIC style and constructs the function, adding a default name.
		/// </summary>
		/// <param name="exprtext"></param>
		public BASICStyleParsedFunction(string exprtext) 
		{
			InternalConstruct(exprtext, "_unnamed_BASIC_style_parsed_function_");
		}


		/// <summary>
		/// Parses an expression in BASIC style and constructs the function, adding a user-specified name.
		/// </summary>
		/// <param name="exprtext"></param>
		/// <param name="name"></param>
		public BASICStyleParsedFunction(string exprtext, string name) 
		{
			InternalConstruct(exprtext, name);
		}


		/// <summary>
		/// Parses the expression string in BASIC style and creates the execution tree.
		/// </summary>
		/// <param name="exprtext"></param>
		/// <param name="name"></param>
		protected void InternalConstruct(string exprtext, string name)
		{
			iName = name;

			string x = (string)exprtext.Clone();
			ManageOperators(ref x);
			ManageSpaces(ref x);
			string [] tokens = x.Split(' ');

			ArrayList tParameters = new ArrayList();
			Tree = Parse(tokens, 0, tokens.Length, ref tParameters);
			tParameters.Sort();
			Parameters = (Parameter [])tParameters.ToArray(typeof(Parameter));
		}


		/// <summary>
		/// Removes useless spaces
		/// </summary>
		/// <param name="s"></param>
		protected void ManageSpaces(ref string s)
		{
			string n;
			s = s.Replace("\t", " ").Replace("\n", " ").Trim();
			do
			{
				n = s;
				s = n.Replace("  ", " ");
			}
			while (s.Length < n.Length);
		}


		/// <summary>
		/// Prepares operators for tokenization.
		/// </summary>
		/// <param name="s"></param>
		protected void ManageOperators(ref string s)
		{
			if (s.IndexOf(']') >= 0) throw new SyntaxException("Character ']' is not accepted.");
			s = s.Replace(">=", "]");
			if (s.IndexOf('[') >= 0) throw new SyntaxException("Character '[' is not accepted.");
			s = s.Replace("<=", "[");
			if (s.IndexOf('?') >= 0) throw new SyntaxException("Character '?' is not accepted.");
			s = s.Replace("<>", "?");
			if (s.IndexOf(':') >= 0) throw new SyntaxException("Character '?' is not accepted.");
			s = s.Replace("=", ":");
			
			s = s.Replace("+", " + ");
			s = s.Replace("-", " - ");
			s = s.Replace("*", " * ");
			s = s.Replace("/", " / ");
			s = s.Replace("^", " ^ ");
			s = s.Replace(">", " > ");
			s = s.Replace("<", " < ");
			s = s.Replace("(", " ( ");
			s = s.Replace(")", " ) ");
			s = s.Replace(",", " , ");

			s = s.Replace("?", " <> ");
			s = s.Replace(":", " = ");
			s = s.Replace("[", " <= ");
			s = s.Replace("]", " >= ");
		}


		/// <summary>
		/// Two-operand operators
		/// </summary>
		protected struct Operator2
		{
			/// <summary>
			/// Character representation of the operator.
			/// </summary>
			public string Chars;
			/// <summary>
			/// Prioriy Level of the operator.
			/// </summary>
			public PriorityLevel Priority;
			/// <summary>
			/// Type of the operator.
			/// </summary>
			public System.Type OpType;

			/// <summary>
			/// Builds a new 2-argument operator.
			/// </summary>
			/// <param name="chars">the character representation of the operator.</param>
			/// <param name="priority">the priority level of the operator.</param>
			/// <param name="optype">the operator type.</param>
			public Operator2(string chars, PriorityLevel priority, System.Type optype)
			{
				Chars = chars;
				Priority = priority;
				OpType = optype;
			}

			/// <summary>
			/// Operator priority levels.
			/// </summary>
			public enum PriorityLevel 
			{ 
				/// <summary>
				/// Or-Xor priority level.
				/// </summary>
				OrXor = 0, 
				/// <summary>
				/// And priority level.
				/// </summary>
				And = 1,
				/// <summary>
				/// Relation operator priority level.
				/// </summary>
				Relational = 2, 
				/// <summary>
				/// Additive operation priority level.
				/// </summary>
				AddSub = 3, 
				/// <summary>
				/// Multiplicative operation priority level.
				/// </summary>
				MulDiv = 4, 
				/// <summary>
				/// Power operation priority level.
				/// </summary>
				Pow = 5, 
				/// <summary>
				/// Unknown priority level.
				/// </summary>
				Unknown = 1000 
			}
		}


		/// <summary>
		/// Known operators
		/// </summary>
		protected static Operator2 [] KnownOperators =
		{
			new Operator2("+", Operator2.PriorityLevel.AddSub, typeof(MathNode.Add)), 
			new Operator2("-", Operator2.PriorityLevel.AddSub, typeof(MathNode.Sub)),
			new Operator2("*", Operator2.PriorityLevel.MulDiv, typeof(MathNode.Mul)),
			new Operator2("/", Operator2.PriorityLevel.MulDiv, typeof(MathNode.Div)),
			new Operator2("^", Operator2.PriorityLevel.Pow, typeof(MathNode.Pow)),
			new Operator2("=", Operator2.PriorityLevel.Relational, typeof(MathNode.Eq)),
			new Operator2("<=", Operator2.PriorityLevel.Relational, typeof(MathNode.LessEq)),
			new Operator2("<", Operator2.PriorityLevel.Relational, typeof(MathNode.Less)),
			new Operator2(">=", Operator2.PriorityLevel.Relational, typeof(MathNode.GreaterEq)),
			new Operator2(">", Operator2.PriorityLevel.Relational, typeof(MathNode.Greater)),
			new Operator2("<>", Operator2.PriorityLevel.Relational, typeof(MathNode.NotEq)),
			new Operator2("And", Operator2.PriorityLevel.And, typeof(MathNode.And)),
			new Operator2("Or", Operator2.PriorityLevel.OrXor, typeof(MathNode.Or)),
			new Operator2("Xor", Operator2.PriorityLevel.OrXor, typeof(MathNode.Xor))
		};

		private bool ParseConstant(string [] tokens, int begin, int length, out double d)
		{
			try
			{
				string s = "";
				int i;
				for (i = begin; i < (begin + length); i++) s += tokens[i];
				d = Convert.ToDouble(s, System.Globalization.CultureInfo.InvariantCulture);
				return true;
			}
			catch (Exception) 
			{
				d = 0.0;			
			}
			return false;
		}

		private MathNode.Generic Parse(string [] tokens, int begin, int length, ref ArrayList tParameters)
		{
			int i, j;

			if (tokens[begin] == "(" && tokens[begin + length - 1] == ")") 
			{
				int bracelevel = 1;
				for (i = begin + 1; i < (begin + length - 1); i++)
					if (tokens[i] == "(") bracelevel++;
					else if (tokens[i] == ")") 
						if (--bracelevel == 0) break;
				if (i == (begin + length - 1)) return Parse(tokens, begin + 1, length - 2, ref tParameters);
			}
			
			#region Search for two operands operators
		{
				double dummy;
				int lastopindex = -1;
				int lastop = -1;
				Operator2.PriorityLevel lastpriority = Operator2.PriorityLevel.Unknown;

				for (i = begin; i < (begin + length); i++)
				{
					if (tokens[i] == "(")
					{
						int bracelevel = 1;
						while (++i < (begin + length))
						{
							if (tokens[i] == "(") bracelevel++;
							else if (tokens[i] == ")") 
							{
								if (--bracelevel == 0) break;
							}
						}
						if (bracelevel > 0) throw new SyntaxException("Unmatched parenthesis");
						continue;
					}
					for (j = 0; (j < KnownOperators.Length) && (KnownOperators[j].Chars != tokens[i]); j++);
					if (j == KnownOperators.Length) continue;
					if (KnownOperators[j].Priority <= lastpriority) 
					{
						if (KnownOperators[j].Chars == "+" || KnownOperators[j].Chars == "-" )
							if (i == begin || ParseConstant(tokens, i - 1, 3, out dummy)) continue;
						lastpriority = KnownOperators[j].Priority;
						lastop = j;
						lastopindex = i;
					}
				}
				if (lastopindex >= begin)
				{
					return (MathNode.TwoParameter)System.Activator.CreateInstance(KnownOperators[lastop].OpType, new MathNode.Generic [] {Parse(tokens, begin, lastopindex - begin, ref tParameters), Parse(tokens, lastopindex + 1, begin + length - 1 - lastopindex, ref tParameters)});
				}
			}
			#endregion

			#region Search for known functions
			{
				string s = tokens[begin].ToLower();
				string n = "";
				for (j = 0; j < KnownFunctions.Length; j++)
				{
					n = (string)KnownFunctions[j].GetMethod("Name").Invoke(null, null);
					if (n == null) continue;
					if (s == n.ToLower()) break;
				}
				if (j < KnownFunctions.Length)
				{
					Type t = KnownFunctions[j];
					int paramnum = (int)t.BaseType.GetMethod("ParameterCount").Invoke(null, null);
					if (paramnum == 0)
					{
						if (length > 1) throw new SyntaxException("No tokens expected after " + n + ".");
						return (MathNode.Generic)System.Activator.CreateInstance(t);
					}
					if (tokens[begin + 1] != "(" || tokens[begin + length - 1] != ")") throw new SyntaxException("Function arguments must be enclosed in parentheses.");
					ArrayList arglist = new ArrayList();
					j = begin + 2;
					int jend;
					do
					{
						jend = j;
					while (jend < (begin + length - 1) && tokens[jend] != ",") jend++;
						if (jend == j) throw new SyntaxException("Functions cannot have null arguments");
						arglist.Add(Parse(tokens, j, jend - j, ref tParameters));
						j = jend + 1;
					} while (j < (begin + length - 1));
					if (arglist.Count != paramnum) throw new SyntaxException("Wrong number of arguments in function.");
					MathNode.Generic [] constrargs = (MathNode.Generic [])arglist.ToArray(typeof(MathNode.Generic));
					return (MathNode.Generic)System.Activator.CreateInstance(t, constrargs);
				}
			}
			#endregion

			#region Search for parameter name
			if (length == 1)
			{
				char ch = tokens[begin][0];
				if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
				{
					for (j = 1; j < tokens[begin].Length; j++)
					{
						ch = tokens[begin][j];
						if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || (ch == '_')) continue;
						break;
					}
					if (j != tokens[begin].Length) throw new SyntaxException("Parameter name is not well-formed.");
					{
						string np = tokens[begin].ToLower();
						foreach (Parameter p in tParameters)
							if (np == p.iName.ToLower())
								return new MathNode.Parameter(p);
						Parameter q = new Parameter(tokens[begin]);
						tParameters.Add(q);
						return new MathNode.Parameter(q);
					}
				}
			}
			#endregion

			#region	Search for constants
			{
				double d;
				if (ParseConstant(tokens, begin, length, out d))
					return new MathNode.Const(d);
			}
			#endregion

			#region Search for unary operators
			{	
				switch (tokens[begin])
				{
					case "-":	return new MathNode.Minus(Parse(tokens, begin + 1, length - 1, ref tParameters));
					case "Not":	return new MathNode.Not(Parse(tokens, begin + 1, length - 1, ref tParameters));
				}
			}
			#endregion

			throw new SyntaxException("Unknown token type");
		}
	}


}
