using System;

namespace NumericalTools.Scripting
{
	public enum Syntax
	{
		/// <summary>
		/// C-like syntax.
		/// </summary>
		C,
		/// <summary>
		/// Pascal-like syntax.
		/// </summary>
		Pascal,
		/// <summary>
		/// BASIC-like syntax.
		/// </summary>
		BASIC
	}

	/// <summary>
	/// Possible parameter types.
	/// </summary>
	public enum ParameterType
	{
		/// <summary>
		/// Void. Can only be used in specifying return types.
		/// </summary>
		Void,
		/// <summary>
		/// 32-bit integer parameter type.
		/// </summary>
		Int32,
		/// <summary>
		/// 64-bit integer parameter type.
		/// </summary>
		Int64,
		/// <summary>
		/// Double parameter type.
		/// </summary>
		Double,
		/// <summary>
		/// String parameter type.
		/// </summary>
		String,
		/// <summary>
		/// Boolean parameter type.
		/// </summary>
		Bool,
		/// <summary>
		/// Object parameter type.
		/// </summary>
		Object
	}

	/// <summary>
	/// Describes a parameter for a function call.
	/// </summary>
	public class ParameterDescriptor
	{
		/// <summary>
		/// The name of the parameter.
		/// </summary>
		public ParameterType Type;
		/// <summary>
		/// The name of the parameter.
		/// </summary>
		public string Name;
		/// <summary>
		/// Builds a new parameter descriptor of the specified type and with the specified name.
		/// </summary>
		/// <param name="ptype">the type of the parameter.</param>
		/// <param name="pname">the name of the parameter.</param>
		public ParameterDescriptor(ParameterType ptype, string pname)
		{
			Type = ptype;
			Name = pname;
		}
		/// <summary>
		/// Converts a string to a value object of the correct type.
		/// </summary>
		/// <param name="s">the string to be converted.</param>
		/// <returns>the value object created from the string.</returns>
		public object ToType(string s)
		{
			System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;
			switch (Type)
			{
				case ParameterType.Bool:	return Convert.ToBoolean(s);
				case ParameterType.Double:	return Convert.ToDouble(s, InvC);
				case ParameterType.Int32:	return Convert.ToInt32(s);
				case ParameterType.Int64:	return Convert.ToInt64(s);
				case ParameterType.String:	return s;
			}
			throw new Exception("Unsupported type!");
		}
	}

	/// <summary>
	/// A void function call with parameters and a possible return type.
	/// </summary>
	public delegate void Function(ref object ret, object [] parameters);

	/// <summary>
	/// Describes the syntax of a function call and its help strings.
	/// </summary>
	public class FunctionDescriptor : ICloneable, IComparable
	{
		/// <summary>
		/// Lists the parameter types and names.
		/// </summary>
		public ParameterDescriptor [] Parameters;
		/// <summary>
		/// The function name.
		/// </summary>
		public string Name;
		/// <summary>
		/// The return type of the function.
		/// </summary>
		public ParameterType Type;
		/// <summary>
		/// Delegate to the function call.
		/// </summary>
		public Function dFunctionCall;
		/// <summary>
		/// The help string for this function.
		/// </summary>
		public string Help;
		/// <summary>
		/// Retrieves a shallow copy of the object.
		/// </summary>
		/// <returns>the object clone.</returns>
		public object Clone()
		{
			FunctionDescriptor d = new FunctionDescriptor();
			d.dFunctionCall = this.dFunctionCall;
			d.Help = this.Help;
			d.Name = this.Name;
			d.Parameters = this.Parameters;
			d.Type = this.Type;
			return d;
		}
		/// <summary>
		/// Builds an empty function descriptor.
		/// </summary>
		public FunctionDescriptor() {}
		/// <summary>
		/// Builds a function descriptor.
		/// </summary>
		/// <param name="name">the function name.</param>
		/// <param name="help">help string for the function.</param>
		/// <param name="parameters">parameter list.</param>
		/// <param name="returntype">return type.</param>
		/// <param name="funccall">delegate that executes the function.</param>
		public FunctionDescriptor(string name, string help, ParameterDescriptor [] parameters, ParameterType returntype, Function funccall)
		{
			Name = name;
			Help = help;
			Parameters = parameters;
			Type = returntype;
			dFunctionCall = funccall;
		}
		/// <summary>
		/// Compares two function descriptor using the alphabetic order of the function names.		
		/// Comparison is case insensitive.
		/// </summary>
		/// <param name="o">the function descriptor to compare to.</param>
		/// <returns>a number that can be negative, zero or positive depending on the relationship between the function names.</returns>
		public int CompareTo(object o)
		{
			return String.Compare(Name, ((FunctionDescriptor)o).Name, true);
		}
		/// <summary>
		/// Retrieves the function name.
		/// </summary>
		/// <returns>the function name.</returns>
		public override string ToString()
		{
			return (string)Name.Clone();
		}
	}

	/// <summary>
	/// A generic instruction.
	/// </summary>	
	public delegate void Instruction(ref int programcounter, System.Collections.ArrayList programstack);

	/// <summary>
	/// A variable.
	/// </summary>
	public class Variable
	{
		/// <summary>
		/// The variable's name.
		/// </summary>
		public string Name;
		/// <summary>
		/// The variable's type.
		/// </summary>
		public ParameterType Type;
		/// <summary>
		/// The variable's value.
		/// </summary>
		public object Value;
		/// <summary>
		/// Builds a new variable with the specified name and undefined (Void) type.
		/// </summary>
		/// <param name="name">the name of the new variable.</param>
		public Variable(string name)
		{
			Name = name;
			Type = ParameterType.Void;
			Value = null;
		}
	}

	/// <summary>
	/// A tag that marks the beginning of the local stack frame for the current block.
	/// </summary>
	public class BlockBeginTag
	{
		/// <summary>
		/// Builds a block begin tag.
		/// </summary>
		public BlockBeginTag() {}
	}

	/// <summary>
	/// A parsed and executable script.
	/// </summary>
	public class Script
	{
		/// <summary>
		/// The list of known function descriptors.
		/// </summary>
		protected static System.Collections.ArrayList FunctionDescriptors = new System.Collections.ArrayList();

		/// <summary>
		/// Resets the parsing engine and the symbol definitions and bindings.
		/// </summary>
		public static void ResetEngine() 
		{
			FunctionDescriptors.Clear();
			FunctionDescriptors.Add(new FunctionDescriptor("acos", "Computes the arc-cosine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnACOS)));
			FunctionDescriptors.Add(new FunctionDescriptor("asin", "Computes the arc-sine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnASIN)));
			FunctionDescriptors.Add(new FunctionDescriptor("atan", "Computes the arc-tangent of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnATAN)));
			FunctionDescriptors.Add(new FunctionDescriptor("atan2", "Computes the phase of the complex number x + iy.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x"), new ParameterDescriptor(ParameterType.Double, "y") }, ParameterType.Double, new Function(fnATAN2)));
			FunctionDescriptors.Add(new FunctionDescriptor("ceil", "Retrieves the smallest integer not smaller than x.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnCEIL)));
			FunctionDescriptors.Add(new FunctionDescriptor("cos", "Computes the cosine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnCOS)));
			FunctionDescriptors.Add(new FunctionDescriptor("cosh", "Computes the hyperbolic cosine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnCOSH)));
			FunctionDescriptors.Add(new FunctionDescriptor("E", "Retrieves the Neper's number.", new ParameterDescriptor [] {}, ParameterType.Double, new Function(fnE)));
			FunctionDescriptors.Add(new FunctionDescriptor("exp", "Computes the exponential of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnEXP)));
			FunctionDescriptors.Add(new FunctionDescriptor("floor", "Retrieves the largest integer not larger than x.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnFLOOR)));
			FunctionDescriptors.Add(new FunctionDescriptor("log", "Computes the natural logarithm of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnLOG)));
			FunctionDescriptors.Add(new FunctionDescriptor("log10", "Computes the decimal logarithm of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnLOG10)));
			FunctionDescriptors.Add(new FunctionDescriptor("max", "Computes the maximum of two numbers.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x"), new ParameterDescriptor(ParameterType.Double, "y") }, ParameterType.Double, new Function(fnMAX)));
			FunctionDescriptors.Add(new FunctionDescriptor("min", "Computes the minimum of two numbers.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x"), new ParameterDescriptor(ParameterType.Double, "y") }, ParameterType.Double, new Function(fnMIN)));
			FunctionDescriptors.Add(new FunctionDescriptor("PI", "Retrieves PI.", new ParameterDescriptor [] {}, ParameterType.Double, new Function(fnPI)));
			FunctionDescriptors.Add(new FunctionDescriptor("pow", "Computes x raised to y.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x"), new ParameterDescriptor(ParameterType.Double, "y") }, ParameterType.Double, new Function(fnPOW)));
			FunctionDescriptors.Add(new FunctionDescriptor("round", "Retrieves the nearest integer.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnROUND)));
			FunctionDescriptors.Add(new FunctionDescriptor("sign", "Retrieves the sign of the number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnSIGN)));
			FunctionDescriptors.Add(new FunctionDescriptor("sin", "Computes the sine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnSIN)));
			FunctionDescriptors.Add(new FunctionDescriptor("sinh", "Computes the hyperbolic sine of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnSINH)));
			FunctionDescriptors.Add(new FunctionDescriptor("sqrt", "Computes the square root of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnSQRT)));
			FunctionDescriptors.Add(new FunctionDescriptor("tan", "Computes the tangent of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnTAN)));
			FunctionDescriptors.Add(new FunctionDescriptor("tanh", "Computes the hyperbolic tangent of a number.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.Double, "x") }, ParameterType.Double, new Function(fnTANH)));
			FunctionDescriptors.Add(new FunctionDescriptor("tonumber", "Converts an integer or a string to a double, using a neutral format.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "x") }, ParameterType.String, new Function(fnTONUMBER)));
			FunctionDescriptors.Add(new FunctionDescriptor("mid", "Extracts a substring from a string.", new ParameterDescriptor [] { new ParameterDescriptor(ParameterType.String, "str"), new ParameterDescriptor(ParameterType.Int32, "firstchar"), new ParameterDescriptor(ParameterType.Int32, "length") }, ParameterType.String, new Function(fnMID)));
		}

		public static FunctionDescriptor [] GetFunctionDescriptors()
		{
			System.Collections.ArrayList ds = new System.Collections.ArrayList();
			foreach (FunctionDescriptor f in FunctionDescriptors)
				ds.Add(f.Clone());
			ds.Sort();
			return (FunctionDescriptor [])ds.ToArray(typeof(FunctionDescriptor));
		}

		#region predefined functions
		protected static void fnACOS(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Acos(Convert.ToDouble(x[0]));
		}

		protected static void fnASIN(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Asin(Convert.ToDouble(x[0]));
		}
		
		protected static void fnATAN(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Atan(Convert.ToDouble(x[0]));
		}
		
		protected static void fnATAN2(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("2 parameters are required!");
			ret = Math.Atan2(Convert.ToDouble(x[0]), Convert.ToDouble(x[1]));
		}
		
		protected static void fnCEIL(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Ceiling(Convert.ToDouble(x[0]));
		}
		
		protected static void fnCOS(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Cos(Convert.ToDouble(x[0]));
		}
		
		protected static void fnCOSH(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Cosh(Convert.ToDouble(x[0]));
		}
		
		protected static void fnE(ref object ret, object [] x)
		{
			if (x.Length != 0) throw new Exception("No parameter is required!");
			ret = Math.E;
		}
		
		protected static void fnEXP(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Exp(Convert.ToDouble(x[0]));
		}
		
		protected static void fnFLOOR(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Floor(Convert.ToDouble(x[0]));
		}
		
		protected static void fnLOG(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Log(Convert.ToDouble(x[0]));
		}
		
		protected static void fnLOG10(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Log10(Convert.ToDouble(x[0]));
		}
		
		protected static void fnMAX(ref object ret, object [] x)
		{
			if (x.Length != 2) throw new Exception("2 parameters are required!");
			ret = Math.Max(Convert.ToDouble(x[0]), Convert.ToDouble(x[1]));
		}
		
		protected static void fnMIN(ref object ret, object [] x)
		{
			if (x.Length != 2) throw new Exception("2 parameters are required!");
			ret = Math.Min(Convert.ToDouble(x[0]), Convert.ToDouble(x[1]));
		}
		
		protected static void fnPI(ref object ret, object [] x)
		{
			if (x.Length != 0) throw new Exception("No parameter is required!");
			ret = Math.PI;
		}
		
		protected static void fnPOW(ref object ret, object [] x)
		{
			if (x.Length != 2) throw new Exception("2 parameters is required!");
			ret = Math.Pow(Convert.ToDouble(x[0]), Convert.ToDouble(x[1]));
		}
		
		protected static void fnROUND(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Round(Convert.ToDouble(x[0]));
		}
		
		protected static void fnSIGN(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Sign(Convert.ToDouble(x[0]));
		}
		
		protected static void fnSIN(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Sin(Convert.ToDouble(x[0]));
		}
		
		protected static void fnSINH(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Sinh(Convert.ToDouble(x[0]));
		}
		
		protected static void fnSQRT(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Sqrt(Convert.ToDouble(x[0]));
		}
		
		protected static void fnTAN(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Tan(Convert.ToDouble(x[0]));
		}
		
		protected static void fnTANH(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Math.Tanh(Convert.ToDouble(x[0]));
		}
		
		protected static void fnTONUMBER(ref object ret, object [] x)
		{
			if (x.Length != 1) throw new Exception("1 parameter is required!");
			ret = Convert.ToDouble((string)x[0], System.Globalization.CultureInfo.InvariantCulture);
		}
		
		protected static void fnMID(ref object ret, object [] x)
		{
			if (x.Length != 3) throw new Exception("3 parameters are required!");
			ret = ((string)x[0]).Substring((int)x[1], (int)x[2]);
		}
		
		#endregion


		/// <summary>
		/// Adds a function descriptor to the list of known descriptors.
		/// </summary>
		/// <param name="d"></param>
		public static void AddFunctionDescriptor(FunctionDescriptor d)
		{
			FunctionDescriptors.Add(d);
		}

		#region Pascal
		private static System.Text.RegularExpressions.Regex xpr_pascal_open = new System.Text.RegularExpressions.Regex(@"\s*[\(]");
		private static System.Text.RegularExpressions.Regex xpr_pascal_close = new System.Text.RegularExpressions.Regex(@"\s*[\)]");
		private static System.Text.RegularExpressions.Regex xpr_pascal_comma = new System.Text.RegularExpressions.Regex(@"\s*[,]\s*");
		private static System.Text.RegularExpressions.Regex xpr_pascal_number = new System.Text.RegularExpressions.Regex(@"\s*(([-]?\d+[.]\d*|[-]?[.]\d+|[-]?\d+)([E|e][-]?\d+)?)");
		private static System.Text.RegularExpressions.Regex xpr_pascal_string = new System.Text.RegularExpressions.Regex("\\s*\\\"([^\\\"]*)\\\"\\s*");
		private static System.Text.RegularExpressions.Regex xpr_pascal_op = new System.Text.RegularExpressions.Regex(@"\s*(\+|\-|\*|/|\^|\=\=|\!\=|\<\=|\>\=|\<|\>|and|or|)\s*");
		private static System.Text.RegularExpressions.Regex xpr_pascal_function = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z][a-zA-Z_0-9]*)\s*\(");
		private static System.Text.RegularExpressions.Regex xpr_pascal_var = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z][a-zA-Z_0-9]*)\s*");
		private static System.Text.RegularExpressions.Regex xpr_pascal_unaryminus = new System.Text.RegularExpressions.Regex(@"\s*[\-]");
			
		/// <summary>
		/// Parses a Pascal-like expression and fills the program.
		/// </summary>
		/// <param name="pos">position to start from.</param>
		/// <param name="expr">the expression to be parsed.</param>
		/// <param name="appendop">the binary operation that must be appended or null.</param>
		/// <param name="allowvoid">true to allow void function calls, false to disallow.</param>
		/// <param name="program">the instruction list to be filled.</param>
		/// <returns>true if parsing should continue at the same brace level, false otherwise.</returns>
		protected bool PascalExpression(ref int pos, string expr, Instructions.BinaryOp appendop, bool allowvoid, System.Collections.ArrayList program)
		{		
			System.Text.RegularExpressions.Match m;
			System.Collections.ArrayList subprogram = new System.Collections.ArrayList();
			if ((m = xpr_pascal_open.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				PascalExpression(ref pos, expr, null, false, subprogram);
				subprogram.Add(new Instruction(new Instructions.Identity().Exec));
			}
			else if ((m = xpr_pascal_close.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return false;
			}
			else if ((m = xpr_pascal_string.Match(expr, pos)).Success && m.Index == pos)
			{
				subprogram.Add(new Instruction(new Instructions.PushValue(m.Groups[1].Value).Exec));
				pos += m.Length;				
			}
			else if ((m = xpr_pascal_number.Match(expr, pos)).Success && m.Index == pos)
			{
				double d = Convert.ToDouble(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
				subprogram.Add(new Instruction(new Instructions.PushValue(d).Exec));
				pos += m.Length;				
			}
			else if ((m = xpr_pascal_function.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				PascalExpression(ref pos, expr, null, false, subprogram);
				int i;
				for (i = 0; i < FunctionDescriptors.Count; i++)
				{
					FunctionDescriptor fd = (FunctionDescriptor)FunctionDescriptors[i];					
					if (String.Compare(fd.Name, m.Groups[1].Value, true) == 0)
					{
						if (fd.Type == ParameterType.Void && !allowvoid) throw new Exception("Void function " + fd.Name + " not allowed in expression!");
						subprogram.Add(new Instruction(new Instructions.Call(fd).Exec));
						break;
					}
				}	
				if (i == FunctionDescriptors.Count) throw new Exception("Unknown function \"" + m.Groups[1].Value + "\"");								
			}
			else if ((m = xpr_pascal_var.Match(expr, pos)).Success && m.Index == pos)
			{
				subprogram.Add(new Instruction(new Instructions.VarRead(m.Groups[1].Value).Exec));
				pos += m.Length;
			}			
			else if ((m = xpr_pascal_unaryminus.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				PascalExpression(ref pos, expr, new Instructions.BinaryOp("!"), false, subprogram);
			}			
			else throw new Exception("Unexpected token!");
			int insertionpos = program.Count;
			if (appendop != null) 
			{
				Instruction prevop;
				while (insertionpos > 1 && (prevop = (Instruction)program[insertionpos - 1]).Target.GetType() == typeof(Instructions.BinaryOp) && ((Instructions.BinaryOp)prevop.Target).Priority < appendop.Priority)
					insertionpos--;
				subprogram.Add(new Instruction(appendop.Exec));
			}
			while (subprogram.Count > 0)
			{
				program.Insert(insertionpos++, subprogram[0]);
				subprogram.RemoveAt(0);
			}			
			while (pos < expr.Length && PascalExpression2(ref pos, expr, program));
			return false;
		}

		/// <summary>
		/// Parses a Pascal-like expression from a position that is not the first and fills the program.
		/// </summary>
		/// <param name="pos">position to start from.</param>
		/// <param name="expr">the expression to be parsed.</param>
		/// <param name="program">the instruction list to be filled.</param>
		/// <returns>true if parsing should continue at the same brace level, false otherwise.</returns>
		protected bool PascalExpression2(ref int pos, string expr, System.Collections.ArrayList program)
		{		
			System.Text.RegularExpressions.Match m;
			if ((m = xpr_pascal_comma.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return PascalExpression(ref pos, expr, null, false, program);				
			}
			else if ((m = xpr_pascal_close.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return false;
			}
			else if ((m = xpr_pascal_op.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				Instructions.BinaryOp op = new Instructions.BinaryOp(m.Groups[1].Value);
				return PascalExpression(ref pos, expr, op, false, program);
			}
			throw new Exception("Unexpected token found!");
		}

		/// <summary>
		/// Finds the end of an expression within braces.
		/// </summary>
		/// <param name="expr">the expression to be scanned.</param>
		/// <param name="begin">the start position for the scan.</param>
		/// <param name="end">the end position for the scan.</param>
		/// <returns>the end of the expression or -1 is the end is not found within the limits.</returns>
		protected void PascalFindBraces(string expr, ref int begin, ref int end)
		{
			int scan = begin;
			int bracelevel = 0;			
			while (scan < end && expr[scan] != '(') scan++;
			begin = scan;
			if (scan == end) 
			{
				begin = -1;
				return;
			}
			bracelevel = 1;			
			while (scan < end && bracelevel > 0)
			{
				scan++;
				if (expr[scan] == '(') bracelevel++;
				else if (expr[scan] == ')') bracelevel--;
				else if (expr[scan] == '\"')
				{
					do
					{
						scan++;
					}
					while (scan < end && expr[scan] != '\"');
					if (scan == end) 
					{
						begin = -1;
						return;
					}
				}
			}
			end = scan;
		}

		/// <summary>
		/// Parses a Pascal-like script and builds the Script object.
		/// </summary>
		/// <param name="pos">the position where parsing of the script has arrived.</param>
		/// <param name="terminator">the expected terminator for the block.</param>
		/// <param name="scriptstr">the script string.</param>
		/// <param name="program">the list of instructions being built.</param>	
		protected void ParsePascal(ref int pos, string terminator, string scriptstr, System.Collections.ArrayList program)
		{
			if (terminator == null) terminator = "end";
			System.Collections.ArrayList unm_if = new System.Collections.ArrayList();
			System.Collections.ArrayList unm_else = new System.Collections.ArrayList();
			System.Text.RegularExpressions.Regex rgx_line = new System.Text.RegularExpressions.Regex("([^;]*);");
			System.Text.RegularExpressions.Regex rgx_fncall = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(");
			System.Text.RegularExpressions.Regex rgx_token = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*");
			System.Text.RegularExpressions.Regex rgx_assign = new System.Text.RegularExpressions.Regex(@"\s*:=\s*");
			System.Text.RegularExpressions.Regex rgx_string = new System.Text.RegularExpressions.Regex("\\s*\\\"([^\\\"]*)\\\"\\s*");
			System.Text.RegularExpressions.Regex rgx_number = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s*");
			System.Text.RegularExpressions.Regex rgx_bool = new System.Text.RegularExpressions.Regex(@"\s*(true\s*|\s*false)\s*");
			System.Text.RegularExpressions.Regex rgx_for = new System.Text.RegularExpressions.Regex(@"for\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\=\s*(\d+)\s*\.\.\s*(\d+)\s+do\s+");
			scriptstr = scriptstr.Trim();
			System.Text.RegularExpressions.Match m;		
			if (String.Compare(terminator, "end", true) == 0)
			{
				if (String.Compare((m = rgx_token.Match(scriptstr, pos)).Groups[1].Value, "begin", true) != 0) throw new Exception("Block must be enclosed in \"begin...end;\" pairs!");
				pos += m.Length;			
			}
			program.Add(new Instruction(new Instructions.Begin().Exec));
			bool last_was_if_else = false;
			while (pos < scriptstr.Length && (m = rgx_line.Match(scriptstr, pos)).Success)
			{
				pos = pos + m.Length;
				string line = m.Value;
				line = line.Remove(line.Length - 1, 1);
				System.Text.RegularExpressions.Match tm = rgx_token.Match(line);
				System.Text.RegularExpressions.Match wm;
				System.Text.RegularExpressions.Match fm;
				if (tm.Success)
				{				
					if (String.Compare(tm.Groups[1].Value, terminator, true) == 0)
					{
						if (tm.Length != line.Length) throw new Exception("No tokens allowed after \"end\"!");
						foreach (int i in unm_if)
							program[i] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
						foreach (int i in unm_else)
							program[i] = new Instruction(new Instructions.Jump(program.Count).Exec);
						program.Add(new Instruction(new Instructions.End().Exec));
						return;
					}
					else if (String.Compare(tm.Groups[1].Value, "begin", true) == 0)
					{
						pos = pos - m.Length;
						ParsePascal(ref pos, "end", scriptstr, program);
					}
					else if (String.Compare(tm.Groups[1].Value, "var", true) == 0)
					{
						try
						{														
							System.Text.RegularExpressions.Match vm = rgx_token.Match(line, tm.Length);
							if (vm.Length + tm.Length != line.Length) throw new Exception();
							program.Add(new Instruction(new Instructions.Var(new Variable(vm.Groups[1].Value)).Exec));
						}
						catch (Exception)
						{
							throw new Exception("Wrong syntax in variable declaration \"" + line + "\"!");
						}
					}
					else if (String.Compare(tm.Groups[1].Value, "if", true) == 0)
					{
						last_was_if_else = true;
						int bracestart = tm.Length, braceend = line.Length - 1;
						PascalFindBraces(line, ref bracestart, ref braceend);
						if (bracestart < 0) throw new Exception("Condition between parentheses is required after \"if\"!");
						int p = 0;
						PascalExpression(ref p, line.Substring(bracestart, braceend + 1 - bracestart), null, false, program);
						program.Add(new Instruction(new Instructions.JumpIf(false, -1).Exec));
						unm_if.Insert(0, program.Count - 1);
						pos = pos - line.Length + braceend;
						continue;
					}
					else if (String.Compare(tm.Groups[1].Value, "else", true) == 0)
					{
						last_was_if_else = true;
						if (unm_if.Count == 0) throw new Exception("Unexpected 'else' keyword!");						
						program.Add(new Instruction(new Instructions.Jump(-1).Exec));
						program[(int)unm_if[0]] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
						unm_if.RemoveAt(0);
						unm_else.Insert(0, program.Count - 1);
						pos = pos - line.Length + tm.Groups[1].Index + tm.Groups[1].Length;
						continue;
					}
					else if (String.Compare(tm.Groups[1].Value, "while", true) == 0)
					{
						int bracestart = tm.Length, braceend = line.Length - 1;
						PascalFindBraces(line, ref bracestart, ref braceend);
						if (bracestart < 0) throw new Exception("Condition between parentheses is required after \"while\"!");
						int p = 0;
						int checkpos = program.Count;
						PascalExpression(ref p, line.Substring(bracestart, braceend + 1 - bracestart), null, false, program);
						int jumppos = program.Count;
						program.Add(new Instruction(new Instructions.JumpIf(false, -1).Exec));						
						pos = pos - line.Length + braceend;
						ParsePascal(ref pos, "loop", scriptstr, program);
						program.Add(new Instruction(new Instructions.Jump(checkpos).Exec));
						program[jumppos] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
					}
					else if (String.Compare(tm.Groups[1].Value, "for", true) == 0)
					{
						System.Text.RegularExpressions.Match ffm = rgx_for.Match(line, tm.Index);
						if (ffm.Success == false || ffm.Index != tm.Groups[1].Index) throw new Exception("Wrong \"for\" syntax!");
						int startval = Convert.ToInt32(ffm.Groups[2].Value);
						int endval = Convert.ToInt32(ffm.Groups[3].Value);
						int incr = (startval <= endval) ? 1 : -1;
						program.Add(new Instruction(new Instructions.PushValue(startval).Exec));
						program.Add(new Instruction(new Instructions.VarAssign(ffm.Groups[1].Value).Exec));
						int checkpos = program.Count;
						program.Add(new Instruction(new Instructions.VarRead(ffm.Groups[1].Value).Exec));
						program.Add(new Instruction(new Instructions.PushValue(endval).Exec));
						program.Add(new Instruction(new Instructions.BinaryOp((startval <= endval) ? ">" : "<").Exec));
						int jumppos = program.Count;
						program.Add(new Instruction(new Instructions.JumpIf(true, -1).Exec));						
						pos = pos - line.Length + ffm.Length;
						ParsePascal(ref pos, "loop", scriptstr, program);
						program.Add(new Instruction(new Instructions.PushValue(incr).Exec));
						program.Add(new Instruction(new Instructions.VarRead(ffm.Groups[1].Value).Exec));
						program.Add(new Instruction(new Instructions.BinaryOp("+").Exec));
						program.Add(new Instruction(new Instructions.VarAssign(ffm.Groups[1].Value).Exec));
						program.Add(new Instruction(new Instructions.Jump(checkpos).Exec));
						program[jumppos] = new Instruction(new Instructions.JumpIf(true, program.Count).Exec);
					}
					else if ((wm = rgx_assign.Match(line, tm.Length)).Success)
					{
						int p = tm.Length + wm.Length;						
						PascalExpression(ref p, line, null, false, program);
						program.Add(new Instruction(new Instructions.VarAssign(tm.Groups[1].Value).Exec));
					}
					else if ((fm = rgx_fncall.Match(line)).Success)
					{					
						int p = 0;
						PascalExpression(ref p, line, null, true, program);
						if (!((Instructions.Call)((Instruction)program[program.Count - 1]).Target).IsVoid)
							program.Add(new Instruction(new Instructions.Pop().Exec));
					}
					if (last_was_if_else)
					{
						foreach (int i in unm_if)
							program[i] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
						foreach (int i in unm_else)
							program[i] = new Instruction(new Instructions.Jump(program.Count).Exec);
						unm_else.Clear();
						last_was_if_else = false;
					}
					else
					{
						unm_if.Clear();
					}
				}
				else throw new Exception("Unsupported syntax!");
			}
		}
		#endregion


		#region C
		private static System.Text.RegularExpressions.Regex xpr_c_open = new System.Text.RegularExpressions.Regex(@"\s*[\(]");
		private static System.Text.RegularExpressions.Regex xpr_c_close = new System.Text.RegularExpressions.Regex(@"\s*[\)]");
		private static System.Text.RegularExpressions.Regex xpr_c_comma = new System.Text.RegularExpressions.Regex(@"\s*[,]\s*");
		private static System.Text.RegularExpressions.Regex xpr_c_number = new System.Text.RegularExpressions.Regex(@"\s*(([-]?\d+[.]\d*|[-]?[.]\d+|[-]?\d+)([E|e][-]?\d+)?)");
		private static System.Text.RegularExpressions.Regex xpr_c_string = new System.Text.RegularExpressions.Regex("\\s*\\\"([^\\\"]*)\\\"\\s*");
		//private static System.Text.RegularExpressions.Regex xpr_c_incrdecr = new System.Text.RegularExpressions.Regex(@"\s*(\+\+|\-\-)");
		private static System.Text.RegularExpressions.Regex xpr_c_assign = new System.Text.RegularExpressions.Regex(@"\s*(\+\=|\-\=|\*\=|\/\=|&&\=|\|\|\=|\=)\s*");
		private static System.Text.RegularExpressions.Regex xpr_c_op = new System.Text.RegularExpressions.Regex(@"\s*(\=\=|\+|\-|\*|/|\^|\=|\!\=|\<\=|\>\=|\<|\>|&&|\|\|)\s*");
		private static System.Text.RegularExpressions.Regex xpr_c_function = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z][a-zA-Z_0-9]*)\s*\(");
		private static System.Text.RegularExpressions.Regex xpr_c_var = new System.Text.RegularExpressions.Regex(@"\s*(\+\+|\-\-|)\s*([a-zA-Z][a-zA-Z_0-9]*)\s*(\+\+|\-\-|)\s*");
		private static System.Text.RegularExpressions.Regex xpr_c_unaryminus = new System.Text.RegularExpressions.Regex(@"\s*[\-]");
			
		/// <summary>
		/// Parses a C-like expression and fills the program.
		/// </summary>
		/// <param name="pos">position to start from.</param>
		/// <param name="expr">the expression to be parsed.</param>
		/// <param name="appendop">the binary operation that must be appended or null.</param>
		/// <param name="allowvoid">true to allow void function calls, false to disallow.</param>
		/// <param name="program">the instruction list to be filled.</param>
		/// <returns>true if parsing should continue at the same brace level, false otherwise.</returns>
		protected bool CExpression(ref int pos, string expr, Instructions.BinaryOp appendop, bool allowvoid, System.Collections.ArrayList program)
		{		
			System.Text.RegularExpressions.Match m;
			System.Collections.ArrayList subprogram = new System.Collections.ArrayList();
			if ((m = xpr_c_open.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				CExpression(ref pos, expr, null, false, subprogram);
				subprogram.Add(new Instruction(new Instructions.Identity().Exec));
			}
			else if ((m = xpr_c_close.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return false;
			}
			else if ((m = xpr_c_string.Match(expr, pos)).Success && m.Index == pos)
			{
				subprogram.Add(new Instruction(new Instructions.PushValue(m.Groups[1].Value).Exec));
				pos += m.Length;				
			}
			else if ((m = xpr_c_number.Match(expr, pos)).Success && m.Index == pos)
			{
				double d = Convert.ToDouble(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);
				subprogram.Add(new Instruction(new Instructions.PushValue(d).Exec));
				pos += m.Length;				
			}
			else if ((m = xpr_c_function.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				CExpression(ref pos, expr, null, false, subprogram);
				int i;
				for (i = 0; i < FunctionDescriptors.Count; i++)
				{
					FunctionDescriptor fd = (FunctionDescriptor)FunctionDescriptors[i];					
					if (String.Compare(fd.Name, m.Groups[1].Value, true) == 0)
					{
						if (fd.Type == ParameterType.Void && !allowvoid) throw new Exception("Void function " + fd.Name + " not allowed in expression!");
						subprogram.Add(new Instruction(new Instructions.Call(fd).Exec));
						break;
					}
				}	
				if (i == FunctionDescriptors.Count) throw new Exception("Unknown function \"" + m.Groups[1].Value + "\"");								
			}
			else if ((m = xpr_c_var.Match(expr, pos)).Success && m.Index == pos)
			{
				if (m.Groups[1].Value == "++") subprogram.Add(new Instruction(new Instructions.VarIncrDecrC(m.Groups[2].Value, true).Exec));
				else if (m.Groups[1].Value == "--") subprogram.Add(new Instruction(new Instructions.VarIncrDecrC(m.Groups[2].Value, false).Exec));
				subprogram.Add(new Instruction(new Instructions.VarRead(m.Groups[2].Value).Exec));
				if (m.Groups[3].Value == "++") subprogram.Add(new Instruction(new Instructions.VarIncrDecrC(m.Groups[2].Value, true).Exec));
				else if (m.Groups[3].Value == "--") subprogram.Add(new Instruction(new Instructions.VarIncrDecrC(m.Groups[2].Value, false).Exec));
				pos += m.Length;
			}			
			else if ((m = xpr_c_unaryminus.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				CExpression(ref pos, expr, new Instructions.BinaryOp("!"), false, subprogram);
			}			
			else throw new Exception("Unexpected token!");
			int insertionpos = program.Count;
			if (appendop != null) 
			{
				Instruction prevop;
				while (insertionpos > 1 && (prevop = (Instruction)program[insertionpos - 1]).Target.GetType() == typeof(Instructions.BinaryOp) && ((Instructions.BinaryOp)prevop.Target).Priority < appendop.Priority)
					insertionpos--;
				subprogram.Add(new Instruction(appendop.Exec));
			}
			while (subprogram.Count > 0)
			{
				program.Insert(insertionpos++, subprogram[0]);
				subprogram.RemoveAt(0);
			}			
			while (pos < expr.Length && CExpression2(ref pos, expr, program));
			return false;
		}

		/// <summary>
		/// Parses a C-like expression from a position that is not the first and fills the program.
		/// </summary>
		/// <param name="pos">position to start from.</param>
		/// <param name="expr">the expression to be parsed.</param>
		/// <param name="program">the instruction list to be filled.</param>
		/// <returns>true if parsing should continue at the same brace level, false otherwise.</returns>
		protected bool CExpression2(ref int pos, string expr, System.Collections.ArrayList program)
		{		
			System.Text.RegularExpressions.Match m;
			if ((m = xpr_c_comma.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return CExpression(ref pos, expr, null, false, program);				
			}
			else if ((m = xpr_c_close.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				return false;
			}
			else if ((m = xpr_c_assign.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				Instructions.VarAssignC op = null;
				try
				{
					op = ((Instructions.VarRead)(((Instruction)program[program.Count - 1]).Target)).ToAssignment(m.Groups[1].Value);
				}
				catch (Exception)
				{
					throw new Exception("The left member of an assignment operator must be a variable!");
				}				
				program.RemoveAt(program.Count - 1);
				bool ret = CExpression(ref pos, expr, null, false, program);
				program.Add(new Instruction(op.Exec));
				return ret;
			}
			else if ((m = xpr_c_op.Match(expr, pos)).Success && m.Index == pos)
			{
				pos += m.Length;
				Instructions.BinaryOp op = new Instructions.BinaryOp(m.Groups[1].Value);
				return CExpression(ref pos, expr, op, false, program);
			}
			throw new Exception("Unexpected token found!");
		}

		/// <summary>
		/// Finds the end of an expression within braces.
		/// </summary>
		/// <param name="expr">the expression to be scanned.</param>
		/// <param name="begin">the start position for the scan.</param>
		/// <param name="end">the end position for the scan.</param>
		/// <returns>the end of the expression or -1 is the end is not found within the limits.</returns>
		protected void CFindBraces(string expr, ref int begin, ref int end)
		{
			int scan = begin;
			int bracelevel = 0;			
			while (scan < end && expr[scan] != '(') scan++;
			begin = scan;
			if (scan == end) 
			{
				begin = -1;
				return;
			}
			bracelevel = 1;			
			while (scan < end && bracelevel > 0)
			{
				scan++;
				if (expr[scan] == '(') bracelevel++;
				else if (expr[scan] == ')') bracelevel--;
				else if (expr[scan] == '\"')
				{
					do
					{
						scan++;
					}
					while (scan < end && expr[scan] != '\"');
					if (scan == end) 
					{
						begin = -1;
						return;
					}
				}
			}
			end = scan;
		}

		/// <summary>
		/// Parses a C-like script and builds the Script object.
		/// </summary>
		/// <param name="pos">the position where parsing of the script has arrived.</param>
		/// <param name="terminator">the expected terminator for the block.</param>
		/// <param name="scriptstr">the script string.</param>
		/// <param name="oneinstruction">instructs the parser to stop at the end of the current instruction (simple or compound)</param>
		/// <param name="program">the list of instructions being built.</param>	
		protected void ParseC(ref int pos, string terminator, string scriptstr, bool oneinstruction, System.Collections.ArrayList program)
		{
			System.Text.RegularExpressions.Regex rgx_firstchar = new System.Text.RegularExpressions.Regex("\\S");
			System.Text.RegularExpressions.Regex rgx_line = new System.Text.RegularExpressions.Regex("([^;]*;)");
			System.Text.RegularExpressions.Regex rgx_fncall = new System.Text.RegularExpressions.Regex(@"\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\(");
			System.Text.RegularExpressions.Regex rgx_token = new System.Text.RegularExpressions.Regex("\\s*(\\{|\\;|\\\"|[a-zA-Z0-9_]*)\\s*");
			System.Text.RegularExpressions.Regex rgx_do = new System.Text.RegularExpressions.Regex(@"do\s*\{");
			System.Text.RegularExpressions.Regex rgx_for = new System.Text.RegularExpressions.Regex(@"for\s*\(([^\;]*)\;([^\;]*)\;([^\)]*)\)");
			System.Text.RegularExpressions.Regex rgx_while = new System.Text.RegularExpressions.Regex(@"\s*while\s*\(");
			System.Text.RegularExpressions.Regex rgx_endline = new System.Text.RegularExpressions.Regex(@"\s*\;");
			System.Text.RegularExpressions.Regex rgx_string = new System.Text.RegularExpressions.Regex("\\s*\\\"([^\\\"]*)\\\"\\s*");
			System.Text.RegularExpressions.Regex rgx_number = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s*");
			System.Text.RegularExpressions.Regex rgx_bool = new System.Text.RegularExpressions.Regex(@"\s*(true\s*|\s*false)\s*");
			System.Text.RegularExpressions.Match m;		
			if (!oneinstruction) program.Add(new Instruction(new Instructions.Begin().Exec));			
			bool firstinstructiondone = false;
			while ((!oneinstruction || !firstinstructiondone) && pos < scriptstr.Length && (m = rgx_firstchar.Match(scriptstr, pos)).Success)
			{
				if (m.Groups[0].Value == "{")
				{
					pos = m.Index + m.Length;
					ParseC(ref pos, "}", scriptstr, false, program);
				}
				else if (m.Groups[0].Value == "}")
				{
					pos = m.Index + m.Length;
					program.Add(new Instruction(new Instructions.End().Exec));
					return;
				}
				else
				{
					m = rgx_line.Match(scriptstr, pos);
					pos = pos + m.Length;
					string line = m.Value;				
					System.Text.RegularExpressions.Match tm = rgx_token.Match(line);
					if (tm.Success)
					{				
						line = line.Remove(line.Length - 1, 1);
						if (String.Compare(tm.Groups[1].Value, "var", false) == 0 || String.Compare(tm.Groups[1].Value, "int", false) == 0 || String.Compare(tm.Groups[1].Value, "double", false) == 0 || String.Compare(tm.Groups[1].Value, "string", false) == 0 || String.Compare(tm.Groups[1].Value, "long", false) == 0 || String.Compare(tm.Groups[1].Value, "float", false) == 0)
						{
							try
							{														
								System.Text.RegularExpressions.Match vm = rgx_token.Match(line, tm.Length);
								if (vm.Length + tm.Length != line.Length) throw new Exception();
								program.Add(new Instruction(new Instructions.Var(new Variable(vm.Groups[1].Value)).Exec));
							}
							catch (Exception)
							{
								throw new Exception("Wrong syntax in variable declaration \"" + line + "\"!");
							}
						}
						else if (String.Compare(tm.Groups[1].Value, "if", false) == 0)
						{
							int bracestart = tm.Length, braceend = line.Length - 1;
							CFindBraces(line, ref bracestart, ref braceend);
							if (bracestart < 0) throw new Exception("Condition between parentheses is required after \"if\"!");
							int p = 0;
							CExpression(ref p, line.Substring(bracestart, braceend + 1 - bracestart), null, false, program);
							int jumppos = program.Count;
							program.Add(new Instruction(new Instructions.JumpIf(false, -1).Exec));
							pos = pos - line.Length + braceend;
							ParseC(ref pos, null, scriptstr, true, program);
							program[jumppos] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
							tm = rgx_token.Match(scriptstr, pos);
							if (tm.Success && String.Compare(tm.Groups[1].Value, "else", false) == 0)
							{
								pos = tm.Groups[1].Index + tm.Groups[1].Length;
								int elsepos = program.Count;
								program.Add(new Instruction(new Instructions.Jump(-1).Exec));
								program[jumppos] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
								ParseC(ref pos, null, scriptstr, true, program);
								program[elsepos] = new Instruction(new Instructions.Jump(program.Count).Exec);
							}
						}
						else if (String.Compare(tm.Groups[1].Value, "while", false) == 0)
						{
							int bracestart = tm.Length, braceend = line.Length - 1;
							CFindBraces(line, ref bracestart, ref braceend);
							if (bracestart < 0) throw new Exception("Condition between parentheses is required after \"while\"!");
							int p = 0;
							int checkpos = program.Count;
							CExpression(ref p, line.Substring(bracestart, braceend + 1 - bracestart), null, false, program);
							int jumppos = program.Count;
							program.Add(new Instruction(new Instructions.JumpIf(false, -1).Exec));						
							pos = pos - line.Length + braceend;
							ParseC(ref pos, null, scriptstr, true, program);
							program.Add(new Instruction(new Instructions.Jump(checkpos).Exec));
							program[jumppos] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
						}
						else if (String.Compare(tm.Groups[1].Value, "do", false) == 0)
						{
							System.Text.RegularExpressions.Match dm = rgx_do.Match(line, tm.Groups[1].Index);
							if (dm.Success == false || dm.Index != tm.Groups[1].Index) throw new Exception("Braces needed after \"do\"!");
							int jumppos = program.Count;
							pos = pos - m.Length + dm.Index + dm.Length - 1;
							ParseC(ref pos, "}", scriptstr, true, program);
							dm = rgx_while.Match(scriptstr, pos);
							if (dm.Success == false || dm.Index != pos) throw new Exception("\"while\" needed after \"do\" block!");
							int bracestart = dm.Index + dm.Length - 1, braceend = scriptstr.Length - 1;
							CFindBraces(scriptstr, ref bracestart, ref braceend);
							if (bracestart < 0) throw new Exception("Condition between parentheses is required after \"while\"!");
							int p = 0;
							int checkpos = program.Count;
							CExpression(ref p, scriptstr.Substring(bracestart, braceend + 1 - bracestart), null, false, program);
							program.Add(new Instruction(new Instructions.JumpIf(true, jumppos).Exec));
							pos = braceend + 1;
							dm = rgx_endline.Match(scriptstr, pos);
							if (dm.Success == false || dm.Index != pos) throw new Exception("Unexpected token after while condition!");
						}
						else if (String.Compare(tm.Groups[1].Value, "for", false) == 0)
						{
							pos = pos - m.Length + tm.Groups[1].Index;
							System.Text.RegularExpressions.Match fm = rgx_for.Match(scriptstr, pos);
							if (fm.Success == false || fm.Index != pos) throw new Exception("Conditions needed after \"for\"!");
							int p = 0;
							if (fm.Groups[1].Value.Trim().Length > 0)
								CExpression(ref p, fm.Groups[1].Value, null, false, program);
							int jumppos = program.Count;
							p = 0;
							CExpression(ref p, fm.Groups[2].Value, null, false, program);
							int checkpos = program.Count;
							program.Add(new Instruction(new Instructions.JumpIf(false, -1).Exec));
							pos += fm.Length;
							ParseC(ref pos, null, scriptstr, true, program);
							p = 0;
							if (fm.Groups[3].Value.Trim().Length > 0)
								CExpression(ref p, fm.Groups[3].Value, null, false, program);
							program.Add(new Instruction(new Instructions.Jump(jumppos).Exec));
							program[checkpos] = new Instruction(new Instructions.JumpIf(false, program.Count).Exec);
						}
						else if (line.Trim().Length > 0)
						{					
							int p = 0;
							CExpression(ref p, line, null, true, program);
							Instruction i = (Instruction)program[program.Count - 1];
							if (i.Target.GetType() != typeof(Instructions.Call) || !((Instructions.Call)i.Target).IsVoid)
								program.Add(new Instruction(new Instructions.Pop().Exec));
						}
					}
					else throw new Exception("Unsupported syntax!");					
				}
				firstinstructiondone = true;
			}
		}
		#endregion

		/// <summary>
		/// Parses a script using the specified syntax. 
		/// The script is ready for execution when the constructor returns.
		/// </summary>
		/// <param name="scriptstr">the script string.</param>
		/// <param name="usesyntax">the syntax to be used to parse the script. Currently, only Pascal-style syntax is supported.</param>
		public Script(string scriptstr, Syntax usingsyntax)
		{
			//
			// TODO: Add constructor logic here
			//
			System.Collections.ArrayList program = new System.Collections.ArrayList();
			int pos = 0;
			switch (usingsyntax)
			{
				case Syntax.Pascal: ParsePascal(ref pos, null, scriptstr, program); break;
				case Syntax.C: ParseC(ref pos, null, scriptstr, true, program); break;
				default: throw new Exception("Syntax not supported!");
			}			
			m_Instructions = (Instruction [])program.ToArray(typeof(Instruction));
		}


		/// <summary>
		/// The program counter for script execution.
		/// </summary>
		protected int m_ProgramCounter;
		/// <summary>
		/// Shows the program counter. This is safe for multithreaded operation.
		/// </summary>
		public int ProgramCounter 
		{ 
			get 
			{ 
				lock(this) 
					return m_ProgramCounter; 
			} 
		}

		/// <summary>
		/// The program stack.
		/// </summary>
		protected System.Collections.ArrayList m_Stack = new System.Collections.ArrayList();
		/// <summary>
		/// Shows the program stack. This is safe for multithreaded operation (a copy is provided, not the real stack).
		/// </summary>
		public System.Collections.ArrayList Stack
		{
			get
			{
				lock(this)
					return (System.Collections.ArrayList)m_Stack.Clone();
			}
		}

		/// <summary>
		/// The list of the instructions to be executed.
		/// </summary>
		protected Instruction [] m_Instructions;

		/// <summary>
		/// Executes a parsed script.
		/// </summary>
		/// <returns>an object returned by the script or null.</returns>
		public object Execute()
		{
			lock(this)
			{
				while (m_Stack.Count > 0) m_Stack.RemoveAt(0);
				m_ProgramCounter = -1;
			}
			while (++m_ProgramCounter < m_Instructions.Length)			
				lock(this)
					m_Instructions[m_ProgramCounter](ref m_ProgramCounter, m_Stack);
			lock(this)
			{
				if (m_Stack.Count == 1) return m_Stack[0];
				else if (m_Stack.Count > 1) throw new Exception("Script execution stack is not empty at the end of execution!");
			}
			return null;
		}
	}

	namespace Instructions
	{
		/// <summary>
		/// Implements an identity.
		/// </summary>
		public class Identity
		{
			/// <summary>
			/// Builds an identity.
			/// </summary>
			public Identity() {}

			/// <summary>
			/// Executes the instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp) {}
		}

		/// <summary>
		/// Implements a jump to an address.
		/// </summary>
		public class Jump
		{
			/// <summary>
			/// Instruction to jump to.
			/// </summary>			
			protected int m_JumpPoint;

			/// <summary>
			/// Builds a jump to the specified jump point.
			/// </summary>
			/// <param name="newpc">new value of the program counter after the jump.</param>
			public Jump(int newpc) { m_JumpPoint = newpc; }

			/// <summary>
			/// Executes the instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp) 
			{ 
				pc = m_JumpPoint - 1; 			
			}
		}

		/// <summary>
		/// Implements a conditional jump to an address. 
		/// The jump is performed if the value on the top of the stack is zero or boolean false and the jump condition is false, or if it is nonzero or boolean true and the condition is true.
		/// </summary>
		public class JumpIf
		{
			/// <summary>
			/// Instruction to jump to.
			/// </summary>			
			protected int m_JumpPoint;

			/// <summary>
			/// Condition to check.
			/// </summary>
			protected bool m_JumpCond;

			/// <summary>
			/// Builds a conditional jump to the specified jump point.
			/// </summary>
			/// <param name="condition">condition to be checked.</param>
			/// <param name="newpc">new value of the program counter after the jump.</param>
			public JumpIf(bool condition, int newpc) 
			{ 
				m_JumpCond = condition;
				m_JumpPoint = newpc; 
			}

			/// <summary>
			/// Executes the instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp) 
			{ 
				bool b = Convert.ToBoolean(sp[0]);
				sp.RemoveAt(0);
				if (b == m_JumpCond)
					pc = m_JumpPoint - 1;
			}			
		}

		/// <summary>
		/// Implements a push operation on the stack.
		/// </summary>
		public class PushValue
		{
			/// <summary>
			/// Value to be pushed on the stack.
			/// </summary>
			protected object m_Value;
			/// <summary>
			/// Build a new push operation.
			/// </summary>
			/// <param name="val">the value to be pushed on the stack.</param>
			public PushValue(object val)
			{
				m_Value = val;
			}

			/// <summary>
			/// Executes the instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp) 
			{ 
				sp.Insert(0, m_Value);
			}			
		}
	
		/// <summary>
		/// Implements a pop operation on the stack.
		/// </summary>
		public class Pop
		{
			/// <summary>
			/// Build a new pop operation.
			/// </summary>
			public Pop() {}

			/// <summary>
			/// Executes the instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp) 
			{ 
				sp.RemoveAt(0);
			}			
		}

		/// <summary>
		/// Implements a variable declaration.
		/// </summary>
		public class Var
		{
			/// <summary>
			/// The variable to be declared.
			/// </summary>
			protected Variable m_Var;
			/// <summary>
			/// Builds a variable declaration.
			/// </summary>
			public Var(Variable v) 
			{
				m_Var = v;
			}
			/// <summary>
			/// Executes a variable declaration.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				sp.Insert(0, m_Var);
			}
		}

		/// <summary>
		/// Implements an assignment to a variable.
		/// </summary>
		public class VarAssign
		{
			/// <summary>
			/// The name of the variable to be assigned.
			/// </summary>
			protected string m_VarName;
			/// <summary>
			/// Builds an assignment to a variable.
			/// </summary>
			public VarAssign(string varname) 
			{
				m_VarName = varname;
			}
			/// <summary>
			/// Executes an assignment to a variable.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				object o = sp[0];
				sp.RemoveAt(0);
				int i;
				for (i = 0; i < sp.Count; i++)
					if (sp[i].GetType() == typeof(Variable) && String.Compare(((Variable)sp[i]).Name, m_VarName, false) == 0)
					{
						((Variable)sp[i]).Value = o;
						return;
					}
				throw new Exception("Variable " + m_VarName + " not found in current scope!");
			}
		}

		/// <summary>
		/// Implements a C-like variable increment / decrement.
		/// </summary>
		public class VarIncrDecrC
		{
			/// <summary>
			/// The name of the variable to be incremented / decremented.
			/// </summary>
			protected string m_VarName;
			/// <summary>
			/// True if the operation is an increment, false if it is a decrement.
			/// </summary>
			protected bool m_Increment;
			/// <summary>
			/// Builds an a variable increment / decrement.
			/// </summary>
			public VarIncrDecrC(string varname, bool increment) 
			{
				m_VarName = varname;
				m_Increment = increment;
			}
			/// <summary>
			/// Executes a variable increment / decrement.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				int i;
				for (i = 1; i < sp.Count; i++)
					if (sp[i].GetType() == typeof(Variable) && String.Compare(((Variable)sp[i]).Name, m_VarName, false) == 0)
					{
						Variable v = ((Variable)sp[i]);
						System.Type t = v.Value.GetType();
						if (t == typeof(int))
						{
							if (m_Increment) v.Value = (int)v.Value + 1;
							else v.Value = (int)v.Value - 1;
						}
						else if (t == typeof(long))
						{
							if (m_Increment) v.Value = (long)v.Value + 1;
							else v.Value = (long)v.Value - 1;
						}
						else if (t == typeof(double))
						{
							if (m_Increment) v.Value = (double)v.Value + 1;
							else v.Value = (double)v.Value - 1;
						}
						else throw new Exception("Type " + t + " cannot be incremented / decremented!");
						return;
					}
				throw new Exception("Variable " + m_VarName + " not found in current scope!");
			}

		}
			
		/// <summary>
		/// Implements a C-like assignment to a variable.
		/// </summary>
		public class VarAssignC
		{
			private void Assign(Variable v, object x)
			{
				v.Value = x;
			}

			private void Add(Variable v, object x)
			{
				if (v.Value.GetType() == typeof(string) || x.GetType() == typeof(string))
					v.Value = v.Value.ToString() + x.ToString();
				else
					v.Value = Convert.ToDouble(v.Value) + Convert.ToDouble(x);
			}

			private void Sub(Variable v, object x)
			{
				v.Value = Convert.ToDouble(v.Value) - Convert.ToDouble(x);			
			}

			private void Mul(Variable v, object x)
			{
				v.Value = Convert.ToDouble(v.Value) * Convert.ToDouble(x);
			}

			private void Div(Variable v, object x)
			{
				v.Value = Convert.ToDouble(v.Value) / Convert.ToDouble(x);
			}

			private void Pow(Variable v, object x)
			{
				v.Value = Math.Pow(Convert.ToDouble(v.Value), Convert.ToDouble(x));
			}

			private void And(Variable v, object x)
			{
				v.Value = Convert.ToBoolean(v.Value) && Convert.ToBoolean(x);
			}

			private void Or(Variable v, object x)
			{
				v.Value = Convert.ToBoolean(v.Value) || Convert.ToBoolean(x);
			}

			protected delegate void AssignOp(Variable v, object x);

			/// <summary>
			/// The name of the variable to be assigned.
			/// </summary>
			protected string m_VarName;
			/// <summary>
			/// The instruction to be executed.
			/// </summary>
			protected AssignOp m_Assign;
			/// <summary>
			/// Builds an assignment to a variable.
			/// </summary>
			public VarAssignC(string varname, string assigntype) 
			{
				m_VarName = varname;
				if (String.Compare(assigntype, "=", true) == 0)
				{
					m_Assign = new AssignOp(Assign);
				}
				else if (String.Compare(assigntype, "+=", true) == 0)
				{
					m_Assign = new AssignOp(Add);
				}
				else if (String.Compare(assigntype, "-=", true) == 0)
				{
					m_Assign = new AssignOp(Sub);
				}
				else if (String.Compare(assigntype, "*=", true) == 0)
				{
					m_Assign = new AssignOp(Mul);
				}
				else if (String.Compare(assigntype, "/=", true) == 0)
				{
					m_Assign = new AssignOp(Div);
				}
				else if (String.Compare(assigntype, "^=", true) == 0)
				{
					m_Assign = new AssignOp(Pow);
				}
				else if (String.Compare(assigntype, "&&=", true) == 0)
				{
					m_Assign = new AssignOp(And);
				}
				else if (String.Compare(assigntype, "||=", true) == 0)
				{
					m_Assign = new AssignOp(Or);
				}
			}
			/// <summary>
			/// Executes an assignment to a variable.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				object o = sp[0];				
				int i;
				for (i = 1; i < sp.Count; i++)
					if (sp[i].GetType() == typeof(Variable) && String.Compare(((Variable)sp[i]).Name, m_VarName, false) == 0)
					{
						Variable v = ((Variable)sp[i]);
						m_Assign(v, o);
						sp[0] = v.Value;
						return;
					}
				throw new Exception("Variable " + m_VarName + " not found in current scope!");
			}
		}

		/// <summary>
		/// Implements an assignment to a variable.
		/// </summary>
		public class VarRead
		{
			/// <summary>
			/// The name of the variable to be assigned.
			/// </summary>
			protected string m_VarName;
			/// <summary>
			/// Builds an assignment to a variable.
			/// </summary>
			public VarRead(string varname) 
			{
				m_VarName = varname;
			}
			/// <summary>
			/// Executes an assignment to a variable.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				int i;
				for (i = 0; i < sp.Count; i++)
					if (sp[i].GetType() == typeof(Variable) && String.Compare(((Variable)sp[i]).Name, m_VarName, false) == 0)
					{
						sp.Insert(0, ((Variable)sp[i]).Value);
						return;
					}
				throw new Exception("Variable " + m_VarName + " not found in current scope!");
			}
			/// <summary>
			/// Converts a variable read to a variable assignment.
			/// </summary>
			/// <param name="assigntype">the assignment type.</param>
			/// <returns>the new variable assignment object.</returns>
			public VarAssignC ToAssignment(string assigntype)
			{
				return new VarAssignC(m_VarName, assigntype);
			}
		}

		/// <summary>
		/// Implements a block begin instruction.
		/// </summary>
		public class Begin
		{
			/// <summary>
			/// Builds a block begin instruction.
			/// </summary>
			public Begin() {}
			/// <summary>
			/// Executes a block begin instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				sp.Insert(0, new BlockBeginTag());
			}
		}

		/// <summary>
		/// Implements a block end instruction.
		/// </summary>
		public class End
		{
			/// <summary>
			/// Builds a block end instruction.
			/// </summary>
			public End() {}
			/// <summary>
			/// Executes a block end instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				object o;
				do
				{
					o = sp[0];
					sp.RemoveAt(0);
				}
				while (o.GetType() != typeof(BlockBeginTag));
			}
		}

		/// <summary>
		/// Implements a binary operator.
		/// </summary>
		public class BinaryOp
		{
			private void UnaryMinus(ref int pc, System.Collections.ArrayList sp)
			{
				sp[0] = -(double)sp[0];
			}

			private void And(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToBoolean(p) && Convert.ToBoolean(o));
			}

			private void Or(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToBoolean(p) || Convert.ToBoolean(o));
			}

			private void Equal(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				if (p.GetType() == typeof(bool)) sp.Insert(0, (bool)p == (bool)o);
				else if (p.GetType() == typeof(string)) sp.Insert(0, (string)p == (string)o);
				else sp.Insert(0, Convert.ToDouble(p) == Convert.ToDouble(o));
			}

			private void NotEqual(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				if (p.GetType() == typeof(bool)) sp.Insert(0, (bool)p != (bool)o);
				else if (p.GetType() == typeof(string)) sp.Insert(0, (string)p != (string)o);
				else sp.Insert(0, Convert.ToDouble(p) != Convert.ToDouble(o));
			}

			private void Less(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) < Convert.ToDouble(o));
			}

			private void Greater(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) > Convert.ToDouble(o));
			}


			private void LessEq(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) <= Convert.ToDouble(o));
			}

			private void GreaterEq(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) >= Convert.ToDouble(o));
			}

			private void Add(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				if (o.GetType() == typeof(string) || p.GetType() == typeof(string))
					sp.Insert(0, p.ToString() + o.ToString());
				else
					sp.Insert(0, Convert.ToDouble(p) + Convert.ToDouble(o));
			}

			private void Sub(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) - Convert.ToDouble(o));
			}

			private void Mul(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) * Convert.ToDouble(o));
			}

			private void Div(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Convert.ToDouble(p) / Convert.ToDouble(o));
			}

			private void Pow(ref int pc, System.Collections.ArrayList sp)
			{
				object o, p;
				o = sp[0];
				sp.RemoveAt(0);
				p = sp[0];
				sp.RemoveAt(0);
				sp.Insert(0, Math.Pow(Convert.ToDouble(p), Convert.ToDouble(o)));
			}

			/// <summary>
			/// Delegate to the actual instruction to be executed.
			/// </summary>
			protected Instruction m_OpInstr;
			/// <summary>
			/// The operator to be applied.
			/// </summary>
			protected string m_Op;
			/// <summary>
			/// The priority of the operator.
			/// </summary>
			protected int m_Priority;
			/// <summary>
			/// The priority of the operator.
			/// </summary>
			public int Priority { get { return m_Priority; } }
			/// <summary>
			/// Builds a binary operator.
			/// </summary>
			public BinaryOp(string op) 
			{
				m_Op = op;
				if (op == "&&") 
				{
					m_Priority = 0; m_OpInstr = new Instruction(this.And);					
				}
				else if (op == "||") 
				{
					m_Priority = 0; m_OpInstr = new Instruction(this.Or);
				}
				else if (op == "==") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.Equal);
				}
				else if (op == "!=") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.NotEqual);
				}
				else if (op == "<") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.Less);
				}
				else if (op == "<=") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.LessEq);
				}
				else if (op == ">") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.Greater);
				}
				else if (op == ">=") 
				{
					m_Priority = 1; m_OpInstr = new Instruction(this.GreaterEq);
				}
				else if (op == "+") 
				{
					m_Priority = 2; m_OpInstr = new Instruction(this.Add);
				}
				else if (op == "-") 
				{
					m_Priority = 2; m_OpInstr = new Instruction(this.Sub);
				}
				else if (op == "*") 
				{
					m_Priority = 3; m_OpInstr = new Instruction(this.Mul);
				}
				else if (op == "/") 
				{
					m_Priority = 3; m_OpInstr = new Instruction(this.Div);
				}
				else if (op == "^") 
				{
					m_Priority = 4; m_OpInstr = new Instruction(this.Pow);
				}
				else if (op == "!") 
				{
					m_Priority = 100; m_OpInstr = new Instruction(this.UnaryMinus);
				}
			}
			/// <summary>
			/// Executes a binary operator instruction.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				m_OpInstr(ref pc, sp);
			}
		}

		/// <summary>
		/// Implements a function call.
		/// </summary>
		public class Call
		{
			/// <summary>
			/// The function to be called.
			/// </summary>
			protected FunctionDescriptor m_FunctionDesc;

			/// <summary>
			/// Checks whether the function is void.
			/// </summary>
			public bool IsVoid { get { return m_FunctionDesc.Type == ParameterType.Void; } }
			
			/// <summary>
			/// Builds a function call.
			/// </summary>
			/// <param name="funcdesc">the function descriptor of the function to be called.</param>
			public Call(FunctionDescriptor funcdesc) 
			{
				m_FunctionDesc = funcdesc;
			}

			/// <summary>
			/// Calls the specified function.
			/// </summary>
			/// <param name="pc">the program counter.</param>
			/// <param name="sp">the stack.</param>
			public void Exec(ref int pc, System.Collections.ArrayList sp)
			{
				object [] parameters = new object [m_FunctionDesc.Parameters.Length];
				int i;
				for (i = parameters.Length - 1; i >= 0; i--)
				{
					object o = sp[0];
					sp.RemoveAt(0);
					switch (m_FunctionDesc.Parameters[i].Type)
					{
						case ParameterType.Bool:	parameters[i] = Convert.ToBoolean(o); break;
						case ParameterType.Double:	parameters[i] = Convert.ToDouble(o); break;
						case ParameterType.Int32:	parameters[i] = Convert.ToInt32(o); break;
						case ParameterType.Int64:	parameters[i] = Convert.ToInt64(o); break;
						case ParameterType.String:	parameters[i] = o.ToString(); break;
						default:					throw new Exception("Unknown parameter type at parameter " + i + ", function " + m_FunctionDesc.Name + "!");
					}
				}
				object ret = null;
				m_FunctionDesc.dFunctionCall(ref ret, parameters);
				if (m_FunctionDesc.Type != ParameterType.Void) 
				{
					if (ret == null) throw new Exception("Object of type " + m_FunctionDesc.Type + " expected; null found!");
					else sp.Insert(0, ret);
				}
			}
		}
	}
}
