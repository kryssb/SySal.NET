using System;

namespace SySal.Executables.ManualChecks
{
	internal enum CheckMode { Center, Upstream, Downstream }	
	/// <summary>
	/// ManualChecks - command line tool for efficiency estimation and to prepare massive manual checks campaigns.
	/// </summary>
	/// <remarks>
	/// <para>
	/// Usage: <c>ManualChecks &lt;mode&gt; &lt;input TSR&gt; &lt;output TXT&gt; [&lt;selection string&gt;]</c>
	/// <list type="table">
	/// <listheader><term>Mode</term><description>Behaviour</description></listheader>
	/// <item><term>/3</term><description>check on the center plate of 3</description></item>
	/// <item><term>/3f</term><description>check on the center plate of 3 and dump found only</description></item>
	/// <item><term>/3n</term><description>check on the center plate of 3 and dump not found only</description></item>
	/// <item><term>/3a</term><description>check on the center plate of 3 and dump found and not found</description></item>
	/// <item><term>/u</term><description>check on the upstream plate</description></item>
	/// <item><term>/uf</term><description>check on the upstream plate and dump found only</description></item>
	/// <item><term>/un</term><description>check on the upstream plate and dump not found only</description></item>
	/// <item><term>/ua</term><description>check on the upstream plate and dump found and not found</description></item>
	/// <item><term>/d</term><description>check on the downstream plate</description></item>
	/// <item><term>/df</term><description>check on the downstream plate and dump found only</description></item>
	/// <item><term>/dn</term><description>check on the downstream plate and dump not found only</description></item>
	/// <item><term>/da</term><description>check on the downstream plate and dump found and not found</description></item>
	/// </list>
	/// </para>
	/// <para>If the selection string is not null, the tracks on which checks have to be performed are those that pass the selection.</para>	
	/// <para>
	/// Known selection variables:
	/// <list type="table">
	/// <listheader><term>Name</term><description>Meaning</description></listheader>
	/// <item><term>N</term><description>Number of segments</description></item>
	/// <item><term>DZ</term><description>Downstream Z</description></item>
	/// <item><term>DSX, DSY</term><description>Downstream Slope X,Y</description></item>
	/// <item><term>D0X, D0Y</term><description>Downstream Position X,Y (at Z = 0)</description></item>
	/// <item><term>DPX, DPY</term><description>Downstream Position X,Y (at Z = DZ)</description></item>
	/// <item><term>UZ</term><description>Upstream Z</description></item>
	/// <item><term>USX, USY</term><description>Upstream Slope X,Y</description></item>
	/// <item><term>U0X, U0Y</term><description>Upstream Position X,Y (at Z = 0)</description></item>
	/// <item><term>UPX, UPY</term><description>Upstream Position X,Y (at Z = UZ)</description></item>
	/// </list>
	/// </para>
	/// <para>The format of the output file is an ASCII n-tuple such as</para>
	/// <para><c>ID PX PY SX SY PLATES</c></para>	
	/// <para>The <c>PLATES</c> field denotes the number of plates on which the corresponding volume track has been found.</para>
	/// </remarks>
	public class Exe
	{				
		static void ShowExplanation()
		{
			Console.WriteLine("usage: ManualChecks <mode> <input TSR> <output TXT> [<selection string>]");
			Console.WriteLine("Modes:");
			Console.WriteLine("/3 = check on the center plate of 3");
			Console.WriteLine("/3f = check on the center plate of 3 and dump found only");
			Console.WriteLine("/3n = check on the center plate of 3 and dump not found only");
			Console.WriteLine("/3a = check on the center plate of 3 and dump found and not found");
			Console.WriteLine("/u = check on the upstream plate");
			Console.WriteLine("/uf = check on the upstream plate and dump found only");
			Console.WriteLine("/un = check on the upstream plate and dump not found only");
			Console.WriteLine("/ua = check on the upstream plate and dump found and not found");
			Console.WriteLine("/d = check on the downstream plate");
			Console.WriteLine("/df = check on the downstream plate and dump found only");
			Console.WriteLine("/dn = check on the downstream plate and dump not found only");
			Console.WriteLine("/da = check on the downstream plate and dump found and not found");
			Console.WriteLine("If the selection string is not null, the tracks on which checks have to be performed are those that pass the selection.");
			Console.WriteLine("Known selection variables:");
			Console.WriteLine("N: Number of segments");
			Console.WriteLine("DZ: Downstream Z");
			Console.WriteLine("DSX, DSY: Downstream Slope X,Y");
			Console.WriteLine("D0X, D0Y: Downstream Position X,Y (at Z = 0)");
			Console.WriteLine("DPX, DPY: Downstream Position X,Y (at Z = DZ)");
			Console.WriteLine("UZ: Upstream Z");
			Console.WriteLine("USX, USY: Upstream Slope X,Y");
			Console.WriteLine("U0X, U0Y: Upstream Position X,Y (at Z = 0)");
			Console.WriteLine("UPX, UPY: Upstream Position X,Y (at Z = UZ)");
		}
		/// <summary>
		/// The main entry point for the application.
		/// </summary>
		[STAThread]
		static void Main(string[] args)
		{
			//
			// TODO: Add code to start application here
			//
			if (args.Length != 3 && args.Length != 4)
			{
				ShowExplanation();
				return;
			}
			CheckMode mode;
			bool dumpfound = false, dumpnotfound = false;
			if (String.Compare(args[0], "/3", true) == 0)
			{
				mode = CheckMode.Center;
				dumpfound = false;
				dumpnotfound = false;
			}
			else if (String.Compare(args[0], "/3f", true) == 0)
			{
				mode = CheckMode.Center;
				dumpfound = true;
				dumpnotfound = false;
			}
			else if (String.Compare(args[0], "/3n", true) == 0)
			{
				mode = CheckMode.Center;
				dumpfound = false;
				dumpnotfound = true;
			}
			else if (String.Compare(args[0], "/3a", true) == 0)
			{
				mode = CheckMode.Center;
				dumpfound = true;
				dumpnotfound = true;
			}
			else if (String.Compare(args[0], "/u", true) == 0)
			{
				mode = CheckMode.Upstream;
				dumpfound = false;
				dumpnotfound = false;
			}
			else if (String.Compare(args[0], "/uf", true) == 0)
			{
				mode = CheckMode.Upstream;
				dumpfound = true;
				dumpnotfound = false;
			}
			else if (String.Compare(args[0], "/un", true) == 0)
			{
				mode = CheckMode.Upstream;
				dumpfound = false;
				dumpnotfound = true;
			}
			else if (String.Compare(args[0], "/ua", true) == 0)
			{
				mode = CheckMode.Upstream;
				dumpfound = true;
				dumpnotfound = true;
			}
			else if (String.Compare(args[0], "/d", true) == 0)
			{
				mode = CheckMode.Downstream;	
				dumpfound = false;
				dumpnotfound = false;		
			}
			else if (String.Compare(args[0], "/df", true) == 0)
			{
				mode = CheckMode.Downstream;	
				dumpfound = true;
				dumpnotfound = false;		
			}
			else if (String.Compare(args[0], "/dn", true) == 0)
			{
				mode = CheckMode.Downstream;	
				dumpfound = false;
				dumpnotfound = true;		
			}
			else if (String.Compare(args[0], "/da", true) == 0)
			{
				mode = CheckMode.Downstream;	
				dumpfound = true;
				dumpnotfound = true;		
			}
			else 
			{
				ShowExplanation();
				return;
			}
			SySal.TotalScan.Volume v = (SySal.TotalScan.Volume)SySal.OperaPersistence.Restore(args[1], typeof(SySal.TotalScan.Volume));
			if (v.Layers.Length != 3 && mode == CheckMode.Center)
			{
				Console.WriteLine("The TSR data must contain exactly 3 layers to use this check mode.");
				return;
			}
			else if (v.Layers.Length < 3)
			{
				Console.WriteLine("The TSR data must contain at least 3 layers.");
				return;
			}
			Exe x = new Exe(v);
			x.Check(mode, args[2], (args.Length == 4) ? new NumericalTools.CStyleParsedFunction(args[3]) : null, dumpfound, dumpnotfound);
		}

		NumericalTools.Function SelectionFunction = null;

		CheckMode Mode = CheckMode.Center;

		SySal.TotalScan.Volume Vol = null;

		public Exe(SySal.TotalScan.Volume v)
		{
			Vol = v;
			SelectionFunction = null;
		}

		internal void Check(CheckMode mode, string textout, NumericalTools.Function slf, bool dumpfound, bool dumpnotfound)
		{
			Mode = mode;
			SelectionFunction = slf;
			string [] pars = new string[0];
			if (slf != null)
			{
				pars = slf.ParameterList;
				if (pars.Length == 0) throw new Exception("The selection function must have parameters.");				
			}
			System.IO.StreamWriter wr = null;
			int Searched = 0, Found = 0;
			SySal.DAQSystem.Scanning.IntercalibrationInfo Inv = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
			try
			{				
				wr = new System.IO.StreamWriter(textout);
				wr.WriteLine("ID\tPX\tPY\tSX\tSY\tN");
				SySal.TotalScan.Layer lay = null;
				switch (Mode)
				{
					case CheckMode.Center:	lay = Vol.Layers[1]; break;
					case CheckMode.Upstream:	lay = Vol.Layers[Vol.Layers.Length - 1]; break;
					case CheckMode.Downstream:	lay = Vol.Layers[0]; break;
					default:					throw new Exception("Unsupported mode");
				}
				Inv.RX = Vol.RefCenter.X + lay.AlignData.TranslationX;
				Inv.RY = Vol.RefCenter.Y + lay.AlignData.TranslationY;
				double iden = 1.0 / (lay.AlignData.AffineMatrixXX * lay.AlignData.AffineMatrixYY - lay.AlignData.AffineMatrixXY * lay.AlignData.AffineMatrixYX);
				Inv.MXX = lay.AlignData.AffineMatrixYY * iden;
				Inv.MXY = -lay.AlignData.AffineMatrixXY * iden;
				Inv.MYX = -lay.AlignData.AffineMatrixYX * iden;
				Inv.MYY = lay.AlignData.AffineMatrixXX * iden;
				Inv.TX = - lay.AlignData.TranslationX;
				Inv.TY = - lay.AlignData.TranslationY;
				int tkn = Vol.Tracks.Length;
				int t, p;
				SySal.TotalScan.Track tk;
				string ps;
				double ipx, ipy, isx, isy;
				for (t = 0; t < tkn; t++)
				{
					tk = Vol.Tracks[t];
					if (SelectionFunction != null)
					{
						for (p = 0; p < pars.Length; p++)
						{
							ps = pars[p].ToUpper();
							switch (ps)
							{
								case "N":	SelectionFunction[p] = tk.Length; break;
								case "DZ":	SelectionFunction[p] = tk.Downstream_Z; break;
								case "UZ":	SelectionFunction[p] = tk.Upstream_Z; break;
								case "DSX":	SelectionFunction[p] = tk.Downstream_SlopeX; break;
								case "DSY":	SelectionFunction[p] = tk.Downstream_SlopeY; break;
								case "USX":	SelectionFunction[p] = tk.Upstream_SlopeX; break;
								case "USY":	SelectionFunction[p] = tk.Upstream_SlopeY; break;
								case "D0X":	SelectionFunction[p] = tk.Downstream_PosX - tk.Downstream_SlopeX * tk.Downstream_PosZ; break;
								case "D0Y":	SelectionFunction[p] = tk.Downstream_PosY - tk.Downstream_SlopeY * tk.Downstream_PosZ; break;
								case "U0X":	SelectionFunction[p] = tk.Upstream_PosX - tk.Upstream_SlopeX * tk.Upstream_PosZ; break;
								case "U0Y":	SelectionFunction[p] = tk.Upstream_PosY - tk.Upstream_SlopeY * tk.Upstream_PosZ; break;
								case "DPX":	SelectionFunction[p] = tk.Downstream_PosX + tk.Downstream_SlopeX * (tk.Downstream_Z - tk.Downstream_PosZ); break;
								case "DPY":	SelectionFunction[p] = tk.Downstream_PosY + tk.Downstream_SlopeY * (tk.Downstream_Z - tk.Downstream_PosZ); break;
								case "UPX":	SelectionFunction[p] = tk.Upstream_PosX + tk.Upstream_SlopeX * (tk.Upstream_Z - tk.Upstream_PosZ); break;
								case "UPY":	SelectionFunction[p] = tk.Upstream_PosY + tk.Upstream_SlopeY * (tk.Upstream_Z - tk.Upstream_PosZ); break;
								default:	throw new Exception("Unknown parameter " + ps);
							}							
						}
						if (SelectionFunction.Evaluate() == 0) continue;
					}
					switch (Mode)
					{
						case CheckMode.Center:
						{
							if (tk.UpstreamLayerId == 2 && tk.DownstreamLayerId == 0)
							{
								Searched++;
								if (tk.Length == 3)
								{
									Found++;
									if (dumpfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = tk[1].Info;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
								else
								{
									if (dumpnotfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
										info.Intercept.X = tk.Downstream_PosX + tk.Downstream_SlopeX * (lay.RefCenter.Z - tk.Downstream_PosZ);
										info.Intercept.Y = tk.Downstream_PosY + tk.Downstream_SlopeY * (lay.RefCenter.Z - tk.Downstream_PosZ);
										info.Slope.X = tk.Downstream_SlopeX;
										info.Slope.Y = tk.Downstream_SlopeY;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
							}
						}
						break;

						case CheckMode.Upstream:
						{
							if (tk.DownstreamLayerId == 0 && tk.Length >= (Vol.Layers.Length - 1))
							{
								Searched++;
								if (tk.Length == Vol.Layers.Length)
								{
									Found++;
									if (dumpfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = tk[tk.Length - 1].Info;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
								else
								{
									if (dumpnotfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
										info.Intercept.X = tk.Upstream_PosX + tk.Upstream_SlopeX * (lay.RefCenter.Z - tk.Upstream_PosZ);
										info.Intercept.Y = tk.Upstream_PosY + tk.Upstream_SlopeY * (lay.RefCenter.Z - tk.Upstream_PosZ);
										info.Slope.X = tk.Upstream_SlopeX;
										info.Slope.Y = tk.Upstream_SlopeY;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
							}
						}
						break;

						case CheckMode.Downstream:
						{
							if (tk.UpstreamLayerId == (Vol.Layers.Length - 1) && tk.Length >= (Vol.Layers.Length - 1))
							{
								Searched++;
								if (tk.Length == Vol.Layers.Length)
								{
									Found++;
									if (dumpfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = tk[0].Info;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
								else
								{
									if (dumpnotfound)
									{
										SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
										info.Intercept.X = tk.Downstream_PosX + tk.Downstream_SlopeX * (lay.RefCenter.Z - tk.Downstream_PosZ);
										info.Intercept.Y = tk.Downstream_PosY + tk.Downstream_SlopeY * (lay.RefCenter.Z - tk.Downstream_PosZ);
										info.Slope.X = tk.Downstream_SlopeX;
										info.Slope.Y = tk.Downstream_SlopeY;
										ipx = Inv.MXX * (info.Intercept.X - Inv.RX) + Inv.MXY * (info.Intercept.Y - Inv.RY) + Inv.TX + Inv.RX;
										ipy = Inv.MYX * (info.Intercept.X - Inv.RX) + Inv.MYY * (info.Intercept.Y - Inv.RY) + Inv.TY + Inv.RY;
										isx = Inv.MXX * info.Slope.X + Inv.MXY * info.Slope.Y;
										isy = Inv.MYX * info.Slope.X + Inv.MYY * info.Slope.Y;
										wr.WriteLine(tk.Id + "\t" + ipx.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + ipy.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isx.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + isy.ToString("F4", System.Globalization.CultureInfo.InvariantCulture) + "\t" + tk.Length);
									}
								}
							}
						}
						break;
					}
				}				
				wr.Flush();
			}
			catch(Exception x)
			{
				Console.WriteLine("Unrecoverable error:\n" + x.Message);
			}
			finally
			{				
				if (wr != null)	wr.Close();				
				Console.WriteLine("Searched: " + Searched);
				Console.WriteLine("Found: " + Found);
				if (Searched > 1)
				{
					double p = (double)Found / (double)Searched;
					Console.WriteLine("Efficiency: " + (p * 100.0).ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "% \xb1" + (Math.Sqrt(p * (1.0 - p) / Searched) * 100.0).ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "%");
				}
			}
		}
	}
}
