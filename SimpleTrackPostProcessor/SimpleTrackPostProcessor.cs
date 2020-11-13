using System;
using System.Collections;
using SySal;
using SySal.BasicTypes;
using SySal.Management;
using SySal.Tracking;
using System.Xml.Serialization;


namespace SySal.Processing.SimpleTrackPostProcessing
{
	class MIPEmulsionTrack : SySal.Tracking.MIPEmulsionTrack
	{
		public MIPEmulsionTrack(Grain [] g, MIPEmulsionTrackInfo info) { m_Grains = g; m_Info = info; }

        public SySal.Tracking.MIPEmulsionTrackInfo iInfo { set { m_Info = value; } get { return m_Info; } }
	}

	/// <summary>
	/// Configuration class for SimpleTrackPostProcessor
	/// </summary>
	[Serializable]
	[XmlType("SimpleTrackPostProcessing.Configuration")]
	public class Configuration : SySal.Management.Configuration
	{
		public bool UseTransverseResiduals;
		public bool CleanDoubleReconstructions;
		public double MaxSigma0;
		public double MaxSigmaSlope;
		public double MaxDuplicateDistance;

		public override object Clone()
		{
			Configuration C = new Configuration();
			C.UseTransverseResiduals = UseTransverseResiduals;
			C.CleanDoubleReconstructions = CleanDoubleReconstructions;
			C.MaxSigma0 = MaxSigma0;
			C.MaxSigmaSlope = MaxSigmaSlope;
			C.MaxDuplicateDistance = MaxDuplicateDistance;
			return C;
		}

		public Configuration() : base("") {}

		public Configuration(string name) : base(name) {}
	}

	/// <summary>
	/// Trivial distortion correction. Only shrinkage is accounted for.
	/// </summary>
	[Serializable]
	public class TrivialDistortionCorrection : SySal.Tracking.DistortionCorrection
	{
		/// <summary>
		/// Corrects the grains using shrinkage information.
		/// </summary>
		/// <param name="grains"></param>
		public override void Correct(Grain [] grains)
		{
			int i;
			for (i = 0; i < grains.Length; i++)
				grains[i].Position.Z = (grains[i].Position.Z - ZBaseSurface) * Shrinkage + ZBaseSurface;
		}

		/// <summary>
		/// Initializes the distortion correction with shrinkage and reference depths information.
		/// </summary>
		/// <param name="shrinkage"></param>
		/// <param name="zbasesurface"></param>
		/// <param name="zexternalsurface"></param>
		public TrivialDistortionCorrection(double shrinkage, double zbasesurface, double zexternalsurface) : base(0, null, shrinkage, zbasesurface, zexternalsurface) {}
	}

	/// <summary>
	/// Computes the geometrical parameters of tracks from their grains and optionally sweeps out double measurements.
	/// </summary>
	[Serializable]
	[XmlType("SimpleTrackPostProcessing.SimpleTrackPostProcessor")]
	public class SimpleTrackPostProcessor : IMIPPostProcessor, IManageable, IExposeInfo, IGraphicallyManageable
	{
		#region Internals
		[NonSerialized]
		private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

		[NonSerialized]
		private System.Drawing.Icon ClassIcon;

		[NonSerialized]
		private string intName;

		[NonSerialized]
		private int XLoc, YLoc;

		[NonSerialized]
		private Configuration C;

		[NonSerialized]
		private TrivialDistortionCorrection D;
		#endregion

		#region IManageable
		public string Name
		{
			get
			{
				return intName;	
			}
			set
			{
				intName = value;	
			}
		}

		[XmlElement(typeof(SimpleTrackPostProcessing.Configuration))]
		public SySal.Management.Configuration Config
		{
			get
			{
				return C;
			}
			set
			{
				C = (Configuration)value;
			}
		}

		public bool EditConfiguration(ref SySal.Management.Configuration c)
		{
			bool ret;
			EditConfigForm myform = new EditConfigForm();
			myform.Config = (Configuration)c;
			if ((ret = (myform.ShowDialog() == System.Windows.Forms.DialogResult.OK))) c = myform.Config;
			myform.Dispose();
			return ret;
		}

		[XmlIgnore]
		public IConnectionList Connections
		{
			get
			{
				return EmptyConnectionList;
			}
		}

		[XmlIgnore]
		public bool MonitorEnabled
		{
			get
			{
				return false;
			}
			set
			{
				if (value != false) throw new System.Exception("This object has no monitor.");
			}
		}

		public SimpleTrackPostProcessor()
		{
			C = new Configuration("Default Simple Track PostProcessor Configuration");
			C.UseTransverseResiduals = false;
			C.CleanDoubleReconstructions = true;
			C.MaxSigma0 = 2.0f;
			C.MaxSigmaSlope = 0.0f;
			C.MaxDuplicateDistance = 2.0f;
			intName = "Default Simple Tracker";
			XLoc = 0;
			YLoc = 0;
			System.Resources.ResourceManager resman = new System.Resources.ResourceManager("SimpleTrackPostProcessor.SimpleTrackPostProcessorIcon", this.GetType().Assembly);
			ClassIcon = (System.Drawing.Icon)(resman.GetObject("SimpleTrackPostProcessorIcon"));
		}

		[XmlIgnore]
		public System.Drawing.Icon Icon
		{
			get
			{
				return (System.Drawing.Icon)ClassIcon.Clone();
			}
		}

		public int XLocation
		{
			get
			{
				return XLoc;	
			}
			set
			{
				XLoc = value;
			}
		}

		public int YLocation
		{
			get
			{
				return YLoc;	
			}
			set
			{
				YLoc = value;
			}
		}
		#endregion

		#region IExposeInfo
		/// <summary>
		/// Exposes / hides generation of additional info. This is a dummy implementation, since this class has no information to expose.
		/// </summary>
		[XmlIgnore]
		public bool Expose
		{
			get
			{
				return false;
			}
			set
			{
			}
		}


		/// <summary>
		/// Gets the additional information. This is a dummy implementation, since this class has no information to expose.
		/// </summary>
		[XmlIgnore]
		public System.Collections.ArrayList ExposedInfo
		{
			get
			{
				return new ArrayList();
			}
		}
		#endregion

		#region IDisposable
		public void Dispose()
		{
			if (ClassIcon != null)
			{
				ClassIcon.Dispose();
				ClassIcon = null;
				GC.SuppressFinalize(this);
			}
		}

		~SimpleTrackPostProcessor()
		{
			Dispose();
		}
		#endregion

		#region IMIPPostProcessor
		/// <summary>
		/// Retrieves the DistortionCorrection the PostProcessor used to correct the grains.
		/// </summary>
		public DistortionCorrection DistortionInfo
		{
			get
			{
				return D;
			}
		}

		/// <summary>
		/// Processes the grain sequences supplied, optionally corrects them, and computes the geometrical parameters of each track.
		/// </summary>
		public SySal.Tracking.MIPEmulsionTrack [] Process(Grain [][] grainsequences, double zbasesurf, double zextsurf, double shrinkage, bool correctgrains)
		{
			int totaltks = 0;
			D = new TrivialDistortionCorrection(shrinkage, zbasesurf, zextsurf);
			MIPEmulsionTrack [] temptks = new MIPEmulsionTrack[grainsequences.Length];
			if (temptks.Length == 0) return temptks;
			int i, j;
			int ix, iy, iix, iiy, nx, ny;
			double Slope, dx, dy;
			for (i = 0; i < grainsequences.Length; i++)
			{
				MIPEmulsionTrackInfo info = new MIPEmulsionTrackInfo();
				MIPEmulsionTrack tk = temptks[i] = new MIPEmulsionTrack(grainsequences[i], info);
				info.Count = (ushort)temptks[i].Length;
				info.AreaSum = 0;
				for (j = 0; j < tk.Length; j++)
					info.AreaSum += tk[j].Area;
				info.Field = 0;
				info.TopZ = (tk[0].Position.Z - zbasesurf) * shrinkage + zbasesurf;
				info.BottomZ = (tk[tk.Length - 1].Position.Z - zbasesurf) * shrinkage + zbasesurf;
				info.Slope.Z = 1.0f;
				info.Intercept.Z = zbasesurf;
				double [] Zs = new double[tk.Length];
				double [] Cs = new double[tk.Length];
				double [] Ss = new double[tk.Length];
				double a = 0.0, b = 0.0, range = 0.0;
				for (j = 0; j < info.Count; j++)
				{
					Zs[j] = tk[j].Position.Z;
					Cs[j] = tk[j].Position.X;
					Ss[j] = 1.0f;
				}
				NumericalTools.Fitting.LinearFitDE(Zs, Cs, Ss, ref a, ref b, ref range);
				info.Slope.X = a;
				info.Intercept.X = b + a * tk.Info.Intercept.Z;
				for (j = 0; j < info.Count; j++)
					Cs[j] = tk[j].Position.Y;
				NumericalTools.Fitting.LinearFitDE(Zs, Cs, Ss, ref a, ref b, ref range);
				info.Slope.Y = a;
				info.Intercept.Y = b + a * info.Intercept.Z;
				info.Sigma = 0.0f;
				Slope = Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y);
				if (C.UseTransverseResiduals && Slope > 0.0f)
				{
					Vector2 N;
					N.X = info.Slope.Y / Slope;
					N.Y = - info.Slope.X / Slope;
					for (j = 0; j < tk.Length; j++)
					{
						Grain g = tk[j];
						dx = (info.Intercept.X + (g.Position.Z - info.Intercept.Z) * info.Slope.X - g.Position.X) * N.X;
						dy = (info.Intercept.Y + (g.Position.Z - info.Intercept.Z) * info.Slope.Y - g.Position.Y) * N.Y;
						info.Sigma += dx * dx + dy * dy;
					}
				}
				else
					for (j = 0; j < tk.Length; j++)
					{
						Grain g = tk[j];
						dx = info.Intercept.X + (g.Position.Z - info.Intercept.Z) * info.Slope.X - g.Position.X;
						dy = info.Intercept.Y + (g.Position.Z - info.Intercept.Z) * info.Slope.Y - g.Position.Y;
						info.Sigma += dx * dx + dy * dy;
					}
				info.Sigma = Math.Sqrt(info.Sigma / (info.Count - 1));
				if (info.Sigma > (C.MaxSigma0 + Slope * C.MaxSigmaSlope)) temptks[i] = null;
				else
				{
					totaltks++;
					info.Slope.X /= shrinkage;
					info.Slope.Y /= shrinkage;
				}
			}

			if (C.CleanDoubleReconstructions && totaltks > 0)
			{
				Rectangle ext;
				for (i = 0; i < temptks.Length && temptks[i] == null; i++);
				ext.MinX = ext.MaxX = temptks[i].Info.Intercept.X;
				ext.MinY = ext.MaxY = temptks[i].Info.Intercept.Y;
				for (; i < temptks.Length; i++)
					if (temptks[i] != null)
					{
						if (ext.MinX > temptks[i].Info.Intercept.X) ext.MinX = temptks[i].Info.Intercept.X;
						else if (ext.MaxX < temptks[i].Info.Intercept.X) ext.MaxX = temptks[i].Info.Intercept.X;
						if (ext.MinY > temptks[i].Info.Intercept.Y) ext.MinY = temptks[i].Info.Intercept.Y;
						else if (ext.MaxY < temptks[i].Info.Intercept.Y) ext.MaxY = temptks[i].Info.Intercept.Y;
					}
				double cellsize = Math.Max(C.MaxDuplicateDistance, (ext.MaxX - ext.MinX) * (ext.MaxY - ext.MinY) / totaltks); 
				double dzsurf = zextsurf - zbasesurf;
				ext.MinX -= cellsize;
				ext.MaxX += 2 * cellsize;
				ext.MinY -= cellsize;
				ext.MaxY += 2 * cellsize;
				nx = (int)((ext.MaxX - ext.MinX) / cellsize);
				ny = (int)((ext.MaxY - ext.MinY) / cellsize);
				ArrayList [,] cells = new ArrayList[nx, ny];
				foreach (MIPEmulsionTrack tk in temptks)
					if (tk != null)
					{
						ix = (int)((tk.Info.Intercept.X - ext.MinX) / cellsize);
						iy = (int)((tk.Info.Intercept.Y - ext.MinY) / cellsize);
						if (cells[ix, iy] == null) cells[ix, iy] = new ArrayList(4);
						cells[ix, iy].Add(tk);
					}
				for (ix = 1; ix < (nx - 1); ix++)
					for (iy = 1; iy < (ny - 1); iy++)
						if (cells[ix, iy] != null)
						{
							int itrack1, itrack2;
							for (iix = ix - 1; iix <= ix + 1; iix++)
								for (iiy = iy - 1; iiy <= iy + 1; iiy++)
								{
									if (cells[iix, iiy] == null) continue;
									for (itrack1 = 0; itrack1 < cells[ix, iy].Count; itrack1++)
									{
										MIPEmulsionTrack tk1 = (MIPEmulsionTrack)cells[ix, iy][itrack1];
										for (itrack2 = (iix == ix && iiy == iy) ? (itrack1 + 1) : 0; itrack2 < cells[iix, iiy].Count; itrack2++)
										{
											MIPEmulsionTrack tk2 = (MIPEmulsionTrack)cells[iix, iiy][itrack2];
											Vector2 DBase, DExt;
											DBase.X = tk1.Info.Intercept.X - tk2.Info.Intercept.X;
											DBase.Y = tk1.Info.Intercept.Y - tk2.Info.Intercept.Y;
											DExt.X = DBase.X + dzsurf * (tk1.Info.Slope.X - tk2.Info.Slope.X);
											DExt.Y = DBase.Y + dzsurf * (tk1.Info.Slope.Y - tk2.Info.Slope.Y);
											if (Math.Max(Math.Max(DBase.X, DBase.Y), Math.Max(DExt.X, DExt.Y)) < C.MaxDuplicateDistance)
											{
												if (tk1.iInfo.Count >= tk2.iInfo.Count)
												{
													tk2.iInfo.Sigma = -1.0;                                                    
												}
												else
												{
													tk1.iInfo.Sigma = -1.0;
												}
											}
										
										}
									}
								}

						}
				totaltks = 0;
                foreach (ArrayList ar in cells)
                    if (ar != null)
                        foreach (MIPEmulsionTrack tk in ar)
                            if (tk.Info.Sigma >= 0.0f)
                                totaltks++;
                temptks = new MIPEmulsionTrack[totaltks];
                totaltks = 0;
				foreach (ArrayList ar in cells)
					if (ar != null)
						foreach (MIPEmulsionTrack tk in ar)
							if (tk.Info.Sigma >= 0.0f)
								temptks[totaltks++] = tk;
			}
			else
			{
				MIPEmulsionTrack [] newtemptks = new MIPEmulsionTrack[totaltks];
				for (i = j = 0; i < temptks.Length; i++)
					if (temptks[i] != null)
						newtemptks[j++] = temptks[i];
				temptks = newtemptks;
			}

			if (correctgrains)
			{
				foreach (MIPEmulsionTrack tk in temptks)
					for (i = 0; i < tk.Info.Count; i++)
						tk[i].Position.Z = (tk[i].Position.Z - zbasesurf) * shrinkage + zbasesurf;
			}

			return temptks;
		}
		#endregion
	}
}
