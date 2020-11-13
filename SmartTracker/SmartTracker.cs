#define ENABLE_BITMAP

using System;
using System.Collections;
using System.Runtime.Serialization;
using SySal.BasicTypes;
using SySal.Management;
using System.Xml.Serialization;
using SySal.Tracking;

namespace SySal.Processing.SmartTracking
{
	/// <summary>
	/// Configuration class for SmartTracker
	/// </summary>
	[Serializable]
	[XmlType("SmartTracking.Configuration")]
	public class Configuration : SySal.Management.Configuration
	{
        /// <summary>
        /// A track search starts when one or more grains on the TriggerLayers are found aligned with the line of two grains on TopLayer and BottomLayer.
        /// </summary>
		public struct TriggerInfo
		{
            /// <summary>
            /// Grains on this top layer define one end of the line.
            /// </summary>
			public uint TopLayer;
            /// <summary>
            /// Grains on this bottom layer define one end of the line.
            /// </summary>
			public uint BottomLayer;
            /// <summary>
            /// The list of layers to search for aligned grains.
            /// </summary>
			public uint [] TriggerLayers;
            /// <summary>
            /// Builds a new TriggerInfo.
            /// </summary>
            /// <param name="toplayer">top layer for the trigger.</param>
            /// <param name="bottomlayer">bottom layer for the trigger.</param>
            /// <param name="triggerlayers">set of trigger layers.</param>
            public TriggerInfo(uint toplayer, uint bottomlayer, uint[] triggerlayers)
            {
                TopLayer = toplayer;
                BottomLayer = bottomlayer;
                TriggerLayers = triggerlayers;
            }
		}

        /// <summary>
        /// Maximum slope sought.
        /// </summary>
        /// <remarks>Tracks with higher slope might be formed and accepted.</remarks>
		public double MaxSlope;
        /// <summary>
        /// Minimum slope sought.
        /// </summary>
		public double MinSlope;
        /// <summary>
        /// Transverse alignment tolerance.
        /// </summary>
		public double AlignTol;
		/// <summary>
		/// The triggers for track formation.
		/// </summary>
		public TriggerInfo [] Triggers;
        /// <summary>
        /// Minimum number of grains for vertical tracks.
        /// </summary>
		public double MinGrainsForVerticalTrack = 6;
        /// <summary>
        /// Minimum number of grains for horizontal tracks.
        /// </summary>
		public double MinGrainsForHorizontalTrack = 6;
        /// <summary>
        /// Minimum number of grains for tracks at tan theta = 0.1 from the vertical.
        /// </summary>
		public double MinGrainsSlope01 = 6;
        /// <summary>
        /// Minimum area of clusters for tracking.
        /// </summary>
		public uint MinArea;
        /// <summary>
        /// Maximum area of clusters for tracking.
        /// </summary>
		public uint MaxArea;
        /// <summary>
        /// Maximum number of grains in a cell.
        /// </summary>
		public uint CellOverflow;
        /// <summary>
        /// Cells in X direction.
        /// </summary>
		public uint CellNumX;
        /// <summary>
        /// Cells in Y direction.
        /// </summary>
        public uint CellNumY;
        /// <summary>
        /// Maximum allowed tracking time in ms; if exceeded, no tracks are produced.
        /// </summary>
        public int MaxTrackingTimeMS;
        /// <summary>
        /// Maximum number of processors to used; a value of <c>0</c> can be specified to ask for an automatic decision.
        /// </summary>
        public uint MaxProcessors;
        /// <summary>
        /// Maximum transverse distance of two replicas of the same grain.
        /// </summary>
        public double ReplicaRadius;
        /// <summary>
        /// Only one grain every <c>ReplicaSampleDivider</c> is used to search for replicas.
        /// </summary>
        public uint ReplicaSampleDivider;
        /// <summary>
        /// Minimum number of replicas to be used for grain map correction.
        /// </summary>
        public uint MinReplicas;
        /// <summary>
        /// Number of grains a track can have on a single plane.
        /// </summary>
        public uint InitialMultiplicity;
        /// <summary>
        /// The longitudinal alignment tolerance is <c>AlignTol + Slope*DeltaZMultiplier</c>; 1 is the default value.
        /// </summary>
        public double DeltaZMultiplier = 1.0;
        /// <summary>
        /// If <c>true</c>, two or more tracks can share one or more grains.
        /// </summary>
        public bool AllowOverlap;
        /// <summary>
        /// Produces a copy of this Configuration.
        /// </summary>
        /// <returns>a copy of this Configuration.</returns>
		public override object Clone()
		{
			int i, j;
			Configuration C = new Configuration();
            C.AlignTol = AlignTol;
            C.AllowOverlap = AllowOverlap;
            C.CellNumX = CellNumX;
            C.CellNumY = CellNumY;
            C.CellOverflow = CellOverflow;
            C.DeltaZMultiplier = DeltaZMultiplier;
            C.InitialMultiplicity = InitialMultiplicity;
            C.MaxArea = MaxArea;
            C.MaxSlope = MaxSlope;
            C.MaxTrackingTimeMS = MaxTrackingTimeMS;
            C.MaxProcessors = MaxProcessors;
            C.MinArea = MinArea;
            C.MinGrainsForHorizontalTrack = MinGrainsForHorizontalTrack;
            C.MinGrainsForVerticalTrack = MinGrainsForVerticalTrack;
            C.MinGrainsSlope01 = MinGrainsSlope01;
            C.MinReplicas = MinReplicas;
            C.MinSlope = MinSlope;
            C.Name = (string)Name.Clone();
            C.ReplicaRadius = ReplicaRadius;
            C.ReplicaSampleDivider = ReplicaSampleDivider;            
            
			C.Triggers = new TriggerInfo[Triggers.Length];
			for (i = 0; i < Triggers.Length; i++)
			{
				C.Triggers[i].TopLayer = Triggers[i].TopLayer;
				C.Triggers[i].BottomLayer = Triggers[i].BottomLayer;
				C.Triggers[i].TriggerLayers = new uint[Triggers[i].TriggerLayers.Length];
				for (j = 0; j < Triggers[i].TriggerLayers.Length; j++)
					C.Triggers[i].TriggerLayers[j] = Triggers[i].TriggerLayers[j];
			}
			return C;
		}
        /// <summary>
        /// Builds a new Configuration.
        /// </summary>
		public Configuration() : base("") { Triggers = new TriggerInfo[0]; }
        /// <summary>
        /// Builds a new Configuration with a specified name.
        /// </summary>
        /// <param name="name">the name to be given to the Configuration.</param>
		public Configuration(string name) : base(name) { Triggers = new TriggerInfo[0]; }

	}

	/// <summary>
	/// SmartTracker track finding class.
	/// </summary>
	[Serializable]
	[XmlType("SmartTracking.SmartTracker")]
	public class SmartTracker : IMIPTracker, IManageable, IExposeInfo, IGraphicallyManageable
	{
		#region Internals
		[NonSerialized]
		private int counter_triggers;

        [NonSerialized]
        private int counter_overflowcells;

        [NonSerialized]
		private int counter_trials;
		
		[NonSerialized]
		private int counter_grains;

		[NonSerialized]
		private Configuration C;

		[NonSerialized]
		private bool TrackingAreaSet;

		[NonSerialized]
		private Vector2 CellMin, CellMax;

        [NonSerialized]
        private double DxCell, DyCell;

		[NonSerialized]
		private Cell [][,] Cells;        

		[NonSerialized]
		private string intName;

		[NonSerialized]
		private bool ActivateExpose;

		[NonSerialized]
		private int XLoc, YLoc;

		[NonSerialized]
		private System.Drawing.Icon ClassIcon =	(System.Drawing.Icon)(new System.Resources.ResourceManager("SmartTracker.SmartTrackerIcon", typeof(SmartTracker).Assembly).GetObject("SmartTrackerIcon"));

		[NonSerialized]
		private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        /// <summary>
        /// Grain position correction information.
        /// </summary>
        public struct GrainCorrectionInfo
        {
            /// <summary>
            /// Layer.
            /// </summary>
            public int Layer;
            /// <summary>
            /// Average displacement in X/Y.
            /// </summary>
            public double AvgLX, AvgLY;
            /// <summary>
            /// 0-th and 1-st order displacements in X/Y.
            /// </summary>
            public double L0X, L1X, L0Y, L1Y;
            /// <summary>
            /// Grains used for correction.
            /// </summary>
            public int Count;
            /// <summary>
            /// Layer Z.
            /// </summary>
            public double Z;
            /// <summary>
            /// Represents the grain correction information in string format.
            /// </summary>
            /// <returns></returns>
            public override string ToString()
            {
                return "Layer " + Layer + " Count " + Count + " AvgLX/Y " + AvgLX + "/" + AvgLY + " L0X/Y " + L0X + "/" + L0Y + " L1X/Y " + L1X + "/" + L1Y;
            }
        }

        [NonSerialized]
        private GrainCorrectionInfo[] LayerCorrections;

		/// <summary>
		/// Track grains include track quality information
		/// </summary>
		internal class TrackGrain2 : Grain2
		{
			public int TrackLength;
			public int TrackIndex;
            public Grain2 Grain2;
            public double Z;

			public TrackGrain2(Grain2 g, double z)
			{
				base.Area = g.Area;
				base.Position = g.Position;
                Grain2 = g;
				TrackLength = 0;
				TrackIndex = 0;
                Z = z;
			}
		}
		
        private double[] ZLayers;
        
		private class Cell
		{
			public uint Fill;
            public TrackGrain2[] Grains;

            public Cell(uint capacity)
			{
				Grains = new TrackGrain2[capacity];
				Fill = 0;
			}

			public int FindGrains(double x, double y, double dirx, double diry, double dirtol, double normtol, TrackGrain2 [] dest, double [] dist)
			{
                int i, j, found = 0;
                double normdist, dirdist;
                double dx, dy;
                double reddirtol = dirtol * 1.5;
                double _reddirtol = -reddirtol;
                double _normtol = -normtol;
                double _dirtol = -dirtol;
                foreach (TrackGrain2 g in Grains)
                {
                    if (g == null) break;
                    if ((dx = (g.Position.X - x)) <= reddirtol && (dx >= _reddirtol))
                        if ((dy = (g.Position.Y - y)) <= reddirtol && (dy >= _reddirtol))
                            if ((normdist = (dy * dirx - dx * diry)) <= normtol && (normdist >= _normtol))
                                if ((dirdist = (dx * dirx + dy * diry)) <= dirtol && (dirdist >= _dirtol))
                                {
                                    if (normdist < 0.0) normdist = -normdist;
                                    for (i = 0; i < dist.Length && dist[i] <= normdist; i++) ;
                                    if (i == dist.Length) continue;
                                    for (j = dist.Length - 1; j > i; j--)
                                    {
                                        dist[j] = dist[j - 1];
                                        dest[j] = dest[j - 1];
                                    }
                                    dest[i] = g;
                                    dist[i] = normdist;
                                    found++;
                                }
                }
                return found;
			}

            public TrackGrain2 FindGrain(double x, double y, double dirx, double diry, double dirtol, double normtol)
            {
                double normdist, dirdist;
                double dx, dy;
                double reddirtol = dirtol * 1.5;
                double _reddirtol = -reddirtol;
                double _normtol = -normtol;
                double _dirtol = -dirtol;
                foreach (TrackGrain2 g in Grains)
                {                    
                    if (g == null) break;
                    if ((dx = (g.Position.X - x)) <= reddirtol && (dx >= _reddirtol))
                        if ((dy = (g.Position.Y - y)) <= reddirtol && (dy >= _reddirtol))
                            if ((normdist = (dy * dirx - dx * diry)) <= normtol && (normdist >= _normtol))
                                if ((dirdist = (dx * dirx + dy * diry)) <= dirtol && (dirdist >= _dirtol))
                                    return g;
                }
                return null;
            }
		}

        private bool[][] PresenceBitmap;
        private double BitmapDCell;
        private int BitmapStrideX, BitmapStrideY;

        private int FindGrains(double expx, double expy, Cell[,] layer, double dirx, double diry, double dirtol, TrackGrain2[] dest, int start, int mult)
        {
            int i, f;
            TrackGrain2[] idest = new TrackGrain2[mult];
            double [] dist = new double[mult];
            for (i = 0; i < dist.Length; i++) dist[i] = C.AlignTol;
            double fix, fiy;
            int eix = (int)(fix = ((expx - CellMin.X) / DxCell));
//            if (eix < 0 || eix >= C.CellNumX) return 0;
            int eiy = (int)(fiy = ((expy - CellMin.Y) / DyCell));
//            if (eiy < 0 || eiy >= C.CellNumY) return 0;

            f = layer[eix, eiy].FindGrains(expx, expy, dirx, diry, dirtol, C.AlignTol, idest, dist);            
            fix -= (0.5 + eix);
            fiy -= (0.5 + eiy);
            if (Math.Abs(fix) >= Math.Abs(fiy))
            {
                if (fix < 0.0 && eix > 0)
                    f += layer[eix - 1, eiy].FindGrains(expx, expy, dirx, diry, dirtol, C.AlignTol, idest, dist);
                else if (fix > 0.0 && eix < C.CellNumX - 1)
                    f += layer[eix + 1, eiy].FindGrains(expx, expy, dirx, diry, dirtol, C.AlignTol, idest, dist);
            }
            else
            {
                if (fiy < 0.0 && eiy > 0)
                    f += layer[eix, eiy - 1].FindGrains(expx, expy, dirx, diry, dirtol, C.AlignTol, idest, dist);
                else if (fiy > 0.0 && eiy < C.CellNumY - 1)
                    f += layer[eix, eiy + 1].FindGrains(expx, expy, dirx, diry, dirtol, C.AlignTol, idest, dist);
            }
            for (i = 0; i < idest.Length; i++) dest[i + start] = idest[i];
            return f;
        }
        		
		#endregion

		#region IMIPTracker
		/// <summary>
		/// The area in the field of view where the tracker operates.
		/// </summary>
		[XmlIgnore]
		public Rectangle TrackingArea
		{
			get
			{
                /*
				SySal.BasicTypes.Rectangle r;
				r.MinX = CellMin.X;
				r.MaxX = CellMax.X;
				r.MinY = CellMin.Y;
				r.MaxY = CellMax.Y;
				return r;
                 */
                return TrackArea;
			}

			set
			{
                TrackArea = value;
                /*
                CellMin.X = value.MinX;
				CellMin.Y = value.MinY;
				CellMax.X = value.MaxX;
				CellMax.Y = value.MaxY;
                 * */
				Cells = null;
                PresenceBitmap = null;
				TrackingAreaSet = true;
			}
		}

        private Rectangle TrackArea;

        internal Vector2 Pix2Micron;

        bool Pix2MicronSet = false;

        /// <summary>
        /// Pixel-to-micron conversion factors.
        /// </summary>
        public Vector2 Pixel2Micron
        {
            get
            {
                Vector2 v;
                v.X = Pix2Micron.X;
                v.Y = Pix2Micron.Y;
                return v;
            }
            set
            {
                Pix2Micron.X = value.X;
                Pix2Micron.Y = value.Y;
                Pix2MicronSet = true;
            }
        }

		/// <summary>
		/// Finds tracks as grain sequences.
		/// </summary>
        /// <param name="tomography">images as planes of grains.</param>
        /// <param name="istopside"><c>true</c> for top side, <c>false</c> for bottom side.</param>
        /// <param name="maxtracks">maximum number of tracks to produce.</param>
        /// <param name="enablepresetslope">if <c>true</c>, enables using a preset track slope, with limited slope acceptance.</param>
        /// <param name="presetslope">preselected slope of tracks to be found.</param>
        /// <param name="presetslopeacc">slope acceptances for preselected track slopes.</param>
		public Grain [][] FindTracks(GrainPlane [] tomography, bool istopside, int maxtracks, bool enablepresetslope, Vector2 presetslope, Vector2 presetslopeacc)
        {
            if (!TrackingAreaSet) throw new Exception("Tracking area not set.");
            if (!Pix2MicronSet) throw new Exception("Pixel-to-micron conversion not set.");
            PutClusters(tomography);
            uint processors = (uint)System.Environment.ProcessorCount;
            if (C.MaxProcessors > 0 && C.MaxProcessors < processors) processors = C.MaxProcessors;
            System.Threading.Thread [] threads = new System.Threading.Thread[processors];
            int p, i, j, k;
            System.DateTime timelimit = System.DateTime.Now.AddMilliseconds(C.MaxTrackingTimeMS);
            TrackGrain2[][][] temptracks = new TrackGrain2[processors][][];
            uint [] tracksfound = new uint[processors];
            for (p = 0; p < processors; p++)
            {
                temptracks[p] = new TrackGrain2[maxtracks][];
                if (p > 0)
                {
                    threads[p] = new System.Threading.Thread((System.Threading.ParameterizedThreadStart)delegate(object q)
                        {
                            int pp = (int)q;
                            PartialGetTracks(0, (uint)tomography.Length - 1, (uint)processors, (uint)pp, temptracks[pp], ref tracksfound[pp], enablepresetslope, presetslope, presetslopeacc, timelimit);
                        }
                    );
                    threads[p].Start(p);
                }
                else threads[0] = System.Threading.Thread.CurrentThread;
            }
            PartialGetTracks(0, (uint)tomography.Length - 1, (uint)processors, 0, temptracks[0], ref tracksfound[0], enablepresetslope, presetslope, presetslopeacc, timelimit);
            for (p = 1; p < processors; p++)
                threads[p].Join();

            int totalfound = 0;
            if (C.AllowOverlap)
            {
                for (p = 0; p < processors; p++)                    
                    totalfound += (int)tracksfound[p];
            }
            else
            {
                for (p = 0; p < processors; p++)
                    for (i = 0; i < (int)tracksfound[p]; i++)
                    {
                        TrackGrain2[] K = temptracks[p][i];
                        if (K == null) continue;
                        for (j = 0; j < K.Length && K[j].TrackLength < K.Length; j++) ;
                        if (j < K.Length)
                        {
                            temptracks[p][i] = null;
                            continue;
                        }
                        totalfound++;
                        for (j = 0; j < K.Length; j++)
                        {
                            if (K[j].TrackIndex > 0)
                            {
                                TrackGrain2[] N = temptracks[(K[j].TrackIndex - 1) / maxtracks][(K[j].TrackIndex - 1) % maxtracks];
                                if (N == null) continue;
                                for (k = 0; k < N.Length; k++)
                                {
                                    N[k].TrackIndex = 0;
                                    N[k].TrackLength = 0;
                                    temptracks[p][K[j].TrackIndex] = null;
                                    totalfound--;
                                }
                            }
                        }
                        for (j = 0; j < K.Length; j++)
                        {
                            K[j].TrackIndex = p * maxtracks + i + 1;
                            K[j].TrackLength = K.Length;
                        }

                    }
            }

            Grain[][] finaltracks = new Grain[totalfound][];

            j = 0;
            double avglx = 0.0;
            double avgly = 0.0;
            double dz;
            if (istopside)
            {
                foreach (GrainCorrectionInfo ginfo in LayerCorrections)
                {
                    avglx += ginfo.AvgLX;
                    avgly += ginfo.AvgLY;
                }
                avglx /= (LayerCorrections.Length - 1);
                avgly /= (LayerCorrections.Length - 1);

                for (p = 0; p < processors; p++)
                    for (i = 0; i < (int)tracksfound[p]; i++)
                    {
                        TrackGrain2[] K = temptracks[p][i];
                        if (K == null) continue;
                        Grain[] gs = new Grain[K.Length];
                        for (k = 0; k < K.Length; k++)
                        {
                            Grain g = new Grain();
                            TrackGrain2 tg2 = K[k];
                            g.Area = tg2.Area;
                            g.Position.X = tg2.Position.X - avglx;
                            g.Position.Y = tg2.Position.Y - avgly;
                            g.Position.Z = tg2.Z;
                            gs[k] = g;
                        }
                        finaltracks[j++] = gs;
                    }
            }
            else
            {
                for (p = 0; p < processors; p++)
                    for (i = 0; i < (int)tracksfound[p]; i++)
                    {
                        TrackGrain2[] K = temptracks[p][i];
                        if (K == null) continue;
                        Grain[] gs = new Grain[K.Length];
                        for (k = 0; k < K.Length; k++)
                        {
                            Grain g = new Grain();
                            TrackGrain2 tg2 = K[k];
                            g.Area = tg2.Area;
                            g.Position.X = tg2.Position.X;
                            g.Position.Y = tg2.Position.Y;
                            g.Position.Z = tg2.Z;
                            gs[k] = g;
                        }
                        finaltracks[j++] = gs;
                    }
            }

            return finaltracks;
        }

        private void PutClusters(GrainPlane[] tomography)
        {
            unchecked
            {
                if (Cells == null || Cells.Length != tomography.Length)
                {
                    Cells = new Cell[tomography.Length][,];
                    PresenceBitmap = new bool[tomography.Length][];
                }
                ZLayers = new double[tomography.Length];
                int Layer;
                counter_grains = counter_overflowcells = 0;
                double DeltaDX = Math.Abs(Pixel2Micron.X);
                double DeltaDY = Math.Abs(Pixel2Micron.Y);
                int DeltaBinsX = (int)Math.Floor(C.ReplicaRadius / DeltaDX + 1.0) * 2 + 5;
                int DeltaBinsY = (int)Math.Floor(C.ReplicaRadius / DeltaDY + 1.0) * 2 + 5;
                int[,] DeltaHisto = new int[DeltaBinsX, DeltaBinsY];
                double PixelToMicronX = Pix2Micron.X;
                double PixelToMicronY = Pix2Micron.Y;
                double PixelToMicronXY = Math.Abs(PixelToMicronX * PixelToMicronY);
                double PixelToMicronX3Y = PixelToMicronX * PixelToMicronX * PixelToMicronXY;
                double PixelToMicronY3X = PixelToMicronY * PixelToMicronY * PixelToMicronXY;
                double PixelToMicronX2Y2 = PixelToMicronXY * PixelToMicronXY;
                int ReplicaDivider = (int)C.ReplicaSampleDivider;
                int ReplicaCount;
                double ReplicaRadius = C.ReplicaRadius;
                double _ReplicaRadius = -ReplicaRadius;
                double CX, CY, DX, DY;
                double L0X, L1X, L0Y, L1Y;
                double AvgLX, AvgLY;
                uint PixMin = C.MinArea;
                uint PixMax = C.MaxArea;
                uint CellNumX = C.CellNumX;
                uint CellNumY = C.CellNumY;
                uint CellOverflow = C.CellOverflow;
                int hix, hiy, pix, piy;
                int ic, ir;
                int DividerCount;
                int maxhisto;
                double[] Xs;
                double[] Ys;
                double[] DXs;
                double[] DYs;
                double CurrDispX, CurrDispY;
                double TotalLX = 0.0;
                double TotalLY = 0.0;

                if (Pix2Micron.X > 0.0)
                {
                    CellMin.X = TrackArea.MinX;
                    CellMax.X = TrackArea.MaxX;
                }
                else
                {
                    CellMin.X = TrackArea.MaxX;
                    CellMax.X = TrackArea.MinX;
                }
                CellMin.X *= Pix2Micron.X;
                CellMax.X *= Pix2Micron.X;
                if (Pix2Micron.Y > 0.0)
                {
                    CellMin.Y = TrackArea.MinY;
                    CellMax.Y = TrackArea.MaxY;
                }
                else
                {
                    CellMin.Y = TrackArea.MaxY;
                    CellMax.Y = TrackArea.MinY;
                }
                CellMin.Y *= Pix2Micron.Y;
                CellMax.Y *= Pix2Micron.Y;
                double ViewCenterX = 0.5 * (CellMin.X + CellMax.X);
                double ViewCenterY = 0.5 * (CellMin.Y + CellMax.Y);

                DxCell = (CellMax.X - CellMin.X) / CellNumX;
                DyCell = (CellMax.Y - CellMin.Y) / CellNumY;
                BitmapDCell = C.AlignTol + C.MaxSlope * Math.Abs((tomography[0].Z - tomography[tomography.Length - 1].Z) / (tomography.Length - 1));
                BitmapStrideX = (int)Math.Ceiling((CellMax.X - CellMin.X) / BitmapDCell);
                BitmapStrideY = (int)Math.Ceiling((CellMax.Y - CellMin.Y) / BitmapDCell);
                int maxgrains = 0;
                foreach (GrainPlane pl in tomography)
                    if (maxgrains < pl.Grains.Length)
                        maxgrains = pl.Grains.Length;
                Xs = new double[maxgrains];
                Ys = new double[maxgrains];
                DXs = new double[maxgrains];
                DYs = new double[maxgrains];

                if (LayerCorrections == null || LayerCorrections.Length != tomography.Length)
                {
                    LayerCorrections = new GrainCorrectionInfo[tomography.Length];
                    for (Layer = 0; Layer < LayerCorrections.Length; Layer++) LayerCorrections[Layer].Layer = Layer;
                }

                for (Layer = 0; Layer < tomography.Length; Layer++)
                {
                    GrainPlane tlayer = tomography[Layer];
                    Cell[,] clayer;
                    bool[] pblayer;
                    if (Cells[Layer] == null)
                    {
                        clayer = Cells[Layer] = new Cell[CellNumX, CellNumY];
                        for (pix = 0; pix < CellNumX; pix++)
                            for (piy = 0; piy < CellNumY; piy++)
                                clayer[pix, piy] = new Cell(CellOverflow);
                        PresenceBitmap[Layer] = pblayer = new bool[BitmapStrideX * BitmapStrideY];
                    }
                    else
                    {
                        clayer = Cells[Layer];
                        foreach (Cell c in clayer)
                        {
                            for (ic = 0; ic < c.Grains.Length; ic++) c.Grains[ic] = null;
                            c.Fill = 0;
                        }
                        pblayer = PresenceBitmap[Layer];
                        for (ir = 0; ir < pblayer.Length; ir++) pblayer[ir] = false;
                    }
                    ZLayers[Layer] = tlayer.Z;
                    if (Layer > 0 && ZLayers[Layer] > ZLayers[Layer - 1]) throw new Exception("Planes must be in descending Z order; inversion found at layer " + Layer + ".");

                    L0X = L0Y = 0.0;
                    L1X = L1Y = 1.0;
                    AvgLX = AvgLY = 0.0;
                    DividerCount = ReplicaCount = 0;
                    if (ReplicaRadius > 0 && Layer > 0)
                    {
                        CurrDispX = CurrDispY = 0.0;
                        ReplicaCount = 0;
                        for (hix = 0; hix < DeltaBinsX; hix++)
                            for (hiy = 0; hiy < DeltaBinsY; hiy++)
                                DeltaHisto[hix, hiy] = 0;
                        foreach (Grain2 g in tlayer.Grains)
                        {
                            CX = g.Position.X * PixelToMicronX;
                            CY = g.Position.Y * PixelToMicronY;
                            if (g.Area >= PixMin && g.Area <= PixMax && CX >= CellMin.X && CX <= CellMax.X && CY >= CellMin.Y && CY <= CellMax.Y)
                            {
                                counter_grains++;
                                pix = (int)((CX - CellMin.X) / DxCell);
                                piy = (int)((CY - CellMin.Y) / DyCell);
                                Cell thiscell = clayer[pix, piy];
                                if (thiscell.Fill < CellOverflow)
                                {
                                    TrackGrain2 tg = thiscell.Grains[thiscell.Fill++] = new TrackGrain2(g, ZLayers[Layer]);
                                    tg.Position.X = CX;
                                    tg.Position.Y = CY;
                                }
                                if (++DividerCount == ReplicaDivider)
                                {
                                    DividerCount = 0;
                                    Cell repcell = Cells[Layer - 1][pix, piy];
                                    for (ir = 0; ir < repcell.Fill; ir++)
                                    {
                                        TrackGrain2 r = repcell.Grains[ir];
                                        if (((DX = (r.Position.X - CX)) < ReplicaRadius) && (DX > _ReplicaRadius) && ((DY = (r.Position.Y - CY)) < ReplicaRadius) && (DY > _ReplicaRadius))
                                        {
                                            Xs[ReplicaCount] = CX;
                                            Ys[ReplicaCount] = CY;
                                            DXs[ReplicaCount] = DX;
                                            DYs[ReplicaCount] = DY;
                                            DeltaHisto[(int)Math.Floor(DX / DeltaDX + 0.5) + DeltaBinsX / 2, (int)Math.Floor(DY / DeltaDY + 0.5) + DeltaBinsY / 2]++;
                                            ReplicaCount++;
                                            break;
                                        }
                                    }
                                }
                            }
                        }
                        maxhisto = 0;
                        hix = DeltaBinsX / 2;
                        hiy = DeltaBinsY / 2;
                        for (pix = 0; pix < DeltaBinsX; pix++)
                            for (piy = 0; piy < DeltaBinsY; piy++)
                                if (DeltaHisto[pix, piy] > DeltaHisto[hix, hiy])
                                {
                                    hix = pix;
                                    hiy = piy;
                                }
                        if (DeltaHisto[hix, hiy] > 0)
                            if ((maxhisto =
                                DeltaHisto[hix - 1, hiy - 1] + DeltaHisto[hix - 1, hiy] + DeltaHisto[hix - 1, hiy + 1] +
                                DeltaHisto[hix, hiy - 1] + DeltaHisto[hix, hiy] + DeltaHisto[hix, hiy + 1] +
                                DeltaHisto[hix + 1, hiy - 1] + DeltaHisto[hix + 1, hiy] + DeltaHisto[hix + 1, hiy + 1]) > C.MinReplicas)
                            {
                                DX = (hix - DeltaBinsX / 2) * DeltaDX;
                                DY = (hiy - DeltaBinsY / 2) * DeltaDY;
                                CurrDispX =
                                    (DX * (DeltaHisto[hix, hiy] + DeltaHisto[hix, hiy - 1] + DeltaHisto[hix, hiy + 1]) +
                                    (DX - DeltaDX) * (DeltaHisto[hix - 1, hiy] + DeltaHisto[hix - 1, hiy - 1] + DeltaHisto[hix - 1, hiy + 1]) +
                                    (DX + DeltaDX) * (DeltaHisto[hix + 1, hiy] + DeltaHisto[hix + 1, hiy - 1] + DeltaHisto[hix + 1, hiy + 1])) /
                                    maxhisto;
                                CurrDispY =
                                    (DY * (DeltaHisto[hix, hiy] + DeltaHisto[hix - 1, hiy] + DeltaHisto[hix + 1, hiy]) +
                                    (DY - DeltaDY) * (DeltaHisto[hix, hiy - 1] + DeltaHisto[hix - 1, hiy - 1] + DeltaHisto[hix + 1, hiy - 1]) +
                                    (DY + DeltaDY) * (DeltaHisto[hix, hiy + 1] + DeltaHisto[hix - 1, hiy + 1] + DeltaHisto[hix + 1, hiy + 1])) /
                                    maxhisto;
                                int rsp, rv;
                                double SumPx = 0.0, SumX = 0.0, SumX2 = 0.0, SumXPx = 0.0, DenX = 0.0;
                                double SumPy = 0.0, SumY = 0.0, SumY2 = 0.0, SumYPy = 0.0, DenY = 0.0;
                                double V, W;
                                for (rsp = rv = 0; rsp < ReplicaCount; rsp++)
                                    if (Math.Abs(DXs[rsp] - CurrDispX) < DeltaDX && Math.Abs(DYs[rsp] - CurrDispY) < DeltaDY)
                                    {
                                        rv++;
                                        V = Xs[rsp];
                                        W = DXs[rsp];
                                        SumX += V;
                                        SumX2 += V * V;
                                        SumPx += W;
                                        SumXPx += V * W;
                                        V = Ys[rsp];
                                        W = DYs[rsp];
                                        SumY += V;
                                        SumY2 += V * V;
                                        SumPy += W;
                                        SumYPy += V * W;
                                    }
                                DenX = 1.0 / (rv * SumX2 - SumX * SumX);
                                L0X = (SumPx * SumX2 - SumX * SumXPx) * DenX;
                                L1X = (rv * SumXPx - SumX * SumPx) * DenX;
                                DenY = 1.0 / (rv * SumY2 - SumY * SumY);
                                L0Y = (SumPy * SumY2 - SumY * SumYPy) * DenY;
                                L1Y = (rv * SumYPy - SumY * SumPy) * DenY;

                                SumPx = 0.0; SumX = 0.0; SumX2 = 0.0; SumXPx = 0.0; DenX = 0.0;
                                SumPy = 0.0; SumY = 0.0; SumY2 = 0.0; SumYPy = 0.0; DenY = 0.0;
                                for (rsp = rv = 0; rsp < ReplicaCount; rsp++)
                                    if (Math.Abs(DXs[rsp] - (Xs[rsp] * L1X + L0X)) < DeltaDX && Math.Abs(DYs[rsp] - (Ys[rsp] * L1Y + L0Y)) < DeltaDY)
                                    {
                                        rv++;
                                        V = Xs[rsp];
                                        W = DXs[rsp];
                                        SumX += V;
                                        SumX2 += V * V;
                                        SumPx += W;
                                        SumXPx += V * W;
                                        V = Ys[rsp];
                                        W = DYs[rsp];
                                        SumY += V;
                                        SumY2 += V * V;
                                        SumPy += W;
                                        SumYPy += V * W;
                                    }
                                DenX = 1.0 / (rv * SumX2 - SumX * SumX);
                                L0X = (SumPx * SumX2 - SumX * SumXPx) * DenX;
                                L1X = (rv * SumXPx - SumX * SumPx) * DenX;
                                DenY = 1.0 / (rv * SumY2 - SumY * SumY);
                                L0Y = (SumPy * SumY2 - SumY * SumYPy) * DenY;
                                L1Y = (rv * SumYPy - SumY * SumPy) * DenY;
                                AvgLX = L0X + L1X * ViewCenterX;
                                AvgLY = L0Y + L1Y * ViewCenterY;
                                L1X += 1.0;
                                L1Y += 1.0;
                                TotalLX += AvgLX;
                                TotalLY += AvgLY;
                            }
                        foreach (Cell c in clayer)
                            if (c.Fill >= CellOverflow)
                            {
                                counter_overflowcells++;
                                c.Grains[0] = null;
                                c.Fill = 0;
                            }
                            else
                                foreach (TrackGrain2 tg1 in c.Grains)
                                    if (tg1 == null) break;
                                    else
                                    {
                                        tg1.Position.X = tg1.Position.X * L1X + L0X;
                                        tg1.Position.Y = tg1.Position.Y * L1Y + L0Y;
                                        pblayer[((int)((tg1.Position.Y - CellMin.Y) / BitmapDCell)) * BitmapStrideX + (int)((tg1.Position.X - CellMin.X) / BitmapDCell)] = true;
                                    }
                        LayerCorrections[Layer].Count = maxhisto;
                        LayerCorrections[Layer].AvgLX = AvgLX;
                        LayerCorrections[Layer].AvgLY = AvgLY;
                        LayerCorrections[Layer].L0X = L0X;
                        LayerCorrections[Layer].L1X = L1X;
                        LayerCorrections[Layer].L0Y = L0Y;
                        LayerCorrections[Layer].L1Y = L1Y;
                    }
                    else
                    {
                        foreach (Grain2 g in tlayer.Grains)
                        {
                            CX = g.Position.X * PixelToMicronX;
                            CY = g.Position.Y * PixelToMicronY;
                            if (g.Area >= PixMin && g.Area <= PixMax && CX >= CellMin.X && CX <= CellMax.X && CY >= CellMin.Y && CY <= CellMax.Y)
                            {
                                counter_grains++;
                                pix = (int)((CX - CellMin.X) / DxCell);
                                piy = (int)((CY - CellMin.Y) / DyCell);
                                Cell thiscell = clayer[pix, piy];
                                if (thiscell.Fill < CellOverflow)
                                {
                                    TrackGrain2 tg = thiscell.Grains[thiscell.Fill++] = new TrackGrain2(g, ZLayers[Layer]);
                                    tg.Position.X = CX;
                                    tg.Position.Y = CY;
                                    pblayer[((int)((tg.Position.Y - CellMin.Y) / BitmapDCell)) * BitmapStrideX + (int)((tg.Position.X - CellMin.X) / BitmapDCell)] = true;
                                }
                            }
                        }
                        foreach (Cell c in clayer)
                            if (c.Fill >= CellOverflow)
                            {
                                counter_overflowcells++;
                                c.Fill = 0;
                                c.Grains[0] = null;
                            }
                        LayerCorrections[Layer].Count = 0;
                        LayerCorrections[Layer].AvgLX = 0.0;
                        LayerCorrections[Layer].AvgLY = 0.0;
                        LayerCorrections[Layer].L0X = 0.0;
                        LayerCorrections[Layer].L1X = 1.0;
                        LayerCorrections[Layer].L0Y = 0.0;
                        LayerCorrections[Layer].L1Y = 1.0;
                    }
                    LayerCorrections[Layer].Z = ZLayers[Layer];
                }
            }
        }

        private void PartialGetTracks(uint StartLayer, uint EndLayer, uint threads, uint threadindex, TrackGrain2[][] tracks, ref uint Found, bool enablepresetslope, Vector2 presetslope, Vector2 presetslopeacc, System.DateTime timelimit)
        {
            unchecked
            {
                Found = 0;
                if (tracks.Length == 0) return;
                if (EndLayer >= Cells.Length) EndLayer = (uint)(Cells.Length - 1);

                double avgdz = (ZLayers[StartLayer] - ZLayers[EndLayer]) / (EndLayer - StartLayer);
                double avgdz2 = avgdz * avgdz;
                double avgds2;
                double DirX, DirY, dirtol;

                double tsl = (C.MinGrainsSlope01 == C.MinGrainsForHorizontalTrack) ? 0.0 : (10.0 * (C.MinGrainsForVerticalTrack - C.MinGrainsSlope01) / (C.MinGrainsSlope01 - C.MinGrainsForHorizontalTrack));
                double tinf = C.MinGrainsForHorizontalTrack;
                double td = (C.MinGrainsForVerticalTrack - C.MinGrainsForHorizontalTrack);

                int bx = (int)(C.CellNumX / threads * threadindex);
                int nx = (int)C.CellNumX;
                int wx = (int)((threadindex == threads - 1) ? C.CellNumX : (C.CellNumX / threads * (threadindex + 1)));
                int ny = (int)C.CellNumY;
                Cell[,] toplayer;
                Cell[,] bottomlayer;
                double tz, idz;
                int ix, iy, iix, iiy, minix, maxix, miniy, maxiy, ifound;
                double SlopeX, SlopeY, Slope2, SlopeS, ISlopeS;
                double MaxSlope = C.MaxSlope;
                double _MaxSlope = -MaxSlope;
                double MaxSlope2 = MaxSlope * MaxSlope;
                double _presetslopeaccx = -presetslopeacc.X;
                double _presetslopeaccy = -presetslopeacc.Y;
                double _d;
                double dz, ldz, cdz, ExpX, ExpY;
                int eix, eiy;
                int seqlength = 0;
                int i, j, lay;
                double CurrDeltaX, CurrDeltaY;
                TrackGrain2[] tempgrains = new TrackGrain2[ZLayers.Length * C.InitialMultiplicity];
                int bmpsx = BitmapStrideX;
                int bmpsy = BitmapStrideY;
                double bmpdcell = BitmapDCell;
                bool[][] pbmp = PresenceBitmap;
                double[] pexpx = new double[ZLayers.Length];
                double[] pexpy = new double[ZLayers.Length];

                counter_trials = counter_triggers = 0;
                if (enablepresetslope)
                {
                    foreach (Configuration.TriggerInfo tr in C.Triggers)
                    {
                        if (tr.TopLayer < StartLayer || tr.BottomLayer > EndLayer) continue;
                        toplayer = Cells[tr.TopLayer];
                        bottomlayer = Cells[tr.BottomLayer];
                        tz = ZLayers[tr.TopLayer];
                        idz = 1.0 / (tz - ZLayers[tr.BottomLayer]);

                        for (ix = bx; ix < wx; ix++)
                        {
                            if (System.DateTime.Now > timelimit)
                            {
                                Found = 0;
                                return;
                            }
                            ifound = 0;
                            minix = ix - 1; if (minix < 0) minix = 0;
                            maxix = ix + 1; if (maxix >= nx) maxix = nx - 1;
                            for (iy = 0; iy < ny; iy++)
                            {
                                miniy = iy - 1; if (miniy < 0) miniy = 0;
                                maxiy = iy + 1; if (maxiy >= ny) maxiy = ny - 1;
                                Cell ct = toplayer[ix, iy];
                                foreach (TrackGrain2 gt in ct.Grains)
                                {
                                    if (gt == null) break;
                                    for (iix = minix; iix <= maxix; iix++)
                                        for (iiy = miniy; iiy <= maxiy; iiy++)
                                        {
                                            Cell cb = bottomlayer[iix, iiy];
                                            foreach (TrackGrain2 gb in cb.Grains)
                                            {
                                                if (gb == null) break;
                                                SlopeX = idz * (gt.Position.X - gb.Position.X);
                                                if (SlopeX >= MaxSlope || SlopeX <= _MaxSlope || ((_d = SlopeX - presetslope.X) > presetslopeacc.X) || (_d < _presetslopeaccx)) continue;
                                                SlopeY = idz * (gt.Position.Y - gb.Position.Y);
                                                if (SlopeY >= MaxSlope || SlopeY <= _MaxSlope || ((_d = SlopeY - presetslope.Y) > presetslopeacc.Y) || (_d < _presetslopeaccy)) continue;
                                                Slope2 = SlopeX * SlopeX + SlopeY * SlopeY;
                                                if (Slope2 < MaxSlope2)
                                                {
                                                    counter_trials++;
                                                    SlopeS = Math.Sqrt(Slope2);
                                                    avgds2 = avgdz2 * Slope2;
                                                    if (SlopeS <= 0.0)
                                                    {
                                                        ISlopeS = 0.0;
                                                        DirX = 1.0;
                                                        DirY = 0.0;
                                                        dirtol = C.AlignTol;
                                                    }
                                                    else
                                                    {
                                                        ISlopeS = 1.0 / SlopeS;
                                                        DirX = SlopeX * ISlopeS;
                                                        DirY = SlopeY * ISlopeS;
                                                        dirtol = C.AlignTol + SlopeS * avgdz * C.DeltaZMultiplier;
                                                    }
                                                    foreach (uint TriggerLayer in tr.TriggerLayers)
                                                    {
                                                        ldz = (tz - ZLayers[TriggerLayer]) * idz;
                                                        cdz = 1.0 - ldz;
                                                        ExpX = ldz * gb.Position.X + cdz * gt.Position.X;
                                                        if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                        ExpY = ldz * gb.Position.Y + cdz * gt.Position.Y;
                                                        if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                        eix = (int)((ExpX - CellMin.X) / DxCell);
                                                        //if (eix < 0 || eix >= nx) continue;
                                                        eiy = (int)((ExpY - CellMin.Y) / DyCell);
                                                        //if (eiy < 0 || eiy >= ny) continue;  
#if ENABLE_BITMAP
                                                        if (pbmp[TriggerLayer][((int)((ExpY - CellMin.Y) / bmpdcell)) * bmpsx + (int)((ExpX - CellMin.X) / bmpdcell)] == false) continue;
#endif
                                                        TrackGrain2 gtrg = Cells[TriggerLayer][eix, eiy].FindGrain(ExpX, ExpY, DirX, DirY, dirtol, C.AlignTol);
                                                        if (gtrg != null)
                                                        {
                                                            counter_triggers++;
                                                            for (i = 0; i < tempgrains.Length; i++) tempgrains[i] = null;
                                                            seqlength = 0;
                                                            CurrDeltaX = Math.Abs(avgdz * SlopeX);
                                                            CurrDeltaY = Math.Abs(avgdz * SlopeY);
#if ENABLE_BITMAP
                                                            for (lay = (int)StartLayer; lay <= EndLayer; lay++)
                                                            {
                                                                dz = ZLayers[lay] - tz;
                                                                pexpx[lay] = ExpX = dz * SlopeX + gt.Position.X;
                                                                if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                                pexpy[lay] = ExpY = dz * SlopeY + gt.Position.Y;
                                                                if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                                if (pbmp[lay][((int)((ExpY - CellMin.Y) / bmpdcell)) * bmpsx + (int)((ExpX - CellMin.X) / bmpdcell)]) seqlength++;
                                                            }
                                                            if ((double)seqlength < (tinf + td / (1.0 + tsl * SlopeS))) break;
                                                            seqlength = 0;
#endif
                                                            for (lay = (int)StartLayer; lay <= EndLayer; lay++)
                                                            {
#if ENABLE_BITMAP
                                                                if (pexpx[lay] < CellMin.X || pexpx[lay] > CellMax.X || pexpy[lay] < CellMin.Y || pexpy[lay] > CellMax.Y) continue;
                                                                if (FindGrains(pexpx[lay], pexpy[lay], Cells[lay], DirX, DirY, dirtol, tempgrains, lay * (int)C.InitialMultiplicity, (int)C.InitialMultiplicity) > 0) seqlength++;
#else
                                                            dz = ZLayers[lay] - tz;
                                                            ExpX = dz * SlopeX + gt.Position.X; 
                                                            if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                            ExpY = dz * SlopeY + gt.Position.Y;
                                                            if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                            if (FindGrains(ExpX, ExpY, Cells[lay], DirX, DirY, dirtol, tempgrains, lay * (int)C.InitialMultiplicity, (int)C.InitialMultiplicity) > 0) seqlength++;
#endif
                                                            }
                                                            if ((double)seqlength > (tinf + td / (1.0 + tsl * SlopeS)))
                                                            {
                                                                if (Found < tracks.Length)
                                                                {
                                                                    i = 0;
                                                                    for (j = 0; j < tempgrains.Length; j++)
                                                                        if (tempgrains[j] != null) i++;
                                                                    TrackGrain2[] N = new TrackGrain2[i];
                                                                    i = 0;
                                                                    for (j = 0; j < tempgrains.Length; j++)
                                                                        if (tempgrains[j] != null)
                                                                            N[i++] = tempgrains[j];
                                                                    tracks[Found++] = N;
                                                                }
                                                            }
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                }
                            }
                        }
                    }
                }
                else
                {
                    foreach (Configuration.TriggerInfo tr in C.Triggers)
                    {
                        if (tr.TopLayer < StartLayer || tr.BottomLayer > EndLayer) continue;
                        toplayer = Cells[tr.TopLayer];
                        bottomlayer = Cells[tr.BottomLayer];
                        tz = ZLayers[tr.TopLayer];
                        idz = 1.0 / (tz - ZLayers[tr.BottomLayer]);

                        for (ix = bx; ix < wx; ix++)
                        {
                            if (System.DateTime.Now > timelimit)
                            {
                                Found = 0;
                                return;
                            }
                            ifound = 0;
                            minix = ix - 1; if (minix < 0) minix = 0;
                            maxix = ix + 1; if (maxix >= nx) maxix = nx - 1;
                            for (iy = 0; iy < ny; iy++)
                            {
                                miniy = iy - 1; if (miniy < 0) miniy = 0;
                                maxiy = iy + 1; if (maxiy >= ny) maxiy = ny - 1;
                                Cell ct = toplayer[ix, iy];
                                foreach (TrackGrain2 gt in ct.Grains)
                                {
                                    if (gt == null) break;
                                    for (iix = minix; iix <= maxix; iix++)
                                        for (iiy = miniy; iiy <= maxiy; iiy++)
                                        {
                                            Cell cb = bottomlayer[iix, iiy];
                                            foreach (TrackGrain2 gb in cb.Grains)
                                            {
                                                if (gb == null) break;
                                                SlopeX = idz * (gt.Position.X - gb.Position.X);
                                                if (SlopeX >= MaxSlope || SlopeX <= _MaxSlope) continue;
                                                SlopeY = idz * (gt.Position.Y - gb.Position.Y);
                                                if (SlopeY >= MaxSlope || SlopeY <= _MaxSlope) continue;
                                                Slope2 = SlopeX * SlopeX + SlopeY * SlopeY;
                                                if (Slope2 < MaxSlope2)
                                                {
                                                    counter_trials++;
                                                    SlopeS = Math.Sqrt(Slope2);
                                                    avgds2 = avgdz2 * Slope2;
                                                    if (SlopeS <= 0.0)
                                                    {
                                                        ISlopeS = 0.0;
                                                        DirX = 1.0;
                                                        DirY = 0.0;
                                                        dirtol = C.AlignTol;
                                                    }
                                                    else
                                                    {
                                                        ISlopeS = 1.0 / SlopeS;
                                                        DirX = SlopeX * ISlopeS;
                                                        DirY = SlopeY * ISlopeS;
                                                        dirtol = C.AlignTol + SlopeS * avgdz * C.DeltaZMultiplier;
                                                    }
                                                    foreach (uint TriggerLayer in tr.TriggerLayers)
                                                    {
                                                        ldz = (tz - ZLayers[TriggerLayer]) * idz;
                                                        cdz = 1.0 - ldz;
                                                        ExpX = ldz * gb.Position.X + cdz * gt.Position.X;
                                                        if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                        ExpY = ldz * gb.Position.Y + cdz * gt.Position.Y;
                                                        if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                        eix = (int)((ExpX - CellMin.X) / DxCell);
                                                        //if (eix < 0 || eix >= nx) continue;
                                                        eiy = (int)((ExpY - CellMin.Y) / DyCell);
                                                        //if (eiy < 0 || eiy >= ny) continue;     
#if ENABLE_BITMAP
                                                        if (pbmp[TriggerLayer][((int)((ExpY - CellMin.Y) / bmpdcell)) * bmpsx + (int)((ExpX - CellMin.X) / bmpdcell)] == false) continue;
#endif
                                                        TrackGrain2 gtrg = Cells[TriggerLayer][eix, eiy].FindGrain(ExpX, ExpY, DirX, DirY, dirtol, C.AlignTol);
                                                        if (gtrg != null)
                                                        {
                                                            counter_triggers++;
                                                            for (i = 0; i < tempgrains.Length; i++) tempgrains[i] = null;
                                                            seqlength = 0;
                                                            CurrDeltaX = Math.Abs(avgdz * SlopeX);
                                                            CurrDeltaY = Math.Abs(avgdz * SlopeY);
#if ENABLE_BITMAP
                                                            for (lay = (int)StartLayer; lay <= EndLayer; lay++)
                                                            {
                                                                dz = ZLayers[lay] - tz;
                                                                pexpx[lay] = ExpX = dz * SlopeX + gt.Position.X;
                                                                if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                                pexpy[lay] = ExpY = dz * SlopeY + gt.Position.Y;
                                                                if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                                if (pbmp[lay][((int)((ExpY - CellMin.Y) / bmpdcell)) * bmpsx + (int)((ExpX - CellMin.X) / bmpdcell)]) seqlength++;
                                                            }
                                                            if ((double)seqlength < (tinf + td / (1.0 + tsl * SlopeS))) break;
                                                            seqlength = 0;
#endif
                                                            for (lay = (int)StartLayer; lay <= EndLayer; lay++)
                                                            {
#if ENABLE_BITMAP
                                                                if (pexpx[lay] < CellMin.X || pexpx[lay] > CellMax.X || pexpy[lay] < CellMin.Y || pexpy[lay] > CellMax.Y) continue;
                                                                if (FindGrains(pexpx[lay], pexpy[lay], Cells[lay], DirX, DirY, dirtol, tempgrains, lay * (int)C.InitialMultiplicity, (int)C.InitialMultiplicity) > 0) seqlength++;
#else
                                                            dz = ZLayers[lay] - tz;
                                                            ExpX = dz * SlopeX + gt.Position.X; 
                                                            if (ExpX < CellMin.X || ExpX >= CellMax.X) continue;
                                                            ExpY = dz * SlopeY + gt.Position.Y;
                                                            if (ExpY < CellMin.Y || ExpY >= CellMax.Y) continue;
                                                            if (FindGrains(ExpX, ExpY, Cells[lay], DirX, DirY, dirtol, tempgrains, lay * (int)C.InitialMultiplicity, (int)C.InitialMultiplicity) > 0) seqlength++;
#endif
                                                            }
                                                            if ((double)seqlength > (tinf + td / (1.0 + tsl * SlopeS)))
                                                            {
                                                                if (Found < tracks.Length)
                                                                {
                                                                    i = 0;
                                                                    for (j = 0; j < tempgrains.Length; j++)
                                                                        if (tempgrains[j] != null) i++;
                                                                    TrackGrain2[] N = new TrackGrain2[i];
                                                                    i = 0;
                                                                    for (j = 0; j < tempgrains.Length; j++)
                                                                        if (tempgrains[j] != null)
                                                                            N[i++] = tempgrains[j];
                                                                    tracks[Found++] = N;
                                                                }
                                                            }
                                                            break;
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                }
                            }
                        }
                    }
                }
            }
        }		
		
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

		[XmlElement(typeof(SmartTracking.Configuration))]
		public SySal.Management.Configuration Config
		{
			get
			{
				return (SySal.Management.Configuration)C.Clone();
			}
			set
			{
				C = (Configuration)(value.Clone());
				Cells = null;
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
        /// <summary>
        /// Builds a new SmartTracker.
        /// </summary>
		public SmartTracker()
		{
			//
			// TODO: Add constructor logic here
			//
			C = new Configuration("Default Smart Tracker Configuration");
            C.AlignTol = 0.23;
            C.AllowOverlap = true;
			C.CellOverflow = 256;
            C.CellNumX = 20;
            C.CellNumY = 16;
            C.DeltaZMultiplier = 1.0;
            C.InitialMultiplicity = 1;
            C.MaxProcessors = 0;            
			C.MaxSlope = 1.0;
            C.MaxTrackingTimeMS = 10000;
			C.MinSlope = 0;			
			C.MinArea = 4;
			C.MaxArea = 36;
			C.MinGrainsForHorizontalTrack = 8.0;
			C.MinGrainsForVerticalTrack = 8.0;
			C.MinGrainsSlope01 = 8.0;
            C.MinReplicas = 40;
            C.ReplicaRadius = 2.0;
            C.ReplicaSampleDivider = 1;
            C.Triggers = new Configuration.TriggerInfo[]
            {
                new Configuration.TriggerInfo(4,14,new uint[]{6,7,9,10,11,12}),
                new Configuration.TriggerInfo(5,13,new uint[]{7,9,10,11}),
                new Configuration.TriggerInfo(2,12,new uint[]{4,5,6,7,8,9,10}),
                new Configuration.TriggerInfo(3,13,new uint[]{5,6,7,8,9,10,11}),
                new Configuration.TriggerInfo(1,8,new uint[]{3,4,5,6}),
                new Configuration.TriggerInfo(2,9,new uint[]{4,5,6,7}),
                new Configuration.TriggerInfo(2,9,new uint[]{4,5,6,7}),
                new Configuration.TriggerInfo(6,13,new uint[]{8,9,10,11}),
                new Configuration.TriggerInfo(7,14,new uint[]{9,10,11,12}),
                new Configuration.TriggerInfo(6,13,new uint[]{8,9,10,11})
            };
			intName = "Default Smart Tracker";
			TrackingAreaSet = false;
			XLoc = 0;
			YLoc = 0;
			System.Resources.ResourceManager resman = new System.Resources.ResourceManager("SmartTracker.SmartTrackerIcon", this.GetType().Assembly);
			ClassIcon = (System.Drawing.Icon)(resman.GetObject("SmartTrackerIcon"));
		}

		[XmlIgnore]
		public System.Drawing.Icon Icon
		{
			get
			{
				return (System.Drawing.Icon)ClassIcon.Clone();
			}
		}
        /// <summary>
        /// X position of the object on a graphical layout.
        /// </summary>
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
        /// <summary>
        /// Y position of the object on a graphical layout.
        /// </summary>
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
		/// Exposes / hides generation of additional info.
		/// </summary>
		[XmlIgnore]
		public bool Expose
		{
			get
			{
				return ActivateExpose;
			}
			set
			{
				ActivateExpose = value;
			}
		}


		/// <summary>
		/// Gets the additional information.
		/// </summary>
		[XmlIgnore]
		public System.Collections.ArrayList ExposedInfo
		{
			get
			{
				if (!ActivateExpose) throw new Exception("No information is presently exposed. Use the Expose property to activate this feature.");
				ArrayList data = new ArrayList();
                foreach (GrainCorrectionInfo gli in LayerCorrections)
                    data.Add(new NamedParameter("LayerCorrections" + gli.Layer, gli));
				data.Add(new NamedParameter("Grains", counter_grains));
                data.Add(new NamedParameter("OverflowCells", counter_overflowcells));
				data.Add(new NamedParameter("Trials", counter_trials));
				data.Add(new NamedParameter("Triggers", counter_triggers));
				return data;
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

		~SmartTracker()
		{
			Dispose();
		}
		#endregion
	}
}
