using System;
using SySal.DAQSystem;
using System.Xml.Serialization;
using SySal.Processing;
using SySal.Processing.FragShiftCorrection;
using System.Linq;
using System.Collections.Generic;

namespace SySal.Executables.BatchLink
{    
    /// <summary>
    /// Transformation matrix.
    /// </summary>
    [Serializable]
    public class MatrixTransform
    {
        public double XX = 1.0;
        public double XY = 0.0;
        public double YX = 0.0;
        public double YY = 1.0;
    }


    /// <summary>
    /// Batch link configuration.
    /// </summary>
    [Serializable]
    [XmlType("BatchLink.Config")]
    public class Config
    {
        /// <summary>
        /// The default base thickness to be used.
        /// </summary>
        public double DefaultBaseThickness = 210.0;
        /// <summary>
        /// If enabled, tracks are understood to be recorded with Z centered around the middle position of the emulsion layer and need to be shifted to the base surface.
        /// </summary>
        public bool CorrectTrackZFromCenterToBase = false;
        /// <summary>
        /// The default thickness of emulsion layers to be used.
        /// </summary>
        public double DefaultEmuThickness = 43.0;
        /// <summary>
        /// The default size for a DB view size (when linking microtracks from DB). Putting this field to 0 will turn on a default value of 500.0 micron.
        /// </summary>
        public double DBViewSize = 500.0;
        /// <summary>
        /// If <c>true</c>, microtrack correction with respect to the center of the field of view is activated.
        /// </summary>
        public bool DBViewCorrection = false;
        /// <summary>
        /// Minimum number of grains for a microtrack to be used to estimate the parameters of microtrack. Ignored if <c>DBViewCorrection</c> is <c>false</c>.
        /// </summary>
        public int DBViewCorrMinGrains = 10;
        /// <summary>
        /// Position tolerance for double measurements of microtracks across a view edge. Ignored if <c>DBViewCorrection</c> is <c>false</c>.
        /// </summary>        
        public double DBViewCorrPosTol = 20.0;
        /// <summary>
        /// Slope tolerance for double measurements of microtracks near a view edge. Ignored if <c>DBViewCorrection</c> is <c>false</c>.
        /// </summary>
        public double DBViewCorrSlopeTol = 0.020;
        /// <summary>
        /// Initial multiplier for X component of slope on the top layer.
        /// </summary>
        public double TopMultSlopeX = 1.0;
        /// <summary>
        /// Initial multiplier for Y component of slope on the top layer.
        /// </summary>
        public double TopMultSlopeY = 1.0;
        /// <summary>
        /// Initial multiplier for X component of slope on the bottom layer.
        /// </summary>
        public double BottomMultSlopeX = 1.0;
        /// <summary>
        /// Initial multiplier for Y component of slope on the bottom layer.
        /// </summary>
        public double BottomMultSlopeY = 1.0;
        /// <summary>
        /// Initial X component of linear distortion correction on the top layer.
        /// </summary>
        public double TopDeltaSlopeX = 0.0;
        /// <summary>
        /// Initial Y component of linear distortion correction on the top layer.
        /// </summary>
        public double TopDeltaSlopeY = 0.0;
        /// <summary>
        /// Initial X component of linear distortion correction on the bottom layer.
        /// </summary>
        public double BottomDeltaSlopeX = 0.0;
        /// <summary>
        /// Initial Y component of linear distortion correction on the bottom layer.
        /// </summary>
        public double BottomDeltaSlopeY = 0.0;
        /// <summary>
        /// This parameter is involved in detection of camera spots. It defines the size of the bin used to determine peaks of fake tracks.
        /// </summary>
        public double MaskBinning = 30;
        /// <summary>
        /// This parameter is involved in detection of camera spots. A camera spot is defined as a bin that has a number of tracks higher than the average by a factor defined by MaskPeakHeightMultiplier.
        /// </summary>
        public double MaskPeakHeightMultiplier = 30;
        /// <summary>
        /// If true, slope multipliers (and linear distortion corrections) are computed automatically after a first link pass; then linking is performed again with corrected parameters.
        /// </summary>
        public bool AutoCorrectMultipliers = true;
        /// <summary>
        /// Shrinkage correction linker configuration details.
        /// </summary>
        public SySal.Processing.StripesFragLink2.Configuration ShrinkLinkerConfig;
        /// <summary>
        /// Minimum slope to be used for automatic shrinkage correction.
        /// </summary>
        public double AutoCorrectMinSlope = 0.03;
        /// <summary>
        /// Maximum slope to be used for automatic shrinkage correction.
        /// </summary>
        public double AutoCorrectMaxSlope = 0.2;
        /// <summary>
        /// Iterations for automatic slope correction.
        /// </summary>
        public int AutoCorrectIterations = 1;
        /// <summary>
        /// Start acceptance for iterative correction procedure.
        /// </summary>
        public double AutoCorrectStartDeltaAcceptance = 1.0;
        /// <summary>
        /// End acceptance for iterative correction procedure.
        /// </summary>
        public double AutoCorrectEndDeltaAcceptance = 1.0;
        /// <summary>
        /// If <c>true</c>, the output is written directly to a data stream.
        /// </summary>
        public bool UseDataStream = false;
        /// <summary>
        /// Linker configuration details.
        /// </summary>
        public SySal.Processing.StripesFragLink2.Configuration LinkerConfig;
        /// <summary>
        /// Matrix that transforms RWD views by left side multiplication (i.e. M' = A * M, M being the original view matrix).
        /// </summary>
        public MatrixTransform RWDFieldCorrectionMatrix = new MatrixTransform();
        /// <summary>
        /// Maximum number of tracks per side in an RWD view. When 0, track clamping is disabled. When different from 0, the tracks are subsampled if needed by skipping so that they don't exceed the specified number.
        /// </summary>
        public uint RWDClampTracks = 0;
    }

    struct PixPos
    {
        public double X, Y;
        public double XTol, YTol;

        public PixPos(double x, double y, double xtol, double ytol)
        {
            X = x;
            Y = y;
            XTol = xtol;
            YTol = ytol;
        }
    }

    class SlopeInfo
    {
        public double Sx, Sy;
        public double SSx, SSy;

        public SlopeInfo(double sx, double sy, double ssx, double ssy)
        {
            Sx = sx;
            Sy = sy;
            SSx = ssx;
            SSy = ssy;
        }

        public static void CorrectMultiplier(SlopeInfo [] data, int maxiterations, double startacceptance, double endacceptance, double maxslope, out double xmult, out double ymult, out double xdelta, out double ydelta, bool hasconsole)
        {
            xmult = ymult = 0.0;
            xdelta = ydelta = 0.0;

            if (data.Length < 3) 
            {
                xmult = 1.0;
                ymult = 1.0;
                xdelta = ydelta = 0.0;
                return;
            }

            if (hasconsole) Console.WriteLine("Correcting X slopes.");
            IterativeFit(data.Select(d => d.SSx).ToArray(), data.Select(d => d.Sx - d.SSx).ToArray(), maxiterations, startacceptance, endacceptance, ref xdelta, ref xmult, hasconsole);
            xmult += 1.0;

            if (hasconsole) Console.WriteLine("Correcting Y slopes.");
            IterativeFit(data.Select(d => d.SSy).ToArray(), data.Select(d => d.Sy - d.SSy).ToArray(), maxiterations, startacceptance, endacceptance, ref ydelta, ref ymult, hasconsole);
            ymult += 1.0;
        }

        static void IterativeFit(double [] indep, double [] dep, int maxiterations, double startacceptance, double endacceptance, ref double intercept, ref double slope, bool hasconsole)
        {
            double dummy = 0.0;
            for (int i = 0; i < maxiterations; i++)
            {
                double acc = startacceptance + i * (endacceptance - startacceptance) / (maxiterations - 1);
                List<double> indep_ = new List<double>();
                List<double> dep_ = new List<double>();
                for (int j = 0; j < indep.Length; j++)                
                    if (Math.Abs(slope * indep[j] + intercept - dep[j]) < acc)
                    {
                        indep_.Add(indep[j]);
                        dep_.Add(dep[j]);
                    }
                NumericalTools.Fitting.LinearFitSE(indep_.ToArray(), dep_.ToArray(), ref slope, ref intercept, ref dummy, ref dummy, ref dummy, ref dummy, ref dummy);
                if (hasconsole) Console.WriteLine("Iteration " + i + " N=" + indep_.Count + " S=" + slope.ToString(System.Globalization.CultureInfo.InvariantCulture) + " I=" + intercept.ToString(System.Globalization.CultureInfo.InvariantCulture));
            }
        }
    }

    class Catalog : SySal.Scanning.Plate.IO.OPERA.RawData.Catalog
    {
        public Catalog(SySal.BasicTypes.Rectangle extents, int xcells, int ycells, double cellstep)
        {
            m_Extents = extents;
            m_Fragments = (uint)(xcells * ycells);
            m_Id.Part0 = 0;
            m_Id.Part1 = 0;
            m_Id.Part2 = 0;
            m_Id.Part3 = 0;
            m_Steps.X = 1.0;
            m_Steps.Y = 1.0;
            FragmentIndices = new uint[ycells,xcells];
            int iy, ix;
            uint it = 0;            
            for (iy = 0; iy < ycells; iy++)
                for (ix = 0; ix < xcells; ix++)
                    FragmentIndices[iy, ix] = ++it;
            m_Steps.X = m_Steps.Y = cellstep;
        }
    }

    class Side : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side
    {
        public Side(SySal.Tracking.MIPEmulsionTrackInfo [] info, double ztop, double zbottom, SySal.BasicTypes.Vector2 center)
        {
            m_TopZ = ztop;
            m_BottomZ = zbottom;
            m_MXX = this.m_IMXX = 1.0;
            m_MYY = this.m_IMYY = 1.0;
            m_MXY = this.m_IMXY = 0.0;
            m_MYX = this.m_IMYX = 0.0;
            m_MapPos.X = m_Pos.X = center.X;
            m_MapPos.Y = m_Pos.Y = center.Y;
            m_Layers = null;
            m_Flags = SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side.SideFlags.OK;
            m_Tracks = new SySal.Scanning.MIPIndexedEmulsionTrack[info.Length];
            int i;
            for (i = 0; i < info.Length; i++)
                m_Tracks[i] = new SySal.Scanning.MIPIndexedEmulsionTrack(info[i], null, i);
        }

        static public void CorrectMapMatrix(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, MatrixTransform A)
        {
            MatrixTransform b = new MatrixTransform() { XX = A.XX * s.MXX + A.XY * s.MYX, XY = A.XX * s.MXY + A.XY * s.MYY, YX = A.YX * s.MXX + A.YY * s.MYX, YY = A.YX * s.MXY + A.YY * s.MYY };
            SetM(s, b.XX, b.XY, b.YX, b.YY);
        }

        static public void SetTransform(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, SySal.DAQSystem.Scanning.IntercalibrationInfo t)
        {
            SetM(s, t.MXX, t.MXY, t.MYX, t.MYY);
            var p = t.Transform(s.Pos);
            SetMapPos(s, p);
        }

        static public void RemoveAllTracks(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s)
        {
            SetTracks(s, new Scanning.MIPIndexedEmulsionTrack[0]);
        }

        static public void AdjustZToTarget(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side s, bool istopside, double ztarget, bool correct_tks_emulyr_center_2_base)
        {            
            double dz = ztarget - (istopside ? s.BottomZ : s.TopZ);
            SetZs(s, s.TopZ + dz, s.BottomZ + dz);
            if (correct_tks_emulyr_center_2_base)
            {
                double z_tks_proj = (istopside ? (s.BottomZ - s.TopZ) : (s.TopZ - s.BottomZ));
                for (int i = 0; i < s.Length; i++)
                {
                    var info = SySal.Executables.BatchLink.Exe.MIPEmulsionTrack.GetInfo(s[i]);
                    info.Intercept.X += (z_tks_proj) * info.Slope.X;
                    info.Intercept.Y += (z_tks_proj) * info.Slope.Y;
                    info.Intercept.Z = ztarget;
                    info.TopZ += dz;
                    info.BottomZ += dz;
                    info.Sigma = z_tks_proj;
                }
            }
            else
                for (int i = 0; i < s.Length; i++)
                {
                    var info = SySal.Executables.BatchLink.Exe.MIPEmulsionTrack.GetInfo(s[i]);
                    info.Intercept.Z += dz;
                    info.TopZ += dz;
                    info.BottomZ += dz;
                }
        }
    }

    class IntTrack
    {
        public SySal.BasicTypes.Vector2 Pos;
        public SySal.BasicTypes.Vector2 Slope;
        public int View;
        public long Zone;
        public SySal.BasicTypes.Vector2 ViewCenter;

        internal class Comparer : System.Collections.IComparer
        {
            #region IComparer Members

            int System.Collections.IComparer.Compare(object x, object y)
            {
                double c = ((IntTrack)x).Pos.Y - ((IntTrack)y).Pos.Y;
                if (c < 0) return -1;
                if (c > 0) return 1;
                return 0;
            }

            #endregion

            public static Comparer TheComparer = new Comparer();
        }
    }

    class IntView
    {
        internal IntTrack[] ttks;
        internal IntTrack[] btks;

        public IntView(System.IO.FileStream fbacker, System.Collections.ArrayList topidlist, System.Collections.ArrayList bottomidlist, int mingrains, double postol)
        {
            System.IO.BinaryReader breader = new System.IO.BinaryReader(fbacker);
            short side;            
            System.Collections.ArrayList tlist = new System.Collections.ArrayList();
            System.Collections.ArrayList ar;
            for (side = 0; side < 2; side++)
            {
                tlist.Clear();
                ar = (side == 0) ? topidlist : bottomidlist;
                foreach (long pos in ar)
                {
                    fbacker.Position = pos;
                    long zone = breader.ReadInt64();
                    /* side = */
                    breader.ReadInt16();
                    breader.ReadInt32();
                    if (breader.ReadInt16() >= mingrains)
                    {
                        IntTrack info = new IntTrack();
                        info.Zone = zone;
                        breader.ReadInt32();
                        info.Pos.X = breader.ReadDouble();
                        info.Pos.Y = breader.ReadDouble();
                        info.Slope.X = breader.ReadDouble();
                        info.Slope.Y = breader.ReadDouble();
                        breader.ReadDouble();
                        info.View = breader.ReadInt32();
                        info.ViewCenter.X = breader.ReadDouble();
                        info.ViewCenter.Y = breader.ReadDouble();
                        tlist.Add(info);
                    }
                }
                IntTrack[] tks;
                tlist.Sort(IntTrack.Comparer.TheComparer);
                if (side == 0)
                {
                    tks = ttks = (IntTrack[])tlist.ToArray(typeof(IntTrack));
                }
                else
                {
                    tks = btks = (IntTrack[])tlist.ToArray(typeof(IntTrack));
                }
            }
        }

        internal void Match(IntView w, double postol, double slopetol, ref double tsxx, ref double tsxy, ref double tsyx, ref double tsyy, ref int tnx, ref int tny, ref double bsxx, ref double bsxy, ref double bsyx, ref double bsyy, ref int bnx, ref int bny)
        {
            short side;
            int i, j;
            IntTrack[] thesetks;
            IntTrack[] othertks;
            for (side = 0; side < 2; side++)
            {
                if (side == 0)
                {
                    thesetks = ttks;
                    othertks = w.ttks;
                }
                else
                {
                    thesetks = btks;
                    othertks = w.btks;
                }
                for (i = 0; i < thesetks.Length; i++)
                {
                    IntTrack tk = thesetks[i];
                    double ystart = tk.Pos.Y - postol - 1.0;
                    double yfinish = tk.Pos.Y + postol + 1.0;
                    int jstart, jend;
                    jstart = 0;
                    jend = othertks.Length - 1;
                    while (jstart < jend - 1)
                    {
                        j = (jstart + jend) / 2;
                        if (othertks[j].Pos.Y < ystart) jstart = j + 1;
                        else if (othertks[j].Pos.Y > ystart) jend = j - 1;
                        else jstart = jend;
                    }
                    if ((j = jstart) > 0)
                        do
                        {
                            double dx, dy;
                            IntTrack otk = othertks[j];
                            if (tk.Zone == otk.Zone && tk.View == otk.View /*tk.Zone != otk.Zone || tk.View == otk.View*/) continue;
                            if (Math.Abs(tk.Slope.X - otk.Slope.X) > slopetol) continue;
                            if (Math.Abs(tk.Slope.Y - otk.Slope.Y) > slopetol) continue;
                            if (Math.Abs(dx = (tk.Pos.X - otk.Pos.X)) > postol) continue;
                            dy = tk.Pos.Y - otk.Pos.Y;
                            //if (Math.Abs(tk.Pos.Y - otk.Pos.Y) > postol) continue;
                            double ox = tk.ViewCenter.X - otk.ViewCenter.X;
                            double oy = tk.ViewCenter.Y - otk.ViewCenter.Y;                            
                            if (Math.Abs(ox) < 3 * postol && Math.Abs(oy) > 3 * postol)
                            {
                                if (side == 0)
                                {
                                    tny++;
                                    tsxy += dx / oy;
                                    tsyy += dy / oy;                                    
                                }
                                else
                                {
                                    bny++;
                                    bsxy += dx / oy;
                                    bsyy += dy / oy;
                                }                                
                            }
                            else if (Math.Abs(ox) > 3 * postol && Math.Abs(oy) < 3 * postol)
                            {
                                if (side == 0)
                                {
                                    tnx++;
                                    tsxx += dx / ox;
                                    tsyx += dy / ox;
                                }
                                else
                                {
                                    bnx++;
                                    bsxx += dx / ox;
                                    bsyx += dy / ox;
                                }                                
                            }
                        }                        
                        while (++j < othertks.Length && othertks[j].Pos.Y <= yfinish);
                }
            }
        }
    }

    class View : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View
    {
        public View(System.IO.FileStream fbacker, System.Collections.ArrayList topidlist, System.Collections.ArrayList bottomidlist, double basethickness, double emuthickness, TilePos tilepos, SySal.BasicTypes.Vector2 center,
            double txx, double txy, double tyx, double tyy, double bxx, double bxy, double byx, double byy, SySal.Scanning.PostProcessing.SlopeCorrections corr)
        {
            System.IO.BinaryReader breader = new System.IO.BinaryReader(fbacker);
            m_Tile = tilepos;                  
            SySal.Tracking.MIPEmulsionTrackInfo[] ttks = new SySal.Tracking.MIPEmulsionTrackInfo[topidlist.Count];
            SySal.Tracking.MIPEmulsionTrackInfo[] btks = new SySal.Tracking.MIPEmulsionTrackInfo[bottomidlist.Count];
            short side;
            System.Collections.ArrayList ar;
            SySal.Tracking.MIPEmulsionTrackInfo[] tks;
            double topz, bottomz, basez;
            for (side = 1; side <= 2; side++)
            {
                if (side == 1)
                {
                    ar = topidlist;
                    tks = ttks;
                    topz = emuthickness;
                    bottomz = 0.0;
                    basez = 0.0;
                }
                else
                {
                    ar = bottomidlist;
                    tks = btks;
                    topz = -basethickness;
                    bottomz = -basethickness - emuthickness;
                    basez = -basethickness;
                }
                int i;
                double vwx, vwy, x, y, sx, sy, ssx, ssy;
                for (i = 0; i < ar.Count; i++)
                {
                    fbacker.Position = (long)ar[i];
                    breader.ReadInt64();
                    /* side = */ breader.ReadInt16();
                    breader.ReadInt32();
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info.Count = (ushort)breader.ReadInt16();
                    info.AreaSum = (uint)breader.ReadInt32();
                    x = breader.ReadDouble();
                    y = breader.ReadDouble();
                    sx = breader.ReadDouble();
                    sy = breader.ReadDouble();                    
                    info.Sigma = breader.ReadDouble();
                    breader.ReadInt32();
                    vwx = breader.ReadDouble();
                    vwy = breader.ReadDouble();
                    if (side == 0)
                    {
                        ssx = sx * corr.TopSlopeMultipliers.X + corr.TopDeltaSlope.X;
                        ssy = sy * corr.TopSlopeMultipliers.Y + corr.TopDeltaSlope.Y;
                        info.Intercept.X = vwx + txx * (x - vwx) + txy * (y - vwy) - center.X;
                        info.Intercept.Y = vwy + tyx * (x - vwx) + tyy * (y - vwy) - center.Y;
                        info.Slope.X = txx * ssx + txy * ssy;
                        info.Slope.Y = tyx * ssx + tyy * ssy;
                    }
                    else
                    {
                        ssx = sx * corr.BottomSlopeMultipliers.X + corr.BottomDeltaSlope.X;
                        ssy = sy * corr.BottomSlopeMultipliers.Y + corr.BottomDeltaSlope.Y;
                        info.Intercept.X = vwx + bxx * (x - vwx) + bxy * (y - vwy) - center.X;
                        info.Intercept.Y = vwy + byx * (x - vwx) + byy * (y - vwy) - center.Y;
                        info.Slope.X = bxx * ssx + bxy * ssy;
                        info.Slope.Y = byx * ssx + byy * ssy;
                    }
                    info.Intercept.Z = basez;
                    info.TopZ = topz;
                    info.BottomZ = bottomz;
                    tks[i] = info;                    
                }
            }
            m_Top = new BatchLink.Side(ttks, emuthickness, 0.0, center);
            m_Bottom = new BatchLink.Side(btks, -basethickness, -basethickness - emuthickness, center);
        }
    }

    class Fragment : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment
    {
        public Fragment(System.IO.FileStream fbacker, System.Collections.ArrayList topidlist, System.Collections.ArrayList bottomidlist, double basethickness, double emuthickness, SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.TilePos tilepos, SySal.BasicTypes.Vector2 center, uint index,
            double txx, double txy, double tyx, double tyy, double bxx, double bxy, double byx, double byy, SySal.Scanning.PostProcessing.SlopeCorrections corr)
        {            
            m_CodingMode = SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.FragmentCoding.GrainSuppression;
            m_Id.Part0 = 0;
            m_Id.Part1 = 0;
            m_Id.Part2 = 0;
            m_Id.Part3 = 0;
            m_Index = index;
            m_StartView = index - 1;
            m_Views = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View[1];
            m_Views[0] = new BatchLink.View(fbacker, topidlist, bottomidlist, basethickness, emuthickness, tilepos, center, txx, txy, tyx, tyy, bxx, bxy, byx, byy, corr);
        }
    }


    /// <summary>
    /// BatchLink - links RWD files into OPERA TLG files.
    /// </summary>
    /// <remarks>
    /// <para>BatchLink uses SySal.Processing.SySal.Processing.StripesFragLink2 for computations. See <see cref="SySal.Processing.StripesFragLink2.StripesFragmentLinker"/> for more inforamtion on the algorithm and its parameters.</para>
    /// <para>
    /// BatchLink can be used in several ways:
    /// <list type="bullet">
    /// <item><term><c>usage: batchlink [/wait] &lt;input RWC path&gt; &lt;output TLG Opera persistence path&gt; &lt;XML config Opera persistence path&gt; [&lt;XML fragment shift correction file&gt;]</c></term></item>
    /// <item><term><c>usage: batchlink /dbquery &lt;DB query to get microtracks&gt; &lt;output TLG Opera persistence path&gt; &lt;XML config Opera persistence path&gt; [&lt;XML fragment shift correction file&gt;]</c></term></item>
    /// <item><term><c>usage: batchlink /dbquerysb &lt;DB query to get microtracks&gt; &lt;output TLG Opera persistence path&gt; &lt;XML config Opera persistence path&gt; [&lt;XML fragment shift correction file&gt;]</c></term></item>
    /// <item><term><c>usage: batchlink /dbqueryrb &lt;File backer for a previously executed DB query&gt; &lt;output TLG Opera persistence path&gt; &lt;XML config Opera persistence path&gt; [&lt;XML fragment shift correction file&gt;]</c></term></item>
    /// </list>
    /// The output path is in the <see cref="SySal.OperaPersistence">GUFS</see> notation, so it can be a file path as well as a DB access string. The same is true for the configuration file (it can be a ProgramSettings entry from the DB).
    /// </para>
    /// <para>The <c>/dbquery</c> and <c>/dbquerysb</c> switches behave in the same way, but if <c>/dbquerysb</c> the results of the query to the DB are saved to a file in the temporary directory, so that further access to the DB is not 
    /// needed when attempting to link the same data. The full path is shown at the end of BatchLink execution, if successful. That file is to be reused with the <c>/dbqueryrb</c> option (see below).</para>
    /// <para>
    /// The DB query (activated by <c>/dbquery</c> or <c>/dbquerysb</c>) must return the following fields in the exact order (field name does not matter):
    /// <c>ID_ZONE SIDE ID_TRACK GRAINS AREASUM POSX POSY SLOPEX SLOPEY SIGMA ID_VIEW VIEWCENTERX VIEWCENTERY</c>
    /// </para>
    /// <para>The <c>/dbqueryrb</c> switch uses the temporary query file generated by <c>/dbquerysb</c> in a previous run as if it were the input from the DB. The path to the file (called <i>File backer</i>) is specified as the first argument
    /// after the switch, in the place that would otherwise be occupied by the DB query.</para>
    /// <para>
    /// XML config syntax:
    /// <code>
    /// &lt;BatchLink.Config&gt;
    ///  &lt;TopMultSlopeX&gt;1&lt;/TopMultSlopeX&gt;
    ///  &lt;TopMultSlopeY&gt;1&lt;/TopMultSlopeY&gt;
    ///  &lt;BottomMultSlopeX&gt;1&lt;/BottomMultSlopeX&gt;
    ///  &lt;BottomMultSlopeY&gt;1&lt;/BottomMultSlopeY&gt;
    ///  &lt;TopDeltaSlopeX&gt;0&lt;/TopDeltaSlopeX&gt;
    ///  &lt;TopDeltaSlopeY&gt;0&lt;/TopDeltaSlopeY&gt;
    ///  &lt;BottomDeltaSlopeX&gt;0&lt;/BottomDeltaSlopeX&gt;
    ///  &lt;BottomDeltaSlopeY&gt;0&lt;/BottomDeltaSlopeY&gt;
    ///  &lt;MaskBinning&gt;30&lt;/MaskBinning&gt;
    ///  &lt;MaskPeakHeightMultiplier&gt;30&lt;/MaskPeakHeightMultiplier&gt;
    ///  &lt;AutoCorrectMultipliers&gt;false&lt;/AutoCorrectMultipliers&gt;
    ///  &lt;AutoCorrectMinSlope&gt;0.03&lt;/AutoCorrectMinSlope&gt;
    ///  &lt;AutoCorrectMaxSlope&gt;0.4&lt;/AutoCorrectMaxSlope&gt;
    ///  &lt;LinkerConfig&gt;
    ///  &lt;Name /&gt;
    ///  &lt;MinGrains&gt;6&lt;/MinGrains&gt;
    ///  &lt;MinSlope&gt;0.0&lt;/MinSlope&gt;
    ///  &lt;MergePosTol&gt;10&lt;/MergePosTol&gt;
    ///  &lt;MergeSlopeTol&gt;0.02&lt;/MergeSlopeTol&gt;
    ///  &lt;PosTol&gt;100&lt;/PosTol&gt;
    ///  &lt;SlopeTol&gt;0.04&lt;/SlopeTol&gt;
    ///  &lt;SlopeTolIncreaseWithSlope&gt;0.3&lt;/SlopeTolIncreaseWithSlope&gt;
    ///  &lt;MemorySaving&gt;3&lt;/MemorySaving&gt;
    ///  &lt;KeepLinkedTracksOnly&gt;true&lt;/KeepLinkedTracksOnly&gt;
    ///  &lt;PreserveViews&gt;true&lt;/PreserveViews&gt;
    ///  &lt;QualityCut&gt;S &amp;lt; 0.13 * N - 1.3&lt;/QualityCut&gt;
    ///  &lt;/LinkerConfig&gt;
    /// &lt;/BatchLink.Config&gt;
    /// </code>
    /// See <see cref="SySal.Processing.StripesFragLink2.StripesFragmentLinker"/> for more information on SySal.Processing.StripesFragLink2 parameters.
    /// </para>
    /// </remarks>
    public class Exe
    {        
        internal class MIPEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
        {
            public static void AdjustSlopes(SySal.Scanning.MIPIndexedEmulsionTrack t, double xslopemult, double yslopemult, double slopedx, double slopedy)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.AccessInfo(t);
                info.Slope.X = info.Slope.X * xslopemult + slopedx;
                info.Slope.Y = info.Slope.Y * yslopemult + slopedy;
            }

            static public void KillTracks(SySal.Tracking.MIPEmulsionTrack t)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.AccessInfo(t);
                t.Info.Count = 0;
                t.Info.AreaSum = 0;
                return;
            }

            static public void KillTracks(SySal.Tracking.MIPEmulsionTrack t, SySal.BasicTypes.Vector2 centerp, PixPos [] mask, uint mingrains)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.AccessInfo(t);
                if (info.Count < mingrains)
                {
                    info.Count = 0;
                    info.AreaSum = 0;
                    return;
                }
                if (double.IsNaN(info.Slope.X) || double.IsInfinity(info.Slope.X) || double.IsNaN(info.Slope.Y) || double.IsInfinity(info.Slope.Y) ||
                    double.IsNaN(info.Intercept.X) || double.IsInfinity(info.Intercept.X) || double.IsNaN(info.Intercept.Y) || double.IsInfinity(info.Intercept.Y) ||
                    double.IsNaN(info.Intercept.Z) || double.IsInfinity(info.Intercept.Z))
                {
                    info.Count = 0;
                    info.AreaSum = 0;
                    return;
                }
                if (Math.Abs(info.Slope.X) > 10.0 || Math.Abs(info.Slope.Y) > 10.0)
                {
                    info.Count = 0;
                    info.AreaSum = 0;
                    return;
                }
                if ((Math.Abs(info.Intercept.X - centerp.X) > 1e6) || (Math.Abs(info.Intercept.Y - centerp.Y) > 1e6))
                {
                    info.Count = 0;
                    info.AreaSum = 0;
                    return;
                }
                int i;
                for (i = 0; i < mask.Length; i++)
                    if (Math.Abs(info.Intercept.X - mask[i].X) < mask[i].XTol &&
                        Math.Abs(info.Intercept.Y - mask[i].Y) < mask[i].YTol)
                    {
                        info.Count = 0;
                        info.AreaSum = 0;
                        return;
                    };
            }

            static public SySal.Tracking.MIPEmulsionTrackInfo GetInfo(SySal.Tracking.MIPEmulsionTrack t) { return MIPEmulsionTrack.AccessInfo(t); }

        }

        bool WaitFragment = true;
        bool DBQuery = false;
        bool DeleteFileBacker = true;
        int MinFragIndex = 0;
        int MaxFragIndex = 0;
        SySal.DAQSystem.Scanning.IntercalibrationInfo? TransformOverride = null;
        /// <summary>
        /// If set to <c>true</c>, the input file is a file backer.
        /// </summary>
        public bool UseFileBacker
        {
            set
            {
                ReadFileBacker = DBQuery = value;
                DeleteFileBacker = false;
            }
        }

        protected bool ReadFileBacker = false;
        protected string BaseName;
        protected double LastPercent;
        System.DateTime LastTime;
        bool HasConsole = false;
        Config C;
        LinearFragmentCorrectionWithHysteresis LHCorr = null;
        ViewList[,] m_ViewLists;
        SySal.BasicTypes.Vector2 m_ViewSize;
        SySal.BasicTypes.Rectangle m_rect;
        double m_tsxx = 1.0, m_tsxy = 0.0, m_tsyx = 0.0, m_tsyy = 1.0;        
        double m_bsxx = 1.0, m_bsxy = 0.0, m_bsyx = 0.0, m_bsyy = 1.0;

        protected SySal.Scanning.Plate.IO.OPERA.RawData.Fragment FBackerDBFragment(uint index)
        {
            SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.TilePos tilepos = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.TilePos();
            tilepos.Y = (int)((index - 1) / m_ViewLists.GetLength(1));
            tilepos.X = (int)((index - 1) - tilepos.Y * m_ViewLists.GetLength(1));
            SySal.BasicTypes.Vector2 center = new SySal.BasicTypes.Vector2();
            center.X = m_rect.MinX + (tilepos.X - 0.5) * DBCellSize;
            center.Y = m_rect.MinY + (tilepos.Y - 0.5) * DBCellSize;
            SySal.Scanning.PostProcessing.SlopeCorrections corr = new SySal.Scanning.PostProcessing.SlopeCorrections();
            corr.TopSlopeMultipliers.X = C.TopMultSlopeX;
            corr.TopSlopeMultipliers.Y = C.TopMultSlopeY;
            corr.TopDeltaSlope.X = C.TopDeltaSlopeX;
            corr.TopDeltaSlope.Y = C.TopDeltaSlopeY;
            corr.BottomSlopeMultipliers.X = C.BottomMultSlopeX;
            corr.BottomSlopeMultipliers.Y = C.BottomMultSlopeY;
            corr.BottomDeltaSlope.X = C.BottomDeltaSlopeX;
            corr.BottomDeltaSlope.Y = C.BottomDeltaSlopeY;
            return new Fragment(m_fbacker, m_ViewLists[tilepos.Y, tilepos.X].topids, m_ViewLists[tilepos.Y, tilepos.X].bottomids, C.DefaultBaseThickness, C.DefaultEmuThickness, tilepos, center, index,
                m_tsxx, m_tsxy, m_tsyx, m_tsyy, m_bsxx, m_bsxy, m_bsyx, m_bsyy, corr);
        }

        protected SySal.Scanning.Plate.IO.OPERA.RawData.Fragment LoadFragment(uint index)
        {
            string fname = BaseName + ".rwd." + index.ToString("X08");
            System.IO.FileStream f = null;
            do
            {
                try
                {
                    f = new System.IO.FileStream(fname, System.IO.FileMode.Open, System.IO.FileAccess.Read);
                    break;
                }
                catch (Exception x)
                {
                    if (!WaitFragment) throw x;
                    else
                    {
                        Console.WriteLine("Waiting for fragment " + fname);
                        System.Threading.Thread.Sleep(2000);
                    }
                }
            }
            while (true);
            SySal.Scanning.Plate.IO.OPERA.RawData.Fragment Frag = null;
            try
            {
                Frag = new SySal.Scanning.Plate.IO.OPERA.RawData.Fragment(f);
                if (MinFragIndex > 0 && MaxFragIndex > 0 && (index < MinFragIndex || index > MaxFragIndex))
                {
                    int j;
                    for (j = 0; j < Frag.Length; j++)
                    {
                        Side.RemoveAllTracks(Frag[j].Top);
                        Side.RemoveAllTracks(Frag[j].Bottom);
                    }
                }
            }
            catch (Exception xc)
            {
                Console.WriteLine("Error opening fragment with index " + index + ":\r\n" + xc.ToString());
                throw xc;
            }
            finally
            {
                f.Close();
            }
            PixPos [] mask = GenerateMask(Frag, m_ViewSize);

            int i;
            for (i = 0; i < Frag.Length; i++)
            {
                SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View v = Frag[i];
                if (TransformOverride != null)
                {
                    Side.SetTransform(v.Top, (SySal.DAQSystem.Scanning.IntercalibrationInfo)TransformOverride);
                    Side.SetTransform(v.Bottom, (SySal.DAQSystem.Scanning.IntercalibrationInfo)TransformOverride);
                }
                else
                {
                    Side.CorrectMapMatrix(v.Top, C.RWDFieldCorrectionMatrix);
                    Side.CorrectMapMatrix(v.Bottom, C.RWDFieldCorrectionMatrix);
                }
                if (C.DefaultBaseThickness != 0.0)
                {
                    Side.AdjustZToTarget(v.Top, true, 0.0, C.CorrectTrackZFromCenterToBase);
                    Side.AdjustZToTarget(v.Bottom, false, -C.DefaultBaseThickness, C.CorrectTrackZFromCenterToBase);
                }
                int j, l;
                l = v.Top.Length;
                for (j = 0; j < l; j++)
                {
                    MIPEmulsionTrack.AdjustSlopes(v.Top[j], C.TopMultSlopeX, C.TopMultSlopeY, C.TopDeltaSlopeX, C.TopDeltaSlopeY);
                    MIPEmulsionTrack.KillTracks(v.Top[j], v.Top.MapPos, mask, (uint)((C.LinkerConfig.KeepLinkedTracksOnly && (C.ShrinkLinkerConfig == null || C.ShrinkLinkerConfig.KeepLinkedTracksOnly)) ? Math.Min(C.LinkerConfig.MinGrains, (C.ShrinkLinkerConfig != null) ? C.ShrinkLinkerConfig.MinGrains : 0) : 0));
                }
                l = v.Bottom.Length;
                for (j = 0; j < l; j++)
                {
                    MIPEmulsionTrack.AdjustSlopes(v.Bottom[j], C.BottomMultSlopeX, C.BottomMultSlopeY, C.BottomDeltaSlopeX, C.BottomDeltaSlopeY);
                    MIPEmulsionTrack.KillTracks(v.Bottom[j], v.Bottom.MapPos, mask, (uint)((C.LinkerConfig.KeepLinkedTracksOnly && (C.ShrinkLinkerConfig == null || C.ShrinkLinkerConfig.KeepLinkedTracksOnly)) ? Math.Min(C.LinkerConfig.MinGrains, (C.ShrinkLinkerConfig != null) ? C.ShrinkLinkerConfig.MinGrains : 0) : 0));
                }
                foreach (var ss in new [] { v.Top, v.Bottom})
                {
                    var l1 = ss.Length;
                    if (C.RWDClampTracks > 0 && l1 > C.RWDClampTracks)
                    {
                        int c = 0; int nc = 0;
                        for (j = 0; j < l1; j++)
                            if (j < nc) MIPEmulsionTrack.KillTracks(ss[j]);
                            else nc = (int)(++c * ((double)l1 / (double)C.RWDClampTracks));
                    }
                }
            }

            if (LHCorr != null) LHCorr.Correct(Frag);

            return Frag;
        }

        /// <summary>
        /// The main entry point for the application.
        /// </summary>
        [STAThread]
        static void Main(string[] args)
        {
#if !(DEBUG)
            try
            {
#endif
                Exe exec = new Exe();
                if (args.Length > 1 && args[0].ToLower() == "/wait") 
                {
                    exec.WaitFragment = true;
                    string [] oldargs = args;
                    int i;
                    args = new string[oldargs.Length - 1];
                    for (i = 1; i < oldargs.Length; i++)
                        args[i - 1] = oldargs[i];
                    oldargs = null;
                }
                else
                {
                    exec.WaitFragment = false;
                    if (args.Length > 1 && (String.Compare(args[0], "/dbquery", true) == 0 || String.Compare(args[0], "/dbquerysb", true) == 0 || String.Compare(args[0], "/dbqueryrb", true) == 0))
                    {
                        exec.DBQuery = true;
                        if (String.Compare(args[0], "/dbquerysb", true) == 0)
                        {
                            exec.ReadFileBacker = false;
                            exec.DeleteFileBacker = false;
                        }
                        else if (String.Compare(args[0], "/dbqueryrb", true) == 0)
                        {
                            exec.ReadFileBacker = true;
                            exec.DeleteFileBacker = false;
                        }
                        string [] oldargs = args;
                        int i;
                        args = new string[oldargs.Length - 1];
                        for (i = 1; i < oldargs.Length; i++)
                            args[i - 1] = oldargs[i];
                        oldargs = null;
                    }
                    else exec.DBQuery = false;
                }

                System.Text.RegularExpressions.Regex rx_frag = new System.Text.RegularExpressions.Regex(@"/frag=(\d+)\-(\d+)");                
                if (args.Length > 1)
                {
                    var m_frag = rx_frag.Match(args[0]);
                    if (m_frag.Success)
                    {
                        exec.MinFragIndex = int.Parse(m_frag.Groups[1].Value);
                        exec.MaxFragIndex = int.Parse(m_frag.Groups[2].Value);

                        string[] oldargs = args;
                        int i;
                        args = new string[oldargs.Length - 1];
                        for (i = 1; i < oldargs.Length; i++)
                            args[i - 1] = oldargs[i];
                        oldargs = null;
                    }
                }
                
                System.Text.RegularExpressions.Regex rx_transform = new System.Text.RegularExpressions.Regex(@"/transform=(\S+),(\S+),(\S+),(\S+),(\S+),(\S+),(\S+),(\S+)");
                if (args.Length > 1)
                {
                    var m_transform = rx_transform.Match(args[0]);
                    if (m_transform.Success)
                    {
                        exec.TransformOverride = new DAQSystem.Scanning.IntercalibrationInfo()
                        {
                            MXX = double.Parse(m_transform.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture),
                            MXY = double.Parse(m_transform.Groups[2].Value, System.Globalization.CultureInfo.InvariantCulture),
                            MYX = double.Parse(m_transform.Groups[3].Value, System.Globalization.CultureInfo.InvariantCulture),
                            MYY = double.Parse(m_transform.Groups[4].Value, System.Globalization.CultureInfo.InvariantCulture),
                            RX = double.Parse(m_transform.Groups[5].Value, System.Globalization.CultureInfo.InvariantCulture),
                            RY = double.Parse(m_transform.Groups[6].Value, System.Globalization.CultureInfo.InvariantCulture),
                            TX = double.Parse(m_transform.Groups[7].Value, System.Globalization.CultureInfo.InvariantCulture),
                            TY = double.Parse(m_transform.Groups[8].Value, System.Globalization.CultureInfo.InvariantCulture),
                            TZ = 0.0
                        };                    

                        string[] oldargs = args;
                        int i;
                        args = new string[oldargs.Length - 1];
                        for (i = 1; i < oldargs.Length; i++)
                            args[i - 1] = oldargs[i];
                        oldargs = null;
                    }
                }
                
                if (args.Length != 3 && args.Length != 4 && args.Length != 5)
                {
                    Console.WriteLine("BatchLink - links RWD files into OPERA TLG files.");
                    Console.WriteLine("usage: batchlink [/wait] [/frag=a-b] [/transform=mxx,mxy,myx,myy,rx,ry,tx,ty] <input RWC path> <output TLG Opera persistence path> <XML config Opera persistence path> [<XML fragment shift correction file>]");
                    Console.WriteLine("  where a and b are the first and last fragment indices.");
                    Console.WriteLine("usage: batchlink /dbquery <DB query to get microtracks> <output TLG Opera persistence path> <XML config Opera persistence path> [<XML fragment shift correction file>]");
                    Console.WriteLine("The DB query must return the following fields in the exact order (field name does not matter): ");
                    Console.WriteLine("ID_ZONE SIDE ID_TRACK GRAINS AREASUM POSX POSY SLOPEX SLOPEY SIGMA ID_VIEW VIEWCENTERX VIEWCENTERY");
                    Console.WriteLine("XML config syntax:");
                    BatchLink.Config C = new BatchLink.Config();
                    C.LinkerConfig = new SySal.Processing.StripesFragLink2.Configuration();
                    System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(BatchLink.Config));
                    System.IO.StringWriter ss = new System.IO.StringWriter();
                    xmls.Serialize(ss, C);
                    Console.WriteLine(ss.ToString());
                    ss.Close();
                    return;
                }
                SySal.OperaDb.OperaDbCredentials testcred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();                
                string xmlconfig = ((SySal.OperaDb.ComputingInfrastructure.ProgramSettings)SySal.OperaPersistence.Restore(args[2], typeof(SySal.OperaDb.ComputingInfrastructure.ProgramSettings))).Settings;
                Console.WriteLine(xmlconfig);                
                exec.HasConsole = true;
                string corrfile = (args.Length == 3) ? null : args[3];
                string corrstring = null;
                if (corrfile != null)
                {
                    System.IO.StreamReader corrr = new System.IO.StreamReader(corrfile);
                    corrstring = corrr.ReadToEnd();
                    corrr.Close();
                }
                exec.ProcessData(args[0], args[1], xmlconfig, corrstring);
                Console.WriteLine("OK!");
#if !(DEBUG)
            }
            catch (Exception x)
            {
                Console.Error.WriteLine(x.ToString());
            }
#endif
        }

        void Progress(double percent)
        {
            if (percent == 100.0 || percent == 0.0 || (percent - LastPercent >= 1.0 && ((System.TimeSpan)(System.DateTime.Now - LastTime)).TotalSeconds > 2.0))
            {
                Console.WriteLine("{0}%", (int)percent);
                LastPercent = percent;
                LastTime = System.DateTime.Now;
            }
        }

        PixPos [] GenerateMask(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment frag, SySal.BasicTypes.Vector2 viewsize)
        {
            int v, s, i;
            int tkcount = 0;
            for (v = 0; v < frag.Length; v++)
                for (s = 0; s < 2; s++)
                {
                    SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side side = (s == 0) ? frag[v].Top : frag[v].Bottom;
                    tkcount += side.Length;
                }
            double [] fpx = new double[tkcount];
            double [] fpy = new double[tkcount];
            tkcount = 0;
            for (v = 0; v < frag.Length; v++)
                for (s = 0; s < 2; s++)
                {
                    SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side side = (s == 0) ? frag[v].Top : frag[v].Bottom;
                    for (i = 0; i < side.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.GetInfo(side[i]);
                        fpx[tkcount] = side[i].Info.Intercept.X;
                        fpy[tkcount] = side[i].Info.Intercept.Y;
                        tkcount++;
                    }
                }
            if (tkcount == 0) return new PixPos[0];
            double [] xc;
            double [] yc;
            double [,] zc;
            double [,] znc;
            try
            {
                NumericalTools.Fitting.Prepare_2DCustom_Distribution(fpx, fpy, C.MaskBinning, C.MaskBinning, -viewsize.X, viewsize.X, -viewsize.Y, viewsize.Y, out xc, out yc, out zc, out znc);
                System.Collections.ArrayList pixl = new System.Collections.ArrayList();
                int xs = xc.Length, ys = yc.Length;
                int xi, yi;
                double max = 0.0;
                for (xi = 0; xi < xs; xi++)
                    for (yi = 0; yi < ys; yi++)
                        max += zc[xi, yi];
                max *= C.MaskPeakHeightMultiplier / (xs * ys);
                for (xi = 0; xi < xs; xi++)
                    for (yi = 0; yi < ys; yi++)
                        if (zc[xi, yi] > max)
                            pixl.Add(new PixPos(xc[xi], yc[yi], C.MaskBinning, C.MaskBinning));
                return (PixPos[])pixl.ToArray(typeof(PixPos));
            }
            catch (Exception)
            {
                return new PixPos[0];
            }
        }

        double DBCellSize = 500.0;

        class ViewList
        {
            public System.Collections.ArrayList topids = new System.Collections.ArrayList();
            public System.Collections.ArrayList bottomids = new System.Collections.ArrayList();
        }

        System.IO.FileStream m_fbacker;
        string m_TempFileBacker;

        /// <summary>
        /// This method does the actual processing. It can be called explicitly, thus using BatchLink as a computation library instead of an executable.
        /// </summary>
        /// <param name="input">the input string. Can be an OPERA persistence path or a DB query.</param>
        /// <param name="output">the output OPERA persistence path.</param>
        /// <param name="programsettings">the string containing the program settings.</param>
        /// <param name="corrstring">the path to the fragment correction file. Can be null if not needed.</param>
        /// <returns>the <see cref="SySal.Scanning.Plate.IO.OPERA.LinkedZone"/> obtained.</returns>
        public SySal.Scanning.Plate.IO.OPERA.LinkedZone ProcessData(string input, string output, string programsettings, string corrstring)
        {
            SySal.Scanning.PostProcessing.SlopeCorrections ReturnSlopeCorrections = new SySal.Scanning.PostProcessing.SlopeCorrections();
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(Config));
            C = (Config)xmls.Deserialize(new System.IO.StringReader(programsettings));            
            SySal.Processing.StripesFragLink2.StripesFragmentLinker SFL = new SySal.Processing.StripesFragLink2.StripesFragmentLinker();
            if (C.AutoCorrectMultipliers == true && C.ShrinkLinkerConfig != null)
                SFL.Config = C.ShrinkLinkerConfig;
            else
                SFL.Config = C.LinkerConfig;
            if (C.DBViewSize > 0.0) DBCellSize = C.DBViewSize;

            if (HasConsole)
            {
                xmls.Serialize(Console.Out, C);
                Console.WriteLine();
                Console.WriteLine(C.TopMultSlopeX);
                Console.WriteLine(C.TopMultSlopeY);
                Console.WriteLine(C.TopDeltaSlopeX);
                Console.WriteLine(C.TopDeltaSlopeY);
                Console.WriteLine(C.BottomMultSlopeX);
                Console.WriteLine(C.BottomMultSlopeY);
                Console.WriteLine(C.BottomDeltaSlopeX);
                Console.WriteLine(C.BottomDeltaSlopeY);
            }
            
            if (corrstring != null)
            {
                System.IO.StringReader rc = new System.IO.StringReader(corrstring);
                xmls = new System.Xml.Serialization.XmlSerializer(typeof(LinearFragmentCorrectionWithHysteresis));
                LHCorr = (LinearFragmentCorrectionWithHysteresis)xmls.Deserialize(rc);
                rc.Close();
            }

            SySal.Scanning.Plate.IO.OPERA.RawData.Catalog Cat = null;

            System.IO.BinaryWriter bwriter = null;
            System.IO.BinaryReader breader = null;

            if (DBQuery)
            {
                if (ReadFileBacker)
                {
                    m_TempFileBacker = input;
                }
                else
                {
                    m_TempFileBacker = System.Environment.ExpandEnvironmentVariables("%TEMP%\\batchlink_" + System.DateTime.Now.Ticks.ToString() + "_" + System.Diagnostics.Process.GetCurrentProcess().Id.ToString() + ".reader");
                }
                try
                {
                    if (ReadFileBacker)
                    {
                        m_fbacker = new System.IO.FileStream(m_TempFileBacker, System.IO.FileMode.Open, System.IO.FileAccess.Read, System.IO.FileShare.ReadWrite);
                    }
                    else
                    {
                        m_fbacker = new System.IO.FileStream(m_TempFileBacker, System.IO.FileMode.CreateNew, System.IO.FileAccess.ReadWrite, System.IO.FileShare.Read);
                        bwriter = new System.IO.BinaryWriter(m_fbacker);
                    }
                    int total = 0;
                    double x, y;
                    Int16 side;
                    if (!ReadFileBacker)
                    {
                        SySal.OperaDb.OperaDbConnection conn = null;
                        SySal.OperaDb.OperaDbDataReader reader = null;
                        (conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect()).Open();
                        reader = new SySal.OperaDb.OperaDbCommand(input, conn).ExecuteReader();

                        bwriter.Write(total);
                        bwriter.Write(m_rect.MinX);
                        bwriter.Write(m_rect.MaxX);
                        bwriter.Write(m_rect.MinY);
                        bwriter.Write(m_rect.MaxY);
                        if (reader.Read())
                        {                            
                            bwriter.Write(reader.GetInt64(0));                            
                            bwriter.Write((short)reader.GetInt32(1));                            
                            bwriter.Write(reader.GetInt32(2));                            
                            bwriter.Write((short)reader.GetInt32(3));
                            bwriter.Write(reader.GetInt32(4));
                            bwriter.Write(x = reader.GetDouble(5));
                            bwriter.Write(y = reader.GetDouble(6));
                            bwriter.Write(reader.GetDouble(7));
                            bwriter.Write(reader.GetDouble(8));
                            bwriter.Write(reader.GetDouble(9));                            
                            bwriter.Write(reader.GetInt32(10));
                            bwriter.Write(reader.GetDouble(11));
                            bwriter.Write(reader.GetDouble(12));
                            m_rect.MinX = m_rect.MaxX = x;
                            m_rect.MinY = m_rect.MaxY = y;                            
                            total++;
                        }
                        while (reader.Read())
                        {
                            bwriter.Write(reader.GetInt64(0));
                            bwriter.Write((short)reader.GetInt32(1));
                            bwriter.Write(reader.GetInt32(2));
                            bwriter.Write((short)reader.GetInt32(3));
                            bwriter.Write(reader.GetInt32(4));
                            bwriter.Write(x = reader.GetDouble(5));
                            bwriter.Write(y = reader.GetDouble(6));
                            bwriter.Write(reader.GetDouble(7));
                            bwriter.Write(reader.GetDouble(8));
                            bwriter.Write(reader.GetDouble(9));
                            bwriter.Write(reader.GetInt32(10));
                            bwriter.Write(reader.GetDouble(11));
                            bwriter.Write(reader.GetDouble(12));
                            if (m_rect.MinX > x) m_rect.MinX = x;
                            else if (m_rect.MaxX < x) m_rect.MaxX = x;
                            if (m_rect.MinY > y) m_rect.MinY = y;
                            else if (m_rect.MaxY < y) m_rect.MaxY = y;
                            total++;
                        }
                        bwriter.Flush();
                        conn.Close();
                    }
                    else
                    {
                        breader = new System.IO.BinaryReader(m_fbacker);
                        total = breader.ReadInt32();
                        m_rect.MinX = breader.ReadDouble();
                        m_rect.MaxX = breader.ReadDouble();
                        m_rect.MinY = breader.ReadDouble();
                        m_rect.MaxY = breader.ReadDouble();
                    }
                    int XCells = (int)(1.0 + (m_rect.MaxX - m_rect.MinX) / DBCellSize);
                    int YCells = (int)(1.0 + (m_rect.MaxY - m_rect.MinY) / DBCellSize);
                    Console.WriteLine("Extents = " + m_rect.MinX + " " + m_rect.MaxX + " " + m_rect.MinY + " " + m_rect.MaxY);
                    Console.WriteLine("DBViewSize = " + C.DBViewSize + ", DBCellSize = " + DBCellSize);
                    Console.WriteLine("XCells = " + XCells + ", YCells = " + YCells);
                    m_ViewLists = new ViewList[YCells, XCells];
                    m_ViewSize.X = m_ViewSize.Y = DBCellSize;
                    int ix, iy;
                    for (iy = 0; iy < YCells; iy++)
                        for (ix = 0; ix < XCells; ix++)
                            m_ViewLists[iy, ix] = new ViewList();
                    if (!ReadFileBacker)
                    {
                        m_fbacker.Seek(0, System.IO.SeekOrigin.Begin);
                        bwriter.Write(total);
                        bwriter.Write(m_rect.MinX);
                        bwriter.Write(m_rect.MaxX);
                        bwriter.Write(m_rect.MinY);
                        bwriter.Write(m_rect.MaxY);
                        breader = new System.IO.BinaryReader(m_fbacker);
                    }
                    while (total-- > 0)
                    {
                        long pos = m_fbacker.Position;
                        breader.ReadInt64();
                        side = breader.ReadInt16();
                        breader.ReadInt32();
                        breader.ReadInt16();
                        breader.ReadInt32();
                        ix = (int)((breader.ReadDouble() - m_rect.MinX) / DBCellSize);
                        iy = (int)((breader.ReadDouble() - m_rect.MinY) / DBCellSize);
                        breader.ReadDouble();
                        breader.ReadDouble();
                        breader.ReadDouble();
                        breader.ReadInt32();
                        breader.ReadDouble();
                        breader.ReadDouble();
                        if (side == 1) m_ViewLists[iy, ix].topids.Add(pos);
                        else if (side == 2) m_ViewLists[iy, ix].bottomids.Add(pos);
                        else throw new Exception("Only 1 and 2 are supported as side values. side value = " + side + " found. Aborting.");
                    }
                    foreach (ViewList v in m_ViewLists)
                    {
                        v.topids.Sort();
                        v.bottomids.Sort();                        
                    }
                    if (C.DBViewCorrection)
                    {
                        m_bsxx = m_bsxy = m_bsyx = m_bsyy = m_tsxx = m_tsxy = m_tsyx = m_tsyy = 0.0;
                        Console.Write("Computing correction...");
                        int tnx = 0, tny = 0;
                        int bnx = 0, bny = 0;
                        for (iy = 0; iy < YCells; iy++)
                            for (ix = 0; ix < XCells; ix++)
                            {
                                IntView v = new IntView(m_fbacker, m_ViewLists[iy, ix].topids, m_ViewLists[iy, ix].bottomids, C.DBViewCorrMinGrains, C.DBViewCorrPosTol);
                                v.Match(v, C.DBViewCorrPosTol, C.DBViewCorrSlopeTol, ref m_tsxx, ref m_tsxy, ref m_tsyx, ref m_tsyy, ref tnx, ref tny, ref m_bsxx, ref m_bsxy, ref m_bsyx, ref m_bsyy, ref bnx, ref bny);
                                if (ix + 1 < XCells)
                                    v.Match(new IntView(m_fbacker, m_ViewLists[iy, ix + 1].topids, m_ViewLists[iy, ix + 1].bottomids, C.DBViewCorrMinGrains, C.DBViewCorrPosTol),
                                         C.DBViewCorrPosTol, C.DBViewCorrSlopeTol, ref m_tsxx, ref m_tsxy, ref m_tsyx, ref m_tsyy, ref tnx, ref tny, ref m_bsxx, ref m_bsxy, ref m_bsyx, ref m_bsyy, ref bnx, ref bny);                                
                                if (iy + 1 < YCells)
                                    v.Match(new IntView(m_fbacker, m_ViewLists[iy + 1, ix].topids, m_ViewLists[iy + 1, ix].bottomids, C.DBViewCorrMinGrains, C.DBViewCorrPosTol),
                                         C.DBViewCorrPosTol, C.DBViewCorrSlopeTol, ref m_tsxx, ref m_tsxy, ref m_tsyx, ref m_tsyy, ref tnx, ref tny, ref m_bsxx, ref m_bsxy, ref m_bsyx, ref m_bsyy, ref bnx, ref bny);                                
                            }
                        if (tnx > 0 && tny > 0 && bnx > 0 && bny > 0)
                        {
                            m_tsxx = 1.0 + m_tsxx / tnx;
                            m_tsyx /= tnx;
                            m_tsxy /= tny;
                            m_tsyy = 1.0 + m_tsyy / tny;
                            m_bsxx = 1.0 + m_bsxx / bnx;
                            m_bsyx /= bnx;
                            m_bsxy /= bny;
                            m_bsyy = 1.0 + m_bsyy / bny;
                            System.Globalization.CultureInfo InvC = System.Globalization.CultureInfo.InvariantCulture;
                            Console.WriteLine("Correction factors: " + m_tsxx.ToString("F5", InvC) + " " + m_tsxy.ToString("F5", InvC) + " " + m_tsyx.ToString("F5", InvC) + " " + m_tsyy.ToString("F5", InvC) +
                                " " + m_bsxx.ToString("F5", InvC) + " " + m_bsxy.ToString("F5", InvC) + " " + m_bsyx.ToString("F5", InvC) + " " + m_bsyy.ToString("F5", InvC));
                        }
                        else
                        {
                            Console.WriteLine("Correction unreliable, switching to no-correction.");
                            m_tsxx = m_tsyy = m_bsxx = m_bsyy = 1.0;
                            m_tsxy = m_tsyx = m_bsxy = m_bsyx = 0.0;
                        }
                    }                   
                }
                catch (Exception x)
                {
                    if (bwriter != null) bwriter = null;
                    if (breader != null) breader = null;
                    if (m_fbacker != null)
                    {
                        m_fbacker.Close();
                        m_fbacker = null;
                    }
                    if (DeleteFileBacker && System.IO.File.Exists(m_TempFileBacker))
                        System.IO.File.Delete(m_TempFileBacker);
                    throw x;
                }

                Console.WriteLine("ViewMap " + m_ViewLists.GetLength(0) + " x " + m_ViewLists.GetLength(1));
                Cat = new Catalog(m_rect, m_ViewLists.GetLength(1), m_ViewLists.GetLength(0), DBCellSize);                

                SFL.Load = new SySal.Scanning.PostProcessing.dLoadFragment(FBackerDBFragment);
            }
            else
            {
                BaseName = input.ToLower().EndsWith(".rwc") ? (input.Substring(0, input.Length - 4)) : input;

                System.IO.FileStream f = null;
                do
                {
                    try
                    {
                        f = new System.IO.FileStream(BaseName + ".rwc", System.IO.FileMode.Open, System.IO.FileAccess.Read);
                        break;
                    }
                    catch (Exception x)
                    {
                        if (!WaitFragment) throw x;
                        else
                        {
                            Console.WriteLine("Waiting for catalog " + BaseName + ".rwc");
                            System.Threading.Thread.Sleep(2000);
                        }
                    }
                }
                while (true);
                Cat = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(f);               
                f.Close();
                m_ViewSize.X = Math.Abs(2.0 * Cat.Steps.X);
                m_ViewSize.Y = Math.Abs(2.0 * Cat.Steps.Y);
            
                SFL.Load = new SySal.Scanning.PostProcessing.dLoadFragment(LoadFragment);
            }
            LastPercent = 0.0;    
            LastTime = System.DateTime.Now.AddSeconds(-2.0);
            string tempoutput = System.Environment.ExpandEnvironmentVariables("%TEMP%\\batchlink_" + System.DateTime.Now.Ticks.ToString() + "_" + System.Diagnostics.Process.GetCurrentProcess().Id.ToString() + ".tlg");

            if (HasConsole) SFL.Progress = new SySal.Scanning.PostProcessing.dProgress(Progress);
            SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = null;
            if (C.UseDataStream)
            {
                SFL.LinkToFile(Cat, typeof(SySal.DataStreams.OPERALinkedZone), C.AutoCorrectMultipliers ? tempoutput : output);
                lz = new SySal.DataStreams.OPERALinkedZone(C.AutoCorrectMultipliers ? tempoutput : output);
            }
            else lz = SFL.Link(Cat);
            Console.WriteLine("Linked tracks (including promotion): " + lz.Length);
            ReturnSlopeCorrections.TopThickness = lz.Top.TopZ - lz.Top.BottomZ;
            ReturnSlopeCorrections.BaseThickness = lz.Top.BottomZ - lz.Bottom.TopZ;
            ReturnSlopeCorrections.BottomThickness = lz.Bottom.TopZ - lz.Bottom.BottomZ;
            ReturnSlopeCorrections.BottomDeltaSlope.X = C.BottomDeltaSlopeX;
            ReturnSlopeCorrections.BottomDeltaSlope.Y = C.BottomDeltaSlopeY;
            ReturnSlopeCorrections.BottomSlopeMultipliers.X = C.BottomMultSlopeX;
            ReturnSlopeCorrections.BottomSlopeMultipliers.Y = C.BottomMultSlopeY;
            ReturnSlopeCorrections.TopDeltaSlope.X = C.TopDeltaSlopeX;
            ReturnSlopeCorrections.TopDeltaSlope.Y = C.TopDeltaSlopeY;
            ReturnSlopeCorrections.TopSlopeMultipliers.X = C.TopMultSlopeX;
            ReturnSlopeCorrections.TopSlopeMultipliers.Y = C.TopMultSlopeY;
            if (C.UseDataStream)
            {
                ((SySal.DataStreams.OPERALinkedZone)lz).Dispose();
                lz = null;
            }
            if (C.AutoCorrectMultipliers)
            {
                System.IO.FileStream o = null;
                if (!C.UseDataStream)
                {
                    o = new System.IO.FileStream(tempoutput, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                    lz.Save(o);
                    o.Flush();
                }
                else lz = new SySal.DataStreams.OPERALinkedZone(C.AutoCorrectMultipliers ? tempoutput : output);

                try
                {
                    int i;
                    System.Collections.ArrayList tsl = new System.Collections.ArrayList(lz.Length);
                    System.Collections.ArrayList bsl = new System.Collections.ArrayList(lz.Length);
                    for (i = 0; i < lz.Length; i++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo info = lz[i].Info;
                        if (Math.Sqrt(info.Slope.X * info.Slope.X + info.Slope.Y * info.Slope.Y) > C.AutoCorrectMinSlope)
                        {
                            tsl.Add(new SlopeInfo(info.Slope.X, info.Slope.Y, lz[i].Top.Info.Slope.X, lz[i].Top.Info.Slope.Y));
                            bsl.Add(new SlopeInfo(info.Slope.X, info.Slope.Y, lz[i].Bottom.Info.Slope.X, lz[i].Bottom.Info.Slope.Y));
                        }
                    }
                    double xm, xd, ym, yd;
                    if (HasConsole) Console.WriteLine("Correcting top side slopes");
                    SlopeInfo.CorrectMultiplier((SlopeInfo [])tsl.ToArray(typeof(SlopeInfo)), C.AutoCorrectIterations, C.AutoCorrectStartDeltaAcceptance, C.AutoCorrectEndDeltaAcceptance, C.AutoCorrectMaxSlope, out xm, out ym, out xd, out yd, HasConsole);
                    C.TopMultSlopeX *= xm;
                    C.TopMultSlopeY *= ym;
                    C.TopDeltaSlopeX = xm * C.TopDeltaSlopeX + xd;
                    C.TopDeltaSlopeY = ym * C.TopDeltaSlopeY + yd;
                    if (HasConsole) Console.WriteLine("Correcting bottom side slopes");
                    SlopeInfo.CorrectMultiplier((SlopeInfo [])bsl.ToArray(typeof(SlopeInfo)), C.AutoCorrectIterations, C.AutoCorrectStartDeltaAcceptance, C.AutoCorrectEndDeltaAcceptance, C.AutoCorrectMaxSlope, out xm, out ym, out xd, out yd, HasConsole);
                    C.BottomMultSlopeX *= xm;
                    C.BottomMultSlopeY *= ym;
                    C.BottomDeltaSlopeX = xm * C.BottomDeltaSlopeX + xd;
                    C.BottomDeltaSlopeY = ym * C.BottomDeltaSlopeY + yd;
                    if (C.UseDataStream)
                    {
                        ((SySal.DataStreams.OPERALinkedZone)lz).Dispose();
                        lz = null;
                    }
                    if (HasConsole)
                    {
                        Console.WriteLine("TopMultSlopeX: {0}", C.TopMultSlopeX);
                        Console.WriteLine("TopMultSlopeY: {0}", C.TopMultSlopeY);
                        Console.WriteLine("TopDeltaSlopeX: {0}", C.TopDeltaSlopeX);
                        Console.WriteLine("TopDeltaSlopeY: {0}", C.TopDeltaSlopeY);
                        Console.WriteLine("BottomMultSlopeX: {0}", C.BottomMultSlopeX);
                        Console.WriteLine("BottomMultSlopeY: {0}", C.BottomMultSlopeY);
                        Console.WriteLine("BottomDeltaSlopeX: {0}", C.BottomDeltaSlopeX);
                        Console.WriteLine("BottomDeltaSlopeY: {0}", C.BottomDeltaSlopeY);
                    }
                    GC.Collect();
                    LastPercent = 0.0;
                    SFL.Config = C.LinkerConfig;
                    if (C.UseDataStream)
                    {
                        SFL.LinkToFile(Cat, typeof(SySal.DataStreams.OPERALinkedZone), output);
                        lz = new SySal.DataStreams.OPERALinkedZone(output);
                    }
                    else lz = SFL.Link(Cat);
                    Console.WriteLine("Linked tracks (including promotion): " + lz.Length);
                    ReturnSlopeCorrections.TopThickness = lz.Top.TopZ - lz.Top.BottomZ;
                    ReturnSlopeCorrections.BaseThickness = lz.Top.BottomZ - lz.Bottom.TopZ;
                    ReturnSlopeCorrections.BottomThickness = lz.Bottom.TopZ - lz.Bottom.BottomZ;
                    if (C.UseDataStream)
                    {
                        if (lz != null)
                        {
                            ((SySal.DataStreams.OPERALinkedZone)lz).Dispose();
                            lz = null;                            
                        }                        
                    }
                    ReturnSlopeCorrections.TopDeltaSlope.X = C.TopDeltaSlopeX;
                    ReturnSlopeCorrections.TopDeltaSlope.Y = C.TopDeltaSlopeY;
                    ReturnSlopeCorrections.TopSlopeMultipliers.X = C.TopMultSlopeX;
                    ReturnSlopeCorrections.TopSlopeMultipliers.Y = C.TopMultSlopeY;
                    ReturnSlopeCorrections.BottomDeltaSlope.X = C.BottomDeltaSlopeX;
                    ReturnSlopeCorrections.BottomDeltaSlope.Y = C.BottomDeltaSlopeY;
                    ReturnSlopeCorrections.BottomSlopeMultipliers.X = C.BottomMultSlopeX;
                    ReturnSlopeCorrections.BottomSlopeMultipliers.Y = C.BottomMultSlopeY;
                    if (o != null)
                    {
                        o.Close();
                        o = null;
                    }
                    System.IO.File.Delete(tempoutput);
                }
                catch(Exception x)
                {
                    if (o != null)
                    {
                        o.Close();
                        o = null;
                    }
                    Console.WriteLine("Relink failed");
                    Console.WriteLine(x.ToString());
                    if (C.UseDataStream)
                    {
                        if (lz != null)
                        {
                            ((SySal.DataStreams.OPERALinkedZone)lz).Dispose();
                            lz = null;                            
                        }                        
                    }
                    if (System.IO.File.Exists(output)) System.IO.File.Delete(output);
                    System.IO.File.Move(tempoutput, output);                    
                }
                if (o != null) o.Close();
            }
            string outstr = (output == null) ? "OK" : (C.UseDataStream ? output : SySal.OperaPersistence.Persist(output, lz));                        
            if (output != null && DBQuery)
            {
                if (C.UseDataStream) lz = new SySal.DataStreams.OPERALinkedZone(output);
                SySal.OperaDb.Scanning.DBMIPMicroTrackIndex dbmi = new SySal.OperaDb.Scanning.DBMIPMicroTrackIndex();
                dbmi.TopTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[lz.Top.Length];
                dbmi.BottomTracksIndex = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex[lz.Bottom.Length];
                int i;
                int xcells = m_ViewLists.GetLength(1);                
                for (i = 0; i < dbmi.TopTracksIndex.Length; i++)
                {
                    SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack tk = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz.Top[i];
                    if (tk.OriginalRawData.Track < 0) dbmi.TopTracksIndex[i] = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(-1,1,-1);
                    else
                    {
                        int frag = tk.OriginalRawData.Fragment;
                        m_fbacker.Position = (long)m_ViewLists[(frag - 1) / xcells, (frag - 1) % xcells].topids[tk.OriginalRawData.Track];
                        dbmi.TopTracksIndex[i] = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(breader.ReadInt64(), breader.ReadInt16(), breader.ReadInt32());
                    }
                }
                for (i = 0; i < dbmi.BottomTracksIndex.Length; i++)
                {
                    SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack tk = (SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)lz.Bottom[i];
                    if (tk.OriginalRawData.Track < 0) dbmi.BottomTracksIndex[i] = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(-1, 2, -1);
                    else
                    {
                        int frag = tk.OriginalRawData.Fragment;
                        m_fbacker.Position = (long)m_ViewLists[(frag - 1) / xcells, (frag - 1) % xcells].bottomids[tk.OriginalRawData.Track];
                        dbmi.BottomTracksIndex[i] = new SySal.OperaDb.TotalScan.DBMIPMicroTrackIndex(breader.ReadInt64(), breader.ReadInt16(), breader.ReadInt32());
                    }
                }
                if (C.UseDataStream)
                {
                    ((SySal.DataStreams.OPERALinkedZone)lz).Dispose();
                    lz = null;                            
                }
                SySal.OperaPersistence.Persist(output, dbmi);
            }

            if (output != null) SySal.OperaPersistence.Persist(output, ReturnSlopeCorrections);

            if (HasConsole) Progress(100.0);
            SFL = null;
            GC.Collect();
            if (output != null)
                Console.WriteLine("Result written to: " + outstr);

            if (m_fbacker != null)
            {
                m_fbacker.Close();
                m_fbacker = null;
            }
            if (m_TempFileBacker != null && DeleteFileBacker && System.IO.File.Exists(m_TempFileBacker))
                System.IO.File.Delete(m_TempFileBacker);
            if (DeleteFileBacker == false)
                Console.WriteLine("File backer path: " + m_TempFileBacker);

            return lz;
        }
    }
}
