using System;
using System.Collections.Generic;
using System.Text;
using SySal;
using SySal.Management;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Processing.MapMerge
{
    /// <summary>
    /// Configuration of a map-merging object.
    /// </summary>
    [Serializable]
    [XmlType("MapMerge.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable
    {
        /// <summary>
        /// Position tolerance for merging.
        /// </summary>
        public double PosTol;
        /// <summary>
        /// Slope tolerance for merging.
        /// </summary>
        public double SlopeTol;
        /// <summary>
        /// X-Y size of the map.
        /// </summary>
        public double MapSize;
        /// <summary>
        /// Maximum position offset.
        /// </summary>
        public double MaxPosOffset;
        /// <summary>
        /// Minimum number of matching tracks.
        /// </summary>
        public int MinMatches;
        /// <summary>
        /// Set to <c>true</c> to use partial statistics in mapping, <c>false</c> otherwise.
        /// </summary>
        public bool FavorSpeedOverAccuracy;
        /// <summary>
        /// Creates a new configuration with the specified name.
        /// </summary>
        /// <param name="name">the name to be assigned.</param>
        public Configuration(string name) : base(name) { }
        /// <summary>
        /// Creates a new configuration with an empty name.
        /// </summary>
        public Configuration() : base("") { }

        #region ICloneable Members
        /// <summary>
        /// Copies the configuration.
        /// </summary>
        /// <returns>the new configuration.</returns>
        public override object Clone()
        {
            Configuration c = new Configuration();
            c.Name = Name;
            c.PosTol = PosTol;
            c.SlopeTol = SlopeTol;
            c.MapSize = MapSize;                        
            c.MaxPosOffset = MaxPosOffset;
            c.MinMatches = MinMatches;
            c.FavorSpeedOverAccuracy = FavorSpeedOverAccuracy;
            return c;
        }

        #endregion
    }

    /// <summary>
    /// Result of a mapping operation.
    /// </summary>
    public struct MapResult
    {
        /// <summary>
        /// Number of matching track pairs.
        /// </summary>
        public int Matches;
        /// <summary>
        /// Position difference (average).
        /// </summary>
        public SySal.BasicTypes.Vector2 DeltaPos;
        /// <summary>
        /// RMS of position differences.
        /// </summary>
        public SySal.BasicTypes.Vector2 DeltaPosRMS;
        /// <summary>
        /// Slope difference (average).
        /// </summary>
        public SySal.BasicTypes.Vector2 DeltaSlope;
        /// <summary>
        /// RMS of slope difference.
        /// </summary>
        public SySal.BasicTypes.Vector2 DeltaSlopeRMS;
        /// <summary>
        /// <c>true</c> if the mapping procedure succeeded, <c>false</c> otherwise.
        /// </summary>
        public bool Valid;
    }

    /// <summary>
    /// Plate side to be used for mapping.
    /// </summary>
    public enum MapSide 
    { 
        /// <summary>
        /// Use base tracks.
        /// </summary>
        Base = 0, 
        /// <summary>
        /// Use microtracks from top side.
        /// </summary>
        Top = 1, 
        /// <summary>
        /// Use microtracks from bottom side.
        /// </summary>    
        Bottom = 2 
    };

    /// <summary>
    /// Manages track maps.
    /// </summary>
    public class MapManager
    {
        /// <summary>
        /// Generic map filter. 
        /// </summary>
        /// <param name="t">the track to be checked.</param>
        /// <returns><c>true</c> if the track is to be kept, <c>false</c> otherwise.</returns>
        public delegate bool dMapFilter(object t);

        static SySal.Tracking.MIPEmulsionTrackInfo[] lzExtractMap(SySal.Scanning.Plate.LinkedZone lz, MapSide side, dMapFilter flt)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            int n;
            switch (side)
            {
                case MapSide.Base: n = lz.Length; break;
                case MapSide.Top: n = lz.Top.Length; break;
                case MapSide.Bottom: n = lz.Bottom.Length; break;
                default: throw new Exception("Internal inconsistency: side = " + side + " is not supported.");
            }
            int i;
            for (i = 0; i < n; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info;
                switch (side)
                {
                    case MapSide.Base: info = lz[i].Info; break;
                    case MapSide.Top: info = lz.Top[i].Info; break;
                    case MapSide.Bottom: info = lz.Bottom[i].Info; break;
                    default: throw new Exception("Internal inconsistency: side = " + side + " is not supported.");
                }
                if (flt == null || flt(info))
                    ar.Add(info);
            }
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
        }

        static SySal.Tracking.MIPEmulsionTrackInfo[] lzExtractMap(SySal.Scanning.Plate.LinkedZone lz, MapSide side, SySal.BasicTypes.Rectangle r, dMapFilter flt)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            int n;
            switch (side)
            {
                case MapSide.Base: n = lz.Length; break;
                case MapSide.Top: n = lz.Top.Length; break;
                case MapSide.Bottom: n = lz.Bottom.Length; break;
                default: throw new Exception("Internal inconsistency: side = " + side + " is not supported.");
            }
            int i;
            for (i = 0; i < n; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info;
                switch (side)
                {
                    case MapSide.Base: info = lz[i].Info; break;
                    case MapSide.Top: info = lz.Top[i].Info; break;
                    case MapSide.Bottom: info = lz.Bottom[i].Info; break;
                    default: throw new Exception("Internal inconsistency: side = " + side + " is not supported.");
                }                
                if (info.Intercept.X < r.MinX || info.Intercept.X > r.MaxX || info.Intercept.Y < r.MinY || info.Intercept.Y > r.MaxY) continue;
                if (flt == null || flt(info))
                    ar.Add(info);
            }
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
        }

        static SySal.Tracking.MIPEmulsionTrackInfo[] layExtractMap(SySal.TotalScan.Layer lay, dMapFilter flt, bool useoriginal)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            int n = lay.Length;
            int i;
            for (i = 0; i < n; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = useoriginal ? lay[i].OriginalInfo : lay[i].Info;
                if (flt == null || flt(lay[i]))
                    ar.Add(info);
            }
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
        }

        static SySal.Tracking.MIPEmulsionTrackInfo[] layExtractMap(SySal.TotalScan.Layer lay, SySal.BasicTypes.Rectangle r, dMapFilter flt, bool useoriginal)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            int n = lay.Length;
            int i;
            for (i = 0; i < n; i++)            
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = useoriginal ? lay[i].OriginalInfo : lay[i].Info;
                if (info.Intercept.X < r.MinX || info.Intercept.X > r.MaxX || info.Intercept.Y < r.MinY || info.Intercept.Y > r.MaxY) continue;
                if (flt == null || flt(lay[i]))
                    ar.Add(info);
            }
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
        }

        static SySal.Tracking.MIPEmulsionTrackInfo[] tkExtractMap(SySal.Tracking.MIPEmulsionTrackInfo[] tks, SySal.BasicTypes.Rectangle r, dMapFilter flt)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            foreach (SySal.Tracking.MIPEmulsionTrackInfo info in tks)
            {
                if (info.Intercept.X < r.MinX || info.Intercept.X > r.MaxX || info.Intercept.Y < r.MinY || info.Intercept.Y > r.MaxY) continue;
                if (flt == null || flt(info))
                    ar.Add(info);
            }
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));

        }

        static SySal.Tracking.MIPEmulsionTrackInfo[] tkExtractMap(SySal.Tracking.MIPEmulsionTrackInfo[] tks, dMapFilter flt)
        {
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            foreach (SySal.Tracking.MIPEmulsionTrackInfo info in tks)
                if (flt == null || flt(info))
                    ar.Add(info);
            return (SySal.Tracking.MIPEmulsionTrackInfo[])ar.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
        }

        /// <summary>
        /// Extracts a track map from the data.
        /// </summary>
        /// <param name="data">the input data; can be an array of microtracks, a linked zone or a TotalScan volume.</param>
        /// <param name="side">the side to be used.</param>
        /// <param name="flt">the track filter; leave <c>null</c> to skip filtering.</param>
        /// <param name="useoriginal">if <c>true</c>, the original (anti-transformed) tracks are used; ignored if the input data is other than a TotalScan volume.</param>
        /// <returns>the subset of tracks to be used for mapping.</returns>        
        public static SySal.Tracking.MIPEmulsionTrackInfo[] ExtractMap(object data, MapSide side, dMapFilter flt, bool useoriginal)
        {
            if (data is SySal.Scanning.Plate.LinkedZone)
            {
                return lzExtractMap((SySal.Scanning.Plate.LinkedZone)data, side, flt);
            }
            else if (data is SySal.TotalScan.Layer)
            {
                int s = ((SySal.TotalScan.Layer)data).Side;
                if (s != (int)side) throw new Exception("Expected side = " + side + " but found " + s + ".");
                return layExtractMap((SySal.TotalScan.Layer)data, flt, useoriginal);
            }
            else if (data is SySal.Tracking.MIPEmulsionTrackInfo[])
            {
                return tkExtractMap((SySal.Tracking.MIPEmulsionTrackInfo [])data, flt);
            }
            throw new Exception("Map extraction from type " + data.GetType() + " is not supported.");
        }

        /// <summary>
        /// Extracts a track map from the data.
        /// </summary>
        /// <param name="data">the input data; can be an array of microtracks, a linked zone or a TotalScan volume.</param>
        /// <param name="side">the side to be used.</param>
        /// <param name="r">the rectangle that sets the bounds for the track map to be extracted.</param>
        /// <param name="flt">the track filter; leave <c>null</c> to skip filtering.</param>
        /// <param name="useoriginal">if <c>true</c>, the original (anti-transformed) tracks are used; ignored if the input data is other than a TotalScan volume.</param>
        /// <returns>the subset of tracks to be used for mapping.</returns>
        public static SySal.Tracking.MIPEmulsionTrackInfo[] ExtractMap(object data, MapSide side, SySal.BasicTypes.Rectangle r, dMapFilter flt, bool useoriginal)
        {
            if (data is SySal.Scanning.Plate.LinkedZone)
            {
                return lzExtractMap((SySal.Scanning.Plate.LinkedZone)data, side, r, flt);
            }
            else if (data is SySal.TotalScan.Layer)
            {
                int s = ((SySal.TotalScan.Layer)data).Side;
                if (s != (int)side) throw new Exception("Expected side = " + side + " but found " + s + ".");
                return layExtractMap((SySal.TotalScan.Layer)data, r, flt, useoriginal);
            }
            else if (data is SySal.Tracking.MIPEmulsionTrackInfo[])
            {
                return tkExtractMap((SySal.Tracking.MIPEmulsionTrackInfo[])data, r, flt);
            }
            throw new Exception("Map extraction from type " + data.GetType() + " is not supported.");
        }
    }

    /// <summary>
    /// Works out parameters for 2D transformation from mapping data.
    /// </summary>
    public class TransformFitter
    {
        /// <summary>
        /// Computes the transformation parameters for a translation.
        /// </summary>
        /// <param name="xypairs">the set of x,y pairs where displacements are known.</param>
        /// <param name="dxdypairs">the set of deltax,deltay pairs measured.</param>
        /// <returns>the transformation parameters.</returns>
        public static SySal.DAQSystem.Scanning.IntercalibrationInfo FindTranslation(double[,] xypairs, double[,] dxdypairs)
        {
            double dx = 0.0, dy = 0.0;
            int i, n;
            n = dxdypairs.GetLength(0);
            for (i = 0; i < n; i++)
            {
                dx += dxdypairs[i, 0];
                dy += dxdypairs[i, 1];
            }
            SySal.DAQSystem.Scanning.IntercalibrationInfo cal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            cal.MXX = cal.MYY = 1.0;
            cal.MXY = cal.MYX = 0.0;
            cal.TX = dx / n;
            cal.TY = dy / n;
            return cal;
        }
        /// <summary>
        /// Computes the transformation parameters for a rototranslation with expansion.
        /// </summary>
        /// <param name="xypairs">the set of x,y pairs where displacements are known.</param>
        /// <param name="dxdypairs">the set of deltax,deltay pairs measured.</param>
        /// <returns>the transformation parameters.</returns>
        public static SySal.DAQSystem.Scanning.IntercalibrationInfo FindRototranslation(double[,] xypairs, double[,] dxdypairs)
        {
            double[,] cmat = new double[4, 4];
            double[] v = new double[4];
            int i, n;
            n = dxdypairs.GetLength(0);
            double avgx = 0.0, avgy = 0.0, x, y;
            for (i = 0; i < n; i++)
            {
                avgx += xypairs[i, 0];
                avgy += xypairs[i, 1];
            }
            avgx /= n;
            avgy /= n;
            SySal.DAQSystem.Scanning.IntercalibrationInfo cal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            cal.RX = avgx;
            cal.RY = avgy;
            for (i = 0; i < n; i++)
            {
                x = xypairs[i, 0] - avgx;
                y = xypairs[i, 1] - avgy;
                
                cmat[0, 0] += x * x + y * y;
                cmat[0, 2] += x;
                cmat[0, 3] += y;
                v[0] += dxdypairs[i, 0] * x + dxdypairs[i, 1] * y;

                cmat[1, 1] += x * x + y * y;
                cmat[1, 2] += y;
                cmat[1, 3] -= x;
                v[1] += dxdypairs[i, 0] * y - dxdypairs[i, 1] * x;

                cmat[2, 0] += x;
                cmat[2, 1] += y;
                cmat[2, 2] += 1.0;
                v[2] += dxdypairs[i, 0];

                cmat[3, 0] += y;
                cmat[3, 1] -= x;
                cmat[3, 3] += 1.0;
                v[3] += dxdypairs[i, 1];
            }
            NumericalTools.Cholesky ch = new NumericalTools.Cholesky(cmat, 1.0e-8);
            v = ch.Solve(v);
            cal.MXX = cal.MYY = 1.0 + v[0];
            cal.MXY = v[1];
            cal.MYX = -v[1];
            cal.TX = v[2];
            cal.TY = v[3];
            return cal;
        }
        /// <summary>
        /// Computes the transformation parameters for a full affine transformation.
        /// </summary>
        /// <param name="xypairs">the set of x,y pairs where displacements are known.</param>
        /// <param name="dxdypairs">the set of deltax,deltay pairs measured.</param>
        /// <returns>the transformation parameters.</returns>
        public static SySal.DAQSystem.Scanning.IntercalibrationInfo FindAffineTransformation(double[,] xypairs, double[,] dxdypairs)
        {
            double[,] cmat = new double[6, 6];
            double[] v = new double[6];
            int i, n;
            n = dxdypairs.GetLength(0);
            double avgx = 0.0, avgy = 0.0, x, y;
            for (i = 0; i < n; i++)
            {
                avgx += xypairs[i, 0];
                avgy += xypairs[i, 1];
            }
            avgx /= n;
            avgy /= n;
            SySal.DAQSystem.Scanning.IntercalibrationInfo cal = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            cal.RX = avgx;
            cal.RY = avgy;
            for (i = 0; i < n; i++)
            {
                x = xypairs[i, 0] - avgx;
                y = xypairs[i, 1] - avgy;

                cmat[0, 0] += x * x;
                cmat[0, 1] += x * y;
                cmat[0, 4] += x;
                v[0] += dxdypairs[i, 0] * x;

                cmat[1, 0] += x * y;
                cmat[1, 1] += y * y;
                cmat[1, 4] += y;
                v[1] += dxdypairs[i, 0] * y;

                cmat[2, 2] += x * x;
                cmat[2, 3] += x * y;
                cmat[2, 5] += x;
                v[2] += dxdypairs[i, 1] * x;

                cmat[3, 2] += x * y;
                cmat[3, 3] += y * y;
                cmat[3, 5] += y;
                v[3] += dxdypairs[i, 1] * y;

                cmat[4, 0] += x;
                cmat[4, 1] += y;
                cmat[4, 4] += 1.0;
                v[4] += dxdypairs[i, 0];

                cmat[5, 2] += x;
                cmat[5, 3] += y;
                cmat[5, 5] += 1.0;
                v[5] += dxdypairs[i, 1];
            }
            NumericalTools.Cholesky ch = new NumericalTools.Cholesky(cmat, 1.0e-8);
            v = ch.Solve(v);
            cal.MXX = 1.0 + v[0];
            cal.MXY = v[1];
            cal.MYX = v[2];
            cal.MYY = 1.0 + v[3];
            cal.TX = v[4];
            cal.TY = v[5];
            return cal;
        }
    }

    /// <summary>
    /// Map Merger.
    /// </summary>
    [Serializable]
    [XmlType("MapMerge.MapMerger")]
    public class MapMerger : SySal.Management.IManageable
    {
        /// <summary>
        /// Retrieves the class name.
        /// </summary>
        /// <returns>the class type.</returns>
        public override string ToString()
        {
            return "Map Merger";
        }

        [NonSerialized]
        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        /// <summary>
        /// Builds a new Map Merger object.
        /// </summary>
        public MapMerger()
        {
            m_Config = new Configuration();
            m_Config.PosTol = 10.0;
            m_Config.SlopeTol = 0.02;
            m_Config.MaxPosOffset = 100.0;
            m_Config.MapSize = 1000.0;
            m_Config.MinMatches = 2;
        }
        /// <summary>
        /// Internal configuration.
        /// </summary>
        protected Configuration m_Config;
        /// <summary>
        /// Internal QuickMapper.
        /// </summary>
        protected QuickMapping.QuickMapper m_QM = new SySal.Processing.QuickMapping.QuickMapper();

        #region IManageable Members

        /// <summary>
        /// Member field on which the Name property relies.
        /// </summary>
        [NonSerialized]
        protected string m_Name;
        /// <summary>
        /// The name of the Map Merger.
        /// </summary>
        public string Name
        {
            get
            {
                return (string)(m_Name.Clone());
            }
            set
            {
                m_Name = (string)(value.Clone());
            }
        }

        /// <summary>
        /// Gets/sets the object configuration.
        /// </summary>
        [XmlElement(typeof(MapMerge.Configuration))]
        public SySal.Management.Configuration Config
        {
            get
            {
                return (SySal.Management.Configuration)m_Config.Clone();
            }
            set
            {
                m_Config = (Configuration)value;
            }
        }

        /// <summary>
        /// List of connections. It is always empty for MapMerge.
        /// </summary>
        public IConnectionList Connections
        {
            get { return EmptyConnectionList; }
        }

        /// <summary>
        /// GUI editor to configure the algorithm parameters.
        /// </summary>
        /// <param name="c">the configuration to be edited.</param>
        /// <returns><c>true</c> if the configuration is accepted, <c>false</c> otherwise.</returns>
        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            EditConfigForm ec = new EditConfigForm();
            ec.C = (SySal.Processing.MapMerge.Configuration)(c.Clone());
            if (ec.ShowDialog() == System.Windows.Forms.DialogResult.OK)
            {
                c = (SySal.Processing.MapMerge.Configuration)(ec.C.Clone());
                return true;
            }
            return false;
        }
        /// <summary>
        /// Gets/sets the monitor.
        /// </summary>
        /// <remarks>Only <c>false</c> is a persistent status.</remarks>
        public bool MonitorEnabled
        {
            get
            {
                return false;
            }
            set
            {
                ;
            }
        }

        #endregion

        /// <summary>
        /// Performs multiple matching.
        /// </summary>
        /// <param name="refmap">the reference map.</param>
        /// <param name="mmaps">the list of track maps to be merged.</param>
        /// <returns>the result of pattern matching with each track pattern.</returns>
        public MapResult[] Map(SySal.Tracking.MIPEmulsionTrackInfo[] refmap, params SySal.Tracking.MIPEmulsionTrackInfo[][] mmaps)
        {
            SySal.Processing.QuickMapping.Configuration qmc = (SySal.Processing.QuickMapping.Configuration)m_QM.Config;
            qmc.FullStatistics = !m_Config.FavorSpeedOverAccuracy;
            qmc.UseAbsoluteReference = true;
            qmc.PosTol = m_Config.PosTol;
            qmc.SlopeTol = m_Config.SlopeTol;
            MapResult[] mres = new MapResult[mmaps.Length];

            int i;
            for (i = 0; i < mmaps.Length; i++)            
            {
                SySal.Scanning.PostProcessing.PatternMatching.TrackPair [] pairs = m_QM.Match(refmap, mmaps[i], 0.0, m_Config.MaxPosOffset, m_Config.MaxPosOffset);
                mres[i].Valid = (mres[i].Matches = pairs.Length) >= m_Config.MinMatches;
                if (pairs.Length <= 0)
                {
                    mres[i].Valid = false;
                    continue;
                }
                SySal.BasicTypes.Vector2 dp = new SySal.BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 dp2 = new SySal.BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 ds = new SySal.BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 ds2 = new SySal.BasicTypes.Vector2();
                double dx, dy;
                foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair p in pairs)
                {
                    dx = (p.First.Info.Intercept.X - p.Second.Info.Intercept.X);
                    dy = (p.First.Info.Intercept.Y - p.Second.Info.Intercept.Y);
                    dp.X += dx;
                    dp.Y += dy;
                    dp2.X += (dx * dx);
                    dp2.Y += (dy * dy);
                    dx = (p.First.Info.Slope.X - p.Second.Info.Slope.X);
                    dy = (p.First.Info.Slope.Y - p.Second.Info.Slope.Y);
                    ds.X += dx;
                    ds.Y += dy;
                    ds2.X += (dx * dx);
                    ds2.Y += (dy * dy);
                }
                dp.X /= pairs.Length;
                dp.Y /= pairs.Length;
                dp2.X = Math.Sqrt(dp2.X / pairs.Length - dp.X * dp.X);
                dp2.Y = Math.Sqrt(dp2.Y / pairs.Length - dp.Y * dp.Y);
                ds.X /= pairs.Length;
                ds.Y /= pairs.Length;
                ds2.X = Math.Sqrt(ds2.X / pairs.Length - ds.X * ds.X);
                ds2.Y = Math.Sqrt(ds2.Y / pairs.Length - ds.Y * ds.Y);
                mres[i].DeltaPos = dp;
                mres[i].DeltaPosRMS = dp2;
                mres[i].DeltaSlope = ds;
                mres[i].DeltaSlopeRMS = ds2;
            }
            return mres;
        }

        /// <summary>
        /// Maps a pattern of tracks onto another one.
        /// </summary>
        /// <param name="refpattern">the reference pattern.</param>
        /// <param name="mappattern">the pattern to be mapped.</param>
        /// <param name="flt">the filter function for mapping.</param>
        /// <param name="logstrw">the output stream where logging information is written; set to <c>null</c> to disable logging.</param>
        /// <returns>the transformation obtained.</returns>
        public SySal.DAQSystem.Scanning.IntercalibrationInfo MapTransform(SySal.Tracking.MIPEmulsionTrackInfo[] refpattern, SySal.Tracking.MIPEmulsionTrackInfo[] mappattern, MapManager.dMapFilter flt, System.IO.TextWriter logstrw)
        {
            SySal.DAQSystem.Scanning.IntercalibrationInfo calinfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
            try
            {
                if (logstrw != null) logstrw.WriteLine("Begin pattern mapping.");
                calinfo.MXX = calinfo.MYY = 1.0;
                calinfo.MXY = calinfo.MYX = 0.0;
                calinfo.RX = calinfo.RY = calinfo.TX = calinfo.TY = calinfo.TZ = 0.0;
                int nr = refpattern.Length;
                int na = mappattern.Length;
                if (logstrw != null)
                {
                    logstrw.WriteLine("Ref tracks: " + nr);
                    logstrw.WriteLine("Add tracks: " + na);
                }
                if (nr == 0 || na == 0) return calinfo;
                SySal.BasicTypes.Rectangle refrect = new SySal.BasicTypes.Rectangle();
                SySal.BasicTypes.Rectangle addrect = new SySal.BasicTypes.Rectangle();
                SySal.Tracking.MIPEmulsionTrackInfo refinfo = refpattern[0];
                SySal.Tracking.MIPEmulsionTrackInfo addinfo = mappattern[0];
                refrect.MinX = refrect.MaxX = refinfo.Intercept.X;
                refrect.MinY = refrect.MaxY = refinfo.Intercept.Y;
                addrect.MinX = addrect.MaxX = addinfo.Intercept.X;
                addrect.MinY = addrect.MaxY = addinfo.Intercept.Y;
                int i;
                for (i = 1; i < nr; i++)
                {
                    refinfo = refpattern[i];
                    if (refinfo.Intercept.X < refrect.MinX) refrect.MinX = refinfo.Intercept.X;
                    else if (refinfo.Intercept.X > refrect.MaxX) refrect.MaxX = refinfo.Intercept.X;
                    if (refinfo.Intercept.Y < refrect.MinY) refrect.MinY = refinfo.Intercept.Y;
                    else if (refinfo.Intercept.Y > refrect.MaxY) refrect.MaxY = refinfo.Intercept.Y;
                }
                for (i = 1; i < na; i++)
                {
                    addinfo = mappattern[i];
                    if (addinfo.Intercept.X < addrect.MinX) addrect.MinX = addinfo.Intercept.X;
                    else if (addinfo.Intercept.X > addrect.MaxX) addrect.MaxX = addinfo.Intercept.X;
                    if (addinfo.Intercept.Y < addrect.MinY) addrect.MinY = addinfo.Intercept.Y;
                    else if (addinfo.Intercept.Y > addrect.MaxY) addrect.MaxY = addinfo.Intercept.Y;
                }
                SySal.BasicTypes.Rectangle maprect = new SySal.BasicTypes.Rectangle();
                maprect.MinX = Math.Max(refrect.MinX, addrect.MinX);
                maprect.MaxX = Math.Min(refrect.MaxX, addrect.MaxX);
                maprect.MinY = Math.Max(refrect.MinY, addrect.MinY);
                maprect.MaxY = Math.Min(refrect.MaxY, addrect.MaxY);
                int xcells = (int)Math.Ceiling((maprect.MaxX - maprect.MinX) / m_Config.MapSize);
                int ycells = (int)Math.Ceiling((maprect.MaxY - maprect.MinY) / m_Config.MapSize);
                if (logstrw != null)
                {
                    logstrw.WriteLine("Ref rect: " + refrect.MinX + " " + refrect.MaxX + " " + refrect.MinY + " " + refrect.MaxY);
                    logstrw.WriteLine("Map rect: " + addrect.MinX + " " + addrect.MaxX + " " + addrect.MinY + " " + addrect.MaxY);
                    logstrw.WriteLine("Common rect: " + maprect.MinX + " " + maprect.MaxX + " " + maprect.MinY + " " + maprect.MaxY);
                    logstrw.WriteLine("X cells: " + xcells + " Y cells: " + ycells);
                }
                if (xcells <= 0 || ycells <= 0) return calinfo;
                int ix, iy;
                System.Collections.ArrayList[,] rmaps = new System.Collections.ArrayList[ycells, xcells];
                System.Collections.ArrayList[,] amaps = new System.Collections.ArrayList[ycells, xcells];
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        rmaps[iy, ix] = new System.Collections.ArrayList();
                        amaps[iy, ix] = new System.Collections.ArrayList();
                    }
                for (i = 0; i < nr; i++)
                {
                    refinfo = refpattern[i];
                    ix = (int)((refinfo.Intercept.X - maprect.MinX) / m_Config.MapSize);
                    if (ix < 0 || ix >= xcells) continue;
                    iy = (int)((refinfo.Intercept.Y - maprect.MinY) / m_Config.MapSize);
                    if (iy < 0 || iy >= ycells) continue;
                    if (flt == null || flt(refinfo))
                        rmaps[iy, ix].Add(refinfo);
                }
                for (i = 0; i < na; i++)
                {
                    addinfo = mappattern[i];
                    ix = (int)((addinfo.Intercept.X - maprect.MinX) / m_Config.MapSize);
                    if (ix < 0 || ix >= xcells) continue;
                    iy = (int)((addinfo.Intercept.Y - maprect.MinY) / m_Config.MapSize);
                    if (iy < 0 || iy >= ycells) continue;
                    if (flt == null || flt(addinfo))
                        amaps[iy, ix].Add(addinfo);
                }
                System.Collections.ArrayList mres = new System.Collections.ArrayList();
                for (ix = 0; ix < xcells; ix++)
                    for (iy = 0; iy < ycells; iy++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo[] ri = (SySal.Tracking.MIPEmulsionTrackInfo[])rmaps[iy, ix].ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
                        if (ri.Length <= 0) continue;
                        SySal.Tracking.MIPEmulsionTrackInfo[] ai = (SySal.Tracking.MIPEmulsionTrackInfo[])amaps[iy, ix].ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo));
                        if (ai.Length <= 0) continue;
                        MapResult mr = Map(ri, ai)[0];
                        if (mr.Valid)
                        {
                            SySal.BasicTypes.Vector2 p = new SySal.BasicTypes.Vector2();
                            p.X = maprect.MinX + m_Config.MapSize * (ix + 0.5);
                            p.Y = maprect.MinY + m_Config.MapSize * (iy + 0.5);
                            mres.Add(new object[] { p, mr });
                            logstrw.WriteLine("Z ix " + ix + " iy " + iy + " matches " + mr.Matches + " X " + p.X + " Y " + p.Y + " DeltaX/Y " + mr.DeltaPos.X + "/" + mr.DeltaPos.Y + " RMSX/Y " + mr.DeltaPosRMS.X + "/" + mr.DeltaPosRMS.Y + " DeltaSX/Y " + mr.DeltaSlope.X + "/" + mr.DeltaSlope.Y + " RMSX/Y " + mr.DeltaSlopeRMS.X + "/" + mr.DeltaSlopeRMS.Y);
                        }
                        else if (logstrw != null)
                            logstrw.WriteLine("Z ix " + ix + " iy " + iy + " matches " + mr.Matches);
                    }
                double[,] inXY = new double[mres.Count, 2];
                double[,] dXY = new double[mres.Count, 2];
                for (i = 0; i < mres.Count; i++)
                {
                    object[] o = (object[])mres[i];
                    inXY[i, 0] = ((SySal.BasicTypes.Vector2)o[0]).X;
                    inXY[i, 1] = ((SySal.BasicTypes.Vector2)o[0]).Y;
                    dXY[i, 0] = -((MapResult)o[1]).DeltaPos.X;
                    dXY[i, 1] = -((MapResult)o[1]).DeltaPos.Y;
                    if (logstrw != null)
                        logstrw.WriteLine("Zone " + i + " matches " + ((MapResult)o[1]).Matches + " " + inXY[i, 0] + " " + inXY[i, 1] + " " + dXY[i, 0] + " " + dXY[i, 1]);
                }
                switch (mres.Count)
                {
                    case 0: return calinfo;
                    case 1: return calinfo = TransformFitter.FindTranslation(inXY, dXY);
                    case 2: return calinfo = TransformFitter.FindRototranslation(inXY, dXY);
                    default:
                        try
                        {
                            return calinfo = TransformFitter.FindAffineTransformation(inXY, dXY);
                        }
                        catch (Exception)
                        {
                            return calinfo = TransformFitter.FindRototranslation(inXY, dXY);
                        }
                }
            }
            finally
            {
                if (logstrw != null)
                    logstrw.WriteLine("End mapping with RX/Y " + calinfo.RX + "/" + calinfo.RY + " MXX/XY/YX/YY " + calinfo.MXX + "/" + calinfo.MXY + "/" + calinfo.MYX + "/" + calinfo.MYY + " TX/Y " + calinfo.TX + "/" + calinfo.TY + ".");
            }
        }

        /// <summary>
        /// Adds segments to an existing layer with a specified mapping transformation.
        /// </summary>
        /// <param name="lay">the layer that is to receive the new segments.</param>
        /// <param name="addsegs">the segments to be added.</param>
        /// <param name="calinfo">the mapping transformation to be used.</param>
        public void AddToLayer(SySal.TotalScan.Flexi.Layer lay, SySal.TotalScan.Flexi.Segment [] addsegs, SySal.DAQSystem.Scanning.IntercalibrationInfo calinfo)
        {
            int i;
            SySal.TotalScan.Segment [] segs = new SySal.TotalScan.Segment[addsegs.Length];
            for (i = 0; i < addsegs.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = addsegs[i].Info;
                info.Slope = calinfo.Deform(info.Slope);
                info.Intercept = calinfo.Transform(info.Intercept);
                addsegs[i].SetInfo(info);
                segs[i] = addsegs[i];
            }
            lay.AddSegments(segs);
        }

        /// <summary>
        /// Adds segments, tracks and vertices of a volume to another one.
        /// </summary>
        /// <param name="refvol">the volume to be augmented with the content of the other.</param>
        /// <param name="addvol">segments, tracks and vertices from this volume are added to the other.</param>
        /// <param name="ds">the dataset that should be assigned to imported tracks.</param>
        /// <param name="fds">the dataset that should be imported; if this parameter is <c>null</c>, all datasets are imported.</param>
        /// <param name="flt">track mapping filter function.</param>
        /// <param name="logstrw">the stream where logging information is to be dumped; set to <c>null</c> to disable logging.</param>
        public void AddToVolume(SySal.TotalScan.Flexi.Volume refvol, SySal.TotalScan.Flexi.Volume addvol, SySal.TotalScan.Flexi.DataSet ds, SySal.TotalScan.Flexi.DataSet fds, MapManager.dMapFilter flt, System.IO.TextWriter logstrw)
        {
            if (logstrw != null) logstrw.WriteLine("Begin AddToVolume.");
#if !DEBUG
            try
            {
#endif
                int i, j, n;
                SySal.DAQSystem.Scanning.IntercalibrationInfo[] calinfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo[addvol.Layers.Length];
                for (i = 0; i < addvol.Layers.Length; i++)
                {
                    for (j = 0; j < refvol.Layers.Length && (refvol.Layers[j].BrickId != addvol.Layers[i].BrickId || refvol.Layers[j].SheetId != addvol.Layers[i].SheetId || refvol.Layers[j].Side != addvol.Layers[i].Side); j++) ;
                    if (j == refvol.Layers.Length) throw new Exception("No reference layer found for Brick/Sheet/Side = " + addvol.Layers[i].BrickId + "/" + addvol.Layers[i].SheetId + "/" + addvol.Layers[i].Side);
                    if (logstrw != null) logstrw.WriteLine("Seeking mapping for layer " + i + " Brick/Sheet/Side " + refvol.Layers[i].BrickId + "/" + refvol.Layers[i].SheetId + "/" + refvol.Layers[i].Side);
                    calinfo[i] = MapTransform(MapManager.ExtractMap(refvol.Layers[j], (MapSide)refvol.Layers[j].Side, flt, true), MapManager.ExtractMap(addvol.Layers[i], (MapSide)refvol.Layers[j].Side, flt, true), null, logstrw);
                }
                for (i = 0; i < addvol.Layers.Length; i++)
                {
                    SySal.TotalScan.Layer lay = addvol.Layers[i];
                    n = lay.Length;
                    SySal.DAQSystem.Scanning.IntercalibrationInfo cinfo = calinfo[i];
                    SySal.DAQSystem.Scanning.IntercalibrationInfo alinfo = new SySal.DAQSystem.Scanning.IntercalibrationInfo();
                    SySal.TotalScan.AlignmentData al = lay.AlignData;
                    alinfo.MXX = al.AffineMatrixXX * cinfo.MXX + al.AffineMatrixXY * cinfo.MYX;
                    alinfo.MXY = al.AffineMatrixXX * cinfo.MXY + al.AffineMatrixXY * cinfo.MYY;
                    alinfo.MYX = al.AffineMatrixYX * cinfo.MXX + al.AffineMatrixYY * cinfo.MYX;
                    alinfo.MYY = al.AffineMatrixYX * cinfo.MXY + al.AffineMatrixYY * cinfo.MYY;
                    double rx = lay.RefCenter.X - cinfo.RX;
                    double ry = lay.RefCenter.Y - cinfo.RY;
                    alinfo.RX = lay.RefCenter.X;
                    alinfo.RY = lay.RefCenter.Y;
                    double dx = cinfo.MXX * rx + cinfo.MXY * ry - rx + cinfo.TX;
                    double dy = cinfo.MYX * rx + cinfo.MYY * ry - ry + cinfo.TY;
                    alinfo.TX = al.AffineMatrixXX * dx + al.AffineMatrixXY * dy + al.TranslationX;
                    alinfo.TY = al.AffineMatrixYX * dx + al.AffineMatrixYY * dy + al.TranslationY;
                    for (j = 0; j < n; j++)
                    {
                        SySal.TotalScan.Flexi.Segment seg = (SySal.TotalScan.Flexi.Segment)lay[j];
                        SySal.Tracking.MIPEmulsionTrackInfo info = seg.OriginalInfo;
                        info.Slope = alinfo.Deform(info.Slope);
                        info.Intercept = alinfo.Transform(info.Intercept);
                        seg.SetInfo(info);
                    }
                }
                if (logstrw != null) logstrw.Write("Importing volume...");
                refvol.ImportVolume(ds, addvol, fds);
                if (logstrw != null) logstrw.WriteLine("Done.");
#if !DEBUG
            }
            catch (Exception x)
            {
                if (logstrw != null) logstrw.WriteLine("Error:\r\n" + x.ToString());
            }
            finally
            {
                if (logstrw != null) logstrw.WriteLine("End AddToVolume.");
            }
#endif
            }
    }
}
