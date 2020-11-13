using System;
using System.Collections.Generic;
using System.Text;
using SySal.BasicTypes;

namespace SySal.Processing.DecaySearchVSept09
{
    /// <summary>
    /// Result of kink search on a track or an array of segments.
    /// </summary>
    public class KinkSearchResult
    {
        /// <summary>
        /// RMS of transverse slope.
        /// </summary>
        public double TransverseSlopeRMS;
        /// <summary>
        /// RMS of longitudinal slope.
        /// </summary>
        public double LongitudinalSlopeRMS;
        /// <summary>
        /// Maximum transverse slope difference.
        /// </summary>
        public double TransverseMaxDeltaSlopeRatio;
        /// <summary>
        /// Maximum longitudinal slope difference.
        /// </summary>
        public double LongitudinalMaxDeltaSlopeRatio;
        /// <summary>
        /// Maximum combined difference.
        /// </summary>
        public double KinkDelta;
        /// <summary>
        /// Index of the segment with largest combined slope difference.
        /// </summary>
        public int KinkIndex;
        /// <summary>
        /// The exception generated during kink search; <c>null</c> if no exception was generated.
        /// </summary>
        public string ExceptionMessage;

        /// <summary>
        /// Computes the slope difference between two tracks.
        /// </summary>
        /// <param name="u">slope of the upstream segment.</param>
        /// <param name="d">slope of the downstream segment.</param>
        /// <param name="uz">upstream Z.</param>
        /// <param name="dz">uownstream Z.</param>
        /// <returns></returns>
        protected SySal.BasicTypes.Vector2 SlopeDiff(SySal.BasicTypes.Vector u, SySal.BasicTypes.Vector d, double uz, double dz)
        {
            SySal.BasicTypes.Vector2 r = new Vector2();
            double leadplates = Math.Round((dz - uz) / 1300.0);
            if (leadplates < 1.0)
            {
                r.X = r.Y = -1.0;
            }
            else
            {
                leadplates = 1.0 / Math.Sqrt(leadplates);
                d.X -= u.X;
                d.Y -= u.Y;
                double n = u.X * u.X + u.Y * u.Y;
                if (n <= 0.0)
                {
                    n = 1.0;
                    u.X = 1.0;
                    u.Y = 0.0;
                }
                else
                {
                    n = 1.0 / Math.Sqrt(n);
                    u.X *= n;
                    u.Y *= n;
                }
                r.X = Math.Abs(d.X * u.X + d.Y * u.Y) * leadplates;
                r.Y = Math.Abs(d.X * u.Y - d.Y * u.X) * leadplates;
            }
            return r;
        }

        /// <summary>
        /// Seeks the kink.
        /// </summary>
        /// <param name="segs">the array of segments to use.</param>
        /// <param name="allowedkink">the number of entries must be equal to the number of segments minus 1, since differences matter; then, the kink is checked for segments whose corresponding entry is set to <c>true</c> in this array.</param>
        protected void ComputeResult(SySal.TotalScan.Segment[] segs, bool[] allowedkink)
        {
            TransverseSlopeRMS = LongitudinalSlopeRMS = TransverseMaxDeltaSlopeRatio = LongitudinalMaxDeltaSlopeRatio = KinkDelta = -1.0;
            KinkIndex = -1;
            KinkDelta = -1.0;
            int i;
            int j = 0;
            double rmst = 0.0, rmsl = 0.0;
            for (i = 1; i < segs.Length; i++)
            {
                SySal.BasicTypes.Vector2 r = SlopeDiff(segs[i].Info.Slope, segs[i - 1].Info.Slope, segs[i].Info.Intercept.Z, segs[i - 1].Info.Intercept.Z);
                if (r.X >= 0.0 && r.Y >= 0.0)
                {
                    j++;
                    rmsl += r.X * r.X;
                    rmst += r.Y * r.Y;
                }
            }
            if (j < 2) return;
            KinkDelta = 0.0;
            double newr;
            TransverseSlopeRMS = Math.Sqrt(rmst / j);
            LongitudinalSlopeRMS = Math.Sqrt(rmsl / j);
            if (allowedkink.Length != segs.Length - 1)
            {
                ExceptionMessage = "The number of segments and of kink search activation flags must be identical";
                return;
            }
            if (allowedkink.Length < 3)
            {
                ExceptionMessage = "Kink search requires at least 3 segments.";
                return;
            }
            for (i = 0; i < allowedkink.Length; i++)
                if (allowedkink[i])
                {
                    SySal.BasicTypes.Vector2 r = SlopeDiff(segs[i + 1].Info.Slope, segs[i].Info.Slope, segs[i + 1].Info.Intercept.Z, segs[i].Info.Intercept.Z);
                    if (r.X < 0.0 || r.Y < 0.0) continue;
                    double rl = r.X / Math.Sqrt((rmsl - r.X * r.X) / (j - 1));
                    double rt = r.Y / Math.Sqrt((rmst - r.Y * r.Y) / (j - 1));
                    if ((newr = (rl * rl + rt * rt)) > KinkDelta)
                    {
                        KinkIndex = i;
                        TransverseMaxDeltaSlopeRatio = rt;                         
                        LongitudinalMaxDeltaSlopeRatio = rl;                        
                        KinkDelta = newr;
                    }
                }
            KinkDelta = Math.Sqrt(KinkDelta);
            ExceptionMessage = null;
        }

        /// <summary>
        /// Builds a null kink search result (no kink).
        /// </summary>
        public KinkSearchResult()
        {
            TransverseSlopeRMS = LongitudinalSlopeRMS = TransverseMaxDeltaSlopeRatio = LongitudinalMaxDeltaSlopeRatio = KinkDelta = -1.0;
            KinkIndex = -1;
            ExceptionMessage = "Kink search not performed.";
        }

        /// <summary>
        /// Seeks a kink for an array of segments.
        /// </summary>        
        /// <param name="segs">the array of segments describing the track.</param>
        /// <param name="allowedkink">the number of entries must be identical to the number of segments; then, the kink is checked for segments whose corresponding entry is set to <c>true</c> in this array.</param>
        public KinkSearchResult(SySal.TotalScan.Segment[] segs, bool[] allowedkink)
        {
            ComputeResult(segs, allowedkink);
        }

        /// <summary>
        /// Seeks a kink in a track.
        /// </summary>        
        /// <param name="t">the track where the kink is to be sought.</param>
        /// <param name="allowedkink">the number of entries must be identical to the number of segments of the track; then, the kink is checked for segments whose corresponding entry is set to <c>true</c> in this array.</param>
        public KinkSearchResult(SySal.TotalScan.Track t, bool[] allowedkink)
        {
            SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[t.Length];
            int i;
            for (i = 0; i < t.Length; i++)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = t[i].Info;                
                segs[i] = new SySal.TotalScan.Segment(info, new SySal.TotalScan.NullIndex());
            }
            ComputeResult(segs, allowedkink);
        }

        /// <summary>
        /// Seeks a kink in a track using the default requirements of the procedure (3/5 base tracks in the first plates, kink sought in the first 4 plates).
        /// </summary>
        /// <param name="tk">the track where the kink is being sought.</param>
        public KinkSearchResult(SySal.TotalScan.Flexi.Track tk)
        {
            SySal.Tracking.MIPEmulsionTrackInfo[] info = tk.BaseTracks;
            System.Collections.ArrayList ar = new System.Collections.ArrayList();
            int i = info.Length - 1;
            double lastz = 0.0;
            if (i >= 0)
            {
                lastz = info[i].TopZ;
                ar.Add(info[i]);
                while (--i >= 0)
                    if (Math.Round((info[i].TopZ - lastz) / 1300.0) >= 1.0)
                    {
                        ar.Add(info[i]);
                        lastz = info[i].TopZ;
                    }
            }
            if (ar.Count < 3)
            {
                TransverseSlopeRMS = LongitudinalSlopeRMS = TransverseMaxDeltaSlopeRatio = LongitudinalMaxDeltaSlopeRatio = KinkDelta = -1.0;
                KinkIndex = -1;
                ExceptionMessage = "At least 3 base tracks are needed to search for a kink.";
                return;
            }
            SySal.TotalScan.Segment[] segs = new SySal.TotalScan.Segment[ar.Count];
            bool [] allowedkink = new bool[segs.Length - 1];
            lastz = ((SySal.Tracking.MIPEmulsionTrackInfo)ar[0]).TopZ;
            for (i = 0; i < segs.Length; i++)
            {
                segs[i] = new SySal.TotalScan.Segment((SySal.Tracking.MIPEmulsionTrackInfo)ar[segs.Length - i - 1], new SySal.TotalScan.NullIndex());
                if (i < segs.Length - 1 && Math.Round((segs[i].Info.TopZ - lastz) / 1300.0) <= 3.0)
                    allowedkink[i] = true;
            }
            ComputeResult(segs, allowedkink);
            if (Math.Round((((SySal.Tracking.MIPEmulsionTrackInfo)ar[2]).TopZ - (lastz = ((SySal.Tracking.MIPEmulsionTrackInfo)ar[0]).TopZ)) / 1300.0) > 4.0)
            {
                TransverseMaxDeltaSlopeRatio = LongitudinalMaxDeltaSlopeRatio = KinkDelta = -1.0;
                KinkIndex = -1;
                ExceptionMessage = "At least 3 base tracks must be found in the 5 most upstream plates to search for a kink.";
                return;
            }
            if (KinkIndex >= 0) KinkIndex = (int)segs[KinkIndex].Info.Field;
        }

        public static bool TrackHasHoles(SySal.TotalScan.Flexi.Track ftk)
        {
            SySal.Tracking.MIPEmulsionTrackInfo[] btks = ftk.BaseTracks;
            int j;
            for (j = 0; j <= 3; j++)
                if (btks.Length - 1 - j >= 0 && Math.Round((btks[btks.Length - 1 - j].TopZ - ftk[ftk.Length - 1].Info.TopZ) / 1300.0) > j) return true;
            return false;
        }

        public static bool TrackHasTooManyHoles(SySal.TotalScan.Flexi.Track ftk)
        {
            SySal.Tracking.MIPEmulsionTrackInfo[] btks = ftk.BaseTracks;
            if (btks.Length < 3 && ftk.Length > 3) return true;
            else if (Math.Round((btks[btks.Length - 3].TopZ - ftk[ftk.Length - 1].Info.TopZ) / 1300.0) > 4.0) return true;
            return false;
        }
    }

    /// <summary>
    /// Class that provides selection of tracks for one-mu events.
    /// </summary>
    public class OneMuOrMultiProngZeroEventExtraTrackFilter
    {
        public SySal.BasicTypes.Vector m_MainVertexPos;

        public OneMuOrMultiProngZeroEventExtraTrackFilter(SySal.TotalScan.Vertex v)
        {
            m_MainVertexPos.X = v.X;
            m_MainVertexPos.Y = v.Y;
            m_MainVertexPos.Z = v.Z;
        }

        public bool Filter(SySal.TotalScan.Track t)
        {
            if (t.Length < 3) return false;            
            double dz = t.Upstream_Z - m_MainVertexPos.Z;
            if (dz < 0.0 || Math.Round(dz / 1300.0) > 4.0) return false;            
            double ipcut = (dz < 1000.0) ? 500.0 : 800.0;
            SySal.Tracking.MIPEmulsionTrackInfo[] segs = ((SySal.TotalScan.Flexi.Track)t).BaseTracks;
            if (segs.Length < 2) return false;
            SySal.BasicTypes.Vector p = t[t.Length - 1].Info.Intercept;
            SySal.BasicTypes.Vector s = segs[segs.Length - 1].Slope;
            s.Z = 1.0 / Math.Sqrt(s.X * s.X + s.Y * s.Y + 1.0);
            s.X *= s.Z;
            s.Y *= s.Z;
            p.X -= m_MainVertexPos.X;
            p.Y -= m_MainVertexPos.Y;
            p.Z -= m_MainVertexPos.Z;
            double dx = p.Y * s.Z - p.Z * s.Y;
            double dy = p.Z * s.X - p.X * s.Z;
            dz = p.X * s.Y - p.Y * s.X;
            return dx * dx + dy * dy + dz * dz <= ipcut * ipcut;
        }
    }

    /// <summary>
    /// Class that provides selection of tracks for one-mu events.
    /// </summary>
    public class ZeroMu123ProngEventExtraTrackFilter
    {
        public SySal.BasicTypes.Vector m_MainVertexPos;

        public ZeroMu123ProngEventExtraTrackFilter(SySal.TotalScan.Vertex v)
        {
            m_MainVertexPos.X = v.X;
            m_MainVertexPos.Y = v.Y;
            m_MainVertexPos.Z = v.Z;
        }

        public bool Filter(SySal.TotalScan.Track t)
        {
            if (t.Length < 3) return false;
            double dz = t.Upstream_Z - m_MainVertexPos.Z;
            if (Math.Round(dz / 1300.0) < -3.0 || Math.Round(dz / 1300.0) > 4.0) return false;
            double ipcut = 800.0;
            SySal.Tracking.MIPEmulsionTrackInfo[] segs = ((SySal.TotalScan.Flexi.Track)t).BaseTracks;
            if (segs.Length < 2) return false;
            SySal.BasicTypes.Vector p = t[t.Length - 1].Info.Intercept;
            SySal.BasicTypes.Vector s = segs[segs.Length - 1].Slope;
            s.Z = 1.0 / Math.Sqrt(s.X * s.X + s.Y * s.Y + 1.0);
            s.X *= s.Z;
            s.Y *= s.Z;
            p.X -= m_MainVertexPos.X;
            p.Y -= m_MainVertexPos.Y;
            p.Z -= m_MainVertexPos.Z;
            double dx = p.Y * s.Z - p.Z * s.Y;
            double dy = p.Z * s.X - p.X * s.Z;
            dz = p.X * s.Y - p.Y * s.X;
            return dx * dx + dy * dy + dz * dz <= ipcut * ipcut;
        }
    }

    /// <summary>
    /// Class that provides selection of tracks for one-mu events.
    /// </summary>
    public class IsolatedTrackEventExtraTrackFilter
    {
        public SySal.BasicTypes.Vector m_MainVertexPos;

        public IsolatedTrackEventExtraTrackFilter(SySal.TotalScan.Vertex v)
        {
            m_MainVertexPos.X = v.X;
            m_MainVertexPos.Y = v.Y;
            m_MainVertexPos.Z = v.Z;
        }

        public bool Filter(SySal.TotalScan.Track t)
        {
            if (t.Length < 3) return false;
            double dz = t.Upstream_Z - m_MainVertexPos.Z;
            if (Math.Round(dz / 1300.0) < -3.0 || Math.Round(dz / 1300.0) > 4.0) return false;
            double ipcut = 800.0;
            SySal.Tracking.MIPEmulsionTrackInfo[] segs = ((SySal.TotalScan.Flexi.Track)t).BaseTracks;
            if (segs.Length < 2) return false;
            SySal.BasicTypes.Vector p = t[t.Length - 1].Info.Intercept;
            SySal.BasicTypes.Vector s = segs[segs.Length - 1].Slope;
            s.Z = 1.0 / Math.Sqrt(s.X * s.X + s.Y * s.Y + 1.0);
            s.X *= s.Z;
            s.Y *= s.Z;
            p.X -= m_MainVertexPos.X;
            p.Y -= m_MainVertexPos.Y;
            p.Z -= m_MainVertexPos.Z;
            double dx = p.Y * s.Z - p.Z * s.Y;
            double dy = p.Z * s.X - p.X * s.Z;
            dz = p.X * s.Y - p.Y * s.X;
            return dx * dx + dy * dy + dz * dz <= ipcut * ipcut;
        }
    }


}
