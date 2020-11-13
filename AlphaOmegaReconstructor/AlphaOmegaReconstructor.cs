//#define FINE

using System;
using System.Collections;
using SySal;
using SySal.BasicTypes;
using SySal.Management;
using NumericalTools;
using SySal.Processing.QuickMapping;
using SySal.TotalScan;
using System.Xml.Serialization;
using SySal.Tracking;
using System.Runtime.Serialization;
using System.Windows.Forms;

namespace SySal.Processing.AlphaOmegaReconstruction
{
    interface IExposeManager
    {
        int AddInfo(int persistid, string info);
        System.Collections.ArrayList GetExposedInfo();
    }

    /// <summary>
    /// Results of the mapping operations.
    /// </summary>
    public enum MappingResult : int
    {
        /// <summary>
        /// Not performed yet.
        /// </summary>
        NotPerformedYet = -5,
        /// <summary>
        /// Bad affine focusing.
        /// </summary>
        BadAffineFocusing = -4,
        /// <summary>
        /// Singularity met during prescan.
        /// </summary>
        SingularityInPrescan = -3,
        /// <summary>
        /// Insufficient prescan.
        /// </summary>
        InsufficientPrescan = -2,
        /// <summary>
        /// One or both zones are empty.
        /// </summary>
        NullInput = -1,
        /// <summary>
        /// Prescan OK.
        /// </summary>
        OK = 0
    }

    /// <summary>
    /// Parameters for mapping 
    /// </summary>
    public struct MappingParameters
    {
        /// <summary>
        /// Number of matches.
        /// </summary>
        public int CoincN;
        /// <summary>
        /// Overlap region.
        /// </summary>
        public Vector2 Overlap;
        /// <summary>
        /// Size of the fixed map.
        /// </summary>
        public Vector2 FixSize;
        /// <summary>
        /// Size of the moving map.
        /// </summary>
        public Vector2 MovSize;
        /// <summary>
        /// Raw translation (not optimized).
        /// </summary>
        public Vector2 RawTransl;
        /// <summary>
        /// Initializes a new instance of the structure.
        /// </summary>
        /// <param name="mCoincN">Number of Matches</param>
        /// <param name="mOverlap">Overlap (X and Y coordinates)</param>
        /// <param name="mFixSize">Size of fixed map (micron)</param>
        /// <param name="mMovSize">Size of moveable map (micron)</param>
        /// <param name="mRawTransl">Raw Translation (micron)</param>
        public MappingParameters(int mCoincN, Vector2 mOverlap, Vector2 mFixSize,
            Vector2 mMovSize, Vector2 mRawTransl)
        {
            CoincN = mCoincN;
            Overlap = mOverlap;
            FixSize = mFixSize;
            MovSize = mMovSize;
            RawTransl = mRawTransl;
        }
    }

    /// <summary>
    /// Data from alignment.
    /// </summary>
    public class AlignmentData : SySal.TotalScan.AlignmentData
    {
        /// <summary>
        /// Result of prescan mapping.
        /// </summary>
        public MappingResult Result;
        /// <summary>
        /// Builds an empty AlignmentData class.
        /// </summary>
        public AlignmentData()
        {
            TranslationX = 0.0;
            TranslationY = 0.0;
            TranslationZ = 0.0;
            AffineMatrixXX = 1.0;
            AffineMatrixXY = 0.0;
            AffineMatrixYX = 0.0;
            AffineMatrixYY = 1.0;
            DShrinkX = 0.0;
            DShrinkY = 0.0;
            SAlignDSlopeX = 0.0;
            SAlignDSlopeY = 0.0;
            Result = MappingResult.NotPerformedYet;

        }
        /// <summary>
        /// Initializes a new instance of AlignmentData.
        /// </summary>
        /// <param name="dShrink">the slope multipliers.</param>
        /// <param name="sAlign_dSlope">the slope deviations.</param>
        /// <param name="Transl">the translation component of the affine transformation.</param>
        /// <param name="AffMat">the deformation component of the affine transformation.</param>
        /// <param name="MapRes">the result of the prescan mapping.</param>
        public AlignmentData(double[] dShrink, double[] sAlign_dSlope,
            double[] Transl, double[,] AffMat, MappingResult MapRes)
        {
            if (AffMat.GetLength(0) != 2 || AffMat.GetLength(1) != 2 ||
                Transl.Length != 3 || dShrink.Length != 2 || sAlign_dSlope.Length != 2) throw new Exception("....");
            TranslationX = Transl[0];
            TranslationY = Transl[1];
            TranslationZ = Transl[2];
            AffineMatrixXX = AffMat[0, 0];
            AffineMatrixXY = AffMat[0, 1];
            AffineMatrixYX = AffMat[1, 0];
            AffineMatrixYY = AffMat[1, 1];
            DShrinkX = dShrink[0];
            DShrinkY = dShrink[1];
            SAlignDSlopeX = sAlign_dSlope[0];
            SAlignDSlopeY = sAlign_dSlope[1];
            Result = MapRes;
        }
        /// <summary>
        /// Initializes a new instance of the AlignmentData class.
        /// </summary>
        /// <param name="dShrink">the slope multipliers.</param>
        /// <param name="sAlign_dSlope">the slope deviations.</param>
        /// <param name="Transformation">the transformation parameter vector.</param>
        /// <param name="MapRes">the results of prescan mapping.</param>
        public AlignmentData(double[] dShrink, double[] sAlign_dSlope, double[] Transformation, MappingResult MapRes)
        {
            if (Transformation.Length != Volume.POS_ALIGN_DATA_LEN || dShrink.Length != 2 ||
                sAlign_dSlope.Length != 2) throw new Exception("....");
            TranslationX = Transformation[4];
            TranslationY = Transformation[5];
            TranslationZ = Transformation[6];
            AffineMatrixXX = Transformation[0];
            AffineMatrixXY = Transformation[1];
            AffineMatrixYX = Transformation[2];
            AffineMatrixYY = Transformation[3];
            DShrinkX = dShrink[0];
            DShrinkY = dShrink[1];
            SAlignDSlopeX = sAlign_dSlope[0];
            SAlignDSlopeY = sAlign_dSlope[1];
            Result = MapRes;
        }
    }
    #region  Cell Array for Fast Linking between Segments (based on positions)
    class Position_CellArray
    {
        public Position_CellArray()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        //		public VolumeScanning.Segment[] CellSegments;

        protected Vector Min;
        protected double DXCell, DYCell;
        protected int XCells, YCells;
        protected double RefZ;

        protected bool Recompute_ConnMatrix;
        protected double m_MaximumLevelSize;
        public double MaximumLevelSize
        {
            get
            {
                return m_MaximumLevelSize;
            }
            set
            {
                m_MaximumLevelSize = value;
                Recompute_ConnMatrix = true;
            }
        }

        double[,] ConnectivityMatrix;

        protected struct Cell
        {
            public int Count;
            public Segment[] Segments;
        };

        protected Cell[] Cells;

        private const int MaxCells = 10000;

        IExposeManager XpMg;

        int XpPersistId1 = -1;
        int XpPersistId2 = -1;

        public Position_CellArray(Segment[] Segs, double ReferenceZ, double XTol, double YTol, double maximumlevelsize, bool autoflush, IExposeManager xpmg)
        {
            XpMg = xpmg;
            RefZ = ReferenceZ;
            DXCell = XTol;
            DYCell = YTol;
            m_AutoFlush = autoflush;

            m_MaximumLevelSize = maximumlevelsize;
            try
            {
                int Count = Segs.Length;
                if (Count == 0) return;
                Recompute_ConnMatrix = false;
                ConnectivityMatrix = Transformation.ConcentricRelativeCoordinates(m_MaximumLevelSize);

                Vector Max;
                int i;
                Vector[] V = new Vector[Count];
                int[] I = new int[Count];
                //SySal.Tracking.MIPEmulsionTrackInfo sInfo = Segs[0].GetInfo();
                SySal.Tracking.MIPEmulsionTrackInfo sInfo = Segs[0].Info;
                V[0].X = sInfo.Intercept.X + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.X;
                V[0].Y = sInfo.Intercept.Y + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.Y;
                V[0].Z = sInfo.Intercept.Z + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.Z;
                //Segs[0].Flush();
                Max = V[0];
                Min = V[0];
                for (i = 1; i < Count; i++)
                {
                    //sInfo = Segs[i].GetInfo();
                    sInfo = Segs[i].Info;
                    V[i].X = sInfo.Intercept.X + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.X;
                    V[i].Y = sInfo.Intercept.Y + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.Y;
                    V[i].Z = sInfo.Intercept.Z + (RefZ - sInfo.Intercept.Z) * sInfo.Slope.Z;
                    //Segs[i].Flush();
                    if (V[i].X < Min.X) Min.X = V[i].X;
                    else if (V[i].X > Max.X) Max.X = V[i].X;
                    if (V[i].Y < Min.Y) Min.Y = V[i].Y;
                    else if (V[i].Y > Max.Y) Max.Y = V[i].Y;
                };
                while (true)
                {
                    XCells = (int)Math.Floor((Max.X - Min.X) / DXCell + 1);
                    YCells = (int)Math.Floor((Max.Y - Min.Y) / DYCell + 1);
                    if (XCells * YCells > MaxCells)
                    {
                        DXCell *= 2.0;
                        DYCell *= 2.0;
                    }
                    else break;
                }
                if (XCells * YCells == 0) return;
                Cells = new Cell[XCells * YCells];
                int[] C = new int[XCells * YCells];
                for (i = 0; i < XCells * YCells; i++)
                {
                    C[i] = Cells[i].Count = 0;
                    Cells[i].Segments = new Segment[0];
                };
                for (i = 0; i < Count; i++)
                    Cells[I[i] = (int)Math.Floor((V[i].X - Min.X) / DXCell) + ((int)Math.Floor((V[i].Y - Min.Y) / DYCell) * XCells)].Count++;
                for (i = 0; i < XCells * YCells; i++)
                    Cells[i].Segments = new Segment[Cells[i].Count];
                for (i = 0; i < Count; i++)
                {
#if false
                    Cells[I[i]].Segments[C[I[i]]] = Segs[i];
#else
                    Cells[I[i]].Segments[C[I[i]]] = (SySal.Processing.AlphaOmegaReconstruction.Segment)(Segs[i].LayerOwner[Segs[i].PosInLayer]);
                    if (autoflush) Segs[i].Flush();
#endif
                    //Cells[I[i]].Segments[C[I[i]]].Flag = i;
                    C[I[i]]++;
                };
                XpPersistId1 = XpMg.AddInfo(XpPersistId1, "");
                XpPersistId2 = XpMg.AddInfo(XpPersistId2, "Positon_CellArray " + XpPersistId1 + " Lock info EMPTY");
                XpPersistId1 = XpMg.AddInfo(XpPersistId1, "Position_CellArray " + XpPersistId1 + " constructor called with " + Segs.Length + " segments.");
            }
            catch (Exception x)
            {
                throw x;
            }
        }

        private Segment[] m_LockedSegments;

        private int m_LockedIX = -1;
        private int m_LockedIY = -1;
        private int m_LockHit = 0;
        private int m_LockMiss = 0;
        private int m_LockCalls = 0;
        private bool m_AutoFlush;

        public void Unlock()
        {
            if (m_LockedSegments != null)
            {
                int i;
                for (i = 0; i < m_LockedSegments.Length; i++)
                    m_LockedSegments[i].Flush();
            }
            m_LockedSegments = null;
        }

        public Segment[] Lock(SySal.Tracking.MIPEmulsionTrackInfo Info)
        {
            m_LockCalls++;
            if (m_LockCalls % 1000 == 0) XpMg.AddInfo(XpPersistId2, "Position_CellArray " + XpPersistId1 + " LockCalls " + m_LockCalls + " Hit " + m_LockHit + " Miss " + m_LockMiss + " Hit/Call " + (100 * m_LockHit / m_LockCalls) + "%");
            Segment[] OutSeg = new Segment[0];
            int Count = 0, n, j, m;
            int cx, cy, i, ix, iy;
            try
            {
                Vector V;
                V.X = Info.Intercept.X + (RefZ - Info.Intercept.Z) * Info.Slope.X;
                V.Y = Info.Intercept.Y + (RefZ - Info.Intercept.Z) * Info.Slope.Y;
                V.Z = Info.Intercept.Z + (RefZ - Info.Intercept.Z) * Info.Slope.Z;

                cx = (int)((V.X - Min.X) / DXCell);
                cy = (int)((V.Y - Min.Y) / DYCell);
                if (cx < -1 || cx > XCells) return OutSeg;
                if (cy < -1 || cy > YCells) return OutSeg;
                if (m_LockedSegments != null)
                {
                    if (cx == m_LockedIX && cy == m_LockedIY)
                    {
                        m_LockHit++;
                        return m_LockedSegments;
                    }
                    if (m_AutoFlush) Unlock();
                }
                m_LockMiss++;

                if (Recompute_ConnMatrix)
                    ConnectivityMatrix = Transformation.ConcentricRelativeCoordinates(m_MaximumLevelSize);

                m = ConnectivityMatrix.GetLength(0);
                for (i = 0; i < m; i++)
                {
                    ix = cx + (int)ConnectivityMatrix[i, 0];
                    iy = cy + (int)ConnectivityMatrix[i, 1];
                    if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells)
                        Count += Cells[ix + iy * XCells].Count;
                };
                if (Count == 0) return OutSeg;
                OutSeg = new Segment[Count];

                Count = 0;
                for (i = 0; i < m; i++)
                {
                    ix = cx + (int)ConnectivityMatrix[i, 0];
                    iy = cy + (int)ConnectivityMatrix[i, 1];
                    if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells)
                    {
                        n = Cells[ix + iy * XCells].Count;
                        for (j = 0; j < n; j++)
                        {
                            OutSeg[Count + j] = Cells[ix + iy * XCells].Segments[j];
                        };
                        Count += n;
                    };
                };
            }
            catch (Exception x)
            {
                throw x;
            }

            m_LockedIX = cx;
            m_LockedIY = cy;
            return (m_LockedSegments = OutSeg);
        }
    }
    #endregion

    #region  Cell Array for Fast Linking between tracks (based on positions)
    class TrackPosition_CellArray
    {
        public TrackPosition_CellArray()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        private const int MaxCells = 10000;

        protected Vector Min;
        protected double DXCell, DYCell, DZCell;
        protected int XCells, YCells, ZCells;
        protected double RefZ;

        protected bool Recompute_ConnMatrix;
        protected double m_MaximumLevelSize;
        public double MaximumLevelSize
        {
            get
            {
                return m_MaximumLevelSize;
            }
            set
            {
                m_MaximumLevelSize = value;
                Recompute_ConnMatrix = true;
            }
        }

        double[,] ConnectivityMatrix;

        protected struct Cell
        {
            public int Count;
            public Track[] Tracks;
        };

        protected Cell[] Cells;

        public TrackPosition_CellArray(Track[] Tks, double gap, double MaxZ, double MinZ, double XTol, double YTol, double ZTol, double maximumlevelsize)
        {

            //RefZ = ReferenceZ;
            int i, j, k, n, m, l;
            DXCell = XTol;
            DYCell = YTol;
            DZCell = ZTol;

            m_MaximumLevelSize = maximumlevelsize;
            int Count = Tks.Length;
            if (Count == 0) return;
            Recompute_ConnMatrix = false;
            ///Controllare!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            double[,] tmpConnectivityMatrix;
            tmpConnectivityMatrix = Transformation.ConcentricRelativeCoordinates(m_MaximumLevelSize);
            n = tmpConnectivityMatrix.GetLength(0);
            m = tmpConnectivityMatrix.GetLength(1);
            l = 1 + (int)Math.Ceiling(m_MaximumLevelSize);
            ConnectivityMatrix = new double[n * l, 3];
            for (k = 0; k < l; k++)
                for (i = 0; i < n; i++)
                    for (j = 0; j < m; j++)
                    {
                        ConnectivityMatrix[i + k * n, j] = tmpConnectivityMatrix[i, j];
                        if (j == m - 1) ConnectivityMatrix[i + k * n, 2] = l / 2 - k;
                    }
            /*			for(k=0; k<l; k++)
                            for(i=0; i<n; i++)
                                ConnectivityMatrix[i+k*n,2] = l/2 - k;
            */
            Vector Max;

            int nseg = 0;
            for (i = 0; i < Count; i++) nseg += Tks[i].Length;


            Vector tV = new Vector();
            //SySal.Tracking.MIPEmulsionTrackInfo sInfo = Segs[0].GetInfo();
            double ztra = Tks[0][0].Info.Intercept.Z;
            if (ztra < MaxZ && ztra > MinZ)
            {
                tV.Z = ztra;
                tV.X = Tks[0][0].Info.Intercept.X;
                tV.Y = Tks[0][0].Info.Intercept.Y;
            }
            Max = tV;
            Min = tV;
            //int nseg = 0;
            for (i = 0; i < Count; i++)
                for (j = 0; j < Tks[i].Length; j++)
                {
                    ztra = Tks[i][j].Info.Intercept.Z;
                    if (ztra < MaxZ && ztra > MinZ)
                    {
                        tV.Z = ztra;
                        tV.X = Tks[i][j].Info.Intercept.X;
                        tV.Y = Tks[i][j].Info.Intercept.Y;
                        if (tV.X < Min.X) Min.X = tV.X;
                        else if (tV.X > Max.X) Max.X = tV.X;
                        if (tV.Y < Min.Y) Min.Y = tV.Y;
                        else if (tV.Y > Max.Y) Max.Y = tV.Y;
                    }
                    Min.Z = MinZ - gap;
                    Max.Z = MaxZ + gap;
                };

            //			Vector[] V = new Vector[nseg];
            //			for (i = 1; i < nseg; i++) V = tV; 
            int Ind;
            //ArrayList arCellIndices = new ArrayList();
            int[] CellIndices = new int[nseg];
            int[] TrackIndices = new int[nseg];

            while (true)
            {
                XCells = (int)Math.Floor((Max.X - Min.X) / DXCell + 1);
                YCells = (int)Math.Floor((Max.Y - Min.Y) / DYCell + 1);
                ZCells = (int)Math.Floor((Max.Z - Min.Z) / DZCell + 1);
                if (XCells * YCells * ZCells > MaxCells)
                {
                    DXCell *= 2.0;
                    DYCell *= 2.0;
                    DZCell *= 2.0;
                }
                else break;
            }

            if (XCells * YCells * ZCells == 0) return;
            Cells = new Cell[XCells * YCells * ZCells];
            int[] C = new int[XCells * YCells * ZCells];
            for (i = 0; i < XCells * YCells * ZCells; i++)
            {
                C[i] = Cells[i].Count = 0;
                Cells[i].Tracks = new Track[0];
            };
            int couy;
            int prevInd;
            k = 0;
            for (i = 0; i < Count; i++)
            {
                prevInd = -1;
                for (j = 0; j < Tks[i].Length; j++)
                {
                    ztra = Tks[i][j].Info.Intercept.Z;
                    if (ztra < MaxZ && ztra > MinZ)
                    {

                        tV.Z = ztra;
                        tV.X = Tks[i][j].Info.Intercept.X;
                        tV.Y = Tks[i][j].Info.Intercept.Y;
                        Ind = (int)Math.Floor((tV.X - Min.X) / DXCell) +
                            ((int)Math.Floor((tV.Y - Min.Y) / DYCell) * XCells) +
                            ((int)Math.Floor((tV.Z - Min.Z) / DZCell) * XCells * YCells);
                        if (Ind != prevInd)
                        {
                            Cells[Ind].Count++;
                            CellIndices[k] = Ind;
                            TrackIndices[k] = i;
                            k++;
                            prevInd = Ind;
                        }
                        else
                        {
                            CellIndices[k] = -1;
                            TrackIndices[k] = -1;
                            k++;
                        }
                    }
                    else
                    {
                        CellIndices[k] = -1;
                        TrackIndices[k] = -1;
                        k++;
                    }
                }
                //arCellIndices.Add(prevInd);

            }
            //CellIndices = (int[])arCellIndices.ToArray(typeof(int));
            for (i = 0; i < XCells * YCells * ZCells; i++)
                Cells[i].Tracks = new Track[Cells[i].Count];

            for (i = 0; i < nseg; i++)
            {
                if (CellIndices[i] != -1)
                {
                    Cells[CellIndices[i]].Tracks[C[CellIndices[i]]] = Tks[TrackIndices[i]];
                    C[CellIndices[i]]++;
                }
            };
        }

        public Track[] Lock(Track t)
        {
            int n, j, m;
            int cx, cy, cz;
            int i, ix, iy, iz = 0;
            int h, nseg;
            nseg = t.Length;
            ArrayList outlist = new ArrayList();

            for (h = 0; h < nseg; h++)
            {
                Vector V;
                V.X = t[h].Info.Intercept.X;
                V.Y = t[h].Info.Intercept.Y;
                V.Z = t[h].Info.Intercept.Z;

                cx = (int)((V.X - Min.X) / DXCell);
                cy = (int)((V.Y - Min.Y) / DYCell);
                cz = (int)((V.Z - Min.Z) / DZCell);

                if (cx >= 0 && cx < XCells && cy >= 0 && cy < YCells && cz >= 0 && cz < ZCells)
                {

                    //Non va bene: non basterebbe per la z
                    //if(Recompute_ConnMatrix)
                    //	ConnectivityMatrix = Transformation.ConcentricRelativeCoordinates(m_MaximumLevelSize);

                    //Controllare!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    m = ConnectivityMatrix.GetLength(0);

                    for (i = 0; i < m; i++)
                    {
                        ix = cx + (int)ConnectivityMatrix[i, 0];
                        iy = cy + (int)ConnectivityMatrix[i, 1];
                        iz = cz + (int)ConnectivityMatrix[i, 2];
                        if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells && iz >= 0 && iz < ZCells)
                        {
                            n = Cells[ix + iy * XCells + iz * XCells * YCells].Count;
                            for (j = 0; j < n; j++)
                            {
                                Track nt = Cells[ix + iy * XCells + iz * XCells * YCells].Tracks[j];
                                if (nt == t) continue;
                                foreach (Track ot in outlist)
                                    if (ot == nt)
                                    {
                                        nt = null;
                                        break;
                                    }
                                if (nt != null)
                                    outlist.Add(nt);
                            };
                        };
                    };
                }
            }

            return (Track[])outlist.ToArray(typeof(Track));
        }

    }
    #endregion

    #region Cell Array for Background Evaluation (based on slopes)
    class Slopes_CellArray
    {
        public Slopes_CellArray()
        {
            //
            // TODO: Add constructor logic here
            //
        }

        //		public VolumeScanning.Segment[] CellSegments;

        protected Vector2 Min;
        protected double DSXCell, DSYCell;
        protected int XCells, YCells;
        protected int SegmentsNumber;
        protected double Risk;
        protected double[,] ConnectivityMatrix;
        protected double MaximumLevelSize;

        protected struct Cell
        {
            public int Count;
            public Segment[] Segments;
        };

        protected Cell[] Cells;

        public Slopes_CellArray(Segment[] Segs, double SXTol, double SYTol, double risk, double maximumlevelsize)
        {
            DSXCell = SXTol;
            DSYCell = SYTol;
            Risk = risk;
            MaximumLevelSize = maximumlevelsize;
            ConnectivityMatrix = Transformation.ConcentricRelativeCoordinates(MaximumLevelSize);

            int Count = Segs.Length;
            SegmentsNumber = Count;
            if (Count > 0)
            {
                Vector2 Max;
                int i;
                Vector2[] V = new Vector2[Count];
                int[] I = new int[Count];
                SySal.Tracking.MIPEmulsionTrackInfo sInfo = Segs[0].GetInfo();
                V[0].X = sInfo.Slope.X;
                V[0].Y = sInfo.Slope.Y;
                Max = V[0];
                Min = V[0];
                for (i = 1; i < Count; i++)
                {
                    sInfo = Segs[i].GetInfo();
                    V[i].X = sInfo.Slope.X;
                    V[i].Y = sInfo.Slope.Y;
                    if (V[i].X < Min.X) Min.X = V[i].X;
                    else if (V[i].X > Max.X) Max.X = V[i].X;
                    if (V[i].Y < Min.Y) Min.Y = V[i].Y;
                    else if (V[i].Y > Max.Y) Max.Y = V[i].Y;
                };
                XCells = (int)Math.Floor((Max.X - Min.X) / DSXCell + 1);
                YCells = (int)Math.Floor((Max.Y - Min.Y) / DSYCell + 1);
                if (XCells * YCells == 0) return;
                Cells = new Cell[XCells * YCells];
                int[] C = new int[XCells * YCells];
                for (i = 0; i < XCells * YCells; i++)
                {
                    C[i] = Cells[i].Count = 0;
                    Cells[i].Segments = new Segment[0];
                };
                for (i = 0; i < Count; i++)
                    Cells[I[i] = (int)Math.Floor((V[i].X - Min.X) / DSXCell) + ((int)Math.Floor((V[i].Y - Min.Y) / DSYCell) * XCells)].Count++;
                for (i = 0; i < XCells * YCells; i++)
                    Cells[i].Segments = new Segment[Cells[i].Count];
                for (i = 0; i < Count; i++)
                {
                    Cells[I[i]].Segments[C[I[i]]] = Segs[i];
                    C[I[i]]++;
                };
            };
        }
#if false
		public SySal.BasicTypes.Vector2 Lock(Segment Seg)
		{
			
			//Segment[] OutSeg = new Segment[0];
			SySal.BasicTypes.Vector2  DOut = new SySal.BasicTypes.Vector2();
			int Count=0, n;
			int cx, cy, i, ix, iy;//, LastGoodIdx=-1;
			double tmpRatio=0, Ratio=0;
			Vector2 V = new Vector2();
			V.X = Seg.Info.Slope.X;
			V.Y = Seg.Info.Slope.Y;
			
			cx = (int)((V.X - Min.X) / DSXCell);
			cy = (int)((V.Y - Min.Y) / DSYCell);
			if (cx < -1 || cx > XCells || cy < -1 || cy > YCells) return DOut;

			n = ConnectivityMatrix.GetLength(0);
			double tmp_condim=-1;
			for (i = 0; i < n-1; i++)
			{
				ix = cx + (int)ConnectivityMatrix[i,0];
				iy = cy + (int)ConnectivityMatrix[i,1];
				if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells) tmpRatio += (float)Cells[ix + iy * XCells].Count/(float)SegmentsNumber;

				if(ConnectivityMatrix[i,2] != ConnectivityMatrix[i+1,2])
				{
					Ratio +=tmpRatio;
					if (Ratio > Risk) break;
					tmpRatio = 0;
					tmp_condim = ConnectivityMatrix[i,2];
				};
			};
			if (tmp_condim == -1) return DOut;
			if (i==n-2)
			{
				ix = cx + (int)ConnectivityMatrix[n-1,0];
				iy = cy + (int)ConnectivityMatrix[n-1,1];
				if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells) tmpRatio += (float)Cells[ix + iy * XCells].Count/(float)SegmentsNumber;
				if (Ratio + tmpRatio < Risk) tmp_condim = ConnectivityMatrix[n-1,2];
			};


			DOut.X = ((V.X - Min.X) % DSXCell);
			if(DSXCell - DOut.X < DOut.X) DOut.X = DSXCell - DOut.X;
			DOut.X += DSXCell*(Math.Sqrt(tmp_condim));				
			DOut.Y = ((V.Y - Min.Y) % DSYCell);
			if(DSYCell - DOut.Y < DOut.Y) DOut.Y = DSYCell - DOut.Y;
			DOut.Y += DSYCell*(Math.Sqrt(tmp_condim));				

			return DOut;

		}
#endif
        public SySal.BasicTypes.Vector2 Lock(SySal.Tracking.MIPEmulsionTrackInfo Info)
        {

            //Segment[] OutSeg = new Segment[0];
            SySal.BasicTypes.Vector2 DOut = new SySal.BasicTypes.Vector2();
            int Count = 0, n;
            int cx, cy, i, ix, iy;//, LastGoodIdx=-1;
            double tmpRatio = 0, Ratio = 0;
            Vector2 V = new Vector2();
            V.X = Info.Slope.X;
            V.Y = Info.Slope.Y;

            cx = (int)((V.X - Min.X) / DSXCell);
            cy = (int)((V.Y - Min.Y) / DSYCell);
            if (cx < -1 || cx > XCells || cy < -1 || cy > YCells) return DOut;

            n = ConnectivityMatrix.GetLength(0);
            double tmp_condim = -1;
            for (i = 0; i < n - 1; i++)
            {
                ix = cx + (int)ConnectivityMatrix[i, 0];
                iy = cy + (int)ConnectivityMatrix[i, 1];
                if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells) tmpRatio += (float)Cells[ix + iy * XCells].Count / (float)SegmentsNumber;

                if (ConnectivityMatrix[i, 2] != ConnectivityMatrix[i + 1, 2])
                {
                    Ratio += tmpRatio;
                    if (Ratio > Risk) break;
                    tmpRatio = 0;
                    tmp_condim = ConnectivityMatrix[i, 2];
                };
            };
            if (tmp_condim == -1) return DOut;
            if (i == n - 2)
            {
                ix = cx + (int)ConnectivityMatrix[n - 1, 0];
                iy = cy + (int)ConnectivityMatrix[n - 1, 1];
                if (ix >= 0 && ix < XCells && iy >= 0 && iy < YCells) tmpRatio += (float)Cells[ix + iy * XCells].Count / (float)SegmentsNumber;
                if (Ratio + tmpRatio < Risk) tmp_condim = ConnectivityMatrix[n - 1, 2];
            };


            DOut.X = ((V.X - Min.X) % DSXCell);
            if (DSXCell - DOut.X < DOut.X) DOut.X = DSXCell - DOut.X;
            DOut.X += DSXCell * (Math.Sqrt(tmp_condim));
            DOut.Y = ((V.Y - Min.Y) % DSYCell);
            if (DSYCell - DOut.Y < DOut.Y) DOut.Y = DSYCell - DOut.Y;
            DOut.Y += DSYCell * (Math.Sqrt(tmp_condim));

            return DOut;

        }
    }

    #endregion

    class Segment : SySal.TotalScan.Segment
    {
        private System.IO.Stream m_Stream;

        internal bool m_IgnoreInAlignment = false;

        internal SySal.Tracking.MIPEmulsionTrackInfo GetInfo()
        {
            if (m_Info == null) ReloadFromDisk();
            return m_Info;
        }

        internal void Flush()
        {
            if (m_Stream != null) m_Info = null;
        }

        public override SySal.Tracking.MIPEmulsionTrackInfo Info
        {
            get
            {
                if (m_Info != null) return (SySal.Tracking.MIPEmulsionTrackInfo)m_Info.Clone();
                ReloadFromDisk();
                SySal.Tracking.MIPEmulsionTrackInfo myInfo = m_Info;
                m_Info = null;
                return myInfo;
            }
        }

        internal void MoveToDisk()
        {
            if (m_Info == null) return;
            m_Stream.Seek(164 * this.m_PosInLayer, System.IO.SeekOrigin.Begin);
            System.IO.BinaryWriter w = new System.IO.BinaryWriter(m_Stream);
            w.Write(m_Info.Count);
            w.Write(m_Info.AreaSum);
            w.Write(m_Info.Field);
            w.Write(m_Info.BottomZ);
            w.Write(m_Info.TopZ);
            w.Write(m_Info.Intercept.X);
            w.Write(m_Info.Intercept.Y);
            w.Write(m_Info.Intercept.Z);
            w.Write(m_Info.Slope.X);
            w.Write(m_Info.Slope.Y);
            w.Write(m_Info.Slope.Z);
            w.Write(m_Info.Sigma);
            m_Info = null;
        }

        private void Backup()
        {
            if (m_Info == null) return;
            m_Stream.Seek(164 * this.m_PosInLayer, System.IO.SeekOrigin.Begin);
            System.IO.BinaryWriter w = new System.IO.BinaryWriter(m_Stream);
            w.Write(m_Info.Count);
            w.Write(m_Info.AreaSum);
            w.Write(m_Info.Field);
            w.Write(m_Info.BottomZ);
            w.Write(m_Info.TopZ);
            w.Write(m_Info.Intercept.X);
            w.Write(m_Info.Intercept.Y);
            w.Write(m_Info.Intercept.Z);
            w.Write(m_Info.Slope.X);
            w.Write(m_Info.Slope.Y);
            w.Write(m_Info.Slope.Z);
            w.Write(m_Info.Sigma);
            w.Write(m_Info.Count);
            w.Write(m_Info.AreaSum);
            w.Write(m_Info.Field);
            w.Write(m_Info.BottomZ);
            w.Write(m_Info.TopZ);
            w.Write(m_Info.Intercept.X);
            w.Write(m_Info.Intercept.Y);
            w.Write(m_Info.Intercept.Z);
            w.Write(m_Info.Slope.X);
            w.Write(m_Info.Slope.Y);
            w.Write(m_Info.Slope.Z);
            w.Write(m_Info.Sigma);
            m_Info = null;
        }

        internal void ReloadFromDisk()
        {
            if (m_Info != null) return;
            m_Info = new MIPEmulsionTrackInfo();
            m_Stream.Seek(164 * this.m_PosInLayer, System.IO.SeekOrigin.Begin);
            System.IO.BinaryReader r = new System.IO.BinaryReader(m_Stream);
            m_Info.Count = r.ReadUInt16();
            m_Info.AreaSum = r.ReadUInt32();
            m_Info.Field = r.ReadUInt32();
            m_Info.BottomZ = r.ReadDouble();
            m_Info.TopZ = r.ReadDouble();
            m_Info.Intercept.X = r.ReadDouble();
            m_Info.Intercept.Y = r.ReadDouble();
            m_Info.Intercept.Z = r.ReadDouble();
            m_Info.Slope.X = r.ReadDouble();
            m_Info.Slope.Y = r.ReadDouble();
            m_Info.Slope.Z = r.ReadDouble();
            m_Info.Sigma = r.ReadDouble();
        }

        internal void RestoreOriginal()
        {
            if (m_Stream != null)
            {
                m_Stream.Seek(164 * this.m_PosInLayer + 82, System.IO.SeekOrigin.Begin);
                System.IO.BinaryReader r = new System.IO.BinaryReader(m_Stream);
                if (m_Info == null)
                {
                    m_Info = new MIPEmulsionTrackInfo();
                }
                m_Info.Count = r.ReadUInt16();
                m_Info.AreaSum = r.ReadUInt32();
                m_Info.Field = r.ReadUInt32();
                m_Info.BottomZ = r.ReadDouble();
                m_Info.TopZ = r.ReadDouble();
                m_Info.Intercept.X = r.ReadDouble();
                m_Info.Intercept.Y = r.ReadDouble();
                m_Info.Intercept.Z = r.ReadDouble();
                m_Info.Slope.X = r.ReadDouble();
                m_Info.Slope.Y = r.ReadDouble();
                m_Info.Slope.Z = r.ReadDouble();
                m_Info.Sigma = r.ReadDouble();
                //MoveToDisk();
            }
        }

        public Segment(SySal.Scanning.MIPBaseTrack tk, Index originalid, Layer layerowner, int posinlayer, System.IO.Stream stream)
        {
            m_Info = tk.Info;
            //m_BaseTrackId = basetrackid;
            m_Index = originalid;
            m_LayerOwner = layerowner;
            m_PosInLayer = posinlayer;
            m_Stream = stream;
            if (m_Stream != null) Backup();
        }

        public Segment(SySal.Tracking.MIPEmulsionTrackInfo tk, Index originalid, Layer layerowner, int posinlayer, System.IO.Stream stream)
        {
            m_Info = tk;
            //m_BaseTrackId = basetrackid;
            m_Index = originalid;
            m_LayerOwner = layerowner;
            m_PosInLayer = posinlayer;
            m_Stream = stream;
            if (m_Stream != null) Backup();
        }

        public Segment(SySal.Tracking.MIPEmulsionTrackInfo t, Index originalid, System.IO.Stream stream)
        {
            m_Info = t;
            //m_BaseTrackId = basetrackid;
            m_Index = originalid;
            m_LayerOwner = null;
            m_PosInLayer = -1;
            m_TrackOwner = null;
            m_PosInTrack = -1;
            m_Stream = stream;
            if (m_Stream != null) Backup();
            //UpstreamLinked = s.UpstreamLinked;
            //DownstreamLinked = s.DownstreamLinked;

        }

        internal void SetLayerOwner(Layer l, int posinlayer)
        {
            m_LayerOwner = l;
            m_PosInLayer = posinlayer;
        }
        internal Segment UpstreamLinked = null;
        internal Segment DownstreamLinked = null;

        //internal int m_Flag;
    }

    internal class UtilitySegment : SySal.TotalScan.Segment
    {
        public static void ResetTrackOwner(SySal.TotalScan.Segment s)
        {
            UtilitySegment.SetTrackOwner(s, null, 0);
        }
    }

    internal class UtilityTrack : SySal.TotalScan.Track
    {
        public new static void SetId(SySal.TotalScan.Track t, int newid)
        {
            SySal.TotalScan.Track.SetId(t, newid);
        }

        public static void MoveToDisk(SySal.TotalScan.Track t)
        {
            int i, n;
            n = t.Length;
            for (i = 0; i < n; i++)
                ((Segment)t[i]).Flush();
        }

        public static void ReloadSegments(SySal.TotalScan.Track t)
        {
            int i, n;
            n = t.Length;
            for (i = 0; i < n; i++)
                ((Segment)t[i]).ReloadFromDisk();
        }
    }

    internal class UtilityVertex : SySal.TotalScan.Vertex
    {
        public new static void SetId(SySal.TotalScan.Vertex v, int newid)
        {
            SySal.TotalScan.Vertex.SetId(v, newid);
        }

        public UtilityVertex() : base() { m_Id = -1; }

        public override void NotifyChanged() { }

        double VertexChi2(AlphaOmegaReconstruction.Configuration C, SySal.BasicTypes.Vector vp, int fitbasetracks)
        {
            double Chi2 = 0.0;
            double range = 0.0;
            double fpx = 0.0, fpy = 0.0, fsx = 0.0, fsy = 0.0;
            double px, py;
            int i, j, fb;
            for (j = 0; j < Tracks.Length; j++)
            {
                SySal.TotalScan.Track tk = Tracks[j];
                fb = Math.Min(fitbasetracks, tk.Length);
                double[] z = new double[2 * fb + 1];
                double[] x = new double[2 * fb + 1];
                double[] y = new double[2 * fb + 1];
                double[] s = new double[2 * fb + 1];
                if (tk.Upstream_Vertex == this)
                {
                    for (i = 0; i < fb; i++)
                    {
                        SySal.TotalScan.Segment seg = tk[tk.Length - i - 1];
                        SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
                        z[2 * i] = seg.LayerOwner.DownstreamZ;
                        z[2 * i + 1] = seg.LayerOwner.UpstreamZ;
                        x[2 * i] = info.Intercept.X + (seg.LayerOwner.DownstreamZ - info.Intercept.Z) * info.Slope.X;
                        x[2 * i + 1] = info.Intercept.X + (seg.LayerOwner.UpstreamZ - info.Intercept.Z) * info.Slope.X;
                        y[2 * i] = info.Intercept.Y + (seg.LayerOwner.DownstreamZ - info.Intercept.Z) * info.Slope.Y;
                        y[2 * i + 1] = info.Intercept.Y + (seg.LayerOwner.UpstreamZ - info.Intercept.Z) * info.Slope.Y;
                        s[2 * i] = Math.Sqrt(C.VtxFitWeightOptStepXY * C.VtxFitWeightOptStepXY + (z[2 * i] - vp.Z) * (z[2 * i] - vp.Z));
                        s[2 * i + 1] = Math.Sqrt(C.VtxFitWeightOptStepXY * C.VtxFitWeightOptStepXY + (z[2 * i + 1] - vp.Z) * (z[2 * i + 1] - vp.Z));
                    }
                }
                else
                {
                    for (i = 0; i < fb; i++)
                    {
                        SySal.TotalScan.Segment seg = tk[i];
                        SySal.Tracking.MIPEmulsionTrackInfo info = seg.Info;
                        z[2 * i] = seg.LayerOwner.DownstreamZ;
                        z[2 * i + 1] = seg.LayerOwner.UpstreamZ;
                        x[2 * i] = info.Intercept.X + (seg.LayerOwner.DownstreamZ - info.Intercept.Z) * info.Slope.X;
                        x[2 * i + 1] = info.Intercept.X + (seg.LayerOwner.UpstreamZ - info.Intercept.Z) * info.Slope.X;
                        y[2 * i] = info.Intercept.Y + (seg.LayerOwner.DownstreamZ - info.Intercept.Z) * info.Slope.Y;
                        y[2 * i + 1] = info.Intercept.Y + (seg.LayerOwner.UpstreamZ - info.Intercept.Z) * info.Slope.Y;
                        s[2 * i] = Math.Sqrt(C.VtxFitWeightOptStepXY * C.VtxFitWeightOptStepXY + (z[2 * i] - vp.Z) * (z[2 * i] - vp.Z));
                        s[2 * i + 1] = Math.Sqrt(C.VtxFitWeightOptStepXY * C.VtxFitWeightOptStepXY + (z[2 * i + 1] - vp.Z) * (z[2 * i + 1] - vp.Z));
                    }
                }
                z[2 * fb] = vp.Z;
                x[2 * fb] = vp.X;
                y[2 * fb] = vp.Y;
                s[2 * fb] = C.VtxFitWeightOptStepXY;
                NumericalTools.Fitting.LinearFitDE(z, x, s, ref fsx, ref fpx, ref range);
                NumericalTools.Fitting.LinearFitDE(z, y, s, ref fsy, ref fpy, ref range);
                for (i = 0; i < fb; i++)
                {
                    px = fpx + fsx * z[2 * i] - x[2 * i];
                    py = fpy + fsy * z[2 * i] - y[2 * i];
                    Chi2 += (px * px + py * py) / (s[2 * i] * s[2 * i]);
                    px = fpx + fsx * z[2 * i + 1] - x[2 * i + 1];
                    py = fpy + fsy * z[2 * i + 1] - y[2 * i + 1];
                    Chi2 += (px * px + py * py) / (s[2 * i + 1] * s[2 * i + 1]);
                }
            }
            return Chi2;
        }

        public void OptimizeVertex(AlphaOmegaReconstruction.Configuration C)
        {
            int fitbasetracks = C.FittingTracks;
            SySal.BasicTypes.Vector VLast = new SySal.BasicTypes.Vector();
            SySal.BasicTypes.Vector VNew = new SySal.BasicTypes.Vector();
            ComputeVertexCoordinates();
            VNew.X = m_X;
            VNew.Y = m_Y;
            VNew.Z = m_Z;
            double Chi2New = VertexChi2(C, VNew, fitbasetracks);
            double Chi2Last = 0.0;
            SySal.BasicTypes.Vector Delta = new SySal.BasicTypes.Vector();
            do
            {
                VLast = VNew;
                Chi2Last = Chi2New;

                SySal.BasicTypes.Vector VIterPlus = new SySal.BasicTypes.Vector();
                SySal.BasicTypes.Vector VIterMinus = new SySal.BasicTypes.Vector();
                double Chi2Plus = 0.0, Chi2Minus = 0.0;

                VIterPlus.X = VIterMinus.X = VNew.X;
                VIterPlus.Y = VIterMinus.Y = VNew.Y;
                VIterPlus.Z = VNew.Z + C.VtxFitWeightOptStepZ;
                VIterMinus.Z = VNew.Z - C.VtxFitWeightOptStepZ;
                Chi2Plus = VertexChi2(C, VIterPlus, fitbasetracks);
                Chi2Minus = VertexChi2(C, VIterMinus, fitbasetracks);
                Delta.Z = (Chi2Plus - Chi2Minus) * C.VtxFitWeightOptStepZ / (2.0 * (2 * Chi2Last - Chi2Plus - Chi2Minus));
                if (Delta.Z > C.VtxFitWeightOptStepZ) Delta.Z = C.VtxFitWeightOptStepZ;
                else if (Delta.Z < -C.VtxFitWeightOptStepZ) Delta.Z = -C.VtxFitWeightOptStepZ;
                VNew.Z = VLast.Z + Delta.Z;
                if ((Chi2New = VertexChi2(C, VNew, fitbasetracks)) < Chi2Last) Chi2Last = Chi2New;
                else VNew.Z = VLast.Z;

                VIterPlus.Y = VIterMinus.Y = VNew.Y;
                VIterPlus.Z = VIterMinus.Z = VNew.Z;
                VIterPlus.X = VNew.X + C.VtxFitWeightOptStepXY;
                VIterMinus.X = VNew.X - C.VtxFitWeightOptStepXY;
                Chi2Plus = VertexChi2(C, VIterPlus, fitbasetracks);
                Chi2Minus = VertexChi2(C, VIterMinus, fitbasetracks);
                Delta.X = (Chi2Plus - Chi2Minus) * C.VtxFitWeightOptStepXY / (2.0 * (2 * Chi2Last - Chi2Plus - Chi2Minus));
                if (Delta.X > C.VtxFitWeightOptStepXY) Delta.X = C.VtxFitWeightOptStepXY;
                else if (Delta.X < -C.VtxFitWeightOptStepXY) Delta.X = -C.VtxFitWeightOptStepXY;
                VNew.X = VLast.X + Delta.X;
                if ((Chi2New = VertexChi2(C, VNew, fitbasetracks)) < Chi2Last) Chi2Last = Chi2New;
                else VNew.X = VLast.X;

                VIterPlus.X = VIterMinus.X = VNew.X;
                VIterPlus.Z = VIterMinus.Z = VNew.Z;
                VIterPlus.Y = VNew.Y + C.VtxFitWeightOptStepXY;
                VIterMinus.Y = VNew.Y - C.VtxFitWeightOptStepXY;
                Chi2Plus = VertexChi2(C, VIterPlus, fitbasetracks);
                Chi2Minus = VertexChi2(C, VIterMinus, fitbasetracks);
                Delta.Y = (Chi2Plus - Chi2Minus) * C.VtxFitWeightOptStepXY / (2.0 * (2 * Chi2Last - Chi2Plus - Chi2Minus));
                if (Delta.Y > C.VtxFitWeightOptStepXY) Delta.Y = C.VtxFitWeightOptStepXY;
                else if (Delta.Y < -C.VtxFitWeightOptStepXY) Delta.Y = -C.VtxFitWeightOptStepXY;
                VNew.Y = VLast.Y + Delta.Y;
                if ((Chi2New = VertexChi2(C, VNew, fitbasetracks)) < Chi2Last) Chi2Last = Chi2New;
                else VNew.Y = VLast.Y;
            }
            while (Math.Sqrt((VLast.X - VNew.X) * (VLast.X - VNew.X) + (VLast.Y - VNew.Y) * (VLast.Y - VNew.Y) + (VLast.Z - VNew.Z) * (VLast.Z - VNew.Z)) > 0.1);
            m_X = VNew.X;
            m_Y = VNew.Y;
            m_Z = VNew.Z;
            m_VertexCoordinatesUpdated = true;
            m_AverageDistance = 0.0;
            int i;
            double dx, dy;
            for (i = 0; i < Tracks.Length; i++)
            {
                if (Tracks[i].Upstream_Vertex == this)
                {
                    dx = Tracks[i].Upstream_PosX + Tracks[i].Upstream_SlopeX * (m_Z - Tracks[i].Upstream_PosZ) - m_X;
                    dy = Tracks[i].Upstream_PosY + Tracks[i].Upstream_SlopeY * (m_Z - Tracks[i].Upstream_PosZ) - m_Y;
                }
                else
                {
                    dx = Tracks[i].Downstream_PosX + Tracks[i].Downstream_SlopeX * (m_Z - Tracks[i].Downstream_PosZ) - m_X;
                    dy = Tracks[i].Downstream_PosY + Tracks[i].Downstream_SlopeY * (m_Z - Tracks[i].Downstream_PosZ) - m_Y;
                }
                m_AverageDistance += Math.Sqrt(dx * dx + dy * dy);
                Tracks[i].NotifyChanged();
            }
            m_AverageDistance /= Tracks.Length;
        }

    }

    class Layer : SySal.TotalScan.Layer
    {
        internal bool IsOnDisk;
        private string Path;
        private System.IO.FileStream SwapFile;

        static System.Random Rnd = new System.Random();

        internal void SetRefCenter(double x, double y, double z)
        {
            m_RefCenter.X = x;
            m_RefCenter.Y = y;
            m_RefCenter.Z = z;
        }

        public void UpdateZ()
        {
            UpdateDownstreamZ();
            UpdateUpstreamZ();
        }

        private void InitFile()
        {
            Path = Environment.ExpandEnvironmentVariables("%TEMP%\\alphaomegareconstruction_r_" + System.DateTime.Now.Ticks.ToString("X16") + Rnd.Next().ToString("X8") + "_i_" + Id.ToString() + ".tmp");
            SwapFile = new System.IO.FileStream(Path, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
            IsOnDisk = false;
        }

        public Layer(SySal.TotalScan.Layer l)
        {
            InitFile();
            m_AlignmentData = (SySal.TotalScan.AlignmentData)l.AlignData.Clone();
            m_Id = l.Id;
            m_RefCenter = l.RefCenter;
            m_SheetId = l.SheetId;
            m_Side = l.Side;
            m_BrickId = l.BrickId;
            int i;
            int n = l.Length;
            Segment[] segs = new Segment[n];
            for (i = 0; i < n; i++)
            {
                //segs[i] = new Segment(l[i].Info, l[i].BaseTrackId, this, i, SwapFile); 
                segs[i] = new Segment(l[i].Info, l[i].Index, this, i, SwapFile);
            }
            Segments = segs;
            for (i = 0; i < n; i++)
                //segs[i] = new Segment(segs[i].Info, segs[i].BaseTrackId, this, i, SwapFile);
                segs[i] = new Segment(segs[i].Info, segs[i].Index, this, i, SwapFile);
            m_UpstreamZ = l.UpstreamZ;
            m_DownstreamZ = l.DownstreamZ;
            m_UpstreamZ_Updated = m_DownstreamZ_Updated = true;
        }

        public Layer(SySal.TotalScan.Layer l, Segment[] segs)
        {
            InitFile();

            m_AlignmentData = (SySal.TotalScan.AlignmentData)l.AlignData.Clone();
            m_Id = l.Id;
            m_RefCenter = l.RefCenter;
            m_SheetId = l.SheetId;
            m_Side = l.Side;
            m_BrickId = l.BrickId;
            Segments = segs;
            int i;
            for (i = 0; i < segs.Length; i++)
                segs[i] = new Segment(segs[i].Info, segs[i].Index, this, i, SwapFile);
            m_UpstreamZ = l.UpstreamZ;
            m_DownstreamZ = l.DownstreamZ;
            m_UpstreamZ_Updated = m_DownstreamZ_Updated = true;
        }

        internal void Flush()
        {
            foreach (Segment s in Segments)
                s.Flush();
            GC.Collect();
            IsOnDisk = true;
        }

        internal void MoveToDisk()
        {
            if (!IsOnDisk)
            {
                SwapFile.Position = 0;
                foreach (Segment s in Segments)
                    s.MoveToDisk();
                SwapFile.Flush();
                GC.Collect();
                IsOnDisk = true;
            }
        }

        internal void ReloadFromDisk()
        {
            if (IsOnDisk)
            {
                if (!IsOnDisk) return;
                SwapFile.Position = 0;
                foreach (Segment s in Segments)
                    s.ReloadFromDisk();
                IsOnDisk = false;
            }
        }

        internal void RestoreOriginalSegments()
        {
            int i;
            for (i = 0; i < Segments.Length; i++)
                ((Segment)Segments[i]).RestoreOriginal();
            IsOnDisk = false;
        }

        ~Layer()
        {
            try
            {
                if (SwapFile != null) SwapFile.Close();
                System.IO.File.Delete(Path);
            }
            catch (Exception) { }
        }

        internal void SetAlignmentIgnoreList(int[] alignignorelist)
        {
            int i, n = Segments.Length;
            for (i = 0; i < n; i++)
                ((Segment)Segments[i]).m_IgnoreInAlignment = false;
            foreach (int ix in alignignorelist)
                ((Segment)Segments[ix]).m_IgnoreInAlignment = true;
        }

        internal int UpstreamLinked = -1;

        internal int DownstreamLinked = -1;

        internal void SetId(int id) { m_Id = id; }

        internal AlignmentData iAlignmentData { get { return (AlignmentData)m_AlignmentData; } set { SetAlignmentData(value); } }
    }


    class Volume : SySal.TotalScan.Volume, IExposeManager
    {
        int NextId = 1;

        System.Collections.ArrayList m_XInfoIndex = new ArrayList();
        System.Collections.ArrayList m_XInfo = new ArrayList();

        public int AddInfo(int persistid, string info)
        {
            lock (m_XInfoIndex)
            {
                int i = m_XInfoIndex.BinarySearch(persistid);
                if (i < 0)
                {
                    persistid = NextId++;
                    m_XInfoIndex.Add(persistid);
                    m_XInfo.Add(info);
                }
                else
                {
                    m_XInfo[i] = info;
                }
                return persistid;
            }
        }

        public System.Collections.ArrayList GetExposedInfo()
        {
            lock (m_XInfoIndex)
            {
                System.Collections.ArrayList ar = new ArrayList();
                foreach (string s in m_XInfo)
                    ar.Add(s.Clone());
                return ar;
            }
        }

        internal const int POS_ALIGN_DATA_LEN = 7;

        internal class AOLayerList : SySal.TotalScan.Volume.LayerList
        {
            internal Layer[] iItems
            {
                get { return (Layer[])Items; }
                set { Items = value; }
            }
        }

        internal class AOTrackList : SySal.TotalScan.Volume.TrackList
        {
            internal Track[] iItems
            {
                get { return Items; }
                set { Items = value; }
            }
        }

        internal class AOVertexList : SySal.TotalScan.Volume.VertexList
        {
            internal Vertex[] iItems
            {
                get { return Items; }
                set { Items = value; }
            }
        }

        internal Volume()
        {
            m_Layers = new AOLayerList();
            ((AOLayerList)m_Layers).iItems = new Layer[0];
            m_Tracks = new AOTrackList();
            ((AOTrackList)m_Tracks).iItems = new Track[0];
            m_Vertices = new AOVertexList();
            ((AOVertexList)m_Vertices).iItems = new Vertex[0];
        }

        internal Volume(SySal.TotalScan.Volume.LayerList llist, SySal.TotalScan.Volume.TrackList tlist, SySal.BasicTypes.Cuboid extents, SySal.BasicTypes.Vector refcenter, SySal.BasicTypes.Identifier id)
        {
            m_Extents = extents;
            m_RefCenter = refcenter;
            m_Id = id;
            m_Layers = llist;
            m_Tracks = tlist;
            m_Vertices = new AOVertexList();
            ((AOVertexList)m_Vertices).iItems = new Vertex[0];
        }

        internal AlignmentData[] m_AlignmentData;

        internal void AddLayer(SySal.TotalScan.Layer l)
        {
            //if (l.Length==0) throw new System.Exception("Layer empty!!!");
            int i;
            AOLayerList aol = (AOLayerList)m_Layers;
            Layer[] tmp = aol.iItems;
            aol.iItems = new Layer[tmp.Length + 1];
            for (i = 0; i < tmp.Length && tmp[i].Id < l.Id; i++) aol.iItems[i] = tmp[i];
            aol.iItems[i++] = new Layer(l);
            for (; i < tmp.Length; i++) aol.iItems[i + 1] = tmp[i];
            m_AlignmentData = new AlignmentData[aol.iItems.Length];
            double[] tx = new double[2];
            double[] tfx = new double[Volume.POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
            for (i = 0; i < aol.iItems.Length; i++) m_AlignmentData[i] = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
            for (i = 0; i < aol.iItems.Length; i++) aol.iItems[i].iAlignmentData = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
        }

        internal void AddLayer(SySal.TotalScan.Layer l, MIPEmulsionTrackInfo[] basetks)
        {
            if (l.Length != 0) throw new System.Exception("Layer not empty!!!");
            int i, k, j;
            AOLayerList aol = (AOLayerList)m_Layers;
            Layer[] tmp = aol.iItems;
            aol.iItems = new Layer[tmp.Length + 1];
            for (i = 0; i < tmp.Length && tmp[i].Id < l.Id; i++) aol.iItems[i] = tmp[i];
            Segment[] segs = new Segment[basetks.Length];
            for (j = 0; j < basetks.Length; j++)
                segs[j] = new Segment(basetks[j], new BaseTrackIndex(j), null);
            aol.iItems[i++] = new Layer(l, segs);
            for (; i < tmp.Length; i++) aol.iItems[i + 1] = tmp[i];

            m_AlignmentData = new AlignmentData[aol.iItems.Length];
            double[] tx = new double[2];
            double[] tfx = new double[Volume.POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
            for (i = 0; i < aol.iItems.Length; i++) m_AlignmentData[i] = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
            for (i = 0; i < aol.iItems.Length; i++) aol.iItems[i].iAlignmentData = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
        }

        internal void AddLayer(SySal.TotalScan.Layer l, SySal.Scanning.Plate.LinkedZone lz)
        {
            if (l.Length != 0) throw new System.Exception("Layer not empty!!!");
            int i, k, j;
            AOLayerList aol = (AOLayerList)m_Layers;
            Layer[] tmp = aol.iItems;
            aol.iItems = new Layer[tmp.Length + 1];
            for (i = 0; i < tmp.Length && tmp[i].Id < l.Id; i++) aol.iItems[i] = tmp[i];
            Segment[] segs = new Segment[lz.Length];
            for (j = 0; j < lz.Length; j++)
                segs[j] = new Segment(lz[j].Info, new BaseTrackIndex(lz[j].Id), null);
            aol.iItems[i++] = new Layer(l, segs);
            for (; i < tmp.Length; i++) aol.iItems[i + 1] = tmp[i];

            m_AlignmentData = new AlignmentData[aol.iItems.Length];
            double[] tx = new double[2];
            double[] tfx = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
            for (i = 0; i < aol.iItems.Length; i++) m_AlignmentData[i] = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
            for (i = 0; i < aol.iItems.Length; i++) aol.iItems[i].iAlignmentData = new AlignmentData(tx, tx, tfx, MappingResult.NotPerformedYet);
        }

        #region Topological Functions
        public static SySal.BasicTypes.Vector IntersectTracks(double ax1, double ay1,
            double x1, double y1, double ax2, double ay2, double x2, double y2,
            double CrossTolerance, double MaximumZ, double MinimumZ, ref double Dmin)
        {

            double yp1, xp1, yp2, xp2;
            SySal.BasicTypes.Vector v = new SySal.BasicTypes.Vector();
            Dmin = 0;
            // Calcola la quota della minima distanza,
            // presuppone tracce alla stessa quota

            double LocalDepth = -((ax1 - ax2) * (x1 - x2) + (ay1 - ay2) * (y1 - y2));
            double den = ((ax1 - ax2) * (ax1 - ax2)) + ((ay1 - ay2) * (ay1 - ay2));
            if (den != 0) LocalDepth /= den;

            // Calcola la minima distanza.
            double argx = ((ax1 - ax2) * LocalDepth) + (x1 - x2);
            double argy = ((ay1 - ay2) * LocalDepth) + (y1 - y2);
            Dmin = Math.Sqrt((argx * argx) + (argy * argy));

            // Calcola le trasverse.
            if (CrossTolerance > Dmin)
            {
                if (MaximumZ < LocalDepth)
                {
                    argx = ((ax1 - ax2) * MaximumZ) + (x1 - x2);
                    argy = ((ay1 - ay2) * MaximumZ) + (y1 - y2);
                    Dmin = Math.Sqrt((argx * argx) + (argy * argy));
                    if (CrossTolerance > Dmin)
                    {
                        xp1 = x1 + (ax1 * MaximumZ);
                        yp1 = y1 + (ay1 * MaximumZ);
                        xp2 = x2 + (ax2 * MaximumZ);
                        yp2 = y2 + (ay2 * MaximumZ);
                        v.Z = (float)MaximumZ;
                        v.Y = (float)(yp1 + yp2) / 2;
                        v.X = (float)(xp1 + xp2) / 2;
                    };
                }
                else if (MinimumZ > LocalDepth)
                {
                    argy = ((ay1 - ay2) * MinimumZ) + (y1 - y2);
                    argx = ((ax1 - ax2) * MinimumZ) + (x1 - x2);
                    Dmin = Math.Sqrt((argx * argx) + (argy * argy));
                    if (CrossTolerance > Dmin)
                    {
                        yp1 = y1 + (ay1 * MinimumZ);
                        xp1 = x1 + (ax1 * MinimumZ);
                        yp2 = y2 + (ay2 * MinimumZ);
                        xp2 = x2 + (ax2 * MinimumZ);
                        v.Z = (float)MinimumZ;
                        v.Y = (float)(yp1 + yp2) / 2;
                        v.X = (float)(xp1 + xp2) / 2;
                    };
                }
                else
                {
                    yp1 = y1 + (ay1 * LocalDepth);
                    xp1 = x1 + (ax1 * LocalDepth);
                    yp2 = y2 + (ay2 * LocalDepth);
                    xp2 = x2 + (ax2 * LocalDepth);
                    v.Z = (float)LocalDepth;
                    v.Y = (float)(yp1 + yp2) / 2;
                    v.X = (float)(xp1 + xp2) / 2;
                };

            };
            return v;

        }

        public static TrackIntersection IntersectTracks(Track t1,
            Track t2, double CrossTolerance, double MaximumZ, double MinimumZ,
            IntersectionType[] IType)
        {

            int i;
            double Dmin = 0; //,xp;
            double y1 = 0, x1 = 0, y2 = 0, x2 = 0;
            double ay1 = 0, ax1 = 0, ay2 = 0, ax2 = 0;
            TrackIntersection v = new TrackIntersection();
            SySal.BasicTypes.Vector tt;
            double[] z1 = new double[2];
            double[] z2 = new double[2];
            int n;
            double dz;

            z1[0] = t1[0].Info.TopZ;
            n = t1.Length;
            z1[1] = t1[n - 1].Info.BottomZ;
            z2[0] = t2[0].Info.TopZ;
            n = t2.Length;
            z2[1] = t2[n - 1].Info.BottomZ;

            //Se dx>0 traccia1 più downstream
            //Se dx<0 traccia2 più downstream
            dz = z1[0] - z2[0];

            for (i = 0; i < IType.Length; i++)
            {
                if (IType[i] == IntersectionType.V || IType[i] == IntersectionType.Y ||
                    IType[i] == IntersectionType.X)
                {
                    ay1 = t1.Upstream_SlopeY;
                    ax1 = t1.Upstream_SlopeX;
                    ay2 = t2.Upstream_SlopeY;
                    ax2 = t2.Upstream_SlopeX;
                    y1 = t1.Upstream_PosY;
                    x1 = t1.Upstream_PosX;
                    y2 = t2.Upstream_PosY;
                    x2 = t2.Upstream_PosX;
                }
                else if (IType[i] == IntersectionType.Lambda)
                {
                    ay1 = t1.Downstream_SlopeY;
                    ax1 = t1.Downstream_SlopeX;
                    ay2 = t2.Downstream_SlopeY;
                    ax2 = t2.Downstream_SlopeX;
                    y1 = t1.Downstream_PosY;
                    x1 = t1.Downstream_PosX;
                    y2 = t2.Downstream_PosY;
                    x2 = t2.Downstream_PosX;
                }
                else //Kink
                {
                    if (dz > 0)
                    {
                        ay1 = t1.Upstream_SlopeY;
                        ax1 = t1.Upstream_SlopeX;
                        ay2 = t2.Downstream_SlopeY;
                        ax2 = t2.Downstream_SlopeX;
                        y1 = t1.Upstream_PosY;
                        x1 = t1.Upstream_PosX;
                        y2 = t2.Downstream_PosY;
                        x2 = t2.Downstream_PosX;
                    }
                    else if (dz < 0)
                    {
                        ay1 = t1.Downstream_SlopeY;
                        ax1 = t1.Downstream_SlopeX;
                        ay2 = t2.Upstream_SlopeY;
                        ax2 = t2.Upstream_SlopeX;
                        y1 = t1.Downstream_PosY;
                        x1 = t1.Downstream_PosX;
                        y2 = t2.Upstream_PosY;
                        x2 = t2.Upstream_PosX;
                    };
                };

                //Incoming and Outgoing quantities are all
                //projected at depth=0


                if ((IType[i] == IntersectionType.Kink && dz != 0) || (IType[i] != IntersectionType.Kink))
                {
                    // Calcola la quota della minima distanza,
                    tt = IntersectTracks(ax1, ay1, x1, y1, ax2, ay2, x2, y2, CrossTolerance, MaximumZ, MinimumZ, ref Dmin);


                    if (tt.X != 0 && tt.Y != 0 && tt.Z != 0)
                    //if (tt != null)
                    {
                        double tempZ = tt.Z;// + xp;

                        // Topology
                        IntersectionType tmpIType = IntersectionType.Unknown;
                        IntersectionSymmetry tmpISimm = IntersectionSymmetry.Unknown;
                        Relationship[] tmpRel = new Relationship[2];

                        CheckTopology(z1, z2, tempZ, ref tmpIType, ref tmpISimm, ref tmpRel[0], ref tmpRel[1]);
                        if (tmpIType == IType[i])
                        {
                            v.Type/*.IntersType*/ = tmpIType;
                            v.Symmetry/*.IntersSymm*/ = tmpISimm;

                            //La quota viene portata nel sistema di rif esterno
                            v.Pos.Z = (float)tempZ;
                            v.Pos.Y = tt.Y;
                            v.Pos.X = tt.X;
                            v.Track1 = t1;
                            v.Track2 = t2;
                            //v.Track1.Relat = tmpRel[0];
                            //v.Track2.Relat = tmpRel[1];
                            v.ClosestApproachDistance/*.Dmin*/ = Dmin;
                            return v;
                        };
                    };
                };
            };
            return null;

        }

        public static SegmentIntersection IntersectSegments(Segment s1,
            Segment s2, double CrossTolerance, double MaximumZ, double MinimumZ)
        {

            double zp, Dmin = 0;
            double y1, x1, y2, x2;
            double ay1, ax1, ay2, ax2;
            Relationship dum = Relationship.Unknown;
            SegmentIntersection v = new SegmentIntersection();
            SySal.BasicTypes.Vector tt;

            //Le due tracce vengono proiettate alla stessa quota,
            // quella più upstream
            SySal.Tracking.MIPEmulsionTrackInfo s1Info = s1.GetInfo();
            SySal.Tracking.MIPEmulsionTrackInfo s2Info = s2.GetInfo();
            double dz = s1Info.Intercept.Z - s2Info.Intercept.Z;

            if (dz > 0)
            {
                zp = s2Info.Intercept.Z;
                //x1 = t1.Nx - dx;
                y1 = s1Info.Intercept.Y - dz * s1Info.Slope.Y;
                x1 = s1Info.Intercept.X - dz * s1Info.Slope.X;
                ay1 = s1Info.Slope.Y;
                ax1 = s1Info.Slope.X;
                //x2 = t2.Nx;
                y2 = s2Info.Intercept.Y;
                x2 = s2Info.Intercept.X;
                ay2 = s2Info.Slope.Y;
                ax2 = s2Info.Slope.X;
            }
            else
            {
                zp = s1Info.Intercept.Z;
                //x1 = t1.Nx;
                y1 = s1Info.Intercept.Y;
                x1 = s1Info.Intercept.X;
                ay1 = s1Info.Slope.Y;
                ax1 = s1Info.Slope.X;
                //x2 = t2.Nx - dx;
                y2 = s2Info.Intercept.Y - dz * s2Info.Slope.Y;
                x2 = s2Info.Intercept.X - dz * s2Info.Slope.X;
                ay2 = s2Info.Slope.Y;
                ax2 = s2Info.Slope.X;
            };

            // Calcola la quota della minima distanza,
            tt = IntersectTracks(ax1, ay1, x1, y1, ax2, ay2, x2, y2, CrossTolerance, MaximumZ, MinimumZ, ref Dmin);

            if (tt.X != 0 && tt.Y != 0 && tt.Z != 0)
            {
                double tempZ = tt.Z + zp;
                //La quota viene portata nel sistema di rif esterno
                v.Pos.Z = (float)tempZ;
                v.Pos.Y = tt.Y;
                v.Pos.X = tt.X;
                v.Segment1 = s1;
                v.Segment2 = s2;
                v.ClosestApproachDistance/*.Dmin*/ = Dmin;

                // Topology
                double[] z1 = new double[2];
                double[] z2 = new double[2];
                IntersectionType IType = IntersectionType.Unknown;
                IntersectionSymmetry ISimm = IntersectionSymmetry.Unknown;

                z1[0] = s1Info.TopZ;
                z1[1] = s1Info.BottomZ;
                z2[0] = s2Info.TopZ;
                z2[1] = s2Info.BottomZ;
                CheckTopology(z1, z2, tempZ, ref IType, ref ISimm, ref dum, ref dum);
                v.Type/*.IntersType*/ = IType;
                v.Symmetry/*.IntersSymm*/ = ISimm;

            }
            else
            {
                v = null;
            };

            return v;

        }

        public static void CheckTopology(double[] z1, double[] z2, double zv,
            ref IntersectionType IType, ref IntersectionSymmetry ISimm,
            ref Relationship Track1Rel, ref Relationship Track2Rel)
        {

            IType = IntersectionType.Unknown;
            ISimm = IntersectionSymmetry.Unknown;

            if (zv < z1[1] && zv < z2[1])
            {
                //V
                IType = IntersectionType.V;
                ISimm = IntersectionSymmetry.Symmetric;
                Track1Rel = Relationship.Daughter;
                Track2Rel = Track1Rel;
            }
            else if (zv < z1[1] && zv > z2[1])
            {
                //Y or Kink
                ISimm = IntersectionSymmetry.Element2Upstream;
                if (zv < z2[0]) //Y 2
                    IType = IntersectionType.Y;
                else // Kink 1
                    IType = IntersectionType.Kink;
                Track1Rel = Relationship.Daughter;
                Track2Rel = Relationship.Mother;

            }
            else if (zv > z1[1] && zv < z2[1])
            {
                //Y or Kink
                ISimm = IntersectionSymmetry.Element1Upstream;
                if (zv < z1[0]) //Y 1
                    IType = IntersectionType.Y;
                else // Kink 2
                    IType = IntersectionType.Kink;
                Track2Rel = Relationship.Daughter;
                Track1Rel = Relationship.Mother;
            }
            else if (zv > z1[1] && zv > z2[1])
            {
                //X or Lambda
                if (zv < z1[0] && zv < z2[0]) //Case X
                {
                    IType = IntersectionType.X;
                    ISimm = IntersectionSymmetry.Symmetric;
                    Track1Rel = Relationship.Daughter;
                    Track2Rel = Track1Rel;
                }
                else //Case Lambda
                {
                    IType = IntersectionType.Lambda;
                    if (zv > z1[0] && zv < z2[0]) //Lambda 2
                    {
                        ISimm = IntersectionSymmetry.Element1Upstream;
                        Track2Rel = Relationship.Daughter;
                        Track1Rel = Relationship.Mother;
                    }
                    else if (zv < z1[0] && zv > z2[0]) //Lambda 1
                    {
                        ISimm = IntersectionSymmetry.Element2Upstream;
                        Track1Rel = Relationship.Daughter;
                        Track2Rel = Relationship.Mother;
                    }
                    else // Symmetric Lambda
                    {
                        ISimm = IntersectionSymmetry.Symmetric;
                        Track1Rel = Relationship.Mother;
                        Track2Rel = Track1Rel;
                    };
                };
            };

        }
        #endregion

        #region Vertexing
        public static Vertex[] ClusterizeIntersections(TrackIntersection[] Ti, Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i, j, k, l, h, l1;
            int n, m = 0, m2 = 0;
            double xi = 0, yi = 0, zi = 0;
            double xj = 0, yj = 0, zj = 0;
            double xm, ym, zm;
            bool join_to_a_vertex, add_this_track1, add_this_track2, change_vertex, replica;
            bool[] vtx_joined;
            Vertex vtx = new Vertex();
            Vertex[] tvtx = new Vertex[2];
            for (i = 0; i < 2; i++) tvtx[i] = new Vertex();
            //Vertex vtxj = new Vertex(); 
            //TrackIntersection[] tmptti= new TrackIntersection[2];

            //ArrayList tempTi = new ArrayList();
            ArrayList tempvtx = new ArrayList();
            ArrayList tempvtx2 = new ArrayList();

            n = Ti.Length;


            //In this loop pairs of crosses become vertices.
            for (i = 0; i < n; i++)
            {
                vtx = new Vertex(m);

                //Nell'incrocio ad X (simmetrico) vengono considerate convenzionalmente tutte e due downstream
                //Nell'incrocio a Y già nella funzione CheckTopology la traccia che taglia è condiserata downstream
                //e quella tagliata upstream
                vtx.AddTrack(Ti[i].Track1,
                            ((Ti[i].Symmetry == IntersectionSymmetry.Element1Upstream) ||
                             (Ti[i].Symmetry == IntersectionSymmetry.Symmetric && Ti[i].Type == IntersectionType.Lambda)) ?
                            true : false);
                vtx.AddTrack(Ti[i].Track2,
                            ((Ti[i].Symmetry == IntersectionSymmetry.Element2Upstream) ||
                             (Ti[i].Symmetry == IntersectionSymmetry.Symmetric && Ti[i].Type == IntersectionType.Lambda)) ?
                            true : false);
                if (vtx.Length > 1)
                {
                    tempvtx.Add(vtx);
                    m++;
                };
            }

            //In this loop two-tracks vertices are joined into vertices.
            //If 2 pairs are near each other they are joined.
            //If there is already a vertex near these two pairs, they
            //are joined to that vertex
            bool vtxjoined = false;
            int jstart;
            double cstartlong = C.StartingClusterToleranceLong;
            double cstarttrans = C.StartingClusterToleranceTrans;

            for (i = 0; i < m - 1; i++)
            {

                if (vtxjoined && i > 0)
                {
                    i--;
                    jstart = 0;
                }
                else
                {
                    jstart = i + 1;
                }
                tvtx[0] = (Vertex)tempvtx[i];
                if (tvtx[0].Length > 0)
                {

                    vtxjoined = false;
                    for (j = jstart; j < m; j++)
                    {
                        tvtx[1] = (Vertex)tempvtx[j];
                        if (tvtx[1].Length > 1 && tvtx[0].Length == 1)
                        {
                            xj = tvtx[1].X;
                            yj = tvtx[1].Y;
                            zj = tvtx[1].Z;

                            xi = tvtx[0][0].Downstream_PosX + tvtx[0][0].Downstream_SlopeX * zj;
                            yi = tvtx[0][0].Downstream_PosY + tvtx[0][0].Downstream_SlopeY * zj;
                            zi = zj;
                        }
                        else if (tvtx[1].Length == 1 && tvtx[0].Length > 1)
                        {
                            xi = tvtx[0].X;
                            yi = tvtx[0].Y;
                            zi = tvtx[0].Z;

                            xj = tvtx[1][0].Downstream_PosX + tvtx[1][0].Downstream_SlopeX * zi;
                            yj = tvtx[1][0].Downstream_PosY + tvtx[1][0].Downstream_SlopeY * zi;
                            zj = zi;
                        }
                        else if (tvtx[1].Length > 1 && tvtx[0].Length > 1)
                        {
                            xi = tvtx[0].X;
                            yi = tvtx[0].Y;
                            zi = tvtx[0].Z;

                            xj = tvtx[1].X;
                            yj = tvtx[1].Y;
                            zj = tvtx[1].Z;
                        }

                        //la condizione i!=j ci vuole perchè nel caso di un join allora j è riportata a 0
                        //perchè è necessario che il nuovo vertice unificato sia sottoposto al join con tutti i vertici
                        if (Math.Abs(zi - zj) < cstartlong &&
                            Math.Abs(yi - yj) < cstarttrans &&
                            Math.Abs(xi - xj) < cstarttrans &&
                            tvtx[1].Length + tvtx[0].Length > 2 &&
                            i != j)
                        {


                            xm = (xi + xj) / 2;
                            ym = (yi + yj) / 2;
                            zm = (zi + zj) / 2;
                            add_this_track1 = true;
                            add_this_track2 = true;
                            for (l = tvtx[1].Length - 1; l > -1; l--)
                            {
                                for (k = 0; k < tvtx[0].Length; k++)
                                {
                                    if (tvtx[1][l].Id == tvtx[0][k].Id)
                                    {
                                        add_this_track1 = false;
                                        break;
                                    }
                                }
                                if (add_this_track1) tvtx[0].AddTrack(tvtx[1][l], (tvtx[1][l].Comment == "UpstreamTrack " + tvtx[0].Id) ? true : false);
                                //if(add_this_track1) tvtx[0].AddTrack(tvtx[1][l],(tvtx[1][l].Downstream_Vertex==tvtx[1])?true:false);
                            };

                            if (j > i)
                            {
                                tempvtx.RemoveAt(j);
                                tempvtx.RemoveAt(i);
                                tempvtx.Insert(i, tvtx[0]);
                            }
                            else
                            {
                                tempvtx.RemoveAt(i);
                                tempvtx.RemoveAt(j);
                                tempvtx.Insert(i - 1, tvtx[0]);
                            }
                            m--;

                            vtxjoined = true;
                            break;

                        };
                    };
                };
            };

            for (i = 0; i < tempvtx.Count - 1; i++)
            {
                tvtx[0] = (Vertex)tempvtx[i];
                if (tvtx[0].Length > 0)
                {
                    for (j = i + 1; j < tempvtx.Count; j++)
                    {
                        tvtx[1] = (Vertex)tempvtx[j];
                        if (tvtx[1].Length > 1)
                        {
                            for (l = tvtx[1].Length - 1; l > -1; l--)
                            {
                                for (k = tvtx[0].Length - 1; k > -1; k--)
                                {
                                    if (tvtx[1][l].Id == tvtx[0][k].Id)
                                    {
                                        if (tvtx[1].Length < tvtx[0].Length)
                                        {
                                            tvtx[1].RemoveTrack(tvtx[1][l]);
                                        }
                                        else if (tvtx[1].Length > tvtx[0].Length)
                                        {
                                            tvtx[0].RemoveTrack(tvtx[0][k]);
                                        }
                                        else
                                        {
                                            if (tvtx[1].DX * tvtx[1].DX + tvtx[1].DY * tvtx[1].DY >
                                               tvtx[0].DX * tvtx[0].DX + tvtx[0].DY * tvtx[0].DY)
                                            {
                                                tvtx[1].RemoveTrack(tvtx[1][l]);
                                            }
                                            else
                                            {
                                                tvtx[0].RemoveTrack(tvtx[0][k]);
                                            }

                                        }

                                        break;
                                    }
                                }
                            };
                            if (tvtx[1].Length < 2)
                            {
                                tempvtx.RemoveAt(j);
                                j--;
                            }
                            if (tvtx[0].Length < 2)
                            {
                                tempvtx.RemoveAt(i);
                                i--;
                                break;
                            }



                        }
                    }
                }

            }

            Vertex[] tvt = (Vertex[])tempvtx.ToArray(typeof(Vertex));
            m = 0; n = tvt.Length;
            char[] tmpchar = new char[1] { ' ' };
            for (i = 0; i < n; i++)
                if (tvt[i].Length > 1)
                {
                    k = tvt[i].Length;
                    //					int previdx = -1;
                    //					bool firsttime = true;
                    //					double primtol=0;
                    for (j = 0; j < k; j++)
                    {
                        string[] tmpstr = tvt[i][j].Comment.Split(tmpchar, 2);
                        if (tmpstr.Length == 2 && tmpstr[0] == "UpstreamTrack" && tmpstr[1] == tvt[i].Id.ToString())
                        {
                            /*							double dx = tvt[i].X - (tvt[i][j].Downstream_PosX + tvt[i][j].Downstream_SlopeX*tvt[i].Z);
                                                        double dy = tvt[i].Y - (tvt[i][j].Downstream_PosY + tvt[i][j].Downstream_SlopeY*tvt[i].Z);
                                                        if (dx*dx+dy*dy< primtol || firsttime)
                                                        {
                                                            primtol = dx*dx+dy*dy;
                                                            firsttime = false;
                            */
                            tvt[i][j].SetDownstreamVertex(tvt[i]);
                            /*								if(previdx!=-1)
                                                            {
                                                                tvt[i][previdx].SetUpstreamVertex(tvt[i]);//.SetDownstreamVertex(null);

                                                            }
                                                            previdx=j;
                                                        }
                            */
                        }
                        else
                        {
                            tvt[i][j].SetUpstreamVertex(tvt[i]);//.SetDownstreamVertex(tvt[i]);
                        }
                    }
                    tempvtx2.Add(tempvtx[i]);
                    m++;
                };
            if (Report != null) Report(m + " Vertices\r\n");
            return (Vertex[])tempvtx2.ToArray(typeof(Vertex));

        }

        public Vertex[] FullVertexReconstruction(SySal.TotalScan.Volume vol, Track[] tr, IntersectionType[] ityp,
            Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            switch (C.VtxAlgorithm)
            {
                case VertexAlgorithm.PairBased: return PairBasedVertexReconstruction(tr, ityp, C, ShouldStop, Progress, Report);

                case VertexAlgorithm.Global: return GlobalVertexReconstruction(vol, tr, C, ShouldStop, Progress, Report);

                default: return new Vertex[0];
            }
        }

        private delegate double dGVtxFilter(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass);

        private static double gvtxfltN(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return tkids.Length; }
        private static double gvtxfltND(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { int i, n; for (i = n = 0; i < tkids.Length; i++) if (tkids[i] % 2 == 0) n++; return n; }
        private static double gvtxfltNU(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { int i, n; for (i = n = 0; i < tkids.Length; i++) if (tkids[i] % 2 == 1) n++; return n; }
        private static double gvtxfltPX(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return v.X; }
        private static double gvtxfltPY(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return v.Y; }
        private static double gvtxfltPZ(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return v.Z; }
        private static double gvtxfltRX(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return vol.RefCenter.X; }
        private static double gvtxfltRY(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return vol.RefCenter.Y; }
        private static double gvtxfltRZ(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return vol.RefCenter.Z; }
        private static double gvtxfltLastPass(SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass) { return lastpass ? 1.0 : 0.0; }

        private static bool GVtxKeep(NumericalTools.CStyleParsedFunction gvf, dGVtxFilter[] fltpars, SySal.BasicTypes.Vector v, int[] tkids, SySal.TotalScan.Volume vol, bool lastpass)
        {
            if (gvf == null) return true;
            int i;
            for (i = 0; i < fltpars.Length; i++)
                gvf[i] = fltpars[i](v, tkids, vol, lastpass);
            return gvf.Evaluate() != 0.0;
        }

        public Vertex[] GlobalVertexReconstruction(SySal.TotalScan.Volume vol, Track[] tr, Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i, j, h, k, s;

            double GVtxZScanStep = 2.0 * C.GVtxRadius / C.GVtxMaxSlopeDivergence;

            NumericalTools.CStyleParsedFunction GVf = null;
            dGVtxFilter[] fltpars = null;
            if (C.GVtxFilter != null && C.GVtxFilter.Trim().Length > 0)
            {
                GVf = new CStyleParsedFunction(C.GVtxFilter);
                fltpars = new dGVtxFilter[GVf.ParameterList.Length];
                for (i = 0; i < GVf.ParameterList.Length; i++)
                {
                    dGVtxFilter flt = null;
                    switch (GVf.ParameterList[i].ToUpper())
                    {
                        case "N": flt = gvtxfltN; break;
                        case "ND": flt = gvtxfltND; break;
                        case "NU": flt = gvtxfltNU; break;
                        case "PX": flt = gvtxfltPX; break;
                        case "PY": flt = gvtxfltPY; break;
                        case "PZ": flt = gvtxfltPZ; break;
                        case "RX": flt = gvtxfltRX; break;
                        case "RY": flt = gvtxfltRY; break;
                        case "RZ": flt = gvtxfltRZ; break;
                        case "LASTPASS": flt = gvtxfltLastPass; break;
                        default: throw new Exception("Parameter " + GVf.ParameterList[i] + " supplied to Global Vertex filtering is unknown.");
                    }
                    fltpars[i] = flt;
                }
            }

            int zmini, zmaxi;
            double minz, maxz;
            double z;
            double VMinZ, VMaxZ;
            double Radius2 = C.GVtxRadius * C.GVtxRadius;

            for (i = 0; i < tr.Length; i++)
            {
                tr[i].SetDownstreamVertex(null);
                tr[i].SetUpstreamVertex(null);
                tr[i].NotifyChanged();
            }
            VMinZ = m_Layers[m_Layers.Length - 1].RefCenter.Z - C.GVtxMaxExt;
            VMaxZ = m_Layers[0].RefCenter.Z + C.GVtxMaxExt;
            ProjMap[] zlev = new ProjMap[(int)Math.Ceiling((VMaxZ - VMinZ) / GVtxZScanStep)];
            ProjMap[] zsens = new ProjMap[zlev.Length];
            for (i = 0; i < zlev.Length; i++)
            {
                zlev[i] = new ProjMap();
                zsens[i] = new ProjMap();
            }

            if (Progress != null) Progress(0.0);
            if (ShouldStop != null && ShouldStop()) return null;
            if (Report != null) Report("Scanning tracks...\r\n");
            for (i = 0; i < tr.Length; i++)
            {
                if (ShouldStop != null && ShouldStop()) return null;
                if (Progress != null) Progress(0.33 * (double)i / (double)tr.Length);
                SySal.TotalScan.Track t = tr[i];
                if (t.Length < C.GVtxMinCount || (t.Comment != null && String.Compare(t.Comment, "NO VERTEX", true) == 0)) continue;
                for (s = 0; s < 2; s++)
                {
                    if (s == 0)
                    {
                        minz = t.Downstream_Z;
                        maxz = t.Downstream_Z + C.GVtxMaxExt;
                    }
                    else
                    {
                        minz = t.Upstream_Z - C.GVtxMaxExt;
                        maxz = t.Upstream_Z;
                    }
                    zmini = (int)Math.Ceiling((minz - VMinZ) / GVtxZScanStep);
                    zmaxi = (int)Math.Floor((maxz - VMinZ) / GVtxZScanStep);
                    if (zmini < 0) zmini = 0;
                    if (zmaxi > zlev.Length - 1) zmaxi = zlev.Length - 1;
                    for (j = zmini; j < zmaxi; j++)
                    {
                        z = j * GVtxZScanStep + VMinZ;
                        Projection p = new Projection();
                        p.Id = i;
                        p.IsDownstream = (s == 0);
                        if (s == 0)
                        {
                            p.P.X = t.Downstream_PosX + (z - t.Downstream_PosZ) * t.Downstream_SlopeX;
                            p.P.Y = t.Downstream_PosY + (z - t.Downstream_PosZ) * t.Downstream_SlopeY;
                        }
                        else
                        {
                            p.P.X = t.Upstream_PosX + (z - t.Upstream_PosZ) * t.Upstream_SlopeX;
                            p.P.Y = t.Upstream_PosY + (z - t.Upstream_PosZ) * t.Upstream_SlopeY;
                        }
                        zlev[j].Add(p);
                    }
                }
            }
            /*
                        for (i = 0; i < zlev.Length; i++)
                            zlev[i].Freeze(C.GVtxRadius);
            */
            ProjMap zlevc, zsensc;
            if (ShouldStop != null && ShouldStop()) return null;
            if (Report != null) Report("Preparing tracks...\r\n");
            for (j = 0; j < zlev.Length; j++)
            {
                if (ShouldStop != null && ShouldStop()) return null;
                if (Progress != null) Progress(0.33 * (1.0 + (double)j / (double)zlev.Length));
                zlevc = zlev[j];
                zlevc.Freeze(C.GVtxRadius);
                for (i = 0; i < zlevc.m_List.Count; i++)
                    ((Projection)zlevc.m_List[i]).Sensitized = false;
                for (i = 0; i < zlevc.m_List.Count; i++)
                {
                    Projection p = (Projection)zlevc.m_List[i];
                    ProjMap.Iterator iter = zlevc.Lock(p);
                    Projection op;
                    while ((op = zlevc.Next(iter)) != null)
                    {
                        if (op == p) continue;
                        if (Math.Sqrt((p.P.X - op.P.X) * (p.P.X - op.P.X) + (p.P.Y - op.P.Y) * (p.P.Y - op.P.Y)) < C.GVtxRadius)
                        {
                            if (!p.Sensitized)
                            {
                                zsens[j].Add(p);
                                p.Sensitized = true;
                            }
                            if (!op.Sensitized)
                            {
                                zsens[j].Add(op);
                                op.Sensitized = true;
                            }
                        }
                    }
                }
                zsens[j].Freeze(C.GVtxRadius);
                zlev[j] = null;
            }

            System.Collections.ArrayList tkgroups = new System.Collections.ArrayList();

            SySal.BasicTypes.Vector2 p12 = new SySal.BasicTypes.Vector2();
            SySal.BasicTypes.Vector2 n12 = new SySal.BasicTypes.Vector2();
            SySal.BasicTypes.Vector2 m12 = new SySal.BasicTypes.Vector2();
            SySal.BasicTypes.Vector2 cg = new SySal.BasicTypes.Vector2();
            if (ShouldStop != null && ShouldStop()) return null;
            if (Report != null) Report("Finding vertices...\r\n");
            for (i = 0; i < zlev.Length; i++)
            {
                if (ShouldStop != null && ShouldStop()) return null;
                if (Progress != null) Progress(0.33 * (2.0 + (double)i / (double)zlev.Length));
                zsensc = zsens[i];
                for (j = 0; j < zsensc.m_List.Count; j++)
                {
                    Projection p1 = (Projection)(zsensc.m_List[j]);
                    ProjMap.Iterator iter = zsensc.Lock(p1);
                    Projection p2 = null;
                    while ((p2 = zsensc.Next(iter)) != null)
                    {
                        if (p2 == p1) continue;
                        p12.X = p1.P.X - p2.P.X;
                        p12.Y = p1.P.Y - p2.P.Y;
                        double p12m = p12.X * p12.X + p12.Y * p12.Y;
                        if (p12m > 0.0 && p12m < Radius2)
                        {
                            double p12n = Math.Sqrt(p12m);
                            double p12ni = 1.0 / p12n;
                            n12.X = p12.X * p12ni;
                            n12.Y = p12.Y * p12ni;
                            m12.X = 0.5 * (p1.P.X + p2.P.X);
                            m12.Y = 0.5 * (p1.P.Y + p2.P.Y);
                            double d = Math.Sqrt(Radius2 - p12m * 0.25);
                            for (s = -1; s <= 1; s += 2)
                            {
                                cg.X = m12.X + d * s * n12.Y;
                                cg.Y = m12.Y - d * s * n12.X;
                                System.Collections.ArrayList gl = new System.Collections.ArrayList();
                                gl.Add(p1.Id * 2 + (p1.IsDownstream ? 1 : 0));
                                gl.Add(p2.Id * 2 + (p2.IsDownstream ? 1 : 0));
                                ProjMap.Iterator itergroup = zsensc.Lock(p1);
                                Projection pq = null;
                                while ((pq = zsensc.Next(itergroup)) != null)
                                {
                                    if (pq == p1 || pq == p2) continue;
                                    double dpx = pq.P.X - cg.X;
                                    double dpy = pq.P.Y - cg.Y;
                                    if ((dpx * dpx + dpy * dpy) < Radius2)
                                    {
                                        gl.Add(pq.Id * 2 + (pq.IsDownstream ? 1 : 0));
                                    }
                                }
#if true
                                int[] group = (int[])gl.ToArray(typeof(int));
                                Vertex vtx = C.VtxFitWeightEnable ? new UtilityVertex() : new Vertex(-1);
                                VertexFit vf = new VertexFit();
                                for (h = 0; h < group.Length; h++)
                                {
                                    Track tk = tr[group[h] / 2];
                                    VertexFit.TrackFit tf = new VertexFit.TrackFit();
                                    if (group[h] % 2 == 0)
                                    {
                                        vtx.AddTrack(tk, true);
                                        tf.Intercept.X = tk.Upstream_PosX + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeX;
                                        tf.Intercept.Y = tk.Upstream_PosY + (tk.Upstream_Z - tk.Upstream_PosZ) * tk.Upstream_SlopeY;
                                        tf.Intercept.Z = tk.Upstream_Z;
                                        tf.Slope.X = tk.Upstream_SlopeX;
                                        tf.Slope.Y = tk.Upstream_SlopeY;
                                        tf.Slope.Z = 1.0;
                                        tf.Weight = Vertex.SlopeScatteringWeight(tk);
                                        tf.MaxZ = tk.Upstream_Z;
                                        tf.MinZ = tf.MaxZ - 1e6;
                                    }
                                    else
                                    {
                                        vtx.AddTrack(tk, false);
                                        tf.Intercept.X = tk.Downstream_PosX + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeX;
                                        tf.Intercept.Y = tk.Downstream_PosY + (tk.Downstream_Z - tk.Downstream_PosZ) * tk.Downstream_SlopeY;
                                        tf.Intercept.Z = tk.Downstream_Z;
                                        tf.Slope.X = tk.Downstream_SlopeX;
                                        tf.Slope.Y = tk.Downstream_SlopeY;
                                        tf.Slope.Z = 1.0;
                                        tf.Weight = Vertex.SlopeScatteringWeight(tk);
                                        tf.MinZ = tk.Downstream_Z;
                                        tf.MaxZ = tf.MinZ + 1e6;
                                    }
                                    tf.Id = new BaseTrackIndex(tk.Id);
                                    vf.AddTrackFit(tf);
                                }
                                try
                                {
                                    if (vf.AvgDistance < 0 || vf.AvgDistance > C.GVtxRadius)
                                    {
                                        group = new int[0];
                                        break;
                                    }
                                }
                                catch (VertexFit.FitException)
                                {
                                    group = new int[0];
                                    break;
                                }

                                for (h = 0; h < group.Length; h++)
                                {
                                    Track tk = tr[group[h] / 2];
                                    if (group[h] % 2 == 0)
                                    {
                                        if (vtx[h].Upstream_Vertex != null)
                                        {
                                            if (vtx.Length > vtx[h].Upstream_Vertex.Length ||
                                                (vtx.Length == vtx[h].Upstream_Vertex.Length && vf.AvgDistance < vtx[h].Upstream_Vertex.AverageDistance))
                                            {
                                                Vertex w = vtx[h].Upstream_Vertex;
                                                //KRYSS: NEW - ONLY DESTROY VERTICES IF ONE TRACK IS LEFT
                                                if (w.Length <= 2)
                                                {
                                                    //END KRYSS
                                                    for (k = 0; k < w.Length; k++)
                                                        if (w[k].Upstream_Vertex == w) w[k].SetUpstreamVertex(null);
                                                        else if (w[k].Downstream_Vertex == w) w[k].SetDownstreamVertex(null);
                                                    //KRYSS: NEW - ONLY DESTROY VERTICES IF ONE TRACK IS LEFT
                                                }
                                                else
                                                {
                                                    w.RemoveTrack(tk);
                                                    try
                                                    {
                                                        w.ComputeVertexCoordinates();
                                                    }
                                                    catch (VertexFit.FitException)
                                                    {
                                                        for (k = 0; k < w.Length; k++)
                                                            if (w[k].Upstream_Vertex == w) w[k].SetUpstreamVertex(null);
                                                            else if (w[k].Downstream_Vertex == w) w[k].SetDownstreamVertex(null);
                                                    }
                                                }
                                                //END KRYSS
                                            }
                                            else if (vtx.Length <= 2)
                                            {
                                                group = new int[0];
                                                break;
                                            }
                                            else
                                            {
                                                vtx.RemoveTrack(tk);
                                                vf.RemoveTrackFit(new BaseTrackIndex(tk.Id));
                                                try
                                                {
                                                    if (vf.AvgDistance < 0 || vf.AvgDistance > C.GVtxRadius)
                                                    {
                                                        group = new int[0];
                                                        break;
                                                    }
                                                }
                                                catch (VertexFit.FitException)
                                                {
                                                    group = new int[0];
                                                    break;
                                                }
                                                int m;
                                                int[] newgroup = new int[group.Length - 1];
                                                for (m = 0; m < h; m++) newgroup[m] = group[m];
                                                for (m = h; m < group.Length - 1; m++) newgroup[m] = group[m + 1];
                                                h--;
                                                group = newgroup;
                                                continue;
                                            }
                                        }
                                    }
                                    else
                                    {
                                        if (vtx[h].Downstream_Vertex != null)
                                        {
                                            if (vtx.Length > vtx[h].Downstream_Vertex.Length ||
                                                (vtx.Length == vtx[h].Downstream_Vertex.Length && vf.AvgDistance < vtx[h].Downstream_Vertex.AverageDistance))
                                            {
                                                Vertex w = vtx[h].Downstream_Vertex;
                                                //KRYSS: NEW - ONLY DESTROY VERTICES IF ONE TRACK IS LEFT
                                                if (w.Length <= 2)
                                                {
                                                    //END KRYSS
                                                    for (k = 0; k < w.Length; k++)
                                                        if (w[k].Upstream_Vertex == w) w[k].SetUpstreamVertex(null);
                                                        else if (w[k].Downstream_Vertex == w) w[k].SetDownstreamVertex(null);
                                                    //KRYSS: NEW - ONLY DESTROY VERTICES IF ONE TRACK IS LEFT
                                                }
                                                else
                                                {
                                                    w.RemoveTrack(tk);
                                                    try
                                                    {
                                                        w.ComputeVertexCoordinates();
                                                    }
                                                    catch (VertexFit.FitException)
                                                    {
                                                        for (k = 0; k < w.Length; k++)
                                                            if (w[k].Upstream_Vertex == w) w[k].SetUpstreamVertex(null);
                                                            else if (w[k].Downstream_Vertex == w) w[k].SetDownstreamVertex(null);
                                                    }
                                                }
                                                //END KRYSS
                                            }
                                            else if (vtx.Length <= 2)
                                            {
                                                group = new int[0];
                                                break;
                                            }
                                            else
                                            {
                                                vtx.RemoveTrack(tk);
                                                vf.RemoveTrackFit(new BaseTrackIndex(tk.Id));
                                                try
                                                {
                                                    if (vf.AvgDistance < 0 || vf.AvgDistance > C.GVtxRadius)
                                                    {
                                                        group = new int[0];
                                                        break;
                                                    }
                                                }
                                                catch (VertexFit.FitException)
                                                {
                                                    group = new int[0];
                                                    break;
                                                }
                                                int m;
                                                int[] newgroup = new int[group.Length - 1];
                                                for (m = 0; m < h; m++) newgroup[m] = group[m];
                                                for (m = h; m < group.Length - 1; m++) newgroup[m] = group[m + 1];
                                                h--;
                                                group = newgroup;
                                                continue;
                                            }
                                        }
                                    }
                                }

                                try
                                {
                                    SySal.BasicTypes.Vector vpos = new Vector();
                                    vpos.X = vf.X;
                                    vpos.Y = vf.Y;
                                    vpos.Z = vf.Z;
                                    if (GVtxKeep(GVf, fltpars, vpos, group, vol, true) == false || vf.AvgDistance < 0 || vf.AvgDistance > C.GVtxRadius)
                                    {
                                        group = new int[0];
                                        break;
                                    }
                                }
                                catch (VertexFit.FitException)
                                {
                                    group = new int[0];
                                    break;
                                }
                                if (h >= group.Length && group.Length > 0)
                                {
                                    for (h = 0; h < group.Length; h++)
                                    {
                                        Track tk = tr[group[h] / 2];
                                        if (group[h] % 2 == 0)
                                            tk.SetUpstreamVertex(vtx);
                                        else
                                            tk.SetDownstreamVertex(vtx);
                                    }
                                    if (C.VtxFitWeightEnable)
                                    {
                                        ((UtilityVertex)vtx).OptimizeVertex(C);
                                    }
                                }

#else
								int [] group = (int [])gl.ToArray(typeof(int));
								for (h = 0; h < tkgroups.Count && group.Length < ((int [])(tkgroups[h])).Length; h++);
								tkgroups.Insert(h, group);
#endif
                            }
                        }
                    }
                }
            }

#if false
			System.Collections.ArrayList vtxgroup = new System.Collections.ArrayList();			
			foreach (int [] group in tkgroups)
			{
				for (i = 0; i < group.Length; i++)
					if ((((group[i] % 2) == 0) ? tr[group[i] / 2].Upstream_Vertex : tr[group[i] / 2].Downstream_Vertex) != null) break;
				if (i == group.Length)
				{
					Vertex vtx = new Vertex(vtxgroup.Count);					
					for (i = 0; i < group.Length; i++)
					{
						Track tk = tr[group[i] /2];
						if (group[i] % 2 == 0)
						{
							tk.SetUpstreamVertex(vtx);
							vtx.AddTrack(tk, true);
						}
						else
						{
							tk.SetDownstreamVertex(vtx);
							vtx.AddTrack(tk, false);
						}
					}
					vtxgroup.Add(vtx);
				}
			}
#else
            System.Collections.ArrayList vtxgroup = new System.Collections.ArrayList();
            for (i = 0; i < vtxgroup.Count; i++)
                ((Vertex)vtxgroup[i]).ComputeVertexCoordinates();
            for (i = 0; i < m_Tracks.Length; i++)
            {
                if (m_Tracks[i].Downstream_Vertex != null)
                {
                    if (m_Tracks[i].Downstream_Vertex.Id == -1)
                    {
                        UtilityVertex.SetId(m_Tracks[i].Downstream_Vertex, vtxgroup.Count);
                        vtxgroup.Add(m_Tracks[i].Downstream_Vertex);
                    }
                }
                if (m_Tracks[i].Upstream_Vertex != null)
                {
                    if (m_Tracks[i].Upstream_Vertex.Id == -1)
                    {
                        UtilityVertex.SetId(m_Tracks[i].Upstream_Vertex, vtxgroup.Count);
                        vtxgroup.Add(m_Tracks[i].Upstream_Vertex);
                    }
                }
            }
#endif
            if (Progress != null) Progress(1.0);
            if (Report != null) Report(vtxgroup.Count + " vertices reconstructed.\r\n");
            return (Vertex[])vtxgroup.ToArray(typeof(Vertex));

        }

        #region GlobalVertex
        class Projection
        {
            public SySal.BasicTypes.Vector2 P;
            public int Id;
            public bool IsDownstream;
            public bool Sensitized;
        }

        struct TkProj
        {
            public int Id;
            public bool IsDownstream;

            public TkProj(int id, bool isdwn)
            {
                Id = id;
                IsDownstream = isdwn;
            }
        }

        class ProjMap
        {
            public System.Collections.ArrayList m_List;

            public SySal.BasicTypes.Rectangle m_Extents;

            public double TileSize;

            public int XTiles, YTiles;

            public class Iterator
            {
                public int Sx, Sy, S, MinSX, MaxSX, MinSY, MaxSY;
            }

            public Projection[,][] Tiles;

            public ProjMap()
            {
                m_List = new System.Collections.ArrayList();
                TileSize = 0.0;
            }

            public void Add(Projection p)
            {
                if (m_List.Count == 0)
                {
                    m_Extents.MinX = m_Extents.MaxX = p.P.X;
                    m_Extents.MinY = m_Extents.MaxY = p.P.Y;
                }
                else
                {
                    if (m_Extents.MinX > p.P.X) m_Extents.MinX = p.P.X;
                    else if (m_Extents.MaxX < p.P.X) m_Extents.MaxX = p.P.X;
                    if (m_Extents.MinY > p.P.Y) m_Extents.MinY = p.P.Y;
                    else if (m_Extents.MaxY < p.P.Y) m_Extents.MaxY = p.P.Y;
                }
                m_List.Add(p);
            }

            public Iterator Lock(Projection p)
            {
                Iterator i = new Iterator();
                i.MinSX = (int)Math.Round((p.P.X - m_Extents.MinX) / TileSize) - 1;
                i.MinSY = (int)Math.Round((p.P.Y - m_Extents.MinY) / TileSize) - 1;
                i.MaxSX = i.MinSX + 2;
                i.MaxSY = i.MinSY + 2;
                i.Sx = i.MinSX; i.Sy = i.MinSY;
                i.S = 0;
                return i;
            }

            public Projection Next(Iterator i)
            {
                while (i.Sy <= i.MaxSY)
                {
                    while (i.Sx <= i.MaxSX)
                    {
                        if (i.S < Tiles[i.Sy, i.Sx].Length) return Tiles[i.Sy, i.Sx][i.S++];
                        i.Sx++;
                        i.S = 0;
                    }
                    i.Sy++;
                    i.Sx = i.MinSX;
                }
                return null;
            }

            private const int MaxTiles = 10000;

            public void Freeze(double tilesize)
            {
                double area = (m_Extents.MaxX - m_Extents.MinX) * (m_Extents.MaxY - m_Extents.MinY);
                if (area == 0.0)
                {
                    XTiles = YTiles = 0;
                    TileSize = 1;
                }
                else TileSize = Math.Sqrt(area / m_List.Count);
                if (TileSize < tilesize) TileSize = tilesize;
                while (true)
                {
                    XTiles = (int)Math.Floor((m_Extents.MaxX - m_Extents.MinX) / TileSize + 1);
                    YTiles = (int)Math.Floor((m_Extents.MaxY - m_Extents.MinY) / TileSize + 1);
                    if (XTiles * YTiles > MaxTiles) TileSize *= 2.0;
                    else break;
                }
                m_Extents.MinX -= 2 * TileSize;
                m_Extents.MaxX += 2 * TileSize;
                m_Extents.MinY -= 2 * TileSize;
                m_Extents.MaxY += 2 * TileSize;
                XTiles += 4;
                YTiles += 4;
                int[,] counts = new int[YTiles, XTiles];
                Tiles = new Projection[YTiles, XTiles][];
                foreach (Projection p in m_List)
                    counts[(int)((p.P.Y - m_Extents.MinY) / TileSize), (int)((p.P.X - m_Extents.MinX) / TileSize)]++;
                int ix, iy;
                for (iy = 0; iy < YTiles; iy++)
                    for (ix = 0; ix < XTiles; ix++)
                        Tiles[iy, ix] = new Projection[counts[iy, ix]];
                foreach (Projection p in m_List)
                {
                    iy = (int)((p.P.Y - m_Extents.MinY) / TileSize);
                    ix = (int)((p.P.X - m_Extents.MinX) / TileSize);
                    Tiles[iy, ix][--counts[iy, ix]] = p;
                }
            }
        }
        #endregion

        public Vertex[] PairBasedVertexReconstruction(Track[] tr, IntersectionType[] ityp,
            Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i, j, k, n, m;
            ArrayList ar_tmpinters;
            TrackIntersection tmpinters;
            TrackIntersection[] Inters;

            ar_tmpinters = new ArrayList();
            n = tr.Length;
            k = 0;
            if (Report != null) Report("Computing Track Intersections... ");
            int nseg = C.MinVertexTracksSegments;
            double ctol = C.CrossTolerance;

            //From relative longitudinal coordinates to external longitudinal coordinates
            double cmaxz = C.MaximumZ + m_Layers[m_Layers.Length - 1].UpstreamZ;
            double cminz = C.MinimumZ + m_Layers[m_Layers.Length - 1].UpstreamZ;

            for (i = j = 0; i < n; i++)
                if (tr[i].Length >= nseg) j++;
            Track[] good = new Track[j];
            for (i = j = 0; i < n; i++)
            {
                tr[i].SetDownstreamVertex(null);
                tr[i].SetUpstreamVertex(null);
                if (tr[i].Length >= nseg)
                {
                    good[j++] = tr[i];
                }
            }
            n = j;

            //CellArray
            int TrackCount = m_Tracks.Length;
            int nl = m_Layers.Length;
            bool cusecell = C.UseCells;
            TrackPosition_CellArray CA = null;
            Track[] tmpt = null;
            if (cusecell)
            {
                CA = new TrackPosition_CellArray(good, m_Layers[0].DownstreamZ - m_Layers[1].DownstreamZ, cmaxz, cminz,
                    //				((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (m_Layers[0].DownstreamZ - m_Layers[nl-1].UpstreamZ) + C.LocalityCellSize,
                    //				((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (m_Layers[0].DownstreamZ - m_Layers[nl-1].UpstreamZ) + C.LocalityCellSize,
                    C.XCellSize, C.YCellSize, C.ZCellSize, C.Matrix);
            }
            //Fine CellArray
            int jstart;
            for (i = 0; i < n; i++)
            {
                if (ShouldStop != null) if (ShouldStop()) return null;
                //for (j = i + 1; j < n; j++)
                //2 righe in più
                if (cusecell)
                {
                    tmpt = CA.Lock(good[i]);
                    m = tmpt.Length;
                    jstart = 0;
                }
                else
                {
                    m = n;
                    jstart = i + 1;
                }
                for (j = jstart; j < m; j++)
                {
                    tmpinters = IntersectTracks(good[i], (cusecell ? tmpt[j] : good[j]), ctol, cmaxz, cminz, ityp);
                    if (tmpinters != null)
                    {
                        ar_tmpinters.Add(tmpinters);
                        k++;
                    };
                }
            }
            good = null;
            GC.Collect();

            if (Report != null) Report(k + " Intersections\r\n");
            if (Report != null) Report("Clusterizing Intersections... ");
            if (ShouldStop != null) if (ShouldStop()) return null;

            Inters = new TrackIntersection[k];
            Inters = (TrackIntersection[])ar_tmpinters.ToArray(typeof(TrackIntersection));
            Vertex[] tmpVertex = ClusterizeIntersections(Inters, C, ShouldStop, Progress, Report);

            int nt = tmpVertex.Length;
            ArrayList tmpvtxs = new ArrayList();
            for (i = j = 0; i < nt; i++)
                if (tmpVertex[i].Length > 0)
                {
                    tmpVertex[i].ComputeVertexCoordinates();
                    UtilityVertex.SetId(tmpVertex[i], j++);
                    tmpvtxs.Add(tmpVertex[i]);
                }
            return (Vertex[])tmpvtxs.ToArray(typeof(Vertex));


        }

        #endregion

        #region Alignment

        private const int MinAlignmentTracks = 10;

        public MappingResult LinkSegments(int OrdinalTopLayerNumber, int OrdinalBottomLayerNumber, Configuration C,
            dShouldStop ShouldStop, dProgress Progress, dReport Report, bool dontresetalignments, bool alignwithtracks)
        {

            int i, n;
            Layer BottomLayer = ((Layer)m_Layers[OrdinalBottomLayerNumber]);
            Layer TopLayer = ((Layer)m_Layers[OrdinalTopLayerNumber]);
            int BottomSegCount = m_Layers[OrdinalBottomLayerNumber].Length;
            int TopSegCount = m_Layers[OrdinalTopLayerNumber].Length;

            BottomLayer.ReloadFromDisk();
            TopLayer.ReloadFromDisk();

            SySal.Tracking.MIPEmulsionTrackInfo[] sBottom = new SySal.Tracking.MIPEmulsionTrackInfo[BottomSegCount];
            SySal.Tracking.MIPEmulsionTrackInfo[] sTop = new SySal.Tracking.MIPEmulsionTrackInfo[TopSegCount];
            bool[] bBottomIgnore = new bool[BottomSegCount];
            bool[] bTopIgnore = new bool[TopSegCount];
            for (i = 0; i < BottomSegCount; i++)
            {
                Segment s = (Segment)m_Layers[OrdinalBottomLayerNumber][i];
                sBottom[i] = s.Info;
                bBottomIgnore[i] = s.m_IgnoreInAlignment;
            }
            for (i = 0; i < TopSegCount; i++)
            {
                Segment s = (Segment)m_Layers[OrdinalTopLayerNumber][i];
                sTop[i] = s.Info;
                bTopIgnore[i] = s.m_IgnoreInAlignment;
            }

            double[] Align = new double[POS_ALIGN_DATA_LEN];

            if (BottomSegCount <= 0 || TopSegCount <= 0)
            {
                double[] tmpfails = new double[2] { 1.0, 1.0 };
                double[] tmpfail = new double[2];
                Align = new double[POS_ALIGN_DATA_LEN] { 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 };
                if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfails, tmpfail, Align, /*MappingResult.NullInput*/ MappingResult.OK);
                //return MappingResult.NullInput;
                return MappingResult.OK;
            };
            Segment[] LinkBottomSegArray;
            Segment[] LinkTopSegArray;

            QuickMapping.QuickMapper qMap = new QuickMapping.QuickMapper();
            QuickMapping.Configuration mConfig = new QuickMapping.Configuration();

            int count;
            double tmpX, tmpY;
            double MinZ, MaxZ;
            MinZ = sBottom[0].BottomZ;
            MaxZ = sBottom[0].TopZ;

            for (i = 1; i < BottomSegCount; i++)
            {
                if (MinZ > sBottom[i].BottomZ) MinZ = sBottom[i].BottomZ;
                if (MaxZ < sBottom[i].TopZ) MaxZ = sBottom[i].TopZ;
            };
            for (i = 0; i < TopSegCount; i++)
            {
                if (MinZ > sTop[i].BottomZ) MinZ = sTop[i].BottomZ;
                if (MaxZ < sTop[i].TopZ) MaxZ = sTop[i].TopZ;
            };

            double[] SAlignDSlope = new double[2];
            double[] SAlignDShrink = new double[2];
            double[] pTopLinkQuality, pBottomLinkQuality;

            int Iteration;
            double PosTol, SlopeTol;
            int MaxIters;
            double RefZ;

            PosTol = C.Initial_D_Pos;
            SlopeTol = C.Initial_D_Slope;
            MaxIters = C.MaxIters;


            Vector2 tmp = new Vector2();
            MappingParameters StartParams;
            if (Report != null) Report((dontresetalignments ? "Linking " : "Finding match between sheet ") + BottomLayer.SheetId + " and " + TopLayer.SheetId + "\r\n");

            SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] tPairs = null;

            int MapPatternChances;
            Vector2[] MapPattern;

            Vector2[] MapPattern1 = new Vector2[1];
            Vector2[] MapPattern4 = new Vector2[4];
            MapPattern4[0].X = -0.5f; MapPattern4[0].Y = -0.5f;
            MapPattern4[1].X = 0.5f; MapPattern4[1].Y = -0.5f;
            MapPattern4[2].X = -0.5f; MapPattern4[2].Y = 0.5f;
            MapPattern4[3].X = 0.5f; MapPattern4[3].Y = 0.5f;

            if (C.PrescanMode == PrescanModeValue.Translation || C.PrescanMode == PrescanModeValue.Rototranslation)
            {
                MapPatternChances = 1;
                MapPattern = MapPattern1;
            }
            else if (C.PrescanMode == PrescanModeValue.Affine)
            {
                MapPatternChances = 4;
                MapPattern = MapPattern4;
            }
            else
            {
                MapPatternChances = 0;
                MapPattern = MapPattern4;
            };

            Vector2[] TempMapPos = new Vector2[4];
            Vector2[] TempMapDelta = new Vector2[4];
            int[] TempMapCount = new int[4];
            Vector2[] MapPos = new Vector2[3];
            Vector2[] MapDelta = new Vector2[3];
            int trial, succ;

            RefZ = 0.5 * (BottomLayer.DownstreamZ + TopLayer.UpstreamZ);

            ArrayList arBottomProj = new ArrayList();
            ArrayList arTopProj = new ArrayList();
            SySal.Tracking.MIPEmulsionTrackInfo[] pBottomProj, pTopProj;
            int botcount, topcount;
            double MinZoneX, MinZoneY, MaxZoneX, MaxZoneY;
            double MinExtX, MinExtY, MaxExtX, MaxExtY;
            SySal.Tracking.MIPEmulsionTrackInfo tmp_mip_track;
            if (!dontresetalignments)
            {
                for (trial = succ = 0; (trial < MapPatternChances); trial++)
                {
                    TempMapPos[trial].X = MapPattern[trial].X * C.LeverArm;
                    TempMapPos[trial].Y = MapPattern[trial].Y * C.LeverArm;
                    MinZoneX = TempMapPos[trial].X + m_RefCenter.X - C.ZoneWidth * 0.5f;
                    MaxZoneX = TempMapPos[trial].X + m_RefCenter.X + C.ZoneWidth * 0.5f;
                    MinZoneY = TempMapPos[trial].Y + m_RefCenter.Y - C.ZoneWidth * 0.5f;
                    MaxZoneY = TempMapPos[trial].Y + m_RefCenter.Y + C.ZoneWidth * 0.5f;
                    MinExtX = TempMapPos[trial].X + m_RefCenter.X - C.Extents * 0.5f;
                    MaxExtX = TempMapPos[trial].X + m_RefCenter.X + C.Extents * 0.5f;
                    MinExtY = TempMapPos[trial].Y + m_RefCenter.Y - C.Extents * 0.5f;
                    MaxExtY = TempMapPos[trial].Y + m_RefCenter.Y + C.Extents * 0.5f;
                    for (i = botcount = 0; i < BottomSegCount; i++)
                    {
                        tmp_mip_track = new SySal.Tracking.MIPEmulsionTrackInfo();
                        Vector T;
                        T.X = sBottom[i].Intercept.X + (RefZ - sBottom[i].Intercept.Z) * sBottom[i].Slope.X;
                        T.Y = sBottom[i].Intercept.Y + (RefZ - sBottom[i].Intercept.Z) * sBottom[i].Slope.Y;
                        T.Z = sBottom[i].Intercept.Z + (RefZ - sBottom[i].Intercept.Z) * sBottom[i].Slope.Z;
                        if (!bBottomIgnore[i] && T.X >= MinZoneX && T.X <= MaxZoneX && T.Y >= MinZoneY && T.Y <= MaxZoneY)
                        {
                            tmp_mip_track.AreaSum = (uint)i;
                            tmp_mip_track.Intercept.X = T.X;
                            tmp_mip_track.Intercept.Y = T.Y;
                            tmp_mip_track.Intercept.Z = T.Z;
                            tmp_mip_track.Slope.X = sBottom[i].Slope.X;
                            tmp_mip_track.Slope.Y = sBottom[i].Slope.Y;
                            tmp_mip_track.Slope.Z = sBottom[i].Slope.Z;
                            tmp_mip_track.BottomZ = sBottom[i].BottomZ;
                            tmp_mip_track.TopZ = sBottom[i].TopZ;
                            arBottomProj.Add(tmp_mip_track);
                            botcount++;
                        };
                    };
                    this.AddInfo(-1, "BottomCount: " + BottomSegCount + " Used: " + botcount);
                    pBottomProj = (SySal.Tracking.MIPEmulsionTrackInfo[])(arBottomProj.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo)));
                    for (i = topcount = 0; i < TopSegCount; i++)
                    {
                        tmp_mip_track = new SySal.Tracking.MIPEmulsionTrackInfo();
                        Vector T;
                        T.X = sTop[i].Intercept.X + (RefZ - sTop[i].Intercept.Z) * sTop[i].Slope.X;
                        T.Y = sTop[i].Intercept.Y + (RefZ - sTop[i].Intercept.Z) * sTop[i].Slope.Y;
                        T.Z = sTop[i].Intercept.Z + (RefZ - sTop[i].Intercept.Z) * sTop[i].Slope.Z;
                        if (!bTopIgnore[i] && T.X >= MinExtX && T.X <= MaxExtX && T.Y >= MinExtY && T.Y <= MaxExtY)
                        {
                            tmp_mip_track.AreaSum = (uint)i;
                            tmp_mip_track.Intercept.X = T.X;
                            tmp_mip_track.Intercept.Y = T.Y;
                            tmp_mip_track.Intercept.Z = T.Z;
                            tmp_mip_track.Slope.X = sTop[i].Slope.X;
                            tmp_mip_track.Slope.Y = sTop[i].Slope.Y;
                            tmp_mip_track.Slope.Z = sTop[i].Slope.Z;
                            tmp_mip_track.BottomZ = sTop[i].BottomZ;
                            tmp_mip_track.TopZ = sTop[i].TopZ;
                            arTopProj.Add(tmp_mip_track);
                            topcount++;
                        };
                    };
                    this.AddInfo(-1, "TopCount: " + TopSegCount + " Used: " + topcount);
                    pTopProj = (SySal.Tracking.MIPEmulsionTrackInfo[])(arTopProj.ToArray(typeof(SySal.Tracking.MIPEmulsionTrackInfo)));
                    //MapTracks(&StartParams, botcount, pBottomProj, topcount, pTopProj, PosTol, SlopeTol, TSRD.PrescanExtents /*, TSRD.LowestPrescanPeak*/);
                    // ritorna delle trackpairs
                    //da qualche parte bisogna mettere i parametri di match
                    mConfig.PosTol = (C.PrescanMode == PrescanModeValue.Rototranslation) ? (PosTol + C.LeverArm) : PosTol;
                    mConfig.SlopeTol = SlopeTol;
                    mConfig.UseAbsoluteReference = true;
                    mConfig.FullStatistics = true;
                    tPairs = new SySal.Scanning.PostProcessing.PatternMatching.TrackPair[0];
                    qMap.Config = mConfig;
                    if (pBottomProj.Length > 0 && pTopProj.Length > 0) tPairs = qMap.Match(pBottomProj, pTopProj, 0f, C.MaximumShift.X, C.MaximumShift.Y);

                    /*
                     * Implementare
                     * 1)Calcolo degli Shifts
                     * 2)Creazione dei segmenti
                     */
                    n = tPairs.Length;
                    LinkBottomSegArray = new Segment[n];
                    LinkTopSegArray = new Segment[n];
                    double[] tempdx = new double[n];
                    double[] tempdy = new double[n];
                    //              System.IO.StreamWriter debugw = new System.IO.StreamWriter("w:\\temp\\aorec_" + OrdinalTopLayerNumber + ".txt");
                    //              debugw.WriteLine("PPX\tPPY\tPSX\tPSY\tDPX\tDPY");
                    for (i = 0; i < n; i++)
                    {
                        LinkBottomSegArray[i] = new Segment(pBottomProj[tPairs[i].First.Index], BottomLayer[(int)tPairs[i].First.Info.AreaSum].Index, BottomLayer, (int)tPairs[i].First.Info.AreaSum, null);
                        LinkTopSegArray[i] = new Segment(pTopProj[tPairs[i].Second.Index], TopLayer[(int)tPairs[i].Second.Info.AreaSum].Index, TopLayer, (int)tPairs[i].Second.Info.AreaSum, null);
                        /*
                                            LinkBottomSegArray[i] = new Segment(pBottomProj[tPairs[i].First.Index], BottomLayer[tPairs[i].First.Index].Index, BottomLayer, tPairs[i].First.Index, null);
                                            LinkTopSegArray[i] = new Segment(pTopProj[tPairs[i].Second.Index], TopLayer[tPairs[i].Second.Index].Index, TopLayer, tPairs[i].Second.Index, null);
                        */
                        LinkBottomSegArray[i].DownstreamLinked = LinkTopSegArray[i]; //tPairs[i].Second.Index;
                        LinkTopSegArray[i].UpstreamLinked = LinkBottomSegArray[i]; //tPairs[i].First.Index;
                        SySal.Tracking.MIPEmulsionTrackInfo fInfo = tPairs[i].First.Info;
                        SySal.Tracking.MIPEmulsionTrackInfo sInfo = tPairs[i].Second.Info;
                        tempdx[i] = fInfo.Intercept.X - sInfo.Intercept.X;//tPairs[i].First.Info.Intercept.X - tPairs[i].Second.Info.Intercept.X;
                        tempdy[i] = fInfo.Intercept.Y - sInfo.Intercept.Y;//tPairs[i].First.Info.Intercept.Y - tPairs[i].Second.Info.Intercept.Y;
                        //                  debugw.WriteLine(sInfo.Intercept.X.ToString("F1") + "\t" + sInfo.Intercept.Y.ToString("F1") + "\t" + sInfo.Slope.X.ToString("F1") + "\t" + sInfo.Slope.Y.ToString("F1") + "\t" + tempdx[i].ToString("F1") + "\t" + tempdy[i]);
                    };
                    //              debugw.Flush();
                    //              debugw.Close();

                    /*
                     * delete [] pBottomProj;
                     * delete [] pTopProj;
                     * free(StartParams.Pair);
                     */
                    Vector2 Shift = new Vector2();
                    Shift.X = Fitting.Average(tempdx);
                    Shift.Y = Fitting.Average(tempdy);
                    StartParams = new MappingParameters(n, tmp, tmp, tmp, Shift);
                    TempMapDelta[trial] = Shift;
                    TempMapCount[trial] = n;
                    if (StartParams.CoincN > 0 /* ??? */)
                    {
                        succ++;
                    };
                };

                if (C.PrescanMode != PrescanModeValue.None)
                    if (succ < ((C.PrescanMode == PrescanModeValue.Translation || C.PrescanMode == PrescanModeValue.Rototranslation) ? 1 : 3))
                    {
                        double[] tmpfail = new double[2];
                        Align = new double[POS_ALIGN_DATA_LEN];
                        if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.InsufficientPrescan);
                        return MappingResult.InsufficientPrescan;
                    };

                int index, compindex;
                for (index = compindex = 0; index < trial; index++)
                {
                    count = 0;
                    for (compindex = 0; compindex < trial; compindex++)
                        if (TempMapCount[compindex] > TempMapCount[index]) count++;
                    if (count < 3)
                    {
                        MapDelta[count] = TempMapDelta[index];
                        MapPos[count] = TempMapPos[index];
                    };
                };
            }

            if (dontresetalignments || C.PrescanMode == PrescanModeValue.None)
            {
                double[] tmpsh = new double[2] { 1, 1 };
                double[] tmpdslo = new double[2] { 0, 0 };
                Align = new double[Volume.POS_ALIGN_DATA_LEN] { 0, 0, 0, 0, 0, 0, 0 };
                if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpsh, tmpdslo, Align, MappingResult.OK);
            }
            else if (C.PrescanMode == PrescanModeValue.Translation)
            {
                Align[0] = 0.0;
                Align[1] = 0.0;
                Align[2] = 0.0;
                Align[3] = 0.0;
                Align[4] = MapDelta[0].X;
                Align[5] = MapDelta[0].Y;
            }
            else if (C.PrescanMode == PrescanModeValue.Rototranslation)
            {
                int j;
                n = tPairs.Length;
                double[] indep_v = new double[n];
                double[] dep_v = new double[n];
                double[] indep_mean;
                double[] dep_mean;
                double[,] dens;
                double[,] n_dens;
                double dummy = 0.0;
                System.Collections.ArrayList indepbins = new System.Collections.ArrayList();
                System.Collections.ArrayList depbins = new System.Collections.ArrayList();

                for (i = 0; i < n; i++)
                {
                    indep_v[i] = tPairs[i].Second.Info.Intercept.Y - m_RefCenter.Y;
                    dep_v[i] = tPairs[i].First.Info.Intercept.X - tPairs[i].Second.Info.Intercept.X;
                }
                Fitting.Prepare_2DCustom_Distribution(indep_v, dep_v, (3.0 * C.ZoneWidth / Math.Sqrt(n)), PosTol * 0.5, out indep_mean, out dep_mean, out dens, out n_dens);
                indepbins.Clear();
                depbins.Clear();

                for (i = 0; i < dens.GetLength(0); i++)
                {
                    double max = n / (indep_mean.Length * dep_mean.Length);
                    max = max + 3.0 * Math.Sqrt(max);
                    int k = -1;
                    for (j = 0; j < dens.GetLength(1); j++)
                        if (dens[i, j] > max)
                        {
                            max = dens[i, j];
                            k = j;
                        }
                    if (k >= 0)
                    {
                        indepbins.Add(indep_mean[i]);
                        depbins.Add(dep_mean[k]);
                    }
                }

                NumericalTools.Fitting.LinearFitSE((double[])indepbins.ToArray(typeof(double)), (double[])depbins.ToArray(typeof(double)), ref Align[1], ref Align[4], ref dummy, ref dummy, ref dummy, ref dummy, ref dummy);

                for (i = 0; i < n; i++)
                {
                    indep_v[i] = tPairs[i].Second.Info.Intercept.X - m_RefCenter.X;
                    dep_v[i] = tPairs[i].First.Info.Intercept.Y - tPairs[i].Second.Info.Intercept.Y;
                }
                Fitting.Prepare_2DCustom_Distribution(indep_v, dep_v, (3.0 * C.ZoneWidth / Math.Sqrt(n)), PosTol * 0.5, out indep_mean, out dep_mean, out dens, out n_dens);
                indepbins.Clear();
                depbins.Clear();

                for (i = 0; i < dens.GetLength(0); i++)
                {
                    double max = n / (indep_mean.Length * dep_mean.Length);
                    max = max + 3.0 * Math.Sqrt(max);
                    int k = -1;
                    for (j = 0; j < dens.GetLength(1); j++)
                        if (dens[i, j] > max)
                        {
                            max = dens[i, j];
                            k = j;
                        }
                    if (k >= 0)
                    {
                        indepbins.Add(indep_mean[i]);
                        depbins.Add(dep_mean[k]);
                    }
                }

                NumericalTools.Fitting.LinearFitSE((double[])indepbins.ToArray(typeof(double)), (double[])depbins.ToArray(typeof(double)), ref Align[2], ref Align[5], ref dummy, ref dummy, ref dummy, ref dummy, ref dummy);
                Report("Rototranslation matches: " + n);
                Report("Rototranslation params: " + Align[0].ToString("F5") + " " + Align[1].ToString("F5") + " " + Align[2].ToString("F5") + " " + Align[3].ToString("F5") + " " + Align[4].ToString("F5") + " " + Align[5].ToString("F5"));
                this.AddInfo(-1, "Rototranslation params: " + Align[0].ToString("F5") + " " + Align[1].ToString("F5") + " " + Align[2].ToString("F5") + " " + Align[3].ToString("F5") + " " + Align[4].ToString("F5") + " " + Align[5].ToString("F5"));
            }
            else if (C.PrescanMode == PrescanModeValue.Affine)
            {
                double Delta = (MapPos[1].X - MapPos[0].X) * (MapPos[2].Y - MapPos[0].Y) - (MapPos[1].Y - MapPos[0].Y) * (MapPos[2].X - MapPos[0].X);
                if (Delta == 0)
                {
                    double[] tmpfail = new double[2];
                    Align = new double[Volume.POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                    if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.SingularityInPrescan);
                    return MappingResult.SingularityInPrescan;
                };
                Delta = 1 / Delta;
                Align[0] = Delta * ((MapDelta[1].X - MapDelta[0].X) * (MapPos[2].Y - MapPos[0].Y) - (MapPos[1].Y - MapPos[0].Y) * (MapDelta[2].X - MapDelta[0].X));
                Align[1] = Delta * ((MapPos[1].X - MapPos[0].X) * (MapDelta[2].X - MapDelta[0].X) - (MapDelta[1].X - MapDelta[0].X) * (MapPos[2].X - MapPos[0].X));
                Align[2] = Delta * ((MapDelta[1].Y - MapDelta[0].Y) * (MapPos[2].Y - MapPos[0].Y) - (MapPos[1].Y - MapPos[0].Y) * (MapDelta[2].Y - MapDelta[0].Y));
                Align[3] = Delta * ((MapPos[1].X - MapPos[0].X) * (MapDelta[2].Y - MapDelta[0].Y) - (MapDelta[1].Y - MapDelta[0].Y) * (MapPos[2].X - MapPos[0].X));
                Align[4] = MapDelta[0].X - Align[0] * MapPos[0].X - Align[1] * MapPos[0].Y;
                Align[5] = MapDelta[0].Y - Align[2] * MapPos[0].X - Align[3] * MapPos[0].Y;
            }


            pBottomLinkQuality = new double[BottomSegCount];
            pTopLinkQuality = new double[TopSegCount];

            Segment[] CellBottomSeg = new Segment[BottomSegCount];
            Segment[] CellTopSeg = new Segment[TopSegCount];

            for (i = 0; i < BottomSegCount; i++) CellBottomSeg[i] = (Segment)BottomLayer[i];
            for (i = 0; i < TopSegCount; i++) CellTopSeg[i] = (Segment)TopLayer[i];// new Segment(sTop[i], TopLayer[i].Index, TopLayer, i, null);

            Position_CellArray CA =
                new Position_CellArray(CellBottomSeg, MinZ,
                ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ) + C.LocalityCellSize,
                ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ) + C.LocalityCellSize, 1.8, false, this);

            if (Progress != null) Progress((double)(OrdinalTopLayerNumber + 0.5) / (double)m_Layers.Length);
            if (Report != null)
                if (C.PrescanMode == PrescanModeValue.None)
                {
                    Report("Prescan not performed\r\n");
                }
                else
                {
                    Report("Prescan performed\r\n");
                }
            if (ShouldStop != null && ShouldStop()) return MappingResult.NotPerformedYet;

            for (Iteration = dontresetalignments ? (MaxIters - 1) : 0; Iteration < MaxIters; Iteration++)
            {

                //Zero-Slope Tolerances
                double PosInc = C.D_PosIncrement;
                double SloInc = C.D_SlopeIncrement;
                PosTol = ((Iteration + 1) * C.D_Pos + (MaxIters - Iteration - 1) * C.Initial_D_Pos) / MaxIters;
                SlopeTol = ((Iteration + 1) * C.D_Slope + (MaxIters - Iteration - 1) * C.Initial_D_Slope) / MaxIters;
                int i1, i2;
                // NEW QUALITY DEFINITION
                for (i1 = 0; i1 < BottomSegCount; i1++) pBottomLinkQuality[i1] = -1.0;
                for (i2 = 0; i2 < TopSegCount; i2++) pTopLinkQuality[i2] = -1.0;

                count = 0;
                Vector2 SlD = new Vector2();
                Vector2 SlSD = new Vector2();
                Vector2 SlS = new Vector2();
                Vector2 SlS2 = new Vector2();

                ArrayList tDx = new ArrayList();
                ArrayList tDy = new ArrayList();
                ArrayList tX = new ArrayList();
                ArrayList tY = new ArrayList();
                ArrayList tSx = new ArrayList();
                ArrayList tSy = new ArrayList();

                double[,] AlignMat = new double[POS_ALIGN_DATA_LEN, POS_ALIGN_DATA_LEN];
                double[] AlignVect = new double[POS_ALIGN_DATA_LEN];

                for (i2 = 0; i2 < TopSegCount; i2++)
                {
                    Segment S2 = CellTopSeg[i2];
                    if ((S2.m_IgnoreInAlignment && (alignwithtracks || Iteration < (MaxIters - 1)))/*KRYSS 26Jun2010*/ || (S2.UpstreamLinked != null)) continue;

                    Vector AP2 = new Vector();
                    Vector AS2 = new Vector();
                    SySal.Tracking.MIPEmulsionTrackInfo S2Info = S2.GetInfo();
                    AS2.X = S2Info.Slope.X;
                    AS2.Y = S2Info.Slope.Y;
                    AS2.Z = 1;

                    AP2.X = ((1 + Align[0]) * (S2Info.Intercept.X - m_RefCenter.X) + Align[1] * (S2Info.Intercept.Y - m_RefCenter.Y) + m_RefCenter.X + Align[4]);
                    AP2.Y = (Align[2] * (S2Info.Intercept.X - m_RefCenter.X) + (1 + Align[3]) * (S2Info.Intercept.Y - m_RefCenter.Y) + m_RefCenter.Y + Align[5]);
                    AP2.Z = (S2Info.Intercept.Z - Align[6]);
                    
                    SySal.Tracking.MIPEmulsionTrackInfo APtrack = new SySal.Tracking.MIPEmulsionTrackInfo();
                    APtrack.Intercept.X = AP2.X;
                    APtrack.Intercept.Y = AP2.Y;
                    APtrack.Intercept.Z = AP2.Z;
                    APtrack.Slope.X = AS2.X;
                    APtrack.Slope.Y = AS2.Y;
                    APtrack.Slope.Z = AS2.Z;
                    APtrack.TopZ = S2Info.TopZ;
                    APtrack.BottomZ = S2Info.BottomZ;
                    Segment AP = new Segment(APtrack, TopLayer[S2.PosInLayer].Index, TopLayer, S2.PosInLayer, null);

                    Segment[] intloopSeg = CA.Lock(APtrack);
                    if (intloopSeg == null) intloopSeg = new Segment[0];

                    int locki1;

                    for (locki1 = 0; locki1 < intloopSeg.Length; locki1++)
                    {
                        Segment S1 = intloopSeg[locki1];
                        if (S1.m_IgnoreInAlignment && (Iteration < (MaxIters - 1))) continue;
                        SySal.Tracking.MIPEmulsionTrackInfo S1Info = S1.GetInfo();

                        if (S1.DownstreamLinked != null) continue;

                        double s2xcom = S2Info.Slope.X;
                        double s2ycom = S2Info.Slope.Y;
                        double s2com = Math.Sqrt(s2xcom * s2xcom + s2ycom * s2ycom);
                        tmpX = (s2xcom - (S1Info.Slope.X + SAlignDSlope[0] + S1Info.Slope.X * SAlignDShrink[0]));
                        tmpY = (s2ycom - (S1Info.Slope.Y + SAlignDSlope[1] + S1Info.Slope.Y * SAlignDShrink[1]));
                        
                        if (Math.Sqrt(tmpX * tmpX + tmpY * tmpY) <= SlopeTol + SloInc * s2com)
                        {
                            Vector S;
                            Vector AS1 = new Vector();
                            AS1.X = ((1 + SAlignDShrink[0]) * S1Info.Slope.X + SAlignDSlope[0]);
                            AS1.Y = ((1 + SAlignDShrink[1]) * S1Info.Slope.Y + SAlignDSlope[1]);
                            AS1.Z = 1.0;
                            S = AS2;

                            Vector2 P1;

                            P1.X = S1Info.Intercept.X + (S1Info.TopZ - S1Info.Intercept.Z) * AS1.X - AP2.X - (S1Info.TopZ - /*S2.Intercept.Z*/AP2.Z) * AS2.X/*S2.Slope*/;
                            P1.Y = S1Info.Intercept.Y + (S1Info.TopZ - S1Info.Intercept.Z) * AS1.Y - AP2.Y - (S1Info.TopZ - /*S2.Intercept.Z*/AP2.Z) * AS2.Y/*S2.Slope*/;
                            Vector2 P2;
                            
                            P2.X = (S1Info.Intercept.X + (S2Info.BottomZ - Align[6] - S1Info.Intercept.Z) * AS1.X - AP2.X - (S2Info.BottomZ - Align[6] - /*S2.Intercept.Z*/AP2.Z) * AS2.X/*S2.Slope*/);
                            P2.Y = (S1Info.Intercept.Y + (S2Info.BottomZ - Align[6] - S1Info.Intercept.Z) * AS1.Y - AP2.Y - (S2Info.BottomZ - Align[6] - /*S2.Intercept.Z*/AP2.Z) * AS2.Y/*S2.Slope*/);

                            
                            double MinDist = Math.Sqrt(P1.X * P1.X + P1.Y * P1.Y);
                            double Dist = Math.Sqrt(P2.X * P2.X + P2.Y * P2.Y);
                            if (Dist < MinDist) MinDist = Dist;
                            double VX = P2.X - P1.X;
                            double VY = P2.Y - P1.Y;
                            double Norm = Math.Sqrt(VX * VX + VY * VY);
                            if (Norm > 0)
                            {
                                VX /= Norm;
                                VY /= Norm;
                                double Proj = P1.X * VX + P1.Y * VY;
                                if (Proj < 0 && Proj > -Norm)
                                {
                                    Dist = Math.Abs(P1.X * VY - P1.Y * VX);
                                    if (Dist < MinDist) MinDist = Dist;
                                };
                            };

                            double InterpSX = (S1Info.Intercept.X - AP2.X) / (S1Info.Intercept.Z - S2Info.Intercept.Z);
                            double InterpSY = (S1Info.Intercept.Y - AP2.Y) / (S1Info.Intercept.Z - S2Info.Intercept.Z);
                            double InterpS = Math.Sqrt(InterpSX * InterpSX + InterpSY * InterpSY);
                            double InterpNX = 1.0, InterpNY = 0.0;
                            if (InterpS > 0.0)
                            {
                                InterpNX = InterpSX / InterpS;
                                InterpNY = InterpSY / InterpS;
                            }
                            if (Math.Abs((AS1.X - InterpSX) * InterpNY - (AS1.Y - InterpSY) * InterpNX) > SlopeTol ||
                                Math.Abs((AS1.X - InterpSX) * InterpNX + (AS1.Y - InterpSY) * InterpNY) > (SlopeTol + SloInc * InterpS) ||
                                Math.Abs((AS2.X - InterpSX) * InterpNY - (AS2.Y - InterpSY) * InterpNX) > SlopeTol ||
                                Math.Abs((AS2.X - InterpSX) * InterpNX + (AS2.Y - InterpSY) * InterpNY) > (SlopeTol + SloInc * InterpS))
                                continue;

                            Dist = Math.Max((AS1.X - InterpSX) * (AS1.X - InterpSX) + (AS1.Y - InterpSY) * (AS1.Y - InterpSY),
                                    (AS2.X - InterpSX) * (AS2.X - InterpSX) + (AS2.Y - InterpSY) * (AS2.Y - InterpSY)) + 0.001;


                            {
                                if (Iteration == (MaxIters - 1))
                                {
                                    int indexTop = 0, indexBottom = 0;
                                    indexTop = i2;
                                    indexBottom = intloopSeg[locki1].PosInLayer;//.Flag;

                                    double IDist = 1.0 / Dist;

                                    if (pBottomLinkQuality[indexBottom] > IDist || pTopLinkQuality[indexTop] > IDist) continue;

                                    if (S1.DownstreamLinked != null)
                                    {
                                        CellTopSeg[S1.DownstreamLinked.PosInLayer].UpstreamLinked = null;
                                        pTopLinkQuality[S1.DownstreamLinked.PosInLayer] = -1.0;
                                        count--;
                                    };

                                    if (S2.UpstreamLinked != null)
                                    {
                                        CellBottomSeg[S2.UpstreamLinked.PosInLayer].DownstreamLinked = null;
                                        pBottomLinkQuality[S2.UpstreamLinked.PosInLayer] = -1.0;
                                        count--;
                                    };

                                    CellTopSeg[indexTop].UpstreamLinked = CellBottomSeg[indexBottom];
                                    CellBottomSeg[indexBottom].DownstreamLinked = CellTopSeg[indexTop];
                                    pTopLinkQuality[indexTop] = pBottomLinkQuality[indexBottom] = IDist;
                                };
                                count++;

                                if (Iteration < (MaxIters - 1))
                                {
                                    tmpX = S1Info.Slope.X - C.AlignBeamSlope.X;
                                    tmpY = S1Info.Slope.Y - C.AlignBeamSlope.Y;

                                    if (!S1.m_IgnoreInAlignment && !S2.m_IgnoreInAlignment && (Math.Sqrt(tmpX * tmpX + tmpY * tmpY) < C.AlignBeamWidth) &&
                                        (!C.AlignOnLinked ||
                                        ((BottomLayer.Id == m_Layers.Length - 1) || S1.UpstreamLinked == null)))
                                    {
                                        RefZ = 0.5f * (S1.Info.Intercept.Z + S2.Info.Intercept.Z);
                                        Vector2 P;
                                        P.X = S2Info.Intercept.X - m_RefCenter.X;
                                        P.Y = S2Info.Intercept.Y - m_RefCenter.Y;
                                        Vector2 D;
                                        D.X = S1Info.Intercept.X + (RefZ - S1Info.Intercept.Z) * AS1.X - S2Info.Intercept.X - (RefZ - S2Info.Intercept.Z) * AS2.X;
                                        D.Y = S1Info.Intercept.Y + (RefZ - S1Info.Intercept.Z) * AS1.Y - S2Info.Intercept.Y - (RefZ - S2Info.Intercept.Z) * AS2.Y;

                                        tX.Add(P.X);
                                        tY.Add(P.Y);
                                        tSx.Add(S.X);
                                        tSy.Add(S.Y);
                                        tDx.Add(D.X);
                                        tDy.Add(D.Y);

                                        Vector2 DS = new Vector2();
                                        DS.X = S2Info.Slope.X - S1Info.Slope.X;
                                        DS.Y = S2Info.Slope.Y - S1Info.Slope.Y;

                                        SlD.X += DS.X;
                                        SlD.Y += DS.Y;

                                        SlS.X += S2Info.Slope.X;
                                        SlS.Y += S2Info.Slope.Y;
                                        SlS2.X += S2Info.Slope.X * S2Info.Slope.X;
                                        SlS2.Y += S2Info.Slope.Y * S2Info.Slope.Y;
                                        SlSD.X += S2Info.Slope.X * DS.X;
                                        SlSD.Y += S2Info.Slope.Y * DS.Y;
                                    };
                                }
                            };
                        };
                    };
                };

                if (dontresetalignments == false && MaxIters > 1 && count < MinAlignmentTracks)
                {
                    double[] tmpfail = new double[2];
                    Align = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                    m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.BadAffineFocusing);
                    Console.WriteLine("Too few tracks (< " + MinAlignmentTracks + "), retrying");
                    return MappingResult.BadAffineFocusing;
                }
                if (count > 0 && Iteration < (MaxIters - 1))
                {
                    try
                    {

                        double[] tvX = (double[])(tX.ToArray(typeof(double)));
                        double[] tvY = (double[])(tY.ToArray(typeof(double)));
                        double[] tvDx = (double[])(tDx.ToArray(typeof(double)));
                        double[] tvDy = (double[])(tDy.ToArray(typeof(double)));
                        double[] tvSx = (double[])(tSx.ToArray(typeof(double)));
                        double[] tvSy = (double[])(tSy.ToArray(typeof(double)));

                        if ((!C.FreezeZ && Fitting.Affine_Focusing(tvDx, tvDy, tvX, tvY, tvSx, tvSy, ref Align) != NumericalTools.ComputationResult.OK) ||
                            (C.FreezeZ && Fitting.Affine(tvDx, tvDy, tvX, tvY, ref Align) != NumericalTools.ComputationResult.OK))
                        {
                            double[] tmpfail = new double[2];
                            Align = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                            if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.BadAffineFocusing);
                            return MappingResult.BadAffineFocusing;
                        }

                        double c = tvDx.Length;
                        double SlDenX, SlDenY;
                        SlDenX = (1 / (c * SlS2.X - SlS.X * SlS.X));
                        SlDenY = (1 / (c * SlS2.Y - SlS.Y * SlS.Y));

                        SAlignDShrink[0] = ((c * SlSD.X - SlS.X * SlD.X) * SlDenX);
                        SAlignDShrink[1] = ((c * SlSD.Y - SlS.Y * SlD.Y) * SlDenY);

                        SAlignDSlope[0] = (SlD.X * SlS2.X - SlS.X * SlSD.X) * SlDenX;
                        SAlignDSlope[1] = (SlD.Y * SlS2.Y - SlS.Y * SlSD.Y) * SlDenY;
                    }
                    catch
                    {
                        double[] tmpfail = new double[2];
                        Align = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                        if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.BadAffineFocusing);
                        return MappingResult.BadAffineFocusing;
                    };
                };

                //if (Progress != null) Progress((double)OrdinalTopLayerNumber + (double)(Iteration + 2.0) / (double)decimal_info_progress);
                if (Progress != null) Progress((double)(OrdinalTopLayerNumber + 1.0) / (double)m_Layers.Length);
                if (Report != null) Report("Iteration #" + (Iteration + 1.0) + ": " + count + " matching tracks\r\n");
                if (ShouldStop != null)
                {
                    Console.WriteLine(ShouldStop.ToString() + " " + ShouldStop());
                    if (ShouldStop()) return MappingResult.NotPerformedYet;
                }

            };

            //Storage dei dati

            //sistema l'allineamento
            //m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(SAlignDShrink, SAlignDSlope, Align);
            //Gli 1 sulla diagonale della matrice
            Align[0] += 1;
            Align[3] += 1;
            /*				SAlignDShrink.X =(1-SAlignDShrink.X);
                            SAlignDShrink.Y =(1-SAlignDShrink.Y);
                            SAlignDSlope.X =-SAlignDSlope.X;
                            SAlignDSlope.Y =-SAlignDSlope.Y;
            */
            if (C.CorrectSlopesAlign)
            {
                SAlignDShrink[0] = (1 + SAlignDShrink[0]);
                SAlignDShrink[1] = (1 + SAlignDShrink[1]);
                SAlignDSlope[0] = SAlignDSlope[0];
                SAlignDSlope[1] = SAlignDSlope[1];
            }
            else
            {
                SAlignDShrink[0] = SAlignDShrink[1] = 1.0;
                SAlignDSlope[0] = SAlignDSlope[1] = 0.0;
            }
            if (!dontresetalignments) m_AlignmentData[OrdinalTopLayerNumber] = new AlignmentData(SAlignDShrink, SAlignDSlope, Align, MappingResult.OK);

            if (Report != null)
                for (i = 0; i < Align.Length; i++)
                    Report("Align[" + i + "] =" + Align[i] + "\r\n");

            //queste due righe rendono ufficiali i linkaggi dei due layers fra loro
            ((Layer)m_Layers[TopLayer.Id]).UpstreamLinked = BottomLayer.Id; //.OrdinalID; 
            ((Layer)m_Layers[BottomLayer.Id]).DownstreamLinked = TopLayer.Id; //.OrdinalID; 

            for (i = 0; i < TopSegCount; i++) ((Segment)m_Layers[TopLayer.Id][i]).UpstreamLinked = CellTopSeg[i].UpstreamLinked;
            for (i = 0; i < BottomSegCount; i++) ((Segment)m_Layers[BottomLayer.Id][i]).DownstreamLinked = CellBottomSeg[i].DownstreamLinked;

            //se un layer non è linkato sopra e sotto allora non si procede allo store
            //se si tratta del primo (ultimo) layer allora basta che sia linkato con il secondo (penultimo) layer

            return MappingResult.OK;
        }

        private void CTransformation(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i, k;
            m_AlignmentData[m_Layers.Length - 1].Result = MappingResult.OK;
            m_AlignmentData[m_Layers.Length - 1].AffineMatrixXX = m_AlignmentData[m_Layers.Length - 1].AffineMatrixYY = 1.0;
            m_AlignmentData[m_Layers.Length - 1].AffineMatrixXY = m_AlignmentData[m_Layers.Length - 1].AffineMatrixYX = 0.0;
            m_AlignmentData[m_Layers.Length - 1].TranslationX = m_AlignmentData[m_Layers.Length - 1].TranslationY = m_AlignmentData[m_Layers.Length - 1].TranslationZ = 0.0;
            m_AlignmentData[m_Layers.Length - 1].SAlignDSlopeX = m_AlignmentData[m_Layers.Length - 1].SAlignDSlopeY = 0.0;
            m_AlignmentData[m_Layers.Length - 1].DShrinkX = m_AlignmentData[m_Layers.Length - 1].DShrinkY = 1.0;
            ((Layer)m_Layers[m_Layers.Length - 1]).iAlignmentData = m_AlignmentData[m_Layers.Length - 1];
            for (i = m_Layers.Length - 2; i > -1; i--)
            {
                AlignmentData oca = ((Layer)m_Layers[i + 1]).iAlignmentData;
                AlignmentData a = m_AlignmentData[i];
                AlignmentData ca = new AlignmentData();
                ca.AffineMatrixXX = a.AffineMatrixXX * oca.AffineMatrixXX + a.AffineMatrixXY * oca.AffineMatrixYX;
                ca.AffineMatrixXY = a.AffineMatrixXX * oca.AffineMatrixXY + a.AffineMatrixXY * oca.AffineMatrixYY;
                ca.AffineMatrixYX = a.AffineMatrixYX * oca.AffineMatrixXX + a.AffineMatrixYY * oca.AffineMatrixYX;
                ca.AffineMatrixYY = a.AffineMatrixYX * oca.AffineMatrixXY + a.AffineMatrixYY * oca.AffineMatrixYY;
                ca.TranslationX = a.AffineMatrixXX * oca.TranslationX + a.AffineMatrixXY * oca.TranslationY + a.TranslationX;
                ca.TranslationY = a.AffineMatrixYX * oca.TranslationX + a.AffineMatrixYY * oca.TranslationY + a.TranslationY;
                ca.TranslationZ = a.TranslationZ + oca.TranslationZ;

                ca.DShrinkX = m_AlignmentData[i].DShrinkX;//new Vector2();
                ca.DShrinkY = m_AlignmentData[i].DShrinkY;//new Vector2();
                ca.SAlignDSlopeX = m_AlignmentData[i].SAlignDSlopeX;//new Vector2();
                ca.SAlignDSlopeY = m_AlignmentData[i].SAlignDSlopeY;//new Vector2();

                if (oca.Result != MappingResult.OK) ca.Result = oca.Result;
                else ca.Result = a.Result;
                ((Layer)m_Layers[i]).iAlignmentData = ca;

                if (ca.Result == MappingResult.OK)
                {
                    ((Layer)m_Layers[i]).ReloadFromDisk();
                    for (k = 0; k < m_Layers[i].Length; k++)
                    {
                        double x, y;
                        SySal.Tracking.MIPEmulsionTrackInfo ttk = ((Segment)m_Layers[i][k]).GetInfo();
                        x = ttk.Intercept.X - m_RefCenter.X;
                        y = ttk.Intercept.Y - m_RefCenter.Y;
                        ttk.Intercept.X = ca.AffineMatrixXX * x + ca.AffineMatrixXY * y + ca.TranslationX + m_RefCenter.X;
                        ttk.Intercept.Y = ca.AffineMatrixYX * x + ca.AffineMatrixYY * y + ca.TranslationY + m_RefCenter.Y;
                        ttk.Intercept.Z -= ca.TranslationZ;
                        ttk.TopZ -= ca.TranslationZ;
                        ttk.BottomZ -= ca.TranslationZ;
                        x = ttk.Slope.X;
                        y = ttk.Slope.Y;
                        ttk.Slope.X = ca.AffineMatrixXX * x + ca.AffineMatrixXY * y;
                        ttk.Slope.Y = ca.AffineMatrixYX * x + ca.AffineMatrixYY * y;
                    }
                    ((Layer)m_Layers[i]).UpdateZ();
                    ((Layer)m_Layers[i]).MoveToDisk();

                }
            };

            for (i = 0; i < m_Layers.Length; i++)
                ((Layer)m_Layers[i]).SetRefCenter(m_RefCenter.X, m_RefCenter.Y, m_Layers[i].RefCenter.Z);
        }

        public void AlignAndLink(Configuration C, bool alignwithtracks, bool dontresetalignments, bool [] layerstoignore, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            float TotalCycles;
            int i, j, k, t;

            Vector RefCenter = new Vector();
            for (i = t = 0; i < m_Layers.Length; i++)
            {
                t += m_Layers[i].Length;
                ((Layer)m_Layers[i]).ReloadFromDisk();
                for (j = 0; j < m_Layers[i].Length; j++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo info = ((Segment)m_Layers[i][j]).GetInfo();
                    RefCenter.X = RefCenter.X + info.Intercept.X +
                        (0.5f * (info.TopZ + info.BottomZ) - info.Intercept.Z) * info.Slope.X;
                    RefCenter.Y = RefCenter.Y + info.Intercept.Y +
                        (0.5f * (info.TopZ + info.BottomZ) - info.Intercept.Z) * info.Slope.Y;
                    RefCenter.Z = RefCenter.Z + info.Intercept.Z +
                        (0.5f * (info.TopZ + info.BottomZ) - info.Intercept.Z) * info.Slope.Z;
                };
                ((Layer)m_Layers[i]).Flush();
            };
            if (t > 0)
            {
                RefCenter.X /= t;
                RefCenter.Y /= t;
                RefCenter.Z /= t;
            };
            m_RefCenter.X = RefCenter.X;
            m_RefCenter.Y = RefCenter.Y;
            m_RefCenter.Z = RefCenter.Z;

            TotalCycles = 0.5f * ((m_Layers.Length - 1) * (m_Layers.Length) - (m_Layers.Length - C.MaxMissingSegments) * (m_Layers.Length - C.MaxMissingSegments - 1));

            int hop;
            int tkstep;
            if (dontresetalignments == false)
                for (i = 0; i < m_Layers.Length; i++)
                    layerstoignore[i] = false;
            for (tkstep = 0; tkstep <= C.ExtraTrackingPasses; tkstep++)
            {
                for (hop = 1; hop <= C.MaxMissingSegments + 1; hop++)
                {
                    for (i = m_Layers.Length - 1 - hop; i >= 0; i--)
                    {
                        if (layerstoignore[i] || layerstoignore[i + hop]) continue;                        
                        //if (i + hop < m_Layers.Length - 1) ((Layer)m_Layers[i - 1]).MoveToDisk();
                        //LinkSegments(i, i + hop, C, ShouldStop, Progress, Report, tkstep > 0 || hop > 1);
                        if (dontresetalignments || tkstep > 0 || hop > 1)
                            LinkSegments(i, i + hop, C, ShouldStop, Progress, Report, true, alignwithtracks);
                        else
                        {
                            Console.WriteLine("Align " + i);
                            int alhop;
                            for (alhop = 1; alhop <= C.MaxMissingSegments + 1 && (i + alhop < m_Layers.Length); alhop++)
                            {
                                MappingResult mapr = LinkSegments(i, i + alhop, C, ShouldStop, Progress, Report, false, alignwithtracks);
                                if (alhop > 1) ((Layer)m_Layers[i + alhop]).MoveToDisk();
                                if (mapr == MappingResult.OK)
                                {
                                    Console.WriteLine("Alignment found");
                                    break;
                                }
                            }
                            if (alhop > C.MaxMissingSegments + 1)
                            {
                                if (C.IgnoreBadLayers)
                                {
                                    layerstoignore[i] = true;
                                    if (Report != null)
                                        Report("Cannot align layer " + i + " Brick " + m_Layers[i].BrickId + " Sheet " + m_Layers[i].SheetId + " Side " + m_Layers[i].Side);
                                }
                                else throw new Exception("Cannot align layer " + i + " Brick " + m_Layers[i].BrickId + " Sheet " + m_Layers[i].SheetId + " Side " + m_Layers[i].Side);
                            }
                            LinkSegments(i, i + 1, C, ShouldStop, Progress, Report, true, alignwithtracks);                            
                        }

                        if (m_Layers[i].UpstreamZ <= m_Layers[i + 1].UpstreamZ)
                            throw new Exception("Layer order is inconsistent with Z coordinate:\r\nLayer " + i + " Brick/Plate/Side " + m_Layers[i].BrickId + "/" + m_Layers[i].SheetId + "/" + m_Layers[i].Side + " Z = " + m_Layers[i].UpstreamZ +
                                "\r\nLayer " + (i + 1) + " Brick/Plate/Side " + m_Layers[i + 1].BrickId + "/" + m_Layers[i + 1].SheetId + "/" + m_Layers[i + 1].Side + " Z = " + m_Layers[i + 1].UpstreamZ);

                        if (Report != null && m_AlignmentData[i].Result != MappingResult.OK) Report("Alignment not successful:\r\n" +
                                                                                                 m_AlignmentData[i].Result.ToString() + "\r\n");
                        if (C.CorrectSlopesAlign)
                        {
                            int SegsCount = m_Layers[i].Length;
                            for (k = 0; k < SegsCount; k++)
                            {
                                SySal.Tracking.MIPEmulsionTrackInfo ttk = ((Segment)m_Layers[i][k]).GetInfo();
                                double sx = ttk.Slope.X;
                                double sy = ttk.Slope.Y;

                                ttk.Slope.X = (float)(m_AlignmentData[i].DShrinkX * sx + m_AlignmentData[i].SAlignDSlopeX);
                                ttk.Slope.Y = (float)(m_AlignmentData[i].DShrinkY * sy + m_AlignmentData[i].SAlignDSlopeY);
                            }
                        }
                    };
                    if ((i + 1) >= 0 && (i + 1) < m_Layers.Length) ((Layer)m_Layers[i + 1]).MoveToDisk();
                    if (dontresetalignments == false && tkstep == 0 && hop == 1) CTransformation(C, ShouldStop, Progress, Report);
                }
            }

            for (i = 0; i < m_Layers.Length - 1; i++)
                ((Layer)m_Layers[i]).MoveToDisk();

            //if (dontresetalignments == false) CTransformation(C, ShouldStop, Progress, Report);
        }

        int TkfltN;
        int TkfltM;
        int TkfltG;
        int TkfltA;
        int TkfltL;

        void ResetTrackFilter()
        {
            TkfltN = TkfltM = TkfltG = TkfltA = TkfltL = 0;
        }

        void UpdateTrackFilter(SySal.TotalScan.Segment myx)
        {
            TkfltN++;
            SySal.Tracking.MIPEmulsionTrackInfo coqui = myx.Info;
            if (coqui.Sigma < 0) TkfltM++;
            TkfltG += coqui.Count;
            TkfltA += (int)coqui.AreaSum;
            Segment mumisy = ((Segment)myx);
            if (mumisy.DownstreamLinked != null) TkfltL += (mumisy.LayerOwner.Id - mumisy.DownstreamLinked.LayerOwner.Id);
            else TkfltL = 1;
        }

        protected void CleanTrack(Segment s, double poserr, double chi2limit)
        {
            System.Collections.ArrayList posl = new ArrayList();
            System.Collections.ArrayList segl = new ArrayList();
            Segment ls = s;
            SySal.BasicTypes.Vector v = new Vector();
            while (ls != null)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = ls.Info;
                SySal.TotalScan.Layer ly = ls.LayerOwner;
                if (ly.Side == 0)
                {
                    if (info.Sigma >= 0.0)
                    {
                        v.Z = ly.DownstreamZ;
                        v.X = info.Intercept.X + (v.Z - info.Intercept.Z) * info.Slope.X;
                        v.Y = info.Intercept.Y + (v.Z - info.Intercept.Z) * info.Slope.Y;
                        posl.Add(v);
                        segl.Add(ls);
                        v.Z = ly.UpstreamZ;
                        v.X = info.Intercept.X + (v.Z - info.Intercept.Z) * info.Slope.X;
                        v.Y = info.Intercept.Y + (v.Z - info.Intercept.Z) * info.Slope.Y;
                        posl.Add(v);
                        segl.Add(ls);
                    }
                    else
                    {
                        v.Z = (Math.Abs(ly.DownstreamZ - info.TopZ) > Math.Abs(ly.UpstreamZ - info.BottomZ)) ? ly.UpstreamZ : ly.DownstreamZ;
                        v.X = info.Intercept.X + (v.Z - info.Intercept.Z) * info.Slope.X;
                        v.Y = info.Intercept.Y + (v.Z - info.Intercept.Z) * info.Slope.Y;
                        posl.Add(v);
                        segl.Add(ls);
                    }
                }
                else
                {
                    posl.Add(info.Intercept);
                    segl.Add(ls);
                }
                ls = ls.UpstreamLinked;
            }
            if (posl.Count <= 3) return;
            SySal.BasicTypes.Vector[] diffs = new Vector[posl.Count - 2];
            int i;
            double dst = 0.0, dsl = 0.0;
            for (i = 0; i < diffs.Length; i++)
            {
                SySal.BasicTypes.Vector vd = (SySal.BasicTypes.Vector)posl[i];
                v = (SySal.BasicTypes.Vector)posl[i + 1];
                SySal.BasicTypes.Vector vu = (SySal.BasicTypes.Vector)posl[i + 2];
                double dz = vd.Z - vu.Z;
                double idz = 1.0 / dz;
                double sx = (vd.X - vu.X) * idz;
                double sy = (vd.Y - vu.Y) * idz;
                double n = sx * sx + sy * sy;
                double nx = 1.0;
                double ny = 0.0;
                if (n > 0.0)
                {
                    n = 1.0 / Math.Sqrt(n);
                    nx = sx * n;
                    ny = sy * n;
                }
                double dx = v.X - vd.X - sx * (v.Z - vd.Z);
                double dy = v.Y - vd.Y - sy * (v.Z - vd.Z);
                double dt = dx * ny - dy * nx;
                double dl = dx * nx + dy * ny;
                diffs[i].X = dl;
                diffs[i].Y = dt;
                diffs[i].Z = dz;
                dst += dt * dt * dz;
                dsl += dl * dl * dz;
            }
            dst /= diffs.Length;
            dsl /= diffs.Length;
            for (i = 0; i < diffs.Length; i++)
                if (Math.Abs(diffs[i].X) > poserr && Math.Abs(diffs[i].Y) > poserr)
                {
                    Segment ns = (Segment)segl[i + 1];
                    if ((diffs[i].X * diffs[i].X / dsl + diffs[i].Y * diffs[i].Y / dst) * diffs[i].Z * 0.5 > chi2limit)
                    {
                        ls = (Segment)segl[i + 1];
                        if (ls.UpstreamLinked != null)
                        {
                            ls.UpstreamLinked.DownstreamLinked = null;
                            ls.UpstreamLinked = null;
                        }
                        else
                        {
                            ls.DownstreamLinked.UpstreamLinked = null;
                            ls.DownstreamLinked = null;
                        }
                        CleanTrack(s, poserr, chi2limit);
                        return;
                    }
                }
        }

        private delegate double dTrackFilter();

        double tffltN() { return TkfltN; }
        double tffltM() { return TkfltM; }
        double tffltG() { return TkfltG; }
        double tffltA() { return TkfltA; }
        double tffltL() { return TkfltL; }

        protected Track[] AttachSegmentsAccordingToLinkingIndex(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            //Numero di segmenti da usare per il fit
            //			System.IO.StreamWriter w = new System.IO.StreamWriter("mem.txt");

            int NFitSeg = 1;

            //Conteggio delle tracce per dimensionamento una-tantum
            int nt = m_Layers[0].Length, SegsCount;
            int i, j, k, h, l;
            int laycount = m_Layers.Length;
            int trksegcount = 0;
            //dimensionamento
            ArrayList tmptracks = new ArrayList();
            Track tmp_Tracks;
            
            nt = 0;
            NumericalTools.CStyleParsedFunction Tf = null;
            dTrackFilter[] fltpars = null;
            if (C.TrackFilter != null && C.TrackFilter.Trim().Length > 0)
            {
                Tf = new CStyleParsedFunction(C.TrackFilter);
                fltpars = new dTrackFilter[Tf.ParameterList.Length];
                for (i = 0; i < fltpars.Length; i++)
                {
                    dTrackFilter flt = null;
                    switch (Tf.ParameterList[i].ToUpper())
                    {
                        case "N": flt = tffltN; break;
                        case "M": flt = tffltM; break;
                        case "G": flt = tffltG; break;
                        case "A": flt = tffltA; break;
                        case "L": flt = tffltL; break;
                        default: throw new Exception("Parameter " + Tf.ParameterList[i] + " supplied to tracking filter is unknown.");
                    }
                    fltpars[i] = flt;
                }
            }

            int critic = C.MinimumCritical - 1;
            if (C.MinimumCritical > 1 && critic <= 1) critic = C.MinimumCritical;
            for (j = 0; j < laycount; j++)
            {                
                SegsCount = m_Layers[j].Length;                
                for (k = 0; k < SegsCount; k++)
                    if (((Segment)m_Layers[j][k]).DownstreamLinked == null)
                    {
                        if (C.TrackCleanChi2Limit > 0.0) CleanTrack((Segment)m_Layers[j][k], C.TrackCleanError, C.TrackCleanChi2Limit);
                        trksegcount = 1;
                        ResetTrackFilter();
                        Segment s = ((Segment)m_Layers[j][k]);
                        UpdateTrackFilter(s);
                        while (s.UpstreamLinked != null)
                        {
                            trksegcount++;
                            s = s.UpstreamLinked;
                            UpdateTrackFilter(s);
                        }

                        if (trksegcount >= critic)
                        {
                            if (Tf != null)
                            {
                                for (i = 0; i < fltpars.Length; i++)
                                    Tf[i] = fltpars[i]();
                                if (Tf.Evaluate() == 0.0) continue;
                            }
                            tmp_Tracks = new Track(nt);
                            s = ((Segment)m_Layers[j][k]);
                            do
                            {
                                tmp_Tracks.AddSegment(s);
                                s = s.UpstreamLinked;
                            }
                            while (s != null);
                            tmp_Tracks.FittingSegments = NFitSeg;
                            tmptracks.Add(tmp_Tracks);
                            nt++;
                        }

                    };
                if (Progress != null) Progress((double)(j + 1) / (double)laycount);
            };

            GC.Collect();
            return (Track[])tmptracks.ToArray(typeof(Track));

        }

        internal class TrackComparer : System.Collections.IComparer
        {

            #region IComparer Members

            public int Compare(object x, object y)
            {
                return (int)Math.Sign(((SySal.TotalScan.Track)x).Upstream_Z - ((SySal.TotalScan.Track)y).Upstream_Z);
            }

            #endregion
        }

        internal class TrackZFinder : System.Collections.IComparer
        {

            #region IComparer Members

            public int Compare(object x, object y)
            {
                return (int)Math.Sign(((SySal.TotalScan.Track)x).Upstream_Z - (double)y);
            }

            #endregion
        }

        internal static void ToUpstream(SySal.TotalScan.Track t, ref double x, ref double y)
        {
            x = t.Upstream_PosX + t.Upstream_SlopeX + (t.Upstream_Z - t.Upstream_PosZ);
            y = t.Upstream_PosY + t.Upstream_SlopeY + (t.Upstream_Z - t.Upstream_PosZ);
        }

        internal static void ToDownstream(SySal.TotalScan.Track t, ref double x, ref double y)
        {
            x = t.Downstream_PosX + t.Downstream_SlopeX + (t.Downstream_Z - t.Downstream_PosZ);
            y = t.Downstream_PosY + t.Downstream_SlopeY + (t.Downstream_Z - t.Downstream_PosZ);
        }

        protected void SortTracksZ(Track[] input, out System.Collections.ArrayList[,] tlarr, out double minx, out double miny, out double stepx, out double stepy)
        {
            int i;
            SySal.BasicTypes.Rectangle rect = new SySal.BasicTypes.Rectangle();
            ToUpstream(input[0], ref rect.MinX, ref rect.MinY);
            rect.MaxX = rect.MinX;
            rect.MaxY = rect.MinY;
            double x = 0.0, y = 0.0;
            for (i = 0; i < input.Length; i++)
            {
                ToUpstream(input[i], ref x, ref y);
                if (rect.MinX > x) rect.MinX = x;
                else if (rect.MaxX < x) rect.MaxX = x;
                if (rect.MinY > y) rect.MinY = y;
                else if (rect.MaxY < y) rect.MaxY = y;
            }
            minx = rect.MinX;
            miny = rect.MinY;
            int sx = (int)Math.Min((rect.MaxX - rect.MinX) / 10000, 9) + 1;
            int sy = (int)Math.Min((rect.MaxY - rect.MinY) / 10000, 9) + 1;
            stepx = (rect.MaxX - rect.MinX) / sx;
            stepy = (rect.MaxY - rect.MinY) / sy;
            tlarr = new System.Collections.ArrayList[sx, sy];
            int ix, iy;
            for (ix = 0; ix < sx; ix++)
                for (iy = 0; iy < sy; iy++)
                    tlarr[ix, iy] = new System.Collections.ArrayList();
            for (i = 0; i < input.Length; i++)
            {
                ToUpstream(input[i], ref x, ref y);
                ix = (int)((x - rect.MinX) / stepx); if (ix == sx) ix--;
                iy = (int)((y - rect.MinY) / stepy); if (iy == sy) iy--;
                tlarr[ix, iy].Add(input[i]);
            }
            for (ix = 0; ix < sx; ix++)
                for (iy = 0; iy < sy; iy++)
                    tlarr[ix, iy].Sort(new TrackComparer());
        }

        public Track[] RelinkTracks(Track[] input, Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            if (C.RelinkEnable == false) return input;
            System.Collections.ArrayList[,] tlarr;
            SySal.BasicTypes.Rectangle rect;
            double minx, miny, stepx, stepy;
            if (Report != null) Report("Sorting... ");
            SortTracksZ(input, out tlarr, out minx, out miny, out stepx, out stepy);
            if (Report != null) Report("Sorting done.\r\n");
            if (Report != null) Report("Merging... ");
            int i, j;
            double aperture2 = C.RelinkAperture * C.RelinkAperture;
            int ix, iy, jx, jy, sx, sy, iix, iiy;
            sx = tlarr.GetLength(0);
            sy = tlarr.GetLength(1);
            double x = 0.0, y = 0.0;
            for (ix = 0; ix < sx; ix++)
                for (iy = 0; iy < sy; iy++)
                    for (i = 0; i < tlarr[ix, iy].Count; i++)
                    {
                        SySal.TotalScan.Track tk = (SySal.TotalScan.Track)tlarr[ix, iy][i];
                        ToDownstream(tk, ref x, ref y);
                        iix = (int)((x - minx) / stepx);
                        iiy = (int)((y - miny) / stepy);
                        for (jx = Math.Max(0, iix - 1); jx <= Math.Min(sx - 1, iix + 1); jx++)
                            for (jy = Math.Max(0, iiy - 1); jy <= Math.Min(sy - 1, iiy + 1); jy++)
                            {
                                j = tlarr[jx, jy].BinarySearch(tk.Downstream_Z, new TrackZFinder());
                                if (j < 0) j = ~j;
                                for (; j < tlarr[jx, jy].Count; j++)
                                {
                                    SySal.TotalScan.Track sk = (SySal.TotalScan.Track)tlarr[jx, jy][j];
                                    if (sk.Upstream_Z < tk.Downstream_Z) continue;
                                    if (sk.Upstream_Z - tk.Downstream_Z > C.RelinkDeltaZ) break;
                                    double dsx = sk.Upstream_SlopeX - tk.Downstream_SlopeX;
                                    double dsy = sk.Upstream_SlopeY - tk.Downstream_SlopeY;
                                    if (dsx * dsx + dsy * dsy > aperture2) continue;
                                    double slx = (sk.Upstream_PosX + (sk.Upstream_Z - sk.Upstream_PosZ) * sk.Upstream_SlopeX - tk.Downstream_PosX - tk.Downstream_SlopeX * (tk.Downstream_Z - tk.Downstream_PosZ)) / (sk.Upstream_Z - tk.Downstream_Z);
                                    double sly = (sk.Upstream_PosY + (sk.Upstream_Z - sk.Upstream_PosZ) * sk.Upstream_SlopeY - tk.Downstream_PosY - tk.Downstream_SlopeY * (tk.Downstream_Z - tk.Downstream_PosZ)) / (sk.Upstream_Z - tk.Downstream_Z);
                                    dsx = slx - sk.Upstream_SlopeX;
                                    dsy = sly - sk.Upstream_SlopeY;
                                    if (dsx * dsx + dsy * dsy > aperture2) continue;
                                    dsx = slx - tk.Downstream_SlopeX;
                                    dsy = sly - tk.Downstream_SlopeY;
                                    if (dsx * dsx + dsy * dsy > aperture2) continue;
                                    {
                                        while (sk.Length > 0)
                                            tk.AddSegment(sk[sk.Length - 1]);
                                        tlarr[jx, jy].RemoveAt(j);
                                        jx = sx; jy = sy;
                                        i--;
                                        break;
                                    }
                                }
                            }
                        double percent;
                        if (ShouldStop != null && ShouldStop()) return new Track[0];
                        if (Progress != null)
                            if ((percent = 10000 * (((double)i / tlarr[ix, iy].Count + ix * sy + iy) / sx / sy)) % 5 == 0)
                                Progress(percent * 0.0001);
                    }
            for (ix = 0; ix < sx; ix++)
                for (iy = 0; iy < sy; iy++)
                    if (ix > 0 || iy > 0)
                    {
                        tlarr[0, 0].AddRange(tlarr[ix, iy]);
                        tlarr[ix, iy] = null;
                    }
            if (Report != null) Report("Merging done. " + tlarr[0, 0].Count + " tracks.\r\n");
            Track[] tkarr = (Track[])tlarr[0, 0].ToArray(typeof(Track));
            tlarr = null;
            for (i = 0; i < tkarr.Length; i++)
                UtilityTrack.SetId(tkarr[i], i);
            return tkarr;
        }


        public Track[] PropagateTracks(Track[] m_Tracks, Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            return m_Tracks;
            int j, n, k, m, i, l, h, SegsCount;
            int nt = m_Tracks.Length;

            int NGap = C.MaxMissingSegments;

            Position_CellArray[] CA = new Position_CellArray[m_Layers.Length];
            Segment[] tmps;

            int critic = C.MinimumCritical - 1;
            for (j = 0; j < m_Layers.Length; j++)
            {
                double MinZ = m_Layers[j].UpstreamZ;
                double MaxZ = m_Layers[j].DownstreamZ;
                SegsCount = m_Layers[j].Length;
                int nrealcou = 0;
                for (i = 0; i < SegsCount; i++)
                    if (((Segment)m_Layers[j][i]).TrackOwner != null)
                        nrealcou++;

                tmps = new Segment[nrealcou];
                for (i = k = 0; i < SegsCount; i++)
                {
                    if (((Segment)m_Layers[j][i]).TrackOwner != null)
                        tmps[k++] = (Segment)m_Layers[j][i];
                };
                CA[j] = new Position_CellArray(tmps, MinZ,
                    ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ) + C.LocalityCellSize,
                    ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ) + C.LocalityCellSize, 1.8, true, this);
            };


            if (Report != null) Report("Linking tracks... ");

            int lt = m_Layers.Length;
            double cdslope = C.D_Slope * C.D_Slope;
            double cdpos = C.D_Pos * C.D_Pos;
            bool kalman = C.KalmanFilter;
            int fitseg = C.FittingTracks;
            int minkal = C.MinKalman;

            SimpleKalmanTracking kt = new SimpleKalmanTracking(2, 2, 0.002, 0.002, 4.5, 4.5, 0.0035, 0.0035, 1300);

            for (j = 0; j < nt; j++) m_Tracks[j].FittingSegments = fitseg;

            for (j = 0; j < nt; j++)
            {
                if (m_Tracks[j].Length > 0 && m_Tracks[j].Length < lt)
                    for (k = 1; k <= NGap + 1; k++)
                    {
                        n = m_Tracks[j].Length;
                        if (n == 0) break;
                        if (m_Tracks[j][0].LayerOwner.Id/*.PosID*/- k >= m_Layers[0].Id/*.OrdinalID*/)
                        {
                            UtilityTrack.ReloadSegments(m_Tracks[j]);
                            
                            SySal.Tracking.MIPEmulsionTrackInfo MIPt = new SySal.Tracking.MIPEmulsionTrackInfo();
                            MIPt.Intercept.X = (m_Tracks[j].Downstream_PosX + m_Layers[m_Tracks[j][0].LayerOwner.Id - k].UpstreamZ * m_Tracks[j].Downstream_SlopeX);
                            MIPt.Intercept.Y = (m_Tracks[j].Downstream_PosY + m_Layers[m_Tracks[j][0].LayerOwner.Id - k].UpstreamZ * m_Tracks[j].Downstream_SlopeY);
                            MIPt.Intercept.Z = m_Layers[m_Tracks[j][0].LayerOwner.Id - k].UpstreamZ;
                            
                            tmps = CA[m_Tracks[j][0].LayerOwner.Id - k].Lock(MIPt);
                            m = tmps.Length;
                            for (i = 1; i < m; i++)
                            {
                                
                                double RefZ = 0.5 * (m_Layers[m_Tracks[j][0].LayerOwner.Id].DownstreamZ + m_Layers[tmps[i].LayerOwner.Id].UpstreamZ);
                                if (tmps[i].TrackOwner != null)
                                {
                                    int it = tmps[i].TrackOwner.Id;
                                    UtilityTrack.ReloadSegments(m_Tracks[it]);
                                    int tmpnn;
                                    if ((tmpnn = m_Tracks[it].Length) > 0 && it != j)
                                    {
                                        Vector2 DSlo;
                                        Vector2 Dpos;
                                        SySal.Tracking.MIPEmulsionTrackInfo mi_info;
                                        SySal.Tracking.MIPEmulsionTrackInfo mj_info;
                                        //Insert here kalman fit
                                        if (kalman && tmpnn > minkal && n > minkal)
                                        {
                                            mi_info = kt.ProjectVolumeTrackAtZ(m_Tracks[it], SySal.TotalScan.ProjectionDirection.UpStream, RefZ);
                                            mj_info = kt.ProjectVolumeTrackAtZ(m_Tracks[j], SySal.TotalScan.ProjectionDirection.DownStream, RefZ);
                                            Dpos.X = mj_info.Intercept.X - mi_info.Intercept.X;
                                            Dpos.Y = mj_info.Intercept.Y - mi_info.Intercept.Y;
                                            DSlo.X = mj_info.Slope.X - mi_info.Slope.X;
                                            DSlo.Y = mj_info.Slope.Y - mi_info.Slope.Y;
                                        }
                                        else
                                        {
                                            Dpos.X = Dpos.Y = 0.0;
                                            DSlo.X = (m_Tracks[j].Downstream_PosX + m_Tracks[j].Downstream_SlopeX * (m_Tracks[j].Downstream_Z - m_Tracks[j].Downstream_PosZ) - m_Tracks[it].Upstream_PosX - m_Tracks[it].Upstream_SlopeX * (m_Tracks[it].Upstream_Z - m_Tracks[it].Upstream_PosZ));
                                            DSlo.Y = (m_Tracks[j].Downstream_PosY + m_Tracks[j].Downstream_SlopeY * (m_Tracks[j].Downstream_Z - m_Tracks[j].Downstream_PosZ) - m_Tracks[it].Upstream_PosY - m_Tracks[it].Upstream_SlopeY * (m_Tracks[it].Upstream_Z - m_Tracks[it].Upstream_PosZ));
                                            DSlo.X = 0.5 * ((DSlo.X - m_Tracks[j].Downstream_SlopeX) + (DSlo.X - m_Tracks[it].Upstream_SlopeX));
                                            DSlo.Y = 0.5 * ((DSlo.Y - m_Tracks[j].Downstream_SlopeY) + (DSlo.Y - m_Tracks[it].Upstream_SlopeY));
                                        }
                                        if (DSlo.X * DSlo.X + DSlo.Y * DSlo.Y < cdslope &&
                                            Dpos.X * Dpos.X + Dpos.Y * Dpos.Y < cdpos)
                                        {
                                            l = m_Tracks[it].Length;
                                            int index_to_add = ((l > m_Tracks[j].Length) ? it : j);
                                            int index_to_remove = ((index_to_add == j) ? it : j);
                                            l = m_Tracks[index_to_remove].Length;
                                            for (h = 0; h < l; h++)
                                            {
                                                bool alreadythere = false;
                                                for (int u = 0; u < m_Tracks[index_to_add].Length; u++) if (m_Tracks[index_to_remove][0].LayerOwner.Id == m_Tracks[index_to_add][u].LayerOwner.Id) alreadythere = true;
                                                if (!alreadythere)
                                                {
                                                    m_Tracks[index_to_add].AddSegment(m_Tracks[index_to_remove][0]);
                                                }
                                                else
                                                {
                                                    m_Tracks[index_to_remove].RemoveSegment(m_Tracks[index_to_remove][0].LayerOwner.Id);
                                                    UtilityTrack.MoveToDisk(m_Tracks[index_to_remove]);
                                                };
                                            };
                                            if (index_to_remove == j) break;
                                        };
                                    }
                                    else
                                    {
                                        UtilityTrack.MoveToDisk(m_Tracks[it]);
                                    }
                                };

                            };
                        };
                        if (m_Tracks[j].Length == 0) break;

                        n = m_Tracks[j].Length;
                        if (m_Tracks[j][n - 1].LayerOwner.Id + k <= m_Layers[m_Layers.Length - 1].Id)
                        {
                            UtilityTrack.ReloadSegments(m_Tracks[j]);
                            SySal.Tracking.MIPEmulsionTrackInfo MIPt = new SySal.Tracking.MIPEmulsionTrackInfo();
                            MIPt.Intercept.X = (m_Tracks[j].Upstream_PosX + m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + k].DownstreamZ/*.TopZ*/* m_Tracks[j].Upstream_SlopeX);
                            MIPt.Intercept.Y = (m_Tracks[j].Upstream_PosY + m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + k].DownstreamZ/*.TopZ*/* m_Tracks[j].Upstream_SlopeY);
                            MIPt.Intercept.Z = m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + k].DownstreamZ/*.TopZ*/;
                            tmps = CA[m_Tracks[j][n - 1].LayerOwner.Id + k].Lock(MIPt);
                            m = tmps.Length;
                            for (i = 1; i < m; i++)
                            {
                                double RefZ = 0.5 * (m_Layers[m_Tracks[j][n - 1].LayerOwner.Id].UpstreamZ/*.BottomZ*/+ m_Layers[tmps[i].LayerOwner.Id].DownstreamZ/*.TopZ*/);
                                if (tmps[i].TrackOwner != null)
                                {
                                    int it = tmps[i].TrackOwner.Id;
                                    int tmpnn;
                                    if ((tmpnn = m_Tracks[it].Length) > 0 && it != j)
                                    {
                                        UtilityTrack.ReloadSegments(m_Tracks[it]);
                                        Vector2 Dpos;
                                        Vector2 DSlo;
                                        SySal.Tracking.MIPEmulsionTrackInfo mi_info;
                                        SySal.Tracking.MIPEmulsionTrackInfo mj_info;
                                        
                                        if (kalman && tmpnn > minkal && n > minkal)
                                        {
                                            mi_info = kt.ProjectVolumeTrackAtZ(m_Tracks[it], SySal.TotalScan.ProjectionDirection.DownStream, RefZ);
                                            mj_info = kt.ProjectVolumeTrackAtZ(m_Tracks[j], SySal.TotalScan.ProjectionDirection.UpStream, RefZ);
                                            Dpos.X = mj_info.Intercept.X - mi_info.Intercept.X;
                                            Dpos.Y = mj_info.Intercept.Y - mi_info.Intercept.Y;
                                            DSlo.X = mj_info.Slope.X - mi_info.Slope.X;
                                            DSlo.Y = mj_info.Slope.Y - mi_info.Slope.Y;
                                        }
                                        else
                                        {
                                            Dpos.X = (m_Tracks[j].Upstream_PosX + RefZ * m_Tracks[j].Upstream_SlopeX -
                                                (m_Tracks[it].Downstream_PosX + RefZ * m_Tracks[it].Downstream_SlopeX));
                                            Dpos.Y = (m_Tracks[j].Upstream_PosY + RefZ * m_Tracks[j].Upstream_SlopeY -
                                                (m_Tracks[it].Downstream_PosY + RefZ * m_Tracks[it].Downstream_SlopeY));
                                            DSlo.X = (m_Tracks[j].Upstream_SlopeX - m_Tracks[it].Downstream_SlopeX);
                                            DSlo.Y = (m_Tracks[j].Upstream_SlopeY - m_Tracks[it].Downstream_SlopeY);
                                        }
                                        if (DSlo.X * DSlo.X + DSlo.Y * DSlo.Y < cdslope &&
                                            Dpos.X * Dpos.X + Dpos.Y * Dpos.Y < cdpos)
                                        {
                                            
                                            l = m_Tracks[it].Length;
                                            int index_to_add = ((l > m_Tracks[j].Length) ? it : j);
                                            int index_to_remove = ((index_to_add == j) ? it : j);
                                            l = m_Tracks[index_to_remove].Length;
                                            for (h = 0; h < l; h++)
                                            {
                                                bool alreadythere = false;
                                                for (int u = 0; u < m_Tracks[index_to_add].Length; u++) if (m_Tracks[index_to_remove][0].LayerOwner.Id == m_Tracks[index_to_add][u].LayerOwner.Id) alreadythere = true;
                                                if (!alreadythere)
                                                {
                                                    m_Tracks[index_to_add].AddSegment(m_Tracks[index_to_remove][0]);
                                                }
                                                else
                                                {
                                                    m_Tracks[index_to_remove].RemoveSegment(m_Tracks[index_to_remove][0].LayerOwner.Id);
                                                    UtilityTrack.MoveToDisk(m_Tracks[index_to_remove]);
                                                };
                                            };
                                            if (index_to_remove == j) break;
                                        };
                                    }
                                    else
                                    {
                                        UtilityTrack.MoveToDisk(m_Tracks[it]);
                                    }
                                };
                            };

                        };

                    };
                if (Progress != null) Progress((double)(j + 1) / (double)nt);
            };


            critic = C.MinimumCritical;
            ArrayList tmptracks = new ArrayList();
            for (i = j = 0; i < nt; i++)
                if (m_Tracks[i].Length >= critic)
                {
                    UtilityTrack.SetId(m_Tracks[i], j++);
                    tmptracks.Add(m_Tracks[i]);
                    UtilityTrack.MoveToDisk(m_Tracks[i]);
                }
                else
                {
                    UtilityTrack.ReloadSegments(m_Tracks[i]);
                    for (k = 0; k < m_Tracks[i].Length; k++)
                        UtilitySegment.ResetTrackOwner(m_Tracks[i][k]);
                    UtilityTrack.MoveToDisk(m_Tracks[i]);
                }

            if (Report != null) Report(tmptracks.Count + " tracks reconstructed\r\n");
            return (Track[])tmptracks.ToArray(typeof(Track));

        }

        public Track[] BuildTracks(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {

            if (Report != null) Report("Attaching segments... ");

            Track[] m_Tracks = AttachSegmentsAccordingToLinkingIndex(C, ShouldStop, Progress, Report);

            int nt = m_Tracks.Length;
            if (Report != null) Report(nt + " tracks reconstructed\r\n");

            int NGap = C.MaxMissingSegments;

            if (NGap < 0)
            {
                if (Report != null) Report(nt + " tracks reconstructed\r\n");
                return m_Tracks;
            }

            if (C.KinkDetection)
                return FindKinks(PropagateTracks(m_Tracks, C, ShouldStop, Progress, Report), C, ShouldStop, Progress, Report);
            else
                return PropagateTracks(m_Tracks, C, ShouldStop, Progress, Report);

        }

        private Track[] FindKinks(Track[] m_Tracks, Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i, j, n, m;
            n = m_Tracks.Length;
            double sig1 = 0, sig2 = 0;
            int idseg = -1;
            int newcounter = 0;
            int idtra;
            int nbre = 0;
            int nmin = C.KinkMinimumSegments;
            double tol_2 = C.KinkMinimumDeltaS * C.KinkMinimumDeltaS;
            double factor = C.KinkFactor;
            double thresh = C.FilterThreshold;
            int filterlen = C.FilterLength;
            double[] filter = new double[filterlen];
            int startfil = -1 * ((int)(0.5 * (0.5 * filterlen - 1)));
            int forlen = (int)(0.5 * filterlen);
            for (i = 0; i < forlen; i++)
            {
                filter[i] = startfil + i;
                filter[2 * forlen - i - 1] = startfil + i; ;
            }

            if (Report != null) Report("Kink detection... ");

            ArrayList tmptracks = new ArrayList();
            for (i = 0; i < n; i++)
            {
                UtilityTrack.ReloadSegments(m_Tracks[i]);
                m = m_Tracks[i].Length;
                if (m >= nmin)
                {
                    double sxd = m_Tracks[i].Downstream_SlopeX;
                    double syd = m_Tracks[i].Downstream_SlopeY;
                    double sxu = m_Tracks[i].Upstream_SlopeX;
                    double syu = m_Tracks[i].Upstream_SlopeY;
                    if ((sxd - sxu) * (sxd - sxu) + (syd - syu) * (syd - syu) > tol_2)
                    {
                        idtra = m_Tracks[i].Id;
                        Track[] t;
                        //check del segmenti mancanti
                        bool missing = (m == m_Tracks[m].Id - m_Tracks[0].Id) ? false : true;

                        if (missing || (!missing && m < filterlen))
                            t = AnalizeTrackByFit(m_Tracks[i], factor, ref sig1, ref sig2, ref idseg);
                        else
                            t = AnalizeTrackByFilter(m_Tracks[i], filter, thresh, ref sig2, ref idseg);

                        if (t.Length == 2)
                        {
                            UtilityTrack.SetId(t[0], newcounter++);
                            tmptracks.Add(t[0]);
                            UtilityTrack.SetId(t[1], newcounter++);
                            tmptracks.Add(t[1]);
                            nbre++;
                        }
                        else
                        {
                            UtilityTrack.SetId(m_Tracks[i], newcounter++);
                            tmptracks.Add(m_Tracks[i]);
                            UtilityTrack.MoveToDisk(m_Tracks[i]);
                        }
                    }
                }
                else
                {
                    UtilityTrack.SetId(m_Tracks[i], newcounter++);
                    tmptracks.Add(m_Tracks[i]);
                    UtilityTrack.MoveToDisk(m_Tracks[i]);
                }
                if (Progress != null) Progress((double)(i + 1) / (double)n);
            }
            if (Report != null) Report(nbre + " kinks\r\n");

            return (Track[])tmptracks.ToArray(typeof(Track));
        }

        private Track[] AnalizeTrackByFit(Track t, double factor, ref double sig1, ref double sig2, ref int idseg)
        {
            int i, j, n, m;
            n = t.Length;
            double dum = 0, tcor1 = 0, tcor2 = 0, cor1 = 0, cor2 = 0, cor3 = 0, cor4 = 0;
            Track[] tt;
            idseg = -1;

            double[] x = new double[n];
            double[] y = new double[n];
            double[] z = new double[n];
            for (j = 0; j < n; j++)
            {
                x[j] = t[j].Info.Intercept.X;
                y[j] = t[j].Info.Intercept.Y;
                z[j] = t[j].Info.Intercept.Z;
            }
            Fitting.LinearFitSE(z, x, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref tcor1);
            Fitting.LinearFitSE(z, y, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref tcor2);
            sig1 = factor * Math.Sqrt(0.5 * (tcor1 * tcor1 + tcor2 * tcor2));
            double prevsig = sig1;
            for (i = 2; i < n - 3; i++)
            {
                x = new double[i + 1];
                y = new double[i + 1];
                z = new double[i + 1];
                //Segment[] s1 = new Segment[i+1];
                for (j = 0; j < i + 1; j++)
                {
                    //s1[j] = t[j];
                    x[j] = t[j].Info.Intercept.X;
                    y[j] = t[j].Info.Intercept.Y;
                    z[j] = t[j].Info.Intercept.Z;
                }
                Fitting.LinearFitSE(z, x, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor1);
                Fitting.LinearFitSE(z, y, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor2);
                x = new double[n - i - 1];
                y = new double[n - i - 1];
                z = new double[n - i - 1];
                //Segment[] s2 = new Segment[n-i-1];
                for (; j < n; j++)
                {
                    //s2[j-i-1] = t[j];
                    x[j - i - 1] = t[j].Info.Intercept.X;
                    y[j - i - 1] = t[j].Info.Intercept.Y;
                    z[j - i - 1] = t[j].Info.Intercept.Z;
                }
                Fitting.LinearFitSE(z, x, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor3);
                Fitting.LinearFitSE(z, y, ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor4);
                //Split
                sig2 = Math.Sqrt(0.25 * (cor1 * cor1 + cor2 * cor2 + cor3 * cor3 + cor4 * cor4));
                if (prevsig < sig2)
                {
                    prevsig = sig2;
                    idseg = i;
                }
            }

            if (idseg != -1)
            {
                /*
                x = new double[idseg+1];
                y = new double[idseg+1];
                z = new double[idseg+1]; 
                //Segment[] s1 = new Segment[idseg+1];
                for(j=0;j<idseg+1;j++)
                {
                    //s1[j] = t[j];
                    x[j]=t[j].Info.Intercept.X; 
                    y[j]=t[j].Info.Intercept.Y; 
                    z[j]=t[j].Info.Intercept.Z; 
                }
                Fitting.LinearFitSE(z,x,ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor1);
                Fitting.LinearFitSE(z,y,ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor2);
                x = new double[n-idseg-1];
                y = new double[n-idseg-1];
                z = new double[n-idseg-1];
                //Segment[] s2 = new Segment[n-idseg-1];
                for(;j<n;j++)
                {
                    //s2[j-idseg-1] = t[j];
                    x[j-idseg-1]=t[j].Info.Intercept.X; 
                    y[j-idseg-1]=t[j].Info.Intercept.Y; 
                    z[j-idseg-1]=t[j].Info.Intercept.Z; 
                }
                Fitting.LinearFitSE(z,x,ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor3);
                Fitting.LinearFitSE(z,y,ref dum, ref dum, ref dum, ref dum, ref dum, ref dum, ref cor4);
                */
                //Split
                tt = new Track[2];
                tt[0] = new Track();
                //n=s1.Length;
                for (j = 0; j < idseg + 1; j++)
                    //for(j=0;j<n;j++)
                    //	tt[0].AddSegment(s1[j]);
                    //	tt[0].AddSegment(t[j]);
                    tt[0].AddSegment(t[0]);
                tt[1] = new Track();
                //n=s2.Length;
                for (; j < n; j++)
                    //for(j=0;j<n;j++)
                    //	tt[1].AddSegment(s2[j]);
                    //	tt[1].AddSegment(t[j]);
                    tt[1].AddSegment(t[0]);
                sig2 = prevsig;
                return tt;
            }

            idseg = -1;
            tt = new SySal.TotalScan.Track[1] { t };
            return tt;
        }

        private Track[] AnalizeTrackByFilter(Track t, double[] filter, double threshold, ref double scalarvalue, ref int idseg)
        {
            int i, j, n, m;
            n = t.Length;
            Track[] tt;
            idseg = -1;
            int flen = filter.Length;

            double[] x = new double[n];
            double[] y = new double[n];
            double[] z = new double[n];
            for (j = 0; j < n; j++)
            {
                x[j] = t[j].Info.Intercept.X;
                y[j] = t[j].Info.Intercept.Y;
                z[j] = t[j].Info.Intercept.Z;
            }

            double min = 0;
            double max = 0;
            scalarvalue = threshold;
            bool firsttime = true;
            for (i = flen - 1; i < n; i++)
            {
                double[] tx = new double[flen];
                double[] ty = new double[flen];
                double[] tz = new double[flen];
                for (j = i - flen + 1; j < flen; j++)
                {
                    tx[j - (i - flen + 1)] = x[j];
                    ty[j - (i - flen + 1)] = y[j];
                    tz[j - (i - flen + 1)] = z[j];
                }

                //Split
                double tmp = Math.Abs(Matrices.ScalarProduct(tx, filter));
                if (max < tmp)
                {
                    max = tmp;
                    if (firsttime)
                    {
                        min = tmp;
                        firsttime = false;
                    }
                }
                else if (min > tmp)
                {
                    min = tmp;
                    if (firsttime)
                    {
                        max = tmp;
                        firsttime = false;
                    }
                }
                if (max / min > scalarvalue)
                {
                    scalarvalue = max / min;
                    idseg = i - (int)(flen / 2);
                }
            }

            if (idseg != -1)
            {
                tt = new Track[2];
                tt[0] = new Track();
                for (j = 0; j < idseg + 1; j++)
                    tt[0].AddSegment(t[0]);
                tt[1] = new Track();
                for (; j < n; j++)
                    tt[1].AddSegment(t[0]);
                return tt;
            }

            idseg = -1;
            tt = new SySal.TotalScan.Track[1] { t };
            return tt;
        }



        public void UpdateTransformations(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            if (Report != null)
            {
                Report("Updating Transformations...\r\n");
            }

            int i, j, k;
            int n, m, l;
            int pairscount;
            double RefZ;
            double[] Align = new double[POS_ALIGN_DATA_LEN];

            //Parameters
            //Minimum number of segments pairs for the transformation to be updated
            int MinPairs = C.MinimumTracksPairs;
            //Minimum segments number for a track to be taken into account
            int MinSeg = C.MinimumSegmentsNumber;

            int ntot = m_Tracks.Length;
            l = m_Layers.Length;
            double[,] SAlignDSlope = new double[l, 2];
            double[,] SAlignDShrink = new double[l, 2];
            double[] tmpSAlignDSlope = new double[2];
            double[] tmpSAlignDShrink = new double[2];

            ArrayList idarr1 = new ArrayList();
            ArrayList idarr2 = new ArrayList();


            for (k = 0; k < l; k++)
            {
                ((Layer)m_Layers[k]).RestoreOriginalSegments();
            }

            for (k = 0; k < l - 1; k++)
            {
                pairscount = 0;
                idarr1.Clear();
                idarr2.Clear();
                for (i = 0; i < ntot; i++)
                {
                    if ((m = m_Tracks[i].Length) >= MinSeg)
                    {
                        for (j = 0; j < m - 1; j++)
                        {
                            if (m_Tracks[i][j].LayerOwner.Id == k)
                            {
                                if (m_Tracks[i][j + 1].LayerOwner.Id == k + 1)
                                {
                                    //Ok, You can do it!
                                    pairscount++;
                                    idarr1.Add(m_Tracks[i][j].PosInLayer);
                                    idarr2.Add(m_Tracks[i][j + 1].PosInLayer);
                                }
                            }
                            else if (m_Tracks[i][j].LayerOwner.Id > k)
                            {
                                break;
                            }
                        }
                    }
                }

                //You can use i,j,n,m with a different meaning
                if (pairscount >= MinPairs)
                {
                    int[] id1 = (int[])idarr1.ToArray(typeof(int));
                    int[] id2 = (int[])idarr2.ToArray(typeof(int));
                    //length is the same for both: they are couple...
                    n = id1.Length;

                    Vector2 DS = new Vector2();
                    Vector2 SlD = new Vector2();
                    Vector2 SlSD = new Vector2();
                    Vector2 SlS = new Vector2();
                    Vector2 SlS2 = new Vector2();

                    for (i = 0; i < n; i++)
                    {
                        Segment S2 = (Segment)m_Layers[k][id1[i]];
                        Segment S1 = (Segment)m_Layers[k + 1][id2[i]];
                        SySal.Tracking.MIPEmulsionTrackInfo S1Info = S1.GetInfo();
                        SySal.Tracking.MIPEmulsionTrackInfo S2Info = S2.GetInfo();

                        DS.X = S2Info.Slope.X - S1Info.Slope.X;
                        DS.Y = S2Info.Slope.Y - S1Info.Slope.Y;

                        SlD.X += DS.X;
                        SlD.Y += DS.Y;

                        SlS.X += S2Info.Slope.X;
                        SlS.Y += S2Info.Slope.Y;
                        SlS2.X += S2Info.Slope.X * S2Info.Slope.X;
                        SlS2.Y += S2Info.Slope.Y * S2Info.Slope.Y;
                        SlSD.X += S2Info.Slope.X * DS.X;
                        SlSD.Y += S2Info.Slope.Y * DS.Y;

                    }

                    double SlDenX, SlDenY;
                    SlDenX = (1 / (n * SlS2.X - SlS.X * SlS.X));
                    SlDenY = (1 / (n * SlS2.Y - SlS.Y * SlS.Y));

                    SAlignDShrink[k, 0] = ((n * SlSD.X - SlS.X * SlD.X) * SlDenX);
                    SAlignDShrink[k, 1] = ((n * SlSD.Y - SlS.Y * SlD.Y) * SlDenY);

                    SAlignDSlope[k, 0] = (SlD.X * SlS2.X - SlS.X * SlSD.X) * SlDenX;
                    SAlignDSlope[k, 1] = (SlD.Y * SlS2.Y - SlS.Y * SlSD.Y) * SlDenY;

                    SAlignDShrink[k, 0] = (1 + SAlignDShrink[k, 0]);
                    SAlignDShrink[k, 1] = (1 + SAlignDShrink[k, 1]);
                    SAlignDSlope[k, 0] = SAlignDSlope[k, 0];
                    SAlignDSlope[k, 1] = SAlignDSlope[k, 1];

                }
            };


            for (i = 0; i < l - 1; i++)
            {
                //Correzione Slopes
                //if (i==0) 
                for (j = i + 1; j < l; j++)
                {
                    //Correzione Slopes
                    int SegsCount = m_Layers[j].Length;
                    for (k = 0; k < SegsCount; k++)
                    {
                        SySal.Tracking.MIPEmulsionTrackInfo ttk = m_Layers[j][k].Info;
                        double sx = ttk.Slope.X;
                        double sy = ttk.Slope.Y;

                        ttk.Slope.X = (float)(m_AlignmentData[i].DShrinkX * sx + m_AlignmentData[i].SAlignDSlopeX);
                        ttk.Slope.Y = (float)(m_AlignmentData[i].DShrinkY * sy + m_AlignmentData[i].SAlignDSlopeY);

                        //Solo l'info va modificato cosi' non si sgarrupano i TopLink ed i BottomLink
                        m_Layers[j][k].Info = ttk;
                    };
                };
            };


            for (k = 0; k < l - 1; k++)
            {
                pairscount = 0;
                idarr1.Clear();
                idarr2.Clear();
                for (i = 0; i < ntot; i++)
                {
                    if ((m = m_Tracks[i].Length) >= MinSeg)
                    {
                        for (j = 0; j < m - 1; j++)
                        {
                            if (m_Tracks[i][j].LayerOwner.Id == k)
                            {
                                if (m_Tracks[i][j + 1].LayerOwner.Id == k + 1)
                                {
                                    //Ok, You can do it!
                                    pairscount++;
                                    idarr1.Add(m_Tracks[i][j].PosInLayer);
                                    idarr2.Add(m_Tracks[i][j + 1].PosInLayer);
                                }
                            }
                            else if (m_Tracks[i][j].LayerOwner.Id > k)
                            {
                                break;
                            }
                        }
                    }
                }

                //You can use i,j,n,m with a different meaning
                if (pairscount >= MinPairs)
                {
                    //((Layer)m_Layers[k]).RestoreOriginalSegments();
                    //((Layer)m_Layers[k+1]).RestoreOriginalSegments();
                    int[] id1 = (int[])idarr1.ToArray(typeof(int));
                    int[] id2 = (int[])idarr2.ToArray(typeof(int));
                    //length is the same for both: they are couple...
                    n = id1.Length;

                    double[] tvX = new double[n];
                    double[] tvY = new double[n];
                    double[] tvDx = new double[n];
                    double[] tvDy = new double[n];
                    double[] tvSx = new double[n];
                    double[] tvSy = new double[n];

                    for (i = 0; i < n; i++)
                    {
                        Segment S2 = (Segment)m_Layers[k][id1[i]];
                        Segment S1 = (Segment)m_Layers[k + 1][id2[i]];
                        SySal.Tracking.MIPEmulsionTrackInfo S1Info = S1.GetInfo();
                        SySal.Tracking.MIPEmulsionTrackInfo S2Info = S2.GetInfo();
                        RefZ = 0.5f * (S1Info.Intercept.Z + S2Info.Intercept.Z);
                        tvX[i] = S1Info.Intercept.X + (RefZ - S1Info.Intercept.Z) * S1Info.Slope.X - m_RefCenter.X;
                        tvY[i] = S1Info.Intercept.Y + (RefZ - S1Info.Intercept.Z) * S1Info.Slope.Y - m_RefCenter.Y;
                        tvSx[i] = S1Info.Slope.X;
                        tvSy[i] = S1Info.Slope.Y;
                        tvDx[i] = tvX[i] - (S2Info.Intercept.X + (RefZ - S2Info.Intercept.Z) * S2Info.Slope.X - m_RefCenter.X);
                        tvDy[i] = tvY[i] - (S2Info.Intercept.Y + (RefZ - S2Info.Intercept.Z) * S2Info.Slope.Y - m_RefCenter.Y);


                    }

                    if ((!C.FreezeZ && Fitting.Affine_Focusing(tvDx, tvDy, tvX, tvY, tvSx, tvSy, ref Align) != NumericalTools.ComputationResult.OK) ||
                        (C.FreezeZ && Fitting.Affine(tvDx, tvDy, tvX, tvY, ref Align) != NumericalTools.ComputationResult.OK))
                    {
                        double[] tmpfail = new double[2];
                        Align = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                        m_AlignmentData[k] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.BadAffineFocusing);
                    }


                    Align[0] += 1;
                    Align[3] += 1;
                    tmpSAlignDShrink[0] = SAlignDShrink[k, 0];
                    tmpSAlignDShrink[1] = SAlignDShrink[k, 1];
                    tmpSAlignDSlope[0] = SAlignDSlope[k, 0];
                    tmpSAlignDSlope[1] = SAlignDSlope[k, 1];
                    m_AlignmentData[k] = new AlignmentData(tmpSAlignDShrink, tmpSAlignDSlope, Align, MappingResult.OK);

                    if (Report != null)
                    {
                        Report("Updated Transformations at layer " + k + ": \r\n");
                        for (i = 0; i < Align.Length; i++)
                            Report("Align[" + i + "] =" + Align[i] + "\r\n");
                    }

                }
                else
                {
                    if (Report != null)
                    {
                        Report("Transformations not updated at layer " + k + "\r\n");
                    }
                }
            };

            CTransformation(C, ShouldStop, Progress, Report);

        }


        public void UpdateTransformations2(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            if (Report != null)
            {
                Report("Updating Transformations...\r\n");
            }

            int i, k, h;
            int n, l;
            int pairscount;
            double[] Align = new double[POS_ALIGN_DATA_LEN];

            int MinPairs = C.MinimumTracksPairs;
            int MinSeg = C.MinimumSegmentsNumber;

            int ntot = m_Tracks.Length;
            l = m_Layers.Length;
            double[,] SAlignDSlope = new double[l, 2];
            double[,] SAlignDShrink = new double[l, 2];
            double[] tmpSAlignDSlope = new double[2];
            double[] tmpSAlignDShrink = new double[2];

            ArrayList idarr1 = new ArrayList();
            ArrayList idarr2 = new ArrayList();

            SySal.Processing.AlphaOmegaReconstruction.AlignmentData zero_a = new SySal.Processing.AlphaOmegaReconstruction.AlignmentData();
            zero_a.AffineMatrixXX = zero_a.AffineMatrixYY = 1.0;
            zero_a.AffineMatrixXY = zero_a.AffineMatrixYX = 0.0;
            zero_a.DShrinkX = zero_a.DShrinkY = 1.0;
            zero_a.SAlignDSlopeX = zero_a.SAlignDSlopeY = 0.0;
            zero_a.TranslationX = zero_a.TranslationY = zero_a.TranslationZ = 0.0;
            ((Layer)m_Layers[l - 1]).iAlignmentData = zero_a;

            for (k = 0; k < l - 1; k++)
            {
                pairscount = 0;
                SySal.TotalScan.Layer templay = m_Layers[k];
                n = templay.Length;
                for (i = 0; i < n; i++)
                {
                    Segment seg = (Segment)templay[i];
                    if (!seg.m_IgnoreInAlignment && seg.TrackOwner != null && seg.TrackOwner.Length >= MinSeg && seg.UpstreamLinked != null && !seg.UpstreamLinked.m_IgnoreInAlignment && seg.LayerOwner.Id == seg.UpstreamLinked.LayerOwner.Id - 1)
                        pairscount++;
                }
                if (pairscount < MinPairs) return;
            }
            ((Layer)m_Layers[l - 1]).RestoreOriginalSegments();

            for (k = l - 1; k > 0; k--)
            {
                ((Layer)m_Layers[k - 1]).RestoreOriginalSegments();
                SySal.TotalScan.Layer templay = m_Layers[k - 1];
                pairscount = 0;
                idarr1.Clear();
                idarr2.Clear();
                n = templay.Length;
                for (i = 0; i < n; i++)
                {
                    Segment seg = (Segment)templay[i];
                    if (!seg.m_IgnoreInAlignment && seg.TrackOwner != null && seg.TrackOwner.Length >= MinSeg && seg.UpstreamLinked != null && !seg.UpstreamLinked.m_IgnoreInAlignment && seg.LayerOwner.Id == seg.UpstreamLinked.LayerOwner.Id - 1)
                    {
                        idarr1.Add(seg.UpstreamLinked.PosInLayer);
                        idarr2.Add(seg.PosInLayer);
                    }
                }
                if (k == l - 1) ((Layer)m_Layers[k]).RestoreOriginalSegments();
                ((Layer)m_Layers[k - 1]).RestoreOriginalSegments();
                int[] id1 = (int[])idarr1.ToArray(typeof(int));
                int[] id2 = (int[])idarr2.ToArray(typeof(int));

                n = id1.Length;

                Vector2 DS = new Vector2();
                Vector2 SlD = new Vector2();
                Vector2 SlSD = new Vector2();
                Vector2 SlS = new Vector2();
                Vector2 SlS2 = new Vector2();

                for (i = 0; i < n; i++)
                {
                    Segment S1 = (Segment)m_Layers[k][id1[i]];
                    Segment S2 = (Segment)m_Layers[k - 1][id2[i]];
                    SySal.Tracking.MIPEmulsionTrackInfo S1Info = S1.GetInfo();
                    SySal.Tracking.MIPEmulsionTrackInfo S2Info = S2.GetInfo();

                    DS.X = S1Info.Slope.X - S2Info.Slope.X;
                    DS.Y = S1Info.Slope.Y - S2Info.Slope.Y;

                    SlD.X += DS.X;
                    SlD.Y += DS.Y;

                    SlS.X += S2Info.Slope.X;
                    SlS.Y += S2Info.Slope.Y;
                    SlS2.X += S2Info.Slope.X * S2Info.Slope.X;
                    SlS2.Y += S2Info.Slope.Y * S2Info.Slope.Y;
                    SlSD.X += S2Info.Slope.X * DS.X;
                    SlSD.Y += S2Info.Slope.Y * DS.Y;

                }

                double SlDenX, SlDenY;
                SlDenX = (1 / (n * SlS2.X - SlS.X * SlS.X));
                SlDenY = (1 / (n * SlS2.Y - SlS.Y * SlS.Y));

                SAlignDShrink[k, 0] = ((n * SlSD.X - SlS.X * SlD.X) * SlDenX);
                SAlignDShrink[k, 1] = ((n * SlSD.Y - SlS.Y * SlD.Y) * SlDenY);

                SAlignDSlope[k, 0] = (SlD.X * SlS2.X - SlS.X * SlSD.X) * SlDenX;
                SAlignDSlope[k, 1] = (SlD.Y * SlS2.Y - SlS.Y * SlSD.Y) * SlDenY;

                SAlignDShrink[k, 0] = (1 + SAlignDShrink[k, 0]);
                SAlignDShrink[k, 1] = (1 + SAlignDShrink[k, 1]);
                SAlignDSlope[k, 0] = SAlignDSlope[k, 0];
                SAlignDSlope[k, 1] = SAlignDSlope[k, 1];

                int SegsCount = m_Layers[k - 1].Length;
                for (h = 0; h < SegsCount; h++)
                {
                    SySal.Tracking.MIPEmulsionTrackInfo ttk = ((Segment)m_Layers[k - 1][h]).GetInfo();
                    double sx = ttk.Slope.X;
                    double sy = ttk.Slope.Y;

                    ttk.Slope.X = (float)(SAlignDShrink[k, 0] * sx + SAlignDSlope[k, 0]);
                    ttk.Slope.Y = (float)(SAlignDShrink[k, 1] * sy + SAlignDSlope[k, 1]);
                };

                double[] tvX = new double[n];
                double[] tvY = new double[n];
                double[] tvDx = new double[n];
                double[] tvDy = new double[n];
                double[] tvSx = new double[n];
                double[] tvSy = new double[n];

                for (i = 0; i < n; i++)
                {
                    Segment S1 = (Segment)m_Layers[k][id1[i]];
                    Segment S2 = (Segment)m_Layers[k - 1][id2[i]];
                    SySal.Tracking.MIPEmulsionTrackInfo S1Info = S1.GetInfo();
                    SySal.Tracking.MIPEmulsionTrackInfo S2Info = S2.GetInfo();

                    tvX[i] = S2Info.Intercept.X - m_RefCenter.X;
                    tvY[i] = S2Info.Intercept.Y - m_RefCenter.Y;
                    tvSx[i] = S2Info.Slope.X;
                    tvSy[i] = S2Info.Slope.Y;
                    tvDx[i] = -tvX[i] + (S1Info.Intercept.X + (S2Info.Intercept.Z - S1Info.Intercept.Z) * S1Info.Slope.X - m_RefCenter.X);
                    tvDy[i] = -tvY[i] + (S1Info.Intercept.Y + (S2Info.Intercept.Z - S1Info.Intercept.Z) * S1Info.Slope.Y - m_RefCenter.Y);

                }

                if ((!C.FreezeZ && Fitting.Affine_Focusing(tvDx, tvDy, tvX, tvY, tvSx, tvSy, ref Align) != NumericalTools.ComputationResult.OK) ||
                    (C.FreezeZ && Fitting.Affine(tvDx, tvDy, tvX, tvY, ref Align) != NumericalTools.ComputationResult.OK))
                {
                    double[] tmpfail = new double[2];
                    Align = new double[POS_ALIGN_DATA_LEN] { 1, 0, 0, 1, 0, 0, 0 };
                    m_AlignmentData[k - 1] = new AlignmentData(tmpfail, tmpfail, Align, MappingResult.BadAffineFocusing);
                }


                Align[0] += 1;
                Align[3] += 1;
                tmpSAlignDShrink[0] = SAlignDShrink[k, 0];
                tmpSAlignDShrink[1] = SAlignDShrink[k, 1];
                tmpSAlignDSlope[0] = SAlignDSlope[k, 0];
                tmpSAlignDSlope[1] = SAlignDSlope[k, 1];
                ((Layer)m_Layers[k - 1]).iAlignmentData = m_AlignmentData[k - 1] = new AlignmentData(tmpSAlignDShrink, tmpSAlignDSlope, Align, MappingResult.OK);

                for (h = 0; h < m_Layers[k - 1].Length; h++)
                {
                    double x, y;
                    AlignmentData ca = m_AlignmentData[k - 1];
                    SySal.Tracking.MIPEmulsionTrackInfo ttk = ((Segment)m_Layers[k - 1][h]).GetInfo();
                    x = ttk.Intercept.X - m_RefCenter.X;
                    y = ttk.Intercept.Y - m_RefCenter.Y;
                    ttk.Intercept.X = ca.AffineMatrixXX * x + ca.AffineMatrixXY * y + ca.TranslationX + m_RefCenter.X;
                    ttk.Intercept.Y = ca.AffineMatrixYX * x + ca.AffineMatrixYY * y + ca.TranslationY + m_RefCenter.Y;
                    ttk.Intercept.Z -= ca.TranslationZ;
                    ttk.TopZ -= ca.TranslationZ;
                    ttk.BottomZ -= ca.TranslationZ;
                    x = ttk.Slope.X;
                    y = ttk.Slope.Y;
                    ttk.Slope.X = ca.AffineMatrixXX * x + ca.AffineMatrixXY * y;
                    ttk.Slope.Y = ca.AffineMatrixYX * x + ca.AffineMatrixYY * y;
                }
                ((Layer)templay).UpdateZ();
                ((Layer)m_Layers[k]).MoveToDisk();

                if (Report != null)
                {
                    Report("Updated Transformations at layer " + k + ": \r\n");
                    for (i = 0; i < Align.Length; i++)
                        Report("Align[" + i + "] =" + Align[i] + "\r\n");
                }

            };
            ((Layer)m_Layers[0]).MoveToDisk();
        }


        #endregion

        #region LowMomentum
        public Track[] ReconstructLowMomentumTracks(Configuration C, dShouldStop ShouldStop, dProgress Progress, dReport Report)
        {
            int i;
            int n = m_AlignmentData.Length;
            for (i = 0; i < n - 1; i++)
                if (m_AlignmentData[i].Result == MappingResult.NotPerformedYet)
                    throw new Exception("Alignment must be performed first!");

            int j, m, l, h, SegsCount;

            Track[] m_Tracks = AttachSegmentsAccordingToLinkingIndex(C, ShouldStop, Progress, Report);
            int nt = m_Tracks.Length;

            //Tentare di unire le tracce tra loro
            //Massima lunghezza del buco? (1, 2 segmenti)
            //Minima lunghezza delle tracce
            Position_CellArray[] Pos_CA = new Position_CellArray[m_Layers.Length];
            Slopes_CellArray[] Slo_CA = new Slopes_CellArray[m_Layers.Length];
            Segment[] tmps;

            //Come PosID si intende il numero ordinale di Layer
            for (j = 0; j < m_Layers.Length; j++)
            {
                double MinZ = m_Layers[j].UpstreamZ;
                double MaxZ = m_Layers[j].DownstreamZ;
                SegsCount = m_Layers[j].Length;
                tmps = new Segment[SegsCount];
                for (i = 0; i < SegsCount; i++)
                    tmps[i] = (Segment)m_Layers[j][i];

                //Le celle devono essere più piccole
                Pos_CA[j] = new Position_CellArray(tmps, MinZ,
                    ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ),
                    ((C.Initial_D_Pos > C.D_Pos) ? C.Initial_D_Pos : C.D_Pos) + ((C.Initial_D_Slope > C.D_Slope) ? C.Initial_D_Slope : C.D_Slope) * (MaxZ - MinZ), 5, true, this);
                Slo_CA[j] = new Slopes_CellArray(tmps, C.SlopesCellSize.X, C.SlopesCellSize.Y, C.RiskFactor, 15);
            };

            for (j = 0; j < nt; j++)
            {
                if (m_Tracks[j].Length > 0 && m_Tracks[j].Length < m_Layers.Length)
                {
                    n = m_Tracks[j].Length;
                    //Bisogna ricontrollare che m_Tracks[j].Length (cioè n) sia >0
                    //perché potrebbe essere stato tolto
                    if (n == 0) break;
                    if (m_Tracks[j][0].LayerOwner.Id > m_Layers[0].Id)
                    {
                        //Bisogna prima proiettare
                        SySal.Tracking.MIPEmulsionTrackInfo MIPt = m_Layers[i][j].Info;
                        //MIPt.Slope.X = m_Tracks[j][0].Info.Slope.X;
                        //MIPt.Slope.Y = m_Tracks[j][0].Info.Slope.Y;
                        //Segment MIPs = new Segment(MIPt);
                        //Sembra che ci sia un errore all'ultimo argomento
                        //Segment MIPs = new Segment(MIPt, (Layer)m_Tracks[j][0].LayerOwner, m_Tracks[j][0].LayerOwner.Id-1);
                        //Ricorda: la coord z di trk è sempre il downstream del layer
                        //perciò, per essere paragonati ai segmenti del cellarray, questi segmenti fittizi
                        //devono essere proiettati alla stessa quota
                        SySal.BasicTypes.Vector2 TollSlo = Slo_CA[m_Tracks[j][0].LayerOwner.Id - 1].Lock(MIPt);
                        Vector I = m_Tracks[j][0].Info.Intercept;
                        Vector S = m_Tracks[j][0].Info.Slope;
                        MIPt.Intercept.X = 0.5 * ((I.X + (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X + TollSlo.X)) +
                            (I.X + (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X - TollSlo.X)));
                        MIPt.Intercept.Y = 0.5 * ((I.Y + (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y + TollSlo.Y)) +
                            (I.Y + (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y - TollSlo.Y)));
                        MIPt.Intercept.Z = m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].UpstreamZ;
                        //MIPs = new Segment(MIPt);

                        //Settare la grandezza dell'area in mezzo al gap
                        //probabilmente bisognerà contemplare il caso asimmetrico
                        //double RefZ = 0.5*(m_Layers[m_Tracks[j][0].LayerOwner.Id-1].UpstreamZ+m_Layers[m_Tracks[j][0].LayerOwner.Id].DownstreamZ);
                        SySal.BasicTypes.Vector2 TollWid;
#if true
                        TollWid.X = (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X + TollSlo.X) - (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X - TollSlo.X);
                        TollWid.Y = (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y + TollSlo.Y) - (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y - TollSlo.Y);
#else
						TollWid.X = 2*(m_Layers[m_Tracks[j][0].LayerOwner.Id-1].DownstreamZ - m_Tracks[j][0].Info.Intercept.Z)*TollSlo.X/Math.Sqrt(3);
						TollWid.Y = 2*(m_Layers[m_Tracks[j][0].LayerOwner.Id-1].DownstreamZ - m_Tracks[j][0].Info.Intercept.Z)*TollSlo.Y/Math.Sqrt(3);
#endif
                        Pos_CA[m_Tracks[j][0].LayerOwner.Id - 1].MaximumLevelSize = ((TollWid.X / C.Initial_D_Pos > TollWid.Y / C.Initial_D_Pos) ? Math.Ceiling(TollWid.X / C.Initial_D_Pos) : Math.Ceiling(TollWid.Y / C.Initial_D_Pos));

                        tmps = Pos_CA[m_Tracks[j][0].LayerOwner.Id - 1].Lock(MIPt);
                        m = tmps.Length;
                        for (i = 1; i < m; i++)
                        {
                            //prova il link
                            double RefZ = 0.5 * (m_Layers[m_Tracks[j][0].LayerOwner.Id].DownstreamZ + m_Layers[tmps[i].LayerOwner.Id].UpstreamZ);
                            TollWid.X = (RefZ - I.Z) * (S.X + TollSlo.X) - (RefZ - I.Z) * (S.X - TollSlo.X);
                            TollWid.Y = (RefZ - I.Z) * (S.Y + TollSlo.Y) - (RefZ - I.Z) * (S.Y - TollSlo.Y);
                            //Certi segmenti possono essere stati tolti da 
                            //una traccia e non più assegnati perché la nuova 
                            //traccia ha già un seg sullo stesso layer del seg tolto(rimisure)
                            //Bisogna perciò controllare quel segmento appartiene ad una traccia
                            if (tmps[i].TrackOwner != null)
                            {
                                int it = tmps[i].TrackOwner.Id;
                                int n_le = m_Tracks[it].Length;
                                if (n_le > 0 && it != j)
                                {
                                    Vector2 Dpos;
                                    SySal.Tracking.MIPEmulsionTrackInfo nInfo = m_Tracks[it][n_le - 1].Info;
                                    Dpos.X = (I.X + (RefZ - I.Z) * S.X -
                                        (nInfo.Intercept.X + (RefZ - nInfo.Intercept.Z) * nInfo.Slope.X));
                                    Dpos.Y = (I.Y + (RefZ - I.Z) * S.Y -
                                        (nInfo.Intercept.Y + (RefZ - nInfo.Intercept.Z) * nInfo.Slope.Y));
                                    Vector2 DSlo;
                                    DSlo.X = (S.X - nInfo.Slope.X);
                                    DSlo.Y = (S.Y - nInfo.Slope.Y);
                                    //Le tolleranze saranno diverse: ricordarsi di aggiustarle
                                    if (DSlo.X * DSlo.X + DSlo.Y * DSlo.Y < TollSlo.X * TollSlo.X + TollSlo.Y * TollSlo.Y &&
                                        Dpos.X * Dpos.X + Dpos.Y * Dpos.Y < TollWid.X * TollWid.X + TollWid.Y * TollWid.Y)
                                    {
                                        //Unisci le due tracce
                                        int index_to_add = ((n_le > m_Tracks[j].Length) ? it : j);
                                        int index_to_remove = ((index_to_add == j) ? it : j);
                                        l = m_Tracks[index_to_remove].Length;
                                        for (h = 0; h < l; h++)
                                        {
                                            bool alreadythere = false;
                                            for (int u = 0; u < m_Tracks[index_to_add].Length; u++) if (m_Tracks[index_to_remove][0].LayerOwner.Id == m_Tracks[index_to_add][u].LayerOwner.Id) alreadythere = true;
                                            if (!alreadythere)
                                            {
                                                //Aggiunge ad una e rimuove automaticamente all'altra
                                                m_Tracks[index_to_add].AddSegment(m_Tracks[index_to_remove][0]);
                                            }
                                            else
                                            {
                                                //per ora si rimuove solo però si dovrà scegliere il segmento che 
                                                //più si adatta con un fit alla traccia che è stata prescelta per sopravvivere
                                                m_Tracks[index_to_remove].RemoveSegment(m_Tracks[index_to_remove][0].LayerOwner.Id);
                                            };
                                        };
                                        //La traccia del loop più esterno è stata rimossa
                                        //non ha più senso continuare con essa
                                        if (index_to_remove == j) break;

                                    };
                                };
                            };

                        };
                    };
                    if (m_Tracks[j].Length == 0) break;

                    //La lunghezza di questa traccia potrebbe essere cambiata sopra
                    //quindi essa va riconsiderata
                    n = m_Tracks[j].Length;
                    if (m_Tracks[j][n - 1].LayerOwner.Id < m_Layers[m_Layers.Length - 1].Id)
                    {
                        //Bisogna prima proiettare
                        SySal.Tracking.MIPEmulsionTrackInfo MIPt = m_Tracks[j][n - 1].Info;//new SySal.Tracking.MIPEmulsionTrackInfo();
                        //MIPt.Slope.X = m_Tracks[j][n-1].Info.Slope.X;
                        //MIPt.Slope.Y = m_Tracks[j][n-1].Info.Slope.Y;
                        //Segment MIPs = new Segment(MIPt);
                        //Bisogna prima proiettare

                        Vector In = m_Tracks[j][n - 1].Info.Intercept;
                        Vector Sn = m_Tracks[j][n - 1].Info.Slope;

                        SySal.BasicTypes.Vector2 TollSlo = Slo_CA[m_Tracks[j][n - 1].LayerOwner.Id - 1].Lock(MIPt);
                        MIPt.Intercept.X = (In.X + m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + 1].DownstreamZ * Sn.X);
                        MIPt.Intercept.Y = (In.Y + m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + 1].DownstreamZ * Sn.Y);
                        MIPt.Intercept.Z = m_Layers[m_Tracks[j][n - 1].LayerOwner.Id + 1].DownstreamZ;
                        //Sembra che ci sia un errore all'ultimo argomento
                        //Segment MIPs = new Segment(MIPt,(Layer)m_Tracks[j][n-1].LayerOwner,m_Tracks[j][n-1].LayerOwner.Id+1);
                        //MIPs = new Segment(MIPt);

                        //Settare la grandezza dell'area in mezzo al gap
                        //probabilmente bisognerà contemplare il caso asimmetrico
                        //double RefZ = 0.5*(m_Layers[m_Tracks[j][0].LayerOwner.Id-1].UpstreamZ+m_Layers[m_Tracks[j][0].LayerOwner.Id].DownstreamZ);
                        SySal.BasicTypes.Vector2 TollWid;
                        Vector I = m_Tracks[j][0].Info.Intercept;
                        Vector S = m_Tracks[j][0].Info.Slope;
                        TollWid.X = (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X + TollSlo.X) - (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.X - TollSlo.X);
                        TollWid.Y = (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y + TollSlo.Y) - (m_Layers[m_Tracks[j][0].LayerOwner.Id - 1].DownstreamZ - I.Z) * (S.Y - TollSlo.Y);

                        tmps = Pos_CA[m_Tracks[j][n - 1].LayerOwner.Id + 1].Lock(MIPt);
                        m = tmps.Length;
                        for (i = 1; i < m; i++)
                        {
                            //prova il link
                            double RefZ = 0.5 * (m_Layers[m_Tracks[j][n - 1].LayerOwner.Id].UpstreamZ + m_Layers[tmps[i].LayerOwner.Id].DownstreamZ);

                            TollWid.X = (RefZ - In.Z) * (Sn.X + TollSlo.X) - (RefZ - In.Z) * (Sn.X - TollSlo.X);
                            TollWid.Y = (RefZ - In.Z) * (Sn.Y + TollSlo.Y) - (RefZ - In.Z) * (Sn.Y - TollSlo.Y);

                            //Certi segmenti possono essere stati tolti da 
                            //una traccia e non più assegnati perché la nuova 
                            //traccia ha già un seg sullo stesso layer del seg tolto(rimisure)
                            //Bisogna perciò controllare quel segmento appartiene ad una traccia
                            if (tmps[i].TrackOwner != null)
                            {
                                int it = tmps[i].TrackOwner.Id;
                                Vector It = m_Tracks[it][0].Info.Intercept;
                                Vector St = m_Tracks[it][0].Info.Slope;
                                int n_le = m_Tracks[it].Length;
                                if (n_le > 0 && it != j)
                                {
                                    Vector2 Dpos;
                                    Vector Io = m_Tracks[j][n_le - 1].Info.Intercept;
                                    Vector So = m_Tracks[j][n_le - 1].Info.Slope;
                                    Dpos.X = (Io.X + RefZ * So.X - (It.X + RefZ * St.X));
                                    Dpos.Y = (Io.Y + RefZ * So.Y - (It.Y + RefZ * St.Y));
                                    Vector2 DSlo;
                                    DSlo.X = (So.X - St.X);
                                    DSlo.Y = (So.Y - St.Y);
                                    //Le tolleranze saranno diverse: ricordarsi di aggiustarle
                                    if (DSlo.X * DSlo.X + DSlo.Y * DSlo.Y < TollSlo.X * TollSlo.X + TollSlo.Y * TollSlo.Y &&
                                        Dpos.X * Dpos.X + Dpos.Y * Dpos.Y < TollWid.X * TollWid.X + TollWid.Y * TollWid.Y)
                                    {
                                        //Unisci le due tracce
                                        int index_to_add = ((n_le > m_Tracks[j].Length) ? it : j);
                                        int index_to_remove = ((index_to_add == j) ? it : j);
                                        l = m_Tracks[index_to_remove].Length;
                                        for (h = 0; h < l; h++)
                                        {
                                            bool alreadythere = false;
                                            for (int u = 0; u < m_Tracks[index_to_add].Length; u++) if (m_Tracks[index_to_remove][0].LayerOwner.Id == m_Tracks[index_to_add][u].LayerOwner.Id) alreadythere = true;
                                            if (!alreadythere)
                                            {
                                                //Aggiunge ad una e rimuove automaticamente all'altra
                                                m_Tracks[index_to_add].AddSegment(m_Tracks[index_to_remove][0]);
                                            }
                                            else
                                            {
                                                //per ora si rimuove solo però si dovrà scegliere il segmento che 
                                                //più si adatta con un fit alla traccia che è stata prescelta per sopravvivere
                                                m_Tracks[index_to_remove].RemoveSegment(m_Tracks[index_to_remove][0].LayerOwner.Id);
                                            };
                                        };
                                        //La traccia del loop più esterno è stata rimossa
                                        //non ha più senso continuare con essa
                                        if (index_to_remove == j) break;
                                    };
                                };
                            };
                        };

                    };

                };
            };


            return null;
        }
        #endregion
    }

    #region Configuration

    /// <summary>
    /// Possible prescan modes.
    /// </summary>
    public enum PrescanModeValue : int
    {
        /// <summary>
        /// One large prescan zone is used, and rototranslation parameters are obtained by analysis of the density of matches.
        /// </summary>
        Rototranslation = 3,
        /// <summary>
        /// Four small prescan zones are used, the best three are selected, and the parameters for affine transformation are computed from local translations.
        /// </summary>
        Affine = 2,
        /// <summary>
        /// One small prescan zone is used and translation is estimated.
        /// </summary>
        Translation = 1,
        /// <summary>
        /// No prescan is performed.
        /// </summary>
        None = 0
    }

    /// <summary>
    /// Possible vertex algorithms.
    /// </summary>
    public enum VertexAlgorithm
    {
        /// <summary>
        /// No vertex reconstruction.
        /// </summary>
        None,
        /// <summary>
        /// Vertex algorithm that finds 2-track crossing and then merges them into vertices.
        /// </summary>
        PairBased,
        /// <summary>
        /// Vertex algorithm that finds multitrack vertices in a single pass.
        /// </summary>
        Global
    }

    /// <summary>
    /// Configuration for AlphaOmegaReconstructor.
    /// </summary>
    [Serializable]
    [XmlType("AlphaOmegaReconstruction.Configuration")]
    public class Configuration : SySal.Management.Configuration, ICloneable//, ISerializable
    {
        /// <summary>
        /// Enables Search for V-Topology Intersections.
        /// </summary>
        public bool TopologyV;
        /// <summary>
        /// Enables Search for Kink-Topology Intersections.
        /// </summary>
        public bool TopologyKink;
        /// <summary>
        /// Enables Search for X-Topology Intersections.
        /// </summary>
        public bool TopologyX;
        /// <summary>
        /// Enables Search for Y-Topology Intersections.
        /// </summary>
        public bool TopologyY;
        /// <summary>
        /// Enables Search for Lambda-Topology Intersections.
        /// </summary>
        public bool TopologyLambda;
        /// <summary>
        /// Minimum number of segments for  a track to be taken into account in the vertex reconstruction.
        /// </summary>
        public int MinVertexTracksSegments;
        /// <summary>
        /// Initial position tolerance at the beginning of the iteratively optimized linking procedure.
        /// </summary>
        public double Initial_D_Pos;
        /// <summary>
        /// Initial slope tolerance at the beginning of the iteratively optimized linking procedure.
        /// </summary>
        public double Initial_D_Slope;
        /// <summary>
        /// Maximum number of iterations.
        /// </summary>
        public int MaxIters;
        /// <summary>
        /// The number of additional linking passes to be done to build tracks. Setting it to <c>0</c> reproduces the old behaviour of AlphaOmegaReconstructor.
        /// </summary>
        public int ExtraTrackingPasses;
        /// <summary>
        /// Coefficient for linear dependence for position tolerance along iterative procedure.
        /// </summary>
        public double D_PosIncrement;
        /// <summary>
        /// Coefficient for linear dependence for slope tolerance along iterative procedure.
        /// </summary>
        public double D_SlopeIncrement;
        /// <summary>
        /// Position tolerance at the end of the iteratively optimized linking procedure.
        /// </summary>
        public double D_Pos;
        /// <summary>
        /// Slope tolerance at the end of the iteratively optimized linking procedure.
        /// </summary>
        public double D_Slope;
        /// <summary>
        /// Size of the locality cell to speed up linking.
        /// </summary>
        public double LocalityCellSize;
        /// <summary>
        /// Slope of the beam to be used for alignment.
        /// </summary>
        public SySal.BasicTypes.Vector2 AlignBeamSlope;
        /// <summary>
        /// Width of the beam to be used for alignment.
        /// </summary>
        public double AlignBeamWidth;
        /// <summary>
        /// If enabled, freezes z longitudinal coordinate to its nominal position.
        /// </summary>
        public bool FreezeZ;
        /// <summary>
        /// If enabled, correct slopes during alignment.
        /// </summary>
        public bool CorrectSlopesAlign;
        /// <summary>
        /// If enabled, only the segments that have been linked on previous layers are used to optimize the alignment.
        /// </summary>
        public bool AlignOnLinked;
        /// <summary>
        /// Maximum number of consecutively missing segments in a track.
        /// </summary>
        public int MaxMissingSegments;
        /// <summary>
        /// Selects the way to obtain a first rough alignment (prescan).
        /// </summary>
        public PrescanModeValue PrescanMode;
        /// <summary>
        /// When the prescan mode is <i>Affine</i>, this parameter is the distance between the centers of two zones during prescan.
        /// When the prescan mode is <i>Rototranslation</i>, this parameter is the increase in the mapping tolerance that is actually applied 
        /// (Rototranslation must account for large shifts due to rotations over large distances).
        /// </summary>
        public double LeverArm;
        /// <summary>
        /// Size of each prescan area (micron).
        /// </summary>
        public double ZoneWidth;
        /// <summary>
        /// Maximum misalignment allowed in each prescan zone (micron).
        /// </summary>
        public double Extents;
        /// <summary>
        /// Risk Factor to attach a background track to a high momentum track (normalized to 1).
        /// </summary>
        public double RiskFactor;
        /// <summary>
        /// Size of cell for slopes distribution (normalized to 1).
        /// </summary>
        public SySal.BasicTypes.Vector2 SlopesCellSize;
        /// <summary>
        /// Maximum shift detectable according to prescan procedure.
        /// </summary>
        public SySal.BasicTypes.Vector2 MaximumShift;
        /// <summary>
        /// Maximum closest mapproach between tracks (Pair Based Vertexing).
        /// </summary>
        public double CrossTolerance;
        /// <summary>
        /// Maximum longitudinal coordinate for intersection (Pair Based Vertexing).
        /// </summary>
        public double MaximumZ;
        /// <summary>
        /// Minimum longitudinal coordinate for intersection (Pair Based Vertexing).
        /// </summary>
        public double MinimumZ;
        /// <summary>
        /// Starting longitudinal tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public double StartingClusterToleranceLong;
        /// <summary>
        /// Maximum longitudinal tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public double MaximumClusterToleranceLong;
        /// <summary>
        /// Starting transverse tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public double StartingClusterToleranceTrans;
        /// <summary>
        /// Starting longitudinal tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public double MaximumClusterToleranceTrans;
        /// <summary>
        /// Starting longitudinal tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public int MinimumTracksPairs;
        /// <summary>
        /// Starting longitudinal tolerance for clusterizing intersections (Pair Based Vertexing).
        /// </summary>
        public int MinimumSegmentsNumber;

        /// <summary>
        /// If enabled, transformations are updated with long tracks.
        /// </summary>
        public bool UpdateTransformations;

        /// <summary>
        /// Matrix for tracks tracks intersections in vertex finding (Pair Based Vertexing).
        /// </summary>
        public double Matrix;
        /// <summary>
        /// Cell Size along X direction for tracks intersections in vertex finding (Pair Based Vertexing).
        /// </summary>
        public double XCellSize;
        /// <summary>
        /// Cell Size along Y direction for tracks intersections in vertex finding (Pair Based Vertexing).
        /// </summary>
        public double YCellSize;
        /// <summary>
        /// Cell Size along Z direction for tracks intersections in vertex finding (Pair Based Vertexing).
        /// </summary>
        public double ZCellSize;
        /// <summary>
        /// If enabled, cells are involved in tracks intersections for vertex finding (Pair Based Vertexing).
        /// </summary>
        public bool UseCells;

        /// <summary>
        /// If enabled, Kalman filter is applied to propagate tracks. Otherwise track-fit-option is applied.
        /// </summary>
        public bool KalmanFilter;
        /// <summary>
        /// Number of micro-track for track fitting when propagating a volume track in track-fit-option.
        /// </summary>
        public int FittingTracks;
        /// <summary>
        /// Minimum number of micro-track for kalman filter to be applied.
        /// </summary>
        public int MinKalman;
        /// <summary>
        /// Critical Parameter: minimum number of microtracks to form a volume track.
        /// </summary>
        public int MinimumCritical;
        /// <summary>
        /// Track filtering function. If not <c>null</c>, is used to check track compatibility with specified criteria. The track is kept if the value of the function is different from <c>0.0</c>.
        /// <remarks>The supported track filtering parameters are listed below:
        /// <para><list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>N</term><description>Number of segments in the vertex.</description></item>
        /// <item><term>M</term><description>Number of segments with <c>Sigma &lt; 0</c>.</description></item>
        /// <item><term>G</term><description>Total number of grains.</description></item>
        /// <item><term>A</term><description>Total area sum.</description></item>
        /// </list></para>
        /// </remarks>
        /// </summary>
        public string TrackFilter;

        /// <summary>
        /// If enabled, kink detection procedure is performed.
        /// </summary>
        public bool KinkDetection;
        /// <summary>
        /// Minimum segemnts for a track to take part in the kink detection procedure.
        /// </summary>
        public int KinkMinimumSegments;
        /// <summary>
        /// Minimum Delat slope between incoming and outgoing slopes of a track to take part to the kink detection procedure.
        /// </summary>
        public double KinkMinimumDeltaS;
        /// <summary>
        /// Threshold factor for tracks fit in kink detection
        /// </summary>
        public double KinkFactor;
        /// <summary>
        /// Threshold factor for tracks fit in kink detection
        /// </summary>
        public double FilterThreshold;
        /// <summary>
        /// Filter Length
        /// </summary>
        public int FilterLength;

        /// <summary>
        /// The vertex algorithm to be used.
        /// </summary>
        public VertexAlgorithm VtxAlgorithm = VertexAlgorithm.None;

        /// <summary>
        /// Maximum divergence (in slope) of tracks belonging to a vertex (Global Vertexing).
        /// </summary>
        public double GVtxMaxSlopeDivergence;
        /// <summary>
        /// Vertex radius for global vertex algorithm (Global Vertexing).
        /// </summary>
        public double GVtxRadius;
        /// <summary>
        /// Maximum track extrapolation depth for global vertex algorithm (Global Vertexing).
        /// </summary>
        public double GVtxMaxExt;
        /// <summary>
        /// Minimum number of base tracks for global vertex algorithm (Global Vertexing).
        /// </summary>
        public int GVtxMinCount;
        /// <summary>
        /// Vertex filtering function. If not <c>null</c>, is used to check vertex compatibility with specified criteria. The vertex is kept if the value of the function is different from <c>0.0</c>.
        /// <remarks>The supported vertex filtering parameters are listed below:
        /// <para><list type="table">
        /// <listheader><term>Name</term><description>Meaning</description></listheader>
        /// <item><term>N</term><description>Number of tracks in the vertex.</description></item>
        /// <item><term>ND</term><description>Number of downstream tracks.</description></item>
        /// <item><term>NU</term><description>Number of upstream tracks.</description></item>
        /// <item><term>PX</term><description>Absolute X position of the vertex.</description></item>
        /// <item><term>PY</term><description>Absolute Y position of the vertex.</description></item>
        /// <item><term>PZ</term><description>Absolute Z position of the vertex.</description></item>
        /// <item><term>RX</term><description>X position of the volume reference center.</description></item>
        /// <item><term>RY</term><description>Y position of the volume reference center.</description></item>
        /// <item><term>RZ</term><description>Z position of the volume reference center.</description></item>
        /// <item><term>LastPass</term><description><c>1.0</c> if the filter is being applied to finalize the vertex list, <c>0.0</c> if this is an intermediate processing step.</description></item>
        /// </list></para>
        /// </remarks>
        /// </summary>
        public string GVtxFilter;

        /// <summary>
        /// If true, weighted vertex fit is enabled (available only with Global Vertexing, ignored otherwise).
        /// </summary>
        public bool VtxFitWeightEnable;
        /// <summary>
        /// Convergence tolerance on weighted vertex fit (available only with Global Vertexing, ignored otherwise).
        /// </summary>
        public double VtxFitWeightTol;
        /// <summary>
        /// Optimization step in XY plane for weighted vertex fit (available only with Global Vertexing, ignored otherwise).
        /// </summary>
        public double VtxFitWeightOptStepXY;
        /// <summary>
        /// Optimization step in Z plane for weighted vertex fit (available only with Global Vertexing, ignored otherwise).
        /// </summary>
        public double VtxFitWeightOptStepZ;

        /// <summary>
        /// Maximum difference in slope between tracks to be linked together. Ignored if <c>RelinkEnable</c> is <c>false</c>.
        /// </summary>
        public double RelinkAperture;
        /// <summary>
        /// Maximum distance between the ends of two tracks to be linked together. Ignored if <c>RelinkEnable</c> is <c>false</c>.
        /// </summary>
        public double RelinkDeltaZ;
        /// <summary>
        /// Activates post-propagation track relinking if set to <c>true</c>.
        /// </summary>
        public bool RelinkEnable;

        /// <summary>
        /// Minimum position measurement error to consider to check track consistency.
        /// </summary>
        public double TrackCleanError;
        /// <summary>
        /// Maximum chi2 for tracks to survive cleaning. Set it to zero or a negative number to disable cleaning.
        /// </summary>
        public double TrackCleanChi2Limit;

        /// <summary>
        /// Minimum number of segments to use a volume track for alignment. Set to <c>0</c> to disable alignment with tracks.
        /// </summary>
        public int TrackAlignMinTrackSegments;
        /// <summary>
        /// Optimization step, in micron, for track-guided alignment translation.
        /// </summary>
        public double TrackAlignTranslationStep;
        /// <summary>
        /// Maximum translation allowed for track-guided alignment.
        /// </summary>
        public double TrackAlignTranslationSweep;
        /// <summary>
        /// Optimization step, in radians, for track-guided alignment rotation.
        /// </summary>
        public double TrackAlignRotationStep;
        /// <summary>
        /// Maximum rotation allowed for track-guided alignment.
        /// </summary>
        public double TrackAlignRotationSweep;
        /// <summary>
        /// Acceptance, in micron, for track-guided alignment.
        /// </summary>
        public double TrackAlignOptAcceptance;
        /// <summary>
        /// Minimum number of segments to align a layer with volume tracks.
        /// </summary>
        public int TrackAlignMinLayerSegments;
        
        /// <summary>
        /// If <c>true</c>, layers that cannot be aligned are skipped, but the reconstruction process does not abort; if <c>false</c>, the reconstruction process aborts if one or more layers cannot be aligned.
        /// </summary>
        public bool IgnoreBadLayers;

        /// <summary>
        /// Builds an unitialized configuration.
        /// </summary>
        public Configuration() : base("") { }

        /// <summary>
        /// Builds a configuration with the specified name.
        /// </summary>
        /// <param name="name"></param>
        public Configuration(string name) : base(name) { }

        /// <summary>
        /// Yields a copy of the configuration.
        /// </summary>
        /// <returns>the cloned configuration.</returns>
        public override object Clone()
        {
            Configuration C = new Configuration(Name);
            C.MinVertexTracksSegments = MinVertexTracksSegments;
            C.VtxAlgorithm = VtxAlgorithm;
            C.Initial_D_Pos = Initial_D_Pos;
            C.Initial_D_Slope = Initial_D_Slope;
            C.MaxIters = MaxIters;
            C.ExtraTrackingPasses = ExtraTrackingPasses;
            C.D_Pos = D_Pos;
            C.D_Slope = D_Slope;
            C.D_PosIncrement = D_PosIncrement;
            C.D_SlopeIncrement = D_SlopeIncrement;
            C.LocalityCellSize = LocalityCellSize;
            C.AlignBeamSlope.X = AlignBeamSlope.X;
            C.AlignBeamSlope.Y = AlignBeamSlope.Y;
            C.AlignBeamWidth = AlignBeamWidth;
            C.AlignOnLinked = AlignOnLinked;
            C.MaxMissingSegments = MaxMissingSegments;
            C.PrescanMode = PrescanMode;
            C.LeverArm = LeverArm;
            C.ZoneWidth = ZoneWidth;
            C.Extents = Extents;
            C.RiskFactor = RiskFactor;
            C.SlopesCellSize.X = SlopesCellSize.X;
            C.SlopesCellSize.Y = SlopesCellSize.Y;
            C.MaximumShift.X = MaximumShift.X;
            C.MaximumShift.Y = MaximumShift.Y;
            C.MaximumZ = MaximumZ;
            C.MinimumZ = MinimumZ;
            C.CrossTolerance = CrossTolerance;
            C.StartingClusterToleranceLong = StartingClusterToleranceLong;
            C.StartingClusterToleranceTrans = StartingClusterToleranceTrans;
            C.MaximumClusterToleranceLong = MaximumClusterToleranceLong;
            C.MaximumClusterToleranceTrans = MaximumClusterToleranceTrans;
            C.CorrectSlopesAlign = CorrectSlopesAlign;
            C.FreezeZ = FreezeZ;
            C.TopologyV = TopologyV;
            C.TopologyKink = TopologyKink;
            C.TopologyX = TopologyX;
            C.TopologyY = TopologyY;
            C.TopologyLambda = TopologyLambda;
            C.MinimumSegmentsNumber = MinimumSegmentsNumber;
            C.MinimumTracksPairs = MinimumTracksPairs;
            C.UpdateTransformations = UpdateTransformations;

            C.XCellSize = XCellSize;
            C.YCellSize = YCellSize;
            C.ZCellSize = ZCellSize;
            C.Matrix = Matrix;
            C.UseCells = UseCells;

            C.KalmanFilter = KalmanFilter;
            C.FittingTracks = FittingTracks;
            C.MinKalman = MinKalman;
            C.MinimumCritical = MinimumCritical;
            C.TrackFilter = (TrackFilter == null) ? null : (string)(TrackFilter.Clone());

            C.KinkDetection = KinkDetection;
            C.KinkMinimumSegments = KinkMinimumSegments;
            C.KinkMinimumDeltaS = KinkMinimumDeltaS;
            C.KinkFactor = KinkFactor;
            C.FilterThreshold = FilterThreshold;
            C.FilterLength = FilterLength;

            C.GVtxMaxExt = GVtxMaxExt;
            C.GVtxMinCount = GVtxMinCount;
            C.GVtxRadius = GVtxRadius;
            C.GVtxMaxSlopeDivergence = GVtxMaxSlopeDivergence;
            C.GVtxFilter = (GVtxFilter == null) ? null : (string)(GVtxFilter.Clone());

            C.VtxFitWeightEnable = VtxFitWeightEnable;
            C.VtxFitWeightOptStepXY = VtxFitWeightOptStepXY;
            C.VtxFitWeightOptStepZ = VtxFitWeightOptStepZ;
            C.VtxFitWeightTol = VtxFitWeightTol;

            C.RelinkAperture = RelinkAperture;
            C.RelinkDeltaZ = RelinkDeltaZ;
            C.RelinkEnable = RelinkEnable;

            C.TrackCleanError = TrackCleanError;
            C.TrackCleanChi2Limit = TrackCleanChi2Limit;

            C.TrackAlignMinLayerSegments = TrackAlignMinLayerSegments;
            C.TrackAlignMinTrackSegments = TrackAlignMinTrackSegments;
            C.TrackAlignOptAcceptance = TrackAlignOptAcceptance;
            C.TrackAlignRotationStep = TrackAlignRotationStep;
            C.TrackAlignRotationSweep = TrackAlignRotationSweep;
            C.TrackAlignTranslationStep = TrackAlignTranslationStep;
            C.TrackAlignTranslationSweep = TrackAlignTranslationSweep;
            C.IgnoreBadLayers = IgnoreBadLayers;

            return C;
        }
    }

    #endregion

    #region AlphaOmegaReconstructor
    /// <summary>
    /// Volume reconstruction 
    /// </summary>
    [Serializable]
    [XmlType("AlphaOmegaReconstruction.AlphaOmegaReconstructor")]
    public class AlphaOmegaReconstructor : IManageable, IVolumeReconstructor, IExposeInfo
    {
        #region Internals

        private int POS_ALIGN_DATA_LEN = 7;

        [NonSerialized]
        private AlphaOmegaReconstruction.Configuration C;

        [NonSerialized]
        private string intName;

        [NonSerialized]
        private dShouldStop intShouldStop;

        [NonSerialized]
        private dProgress intProgress;

        [NonSerialized]
        private dReport intReport;

        [NonSerialized]
        private SySal.Management.FixedConnectionList EmptyConnectionList = new SySal.Management.FixedConnectionList(new FixedTypeConnection.ConnectionDescriptor[0]);

        [NonSerialized]
        private Volume V = new Volume();

        #endregion

        #region Management

        /// <summary>
        /// Constructor. Builds an AlphaOmegaReconstructor with default configuration.
        /// </summary>
        public AlphaOmegaReconstructor()
        {
            C = new Configuration("Default AlphaOmega Configuration");
            C.VtxAlgorithm = VertexAlgorithm.Global;
            C.MinVertexTracksSegments = 3;
            C.TopologyKink = true;
            C.TopologyV = true;
            C.TopologyLambda = false;
            C.TopologyX = false;
            C.TopologyY = false;
            C.GVtxMaxExt = 3900.0;
            C.GVtxMinCount = 2;
            C.GVtxRadius = 30.0;
            C.GVtxMaxSlopeDivergence = 1.2;

            C.AlignBeamSlope.X = C.AlignBeamSlope.Y = 0.0;
            C.AlignBeamWidth = 1;

            C.D_Pos = 30;
            C.D_Slope = 0.03;
            C.D_PosIncrement = 20;
            C.D_SlopeIncrement = 0.025;
            C.Initial_D_Pos = 40;
            C.Initial_D_Slope = 0.04;

            C.AlignOnLinked = false;
            C.CorrectSlopesAlign = false;
            C.FreezeZ = false;

            C.LeverArm = 1500;
            C.ZoneWidth = 2500;
            C.Extents = 1000;
            C.LocalityCellSize = 250;
            C.MaximumShift.X = 1000;
            C.MaximumShift.Y = 1000;

            C.MaxIters = 5;
            C.ExtraTrackingPasses = 0;
            C.MaxMissingSegments = 3;

            C.PrescanMode = PrescanModeValue.Translation;

            C.RiskFactor = 0.01;
            C.SlopesCellSize.X = 0.05;
            C.SlopesCellSize.Y = 0.05;

            C.MaximumZ = 6000;
            C.MinimumZ = -6000;
            C.CrossTolerance = 10;
            C.StartingClusterToleranceLong = 300;
            C.StartingClusterToleranceTrans = 30;
            C.MaximumClusterToleranceLong = 600;
            C.MaximumClusterToleranceTrans = 60;

            C.UpdateTransformations = false;
            C.MinimumSegmentsNumber = 5;
            C.MinimumTracksPairs = 20;

            C.UseCells = true;
            C.XCellSize = 250;
            C.YCellSize = 250;
            C.ZCellSize = 1300;
            C.Matrix = 1.8;

            C.KalmanFilter = false;
            C.FittingTracks = 3;
            C.MinKalman = 4;
            C.MinimumCritical = 1;

            C.KinkDetection = false;
            C.KinkMinimumSegments = 6;
            C.KinkFactor = 1.1;
            C.KinkMinimumDeltaS = 0.02;
            C.FilterThreshold = 500;
            C.FilterLength = 10;

            C.VtxFitWeightEnable = false;
            C.VtxFitWeightOptStepXY = 1.0;
            C.VtxFitWeightOptStepZ = 5.0;
            C.VtxFitWeightTol = 0.1;

            C.RelinkAperture = 0.03;
            C.RelinkDeltaZ = 3000.0;
            C.RelinkEnable = false;

            C.TrackCleanError = 10.0;
            C.TrackCleanChi2Limit = 2.25;

            C.TrackAlignMinTrackSegments = 6;
            C.TrackAlignTranslationStep = 5.0;
            C.TrackAlignTranslationSweep = 100.0;
            C.TrackAlignRotationStep = 0.005;
            C.TrackAlignRotationSweep = 0.02;
            C.TrackAlignOptAcceptance = 15.0;
            C.TrackAlignMinLayerSegments = 3;
            C.IgnoreBadLayers = false;
        }

        /// <summary>
        /// Name of the AlphaOmegaReconstructor instance.
        /// </summary>
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

        /// <summary>
        /// Accesses the AlphaOmegaReconstructor's configuration.
        /// </summary>
        [XmlElement(typeof(AlphaOmegaReconstruction.Configuration))]
        public SySal.Management.Configuration Config
        {
            get
            {
                return (AlphaOmegaReconstruction.Configuration)C.Clone();
            }
            set
            {
                C = (AlphaOmegaReconstruction.Configuration)value.Clone();
            }
        }

        /// <summary>
        /// Allows the user to edit the supplied configuration.
        /// </summary>
        /// <param name="c">the configuration to be edited.</param>
        /// <returns><c>true</c> if the configuration has been modified, <c>false</c> otherwise.</returns>
        public bool EditConfiguration(ref SySal.Management.Configuration c)
        {
            bool ret;
            frmAORecEditConfig myform = new frmAORecEditConfig();
            myform.AOConfig = (AlphaOmegaReconstruction.Configuration)c.Clone();
            if ((ret = (myform.ShowDialog() == DialogResult.OK))) c = myform.AOConfig;
            myform.Dispose();
            return ret;
        }

        /// <summary>
        /// List of connections. It is always empty for AlphaOmegaReconstructors.
        /// </summary>
        [XmlIgnore]
        public IConnectionList Connections
        {
            get
            {
                return EmptyConnectionList;
            }
        }

        /// <summary>
        /// Monitor enable/disable. Monitoring is currently not supported (enabling the monitor results in an exception).
        /// </summary>
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

        #endregion

        #region VolumeReconstruction

        /// <summary>
        /// Callback delegate that can be used to stop the reconstruction process.
        /// </summary>
        public dShouldStop ShouldStop
        {
            get { return intShouldStop; }
            set { intShouldStop = value; }
        }

        /// <summary>
        /// Callback delegate that monitors the reconstruction progress (ranging from 0 to 1).
        /// </summary>
        public dProgress Progress
        {
            get { return intProgress; }
            set { intProgress = value; }
        }

        /// <summary>
        /// Callback delegate that monitors the reconstruction report .
        /// </summary>
        public dReport Report
        {
            get { return intReport; }
            set { intReport = value; }
        }

        /// <summary>
        /// Clears the reconstructor of previously loaded layers.
        /// </summary>
        public void Clear()
        {
            V = new Volume();
        }

        /// <summary>
        /// Adds one layer to the set of layers to use for the reconstruction.
        /// The layer should have been previously filled up with segments.
        /// </summary>
        /// <param name="l">the layer to be added.</param>
        public void AddLayer(SySal.TotalScan.Layer l)
        {
            V.AddLayer(l);
        }

        /// <summary>
        /// Adds one layer to the set of layers to use for the reconstruction, filling it with segments whose geometrical parameters are given by a set of MIPEmulsionTrackInfo.
        /// </summary>
        /// <param name="l">the layer to be added.</param>
        /// <param name="basetks">the base-tracks to be added.</param>        
        public void AddLayer(SySal.TotalScan.Layer l, MIPEmulsionTrackInfo[] basetks)
        {
            V.AddLayer(l, basetks);
        }
        /// <summary>
        /// Adds one layer to the set of layers to use for the reconstruction.
        /// The layer is filled up with tracks from the supplied scanning zone.
        /// This method is used to keep track of unassociated microtracks too, e.g. to search for kinks in the base.
        /// </summary>
        /// <param name="l">the layer to be added.</param>
        /// <param name="zone">the LinkedZone that provides the base-tracks.</param>
        public void AddLayer(SySal.TotalScan.Layer l, SySal.Scanning.Plate.LinkedZone zone)
        {
            V.AddLayer(l, zone);
        }

        /// <summary>
        /// Sets the list of segments to be ignored in alignment for one layer.
        /// </summary>
        /// <param name="layer">the zero-based index of the layer for which the list has to be set.</param>
        /// <param name="alignignorelist">the list of segments to be ignored.</param>
        public void SetAlignmentIgnoreList(int layer, int[] alignignorelist)
        {
            ((Layer)V.Layers[layer]).SetAlignmentIgnoreList(alignignorelist);
        }

        private static int OptimizeAlignment(double[] px, double[] py, double[] dx, double[] dy, double cx, double cy, double psweep, double rsweep, double pstep, double rstep, double pacc, int minentries, ref double[] outpar)
        {
            ArrayList bestsel = new ArrayList();
            int i, n;
            if ((n = px.Length) < minentries) return 0;
            double pxs, pys, rs;
            for (rs = -rsweep; rs <= rsweep; rs += rstep)
                for (pxs = -psweep; pxs <= psweep; pxs += pstep)
                    for (pys = -psweep; pys <= psweep; pys += pstep)
                    {
                        ArrayList sel = new ArrayList();
                        for (i = 0; i < n; i++)
                        {
                            if (Math.Abs(pxs + py[i] * rs - dx[i]) > pacc) continue;
                            if (Math.Abs(pys - px[i] * rs - dy[i]) > pacc) continue;
                            sel.Add(i);
                        }
                        if (sel.Count > bestsel.Count)
                            bestsel = sel;                        
                    }
            if ((n = bestsel.Count) < minentries) return 0;
            double[] fdx = new double[n];
            double[] fdy = new double[n];
            double[] fix = new double[n];
            double[] fiy = new double[n];
            for (i = 0; i < n; i++)
            {
                int j = (int)bestsel[i];
                fdx[i] = dx[j];
                fdy[i] = dy[j];
                fix[i] = px[j];
                fiy[i] = py[j];
            }
            NumericalTools.ComputationResult cr = NumericalTools.Fitting.Affine(fdx, fdy, fix, fiy, ref outpar);
            if (cr != NumericalTools.ComputationResult.OK) return 0;
            return n;
        }

        private static void WorkAlignment(ArrayList reflay, int lay, MIPEmulsionTrackInfo[][] aligntracks, bool [] layerstoignore, double cx, double cy, out double[] px, out double[] py, out double[] dx, out double[] dy)
        {
            string logfile = System.Environment.ExpandEnvironmentVariables("%TEMP%/aorec_wa.txt");
            ArrayList a = new ArrayList();            
            int i;
            double[] d;
            if (layerstoignore[lay] == false)
            {
                foreach (MIPEmulsionTrackInfo[] t in aligntracks)
                {
                    for (i = 1; i < t.Length; i++)
                    {
                        MIPEmulsionTrackInfo s = null, rs = null;
                        if (t[i - 1].Field == lay)
                        {
                            s = t[i - 1];
                            if (reflay.Contains((int)t[i].Field)) rs = t[i];
                            else continue;
                        }
                        else if (t[i].Field == lay)
                        {
                            s = t[i];
                            if (reflay.Contains((int)t[i - 1].Field)) rs = t[i - 1];
                            else continue;
                        }
                        else continue;
                        if (layerstoignore[rs.Field]) continue;
                        d = new double[4];
                        d[0] = s.Intercept.X - cx;
                        d[1] = s.Intercept.Y - cy;
                        d[2] = rs.Intercept.X - (rs.Intercept.Z - s.Intercept.Z) * s.Slope.X - s.Intercept.X;
                        d[3] = rs.Intercept.Y - (rs.Intercept.Z - s.Intercept.Z) * s.Slope.Y - s.Intercept.Y;
                        System.IO.File.AppendAllText(logfile, lay + " " + d[0] + " " + d[1] + " " + d[2] + " " + d[3] + "\r\n");
                        a.Add(d);
                    }
                }
            }
            px = new double[a.Count];
            py = new double[a.Count];
            dx = new double[a.Count];
            dy = new double[a.Count];
            for (i = 0; i < px.Length; i++)
            {
                d = (double[])a[i];
                px[i] = d[0];
                py[i] = d[1];
                dx[i] = d[2];
                dy[i] = d[3];
            }
        }

        private int CheckLayerMatches(Track[] aligntks, int minlayermatches, out MIPEmulsionTrackInfo [][] worktks, dReport report)
        {
            int [] mapnum = new int [V.Layers.Length];
            int i, j;
            worktks = new MIPEmulsionTrackInfo[aligntks.Length][];
            for (j = 0; j < aligntks.Length; j++)            
            {
                Track tk = aligntks[j];
                worktks[j] = new MIPEmulsionTrackInfo[tk.Length];
                for (i = 0; i < tk.Length; i++)
                {
                    SySal.TotalScan.Segment s = tk[i];
                    worktks[j][i] = s.Info;
                    worktks[j][i].Field = (uint)s.LayerOwner.Id;
                    mapnum[s.LayerOwner.Id]++;
                }
            }
            int bestlay = -1;
            int bestmatch = -1;
            int skippedlayers = 0;
            for (i = 0; i < mapnum.Length; i++)
            {
                if (report != null) report("Layer " + i + " matches " + mapnum[i]);
                if (mapnum[i] < minlayermatches)
                {
                    skippedlayers++;
                    if (report != null) report("Layer " + i + " (" + V.Layers[i].BrickId + "/" + V.Layers[i].SheetId + "/" + V.Layers[i].Side + ") has only " + mapnum[i] + " track(s) matching.");
                    continue;
                }
                if (mapnum[i] > bestmatch)
                {
                    bestlay = i;
                    bestmatch = mapnum[i];
                }
            }
            if (skippedlayers > 0 && (bestmatch < 0 || C.IgnoreBadLayers == false)) return -1;
            if (report != null) report("Using Layer " + bestlay + " (" + V.Layers[bestlay].BrickId + "/" + V.Layers[bestlay].SheetId + "/" + V.Layers[bestlay].Side + ") with " + bestmatch + " track(s) as reference.");
            return bestlay;
        }

        private void ApplyAlignmentSet(AlignmentData[] tad)
        {
            int i, j, n;            
            for (i = 0; i < tad.Length; i++)
            {
                AlignmentData ta = tad[i];
                Layer lay = (Layer)V.Layers[i];
                lay.RestoreOriginalSegments();
                n = lay.Length;
                for (j = 0; j < n; j++)
                {
                    MIPEmulsionTrackInfo ainfo = ((Segment)lay[j]).GetInfo();
                    MIPEmulsionTrackInfo oinfo = (MIPEmulsionTrackInfo)ainfo.Clone();
                    ainfo.Slope.X = ta.AffineMatrixXX * oinfo.Slope.X + ta.AffineMatrixXY * oinfo.Slope.Y;
                    ainfo.Slope.Y = ta.AffineMatrixYX * oinfo.Slope.X + ta.AffineMatrixYY * oinfo.Slope.Y;
                    ainfo.Intercept.X = ta.AffineMatrixXX * (oinfo.Intercept.X - V.RefCenter.X) + ta.AffineMatrixXY * (oinfo.Intercept.Y - V.RefCenter.Y) + ta.TranslationX + V.RefCenter.X;
                    ainfo.Intercept.Y = ta.AffineMatrixYX * (oinfo.Intercept.X - V.RefCenter.X) + ta.AffineMatrixYY * (oinfo.Intercept.Y - V.RefCenter.Y) + ta.TranslationY + V.RefCenter.Y;
                    ((Segment)lay[j]).DownstreamLinked = ((Segment)lay[j]).UpstreamLinked = null;
                }
                lay.iAlignmentData = tad[i];
                lay.MoveToDisk();
            }
        }

        /// <summary>
        /// Reconstructs volume tracks and optionally track intersections (vertices), using data that have been previously fed in through AddLayer.
        /// </summary>		
        public SySal.TotalScan.Volume Reconstruct()
        {
            bool[] layerstoignore = new bool[V.Layers.Length];
            if (C.TrackAlignMinTrackSegments <= 0)
            {
                V.AlignAndLink(C, false, false, layerstoignore, intShouldStop, intProgress, intReport);
                ((Volume.AOTrackList)V.Tracks).iItems = V.BuildTracks(C, intShouldStop, intProgress, intReport);
                if (C.UpdateTransformations)
                {
                    V.UpdateTransformations2(C, intShouldStop, intProgress, intReport);
                    ((Volume.AOTrackList)V.Tracks).iItems = V.RelinkTracks(V.PropagateTracks(((Volume.AOTrackList)V.Tracks).iItems, C, intShouldStop, intProgress, intReport), C, intShouldStop, intProgress, intReport);
                }
            }
            else
            {
                if (intReport != null) intReport("Switching to Track Alignment mode");
                Configuration C1 = (Configuration)C.Clone();
                C1.Initial_D_Pos = C1.D_Pos = C.TrackAlignTranslationStep;
                C1.D_Slope = C.Initial_D_Slope;
                V.AlignAndLink(C1, true, false, layerstoignore, intShouldStop, intProgress, intReport);
                if (C1.TrackFilter == null || C1.TrackFilter.Length == 0) C1.TrackFilter = "1";
                C1.TrackFilter = "(L >= " + C1.TrackAlignMinTrackSegments + ") && (" + C1.TrackFilter + ")";
                Track[] aligntracks = V.BuildTracks(C1, intShouldStop, intProgress, intReport);

                ArrayList reflay = new ArrayList(V.Layers.Length);
                MIPEmulsionTrackInfo[][] worktks = null;
                reflay.Add(CheckLayerMatches(aligntracks, C.TrackAlignMinLayerSegments, out worktks, intReport));
                AlignmentData[] tad = new AlignmentData[V.Layers.Length];
                if ((int)reflay[0] >= 0)
                {
                    AlignmentData ta = new AlignmentData();
                    ta.AffineMatrixXX = ta.AffineMatrixYY = 1.0;
                    ta.AffineMatrixXY = ta.AffineMatrixYX = 0.0;
                    ta.DShrinkX = ta.DShrinkY = 1.0;
                    ta.SAlignDSlopeX = ta.SAlignDSlopeY = 0.0;
                    ta.Result = MappingResult.OK;
                    tad[(int)reflay[0]] = ta;
                    int lay;
                    int skippedlayers = 0;
                    while (reflay.Count < V.Layers.Length)
                    {
                        lay = ((int)reflay[0] > (V.Layers.Length - (int)(reflay[reflay.Count - 1]) - 1)) ? ((int)reflay[0] - 1) : ((int)reflay[reflay.Count - 1] + 1);
                        double [] px;
                        double [] py;
                        double [] dx;
                        double [] dy;
                        WorkAlignment(reflay, lay, worktks, layerstoignore, V.RefCenter.X, V.RefCenter.Y, out px, out py, out dx, out dy);
                        if (px.Length <= C.TrackAlignMinLayerSegments)
                        {
                            if (intReport != null) intReport("Not enough matching tracks at layer " + lay + ".");
                            if (C.IgnoreBadLayers)
                            {
                                layerstoignore[lay] = true;
                                skippedlayers++;
                                ta = new AlignmentData();
                                ta.AffineMatrixXX = 1.0;
                                ta.AffineMatrixXY = 0.0;
                                ta.AffineMatrixYX = 0.0;
                                ta.AffineMatrixYY = 1.0;
                                ta.DShrinkX = ta.DShrinkY = 1.0;
                                ta.SAlignDSlopeX = ta.SAlignDSlopeY = 0.0;
                                ta.TranslationX = 0.0;
                                ta.TranslationY = 0.0;
                                ta.TranslationZ = 0.0;
                                ta.Result = MappingResult.InsufficientPrescan;
                                tad[lay] = ta;
                                reflay.Add(lay);
                                reflay.Sort();
                                continue;
                            }
                            else reflay.Clear();
                            break;
                        }
                        double [] outpar = new double[6];
                        int nmatch = OptimizeAlignment(px, py, dx, dy, V.RefCenter.X, V.RefCenter.Y, C.TrackAlignTranslationSweep, C.TrackAlignRotationSweep, C.TrackAlignTranslationStep, C.TrackAlignRotationStep, C.TrackAlignOptAcceptance, C.TrackAlignMinLayerSegments, ref outpar);
                        if (intReport != null)
                        {
                            intReport("Layer " + lay + ", segments suitable for alignment: " + nmatch);
                            if (nmatch > 0)
                                intReport("Alignment parameters: " + outpar[0] + "/" + outpar[1] + "/" + outpar[2] + "/" + outpar[3] + "/" + outpar[4] + "/" + outpar[5]);
                        }
                        ta = new AlignmentData();
                        ta.AffineMatrixXX = 1.0 + outpar[0];
                        ta.AffineMatrixXY = outpar[1];
                        ta.AffineMatrixYX = outpar[2];
                        ta.AffineMatrixYY = 1.0 + outpar[3];
                        ta.DShrinkX = ta.DShrinkY = 1.0;
                        ta.SAlignDSlopeX = ta.SAlignDSlopeY = 0.0;
                        ta.TranslationX = outpar[4];
                        ta.TranslationY = outpar[5];
                        ta.TranslationZ = 0.0;
                        ta.Result = MappingResult.OK;
                        tad[lay] = ta;                        

                        foreach (MIPEmulsionTrackInfo[] tk in worktks)
                        {
                            int i;
                            for (i = 0; i < tk.Length; i++)
                            {
                                MIPEmulsionTrackInfo oinfo = tk[i];
                                MIPEmulsionTrackInfo ainfo = (MIPEmulsionTrackInfo)oinfo.Clone();
                                ainfo.Slope.X = ta.AffineMatrixXX * oinfo.Slope.X + ta.AffineMatrixXY * oinfo.Slope.Y;
                                ainfo.Slope.Y = ta.AffineMatrixYX * oinfo.Slope.X + ta.AffineMatrixYY * oinfo.Slope.Y;
                                ainfo.Intercept.X = ta.AffineMatrixXX * (oinfo.Intercept.X - V.RefCenter.X) + ta.AffineMatrixXY * (oinfo.Intercept.Y - V.RefCenter.Y) + ta.TranslationX + V.RefCenter.X;
                                ainfo.Intercept.Y = ta.AffineMatrixYX * (oinfo.Intercept.X - V.RefCenter.X) + ta.AffineMatrixYY * (oinfo.Intercept.Y - V.RefCenter.Y) + ta.TranslationY + V.RefCenter.Y;
                                tk[i] = ainfo;
                            }
                        }
                        reflay.Add(lay);
                        reflay.Sort();
                    }

                    if (reflay.Count > 0)
                    {
                        if (intReport != null) intReport("Alignment using tracks succeeded, now applying transformations...");

                        ApplyAlignmentSet(tad);

                        if (intReport != null) intReport("Done, new track reconstruction will start.");

                        C1 = (Configuration)C.Clone();
                        C1.MaxIters = 1;
                        C1.Initial_D_Pos = C.D_Pos;
                        C.Initial_D_Slope = C.D_Slope;
                        V.AlignAndLink(C1, false, true, layerstoignore, intShouldStop, intProgress, intReport);
                        ((Volume.AOTrackList)V.Tracks).iItems = V.BuildTracks(C, intShouldStop, intProgress, intReport);
                    }
                    else ((Volume.AOTrackList)V.Tracks).iItems = aligntracks;
                }
                else ((Volume.AOTrackList)V.Tracks).iItems = aligntracks;
            }
            if (C.VtxAlgorithm != VertexAlgorithm.None)
            {
                IntersectionType[] ityp = null;
                if (C.VtxAlgorithm == VertexAlgorithm.PairBased)
                {
                    int i = 0;
                    if (C.TopologyKink) i++;
                    if (C.TopologyLambda) i++;
                    if (C.TopologyV) i++;
                    if (C.TopologyX) i++;
                    if (C.TopologyY) i++;
                    ityp = new IntersectionType[i];
                    i = 0;
                    if (C.TopologyKink) { ityp[i] = IntersectionType.Kink; i++; }
                    if (C.TopologyLambda) { ityp[i] = IntersectionType.Lambda; i++; }
                    if (C.TopologyV) { ityp[i] = IntersectionType.V; i++; }
                    if (C.TopologyX) { ityp[i] = IntersectionType.X; i++; }
                    if (C.TopologyY) ityp[i] = IntersectionType.Y;
                }

                ((Volume.AOVertexList)V.Vertices).iItems =
                    V.FullVertexReconstruction(V, (SySal.TotalScan.Track[])((Volume.AOTrackList)V.Tracks).iItems, ityp, C, intShouldStop, intProgress, intReport);
            }

            return V;
        }

        /// <summary>
        /// Reconstructs volume tracks and low momentum tracks, using data that have been previously fed in through AddLayer.
        /// </summary>
        public SySal.TotalScan.Volume ReconstructLowMomentumTracks()
        {
            bool[] li = new bool[V.Layers.Length];
            V.AlignAndLink(C, false, false, li, intShouldStop, intProgress, intReport);
            ((Volume.AOTrackList)V.Tracks).iItems = V.ReconstructLowMomentumTracks(C, intShouldStop, intProgress, intReport);
            return V;
        }

        /// <summary>
        /// Links layers added to the AlphaOmegaReconstruction.
        /// </summary>
        /// <param name="DownstreamLayerId">Id of the downstream layer to be linked.</param>
        /// <param name="UpstreamLayerId">Id of the upstream layer to be linked.</param>
        /// <param name="C">configuration to be used for linking.</param>
        public void LinkSegments(int DownstreamLayerId, int UpstreamLayerId, Configuration C)
        {
            V.LinkSegments(DownstreamLayerId, UpstreamLayerId, C, intShouldStop, intProgress, intReport, true, false);
        }

        /// <summary>
        /// Recomputes vertices on an existing Volume. Yields a new volume with new vertices, and possibly, also new tracks. Does not recompute layer-to-layer alignment.
        /// </summary>
        public SySal.TotalScan.Volume RecomputeVertices(SySal.TotalScan.Volume v)
        {
            int i = 0;
            if (C.TopologyKink) i++;
            if (C.TopologyLambda) i++;
            if (C.TopologyV) i++;
            if (C.TopologyX) i++;
            if (C.TopologyY) i++;
            IntersectionType[] ityp = new IntersectionType[i];
            i = 0;
            if (C.TopologyKink) { ityp[i] = IntersectionType.Kink; i++; }
            if (C.TopologyLambda) { ityp[i] = IntersectionType.Lambda; i++; }
            if (C.TopologyV) { ityp[i] = IntersectionType.V; i++; }
            if (C.TopologyX) { ityp[i] = IntersectionType.X; i++; }
            if (C.TopologyY) ityp[i] = IntersectionType.Y;


            //			((Volume.AOVertexList)V.Vertices).iItems = null;
            Volume nvol = new Volume(v.Layers, v.Tracks, v.Extents, v.RefCenter, v.Id);
            SySal.TotalScan.Track[] tracks = new SySal.TotalScan.Track[v.Tracks.Length];
            for (i = 0; i < tracks.Length; i++)
                tracks[i] = v.Tracks[i];
            ((Volume.AOVertexList)nvol.Vertices).iItems = nvol.FullVertexReconstruction(nvol, tracks, ityp, C, intShouldStop, intProgress, intReport);
            return nvol;
        }

        #endregion

        #region ExposeInfo

        [NonSerialized]
        bool ActivateExpose = false;

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
                try
                {
                    if (ActivateExpose)
                    {
                        System.Collections.ArrayList ar = V.GetExposedInfo();
                        ar.Add(V.m_AlignmentData);
                        return ar;
                    }
                    return new System.Collections.ArrayList();
                }
                catch (Exception)
                {
                    return new System.Collections.ArrayList();
                }
            }

        }

        #endregion

    }
    #endregion

}


