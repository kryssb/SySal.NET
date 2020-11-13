using System;
using System.Collections.Generic;
using System.Text;

namespace SySal.Executables.NExTScanner
{
    class ChainCluster
    {
        public SySal.Imaging.Cluster Cluster;
        public int Id;
        public double Z;
        public int Plane;
        public ChainCluster Next;
        public uint PreviousSize;
    }

    class BasicGrain3DMaker : SySal.Imaging.IGrain3DMaker
    {
        protected SySal.Executables.NExTScanner.ImagingConfiguration C = new ImagingConfiguration();

        protected SySal.Processing.QuickMapping.QuickMapper QM = new SySal.Processing.QuickMapping.QuickMapper();

        public bool OptimizeDemagCoefficient = false;

        public double[] DemagCoefficients = new double[0];
        public double[] MatchDX = new double[0];
        public double[] MatchDY = new double[0];

        public delegate bool dShouldStop();

        public dShouldStop ShouldStop = null;

        public delegate void dProgress(double x);

        public dProgress Progress = null;

        public SySal.Executables.NExTScanner.ImagingConfiguration Config
        {
            get { return C; }
            set
            {
                C = value;
            }
        }

        #region IGrain3DMaker Members

        public SySal.Imaging.Grain3D[] MakeGrainsFromClusters(SySal.Imaging.Cluster3D[][] cls, SySal.BasicTypes.Vector[] positions)
        {
            this.DemagCoefficients = new double[0];
            this.MatchDX = new double[0];
            this.MatchDY = new double[0];
            System.Collections.ArrayList adx = new System.Collections.ArrayList();
            System.Collections.ArrayList ady = new System.Collections.ArrayList();
            int i;
            int halfw = (int)(C.ImageWidth / 2);
            int halfh = (int)(C.ImageHeight / 2);
            ChainCluster[][] planes = new ChainCluster[positions.Length][];
            System.Collections.ArrayList demags = new System.Collections.ArrayList();
            int plane;            
            double c2x, c2y;
            double px0 = positions[0].X;
            double py0 = positions[0].Y;
            double pz0 = positions[0].Z;            
            for (plane = 0; plane < planes.Length; plane++)
            {
                if (Progress != null) Progress((double)plane / (double)planes.Length * 0.5);
                if (ShouldStop != null && ShouldStop()) return null;
                planes[plane] = new ChainCluster[cls[plane].Length];                
                SySal.Imaging.Cluster3D [] cplanes = cls[plane];
                double px = positions[plane].X;
                double py = positions[plane].Y;
                double pz = positions[plane].Z;
                double pxd = px - px0;
                double pyd = py - py0;
                double pzd = pz - pz0;
                double dmz = OptimizeDemagCoefficient ? 1 : Math.Pow(1.0 + C.DMagDZ, pzd);
                for (i = 0; i < planes[plane].Length; i++)
                {
                    ChainCluster cc = new ChainCluster();
                    cc.Cluster = cplanes[i].Cluster;
                    cc.Id = i;
                    cc.Plane = plane;                    
                    cc.Z = pz;                    
                    cc.Cluster.X = (cc.Cluster.X - halfw) * C.Pixel2Micron.X * dmz;// +px;
                    cc.Cluster.Y = (cc.Cluster.Y - halfh) * C.Pixel2Micron.Y * dmz;// +py;
                    c2y = cc.Cluster.X * cc.Cluster.X * C.XYCurvature * cc.Cluster.Y;
                    c2x = cc.Cluster.Y * cc.Cluster.Y * C.XYCurvature * cc.Cluster.X;
                    cc.Cluster.X += c2x;
                    cc.Cluster.Y += c2y;
                    cc.Z += C.ZCurvature * (cc.Cluster.X * cc.Cluster.X + cc.Cluster.Y * cc.Cluster.Y);
                    cc.Cluster.X += pxd;
                    cc.Cluster.Y += pyd;
                    planes[plane][i] = cc;
                }
            }
            if (Progress != null) Progress(0.5);

            for (i = 0; i < planes.Length - 1; i++)
            {
                if (Progress != null) Progress(0.5 + 0.5 * (double)i / (double)(planes.Length - 1));
                if (ShouldStop != null && ShouldStop()) return null;
                SySal.Tracking.MIPEmulsionTrackInfo[] t1 = new SySal.Tracking.MIPEmulsionTrackInfo[planes[i].Length];
                SySal.Tracking.MIPEmulsionTrackInfo[] t2 = new SySal.Tracking.MIPEmulsionTrackInfo[planes[i + 1].Length];
                int j;
                for (j = 0; j < t1.Length; j++)
                {
                    ChainCluster cc = planes[i][j];
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info.Count = (ushort)cc.Cluster.Area;
                    info.AreaSum = (uint)j;
                    info.Intercept.X = cc.Cluster.X;
                    info.Intercept.Y = cc.Cluster.Y;
                    info.Intercept.Z = 0.0;
                    info.Slope.X = info.Slope.Y = info.Slope.Z = 0.0;
                    t1[j] = info;
                }
                for (j = 0; j < t2.Length; j++)
                {
                    ChainCluster cc = planes[i + 1][j];
                    SySal.Tracking.MIPEmulsionTrackInfo info = new SySal.Tracking.MIPEmulsionTrackInfo();
                    info.Count = (ushort)cc.Cluster.Area;
                    info.AreaSum = (ushort)j;// cls.Id;
                    info.Intercept.X = cc.Cluster.X + 2.0 * (C.ClusterMatchMaxOffset + C.ClusterMatchPositionTolerance);
                    info.Intercept.Y = cc.Cluster.Y + 2.0 * (C.ClusterMatchMaxOffset + C.ClusterMatchPositionTolerance);
                    info.Intercept.Z = 0.0;
                    info.Slope.X = info.Slope.Y = info.Slope.Z = 0.0;
                    t2[j] = info;
                }
                try
                {
                    SySal.Processing.QuickMapping.Configuration qmc = (SySal.Processing.QuickMapping.Configuration)QM.Config;
                    qmc.FullStatistics = false;
                    qmc.UseAbsoluteReference = true;
                    qmc.PosTol = C.ClusterMatchPositionTolerance * 0.5;
                    qmc.SlopeTol = 1.0;
                    QM.Config = qmc;
                    int bkg = QM.Match(t1, t2, 0.0, C.ClusterMatchPositionTolerance * 0.25, C.ClusterMatchPositionTolerance * 0.25).Length;
                    for (j = 0; j < t2.Length; j++)
                    {
                        ChainCluster cc = planes[i + 1][j];
                        t2[j].Intercept.X = cc.Cluster.X;
                        t2[j].Intercept.Y = cc.Cluster.Y;
                    }
                    qmc.PosTol = C.ClusterMatchMaxOffset * 0.5;
                    QM.Config = qmc;
                    SySal.Scanning.PostProcessing.PatternMatching.TrackPair[] tp = QM.Match(t1, t2, 0.0, C.ClusterMatchMaxOffset * 0.25, C.ClusterMatchMaxOffset * 0.25);
                    double deltax = 0.0, deltay = 0.0, demagx = 0.0, demagy = 0.0;
                    double dummy = 0.0;
                    double q = (1.0 - (double)bkg / (double)tp.Length) * 0.25;
                    double[,] deltas = new double[tp.Length, 4];
                    for (j = 0; j < tp.Length; j++)
                    {
                        deltas[j, 0] = tp[j].Second.Info.Intercept.X;
                        deltas[j, 1] = tp[j].Second.Info.Intercept.Y;
                        deltas[j, 2] = tp[j].First.Info.Intercept.X - tp[j].Second.Info.Intercept.X;
                        deltas[j, 3] = tp[j].First.Info.Intercept.Y - tp[j].Second.Info.Intercept.Y;
                    }
                    int[] ids = NumericalTools.Fitting.PeakDataSel(deltas, new double[] { Math.Abs(halfw * C.Pixel2Micron.X) * 0.25, Math.Abs(halfh * C.Pixel2Micron.Y) * 0.25, C.ClusterMatchPositionTolerance * 0.5, C.ClusterMatchPositionTolerance * 0.5 }, -1.0, q);
                    double[] ws = new double[ids.Length];
                    double[] dws = new double[ids.Length];
                    for (j = 0; j < ids.Length; j++)
                    {
                        ws[j] = deltas[ids[j], 0];
                        dws[j] = deltas[ids[j], 2];
                    }
                    NumericalTools.ComputationResult resx = NumericalTools.Fitting.LinearFitSE(ws, dws, ref demagx, ref deltax, ref dummy, ref dummy, ref dummy, ref dummy, ref dummy);
                    for (j = 0; j < ids.Length; j++)
                    {
                        ws[j] = deltas[ids[j], 1];
                        dws[j] = deltas[ids[j], 3];
                    }
                    double dmagdeltaz = positions[i + 1].Z - positions[i].Z;
                    NumericalTools.ComputationResult resy = NumericalTools.Fitting.LinearFitSE(ws, dws, ref demagy, ref deltay, ref dummy, ref dummy, ref dummy, ref dummy, ref dummy);
                    if (resx == NumericalTools.ComputationResult.OK && resy == NumericalTools.ComputationResult.OK)
                    {
                        demags.Add(Math.Log(1.0 + demagx) / dmagdeltaz);
                        demags.Add(Math.Log(1.0 + demagy) / dmagdeltaz);
                    }
                    for (j = 0; j < t2.Length; j++)
                    {
                        ChainCluster cc = planes[i + 1][j];
                        t2[j].Intercept.X += deltax;
                        t2[j].Intercept.Y += deltay;
                        cc.Cluster.X += deltax;
                        cc.Cluster.Y += deltay;
                    }
                    qmc.PosTol = C.ClusterMatchPositionTolerance * 0.5;
                    QM.Config = qmc;
                    tp = QM.Match(t1, t2, 0.0, C.ClusterMatchPositionTolerance * 0.25, C.ClusterMatchPositionTolerance * 0.25);
                    foreach (SySal.Scanning.PostProcessing.PatternMatching.TrackPair tp1 in tp)
                        if (planes[i][tp1.First.Info.AreaSum].Next == null && planes[i + 1][tp1.Second.Info.AreaSum].PreviousSize == 0 &&
                            (planes[i][tp1.First.Info.AreaSum].Cluster.Area < planes[i + 1][tp1.Second.Info.AreaSum].Cluster.Area &&
                            planes[i][tp1.First.Info.AreaSum].Cluster.Area < planes[i][tp1.First.Info.AreaSum].PreviousSize) == false)
                        {                            
                            planes[i][tp1.First.Info.AreaSum].Next = planes[i + 1][tp1.Second.Info.AreaSum];
                            planes[i + 1][tp1.Second.Info.AreaSum].PreviousSize = planes[i][tp1.First.Info.AreaSum].Cluster.Area;
                            adx.Add(tp1.First.Info.Intercept.X - tp1.Second.Info.Intercept.X);
                            ady.Add(tp1.First.Info.Intercept.Y - tp1.Second.Info.Intercept.Y);
                        }
                }
                catch (Exception) { }
            }

            System.Collections.ArrayList achains = new System.Collections.ArrayList();
            System.Collections.ArrayList grains = new System.Collections.ArrayList();
            foreach (ChainCluster[] cplane in planes)
                foreach (ChainCluster cc in cplane)
                {
                    if (cc.Next != null) cc.Next.Id = -1;
                    if (cc.Id >= 0)
                        achains.Add(cc);
                }            
            foreach (ChainCluster cc in achains)
            {
                int totalc = 1;
                ChainCluster cc1 = cc.Next;
                while (cc1 != null)
                {
                    cc1 = cc1.Next;
                    totalc++;
                }
                SySal.Imaging.Cluster3D[] cl3d = new SySal.Imaging.Cluster3D[totalc];
                totalc = 0;
                cc1 = cc;
                uint gvolume = 0;
                while (cc1 != null)
                {
                    double dmz = OptimizeDemagCoefficient ? 1 : Math.Pow(1.0 + C.DMagDZ, positions[cc1.Plane].Z - positions[0].Z);
                    cl3d[totalc].Cluster = cc1.Cluster;
                    cl3d[totalc].Cluster.X = cl3d[totalc].Cluster.X / dmz + positions[0].X;
                    cl3d[totalc].Cluster.Y = cl3d[totalc].Cluster.Y / dmz + positions[0].Y;
                    cl3d[totalc].Layer = (uint)cc1.Plane;
                    cl3d[totalc].Z = cc1.Z;
                    gvolume += cc1.Cluster.Area;
                    cc1 = cc1.Next;
                    totalc++;                                        
                }
                if (gvolume < C.MinGrainVolume) continue;

                grains.Add(SySal.Imaging.Grain3D.FromClusterCenters(cl3d));
            }

            this.DemagCoefficients = (double[])demags.ToArray(typeof(double));
            this.MatchDX = (double[])adx.ToArray(typeof(double));
            this.MatchDY = (double[])ady.ToArray(typeof(double));
            if (Progress != null) Progress(1.0);
            return (SySal.Imaging.Grain3D[])grains.ToArray(typeof(SySal.Imaging.Grain3D));
        }

        #endregion
    }
}
