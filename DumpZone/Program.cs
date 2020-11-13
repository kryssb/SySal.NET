/*
 * 2008-08-21:
 * adjust track
*/
using System;
using System.Data;
using System.Data.Common;
using System.Collections.Generic;
using System.Text;

namespace DumpZone
{
    /// <summary>
    /// Adjust Track Parameter Set.
    /// </summary>
    [Serializable]
    public class AdjustTrackParameterSet
    {
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
    }

    class Program
    {
        private static bool MakeSlopeAdjustment = false;

        private static bool FullZoneDump = false;

        private static AdjustTrackParameterSet C = null;

        private static SySal.OperaDb.OperaDbConnection Conn = null;

        private static SySal.OperaDb.OperaDbTransaction Trans = null;

        private static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo = null;

        static void Main(string[] args)
        {
            if (args.Length > 0 && String.Compare(args[0], "/fulldump", true) == 0)
            {
                FullZoneDump = true;

                string[] oldargs = args;
                int i;
                args = new string[oldargs.Length - 1];
                for (i = 0; i < args.Length; i++)
                    args[i] = oldargs[i + 1];
                oldargs = null;
            }

            System.Xml.Serialization.XmlSerializer xmls = null;
            if (args.Length == 5)
            {
                MakeSlopeAdjustment = true;

                try
                {
                    xmls = new System.Xml.Serialization.XmlSerializer(typeof(AdjustTrackParameterSet));
                    C = new AdjustTrackParameterSet();
                    C = (AdjustTrackParameterSet)xmls.Deserialize(new System.IO.StreamReader(args[4]));
                    xmls = null;
                    C = null;
                }
                catch
                {
                    try
                    {
                        xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.Executables.BatchLink.Config));
                        SySal.Executables.BatchLink.Config config = (SySal.Executables.BatchLink.Config)xmls.Deserialize(new System.IO.StreamReader(args[4]));
                        xmls = null;
                        C = new AdjustTrackParameterSet();
                        C.BottomDeltaSlopeX = config.BottomDeltaSlopeX;
                        C.BottomDeltaSlopeY = config.BottomDeltaSlopeY;
                        C.BottomMultSlopeX = config.BottomMultSlopeX;
                        C.BottomMultSlopeY = config.BottomMultSlopeY;
                        C.TopDeltaSlopeX = config.TopDeltaSlopeX;
                        C.TopDeltaSlopeY = config.TopDeltaSlopeY;
                        C.TopMultSlopeX = config.TopMultSlopeX;
                        C.TopMultSlopeY = config.TopMultSlopeY;
                    }
                    catch
                    {
                        throw new Exception("correction path non correct");
                    }
                }
                string[] oldargs = args;
                int i;
                args = new string[oldargs.Length - 1];
                for (i = 0; i < args.Length; i++)
                    args[i] = oldargs[i];
                oldargs = null;
            }

            if (args.Length != 4)
            {
                Console.WriteLine("usage: DumpZone [/fulldump] <StartupFile path> <RWC path> <TLG path> <series> [XML correction path]");
                Console.WriteLine("XML correction syntax could be the batchlink configuration, otherwise");
                Console.WriteLine("XML correction syntax:");
                xmls = new System.Xml.Serialization.XmlSerializer(typeof(AdjustTrackParameterSet));
                System.IO.StringWriter ss = new System.IO.StringWriter();
                xmls.Serialize(ss, C);
                Console.WriteLine(ss.ToString());
                ss.Close();
                return;
            }

            string startupfile = args[0];
            string rwcpath = args[1];
            string tlgpath = args[2];
            int series = System.Convert.ToInt32(args[3]);

            xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo));
            StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)xmls.Deserialize(new System.IO.StreamReader(startupfile));
            xmls = null;

            Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
            Conn.Open();
            SySal.OperaDb.Schema.DB = Conn;

            Trans = Conn.BeginTransaction();

            SySal.Scanning.Plate.IO.OPERA.LinkedZone lz = null;
            try
            {
                lz = (SySal.Scanning.Plate.IO.OPERA.LinkedZone)SySal.OperaPersistence.Restore(tlgpath, typeof(SySal.Scanning.Plate.IO.OPERA.LinkedZone));

                if (MakeSlopeAdjustment == true && C != null)
                    AdjustSlopes(lz);

                DumpZone(tlgpath, lz, StartupInfo.Plate.BrickId, StartupInfo.Plate.PlateId, StartupInfo.ProcessOperationId, series, StartupInfo.RawDataPath, System.IO.File.GetCreationTime(rwcpath), System.DateTime.Now, Conn, Trans);

                Trans.Commit();
            }
            catch (Exception x)
            {
                if (Trans != null) Trans.Rollback();
                Console.WriteLine(x.Message);
            }
            Conn.Close();
        }

        class MIPEmulsionTrack : SySal.Scanning.MIPIndexedEmulsionTrack
        {
            public static void AdjustSlopes(SySal.Scanning.MIPIndexedEmulsionTrack t, double xslopemult, double yslopemult, double slopedx, double slopedy)
            {
                SySal.Tracking.MIPEmulsionTrackInfo info = MIPEmulsionTrack.AccessInfo(t);
                info.Slope.X = info.Slope.X * xslopemult + slopedx;
                info.Slope.Y = info.Slope.Y * yslopemult + slopedy;
            }
        }

        public static void AdjustSlopes(SySal.Scanning.Plate.IO.OPERA.LinkedZone lz)
        {
            int i, n;
            double multx, multy, deltax, deltay;

            multx = 1.0 / C.TopMultSlopeX;
            multy = 1.0 / C.TopMultSlopeY;
            deltax = -C.TopDeltaSlopeX;
            deltay = -C.TopDeltaSlopeY;
            n = lz.Top.Length;
            for (i = 0; i < n; i++)
            {
                MIPEmulsionTrack.AdjustSlopes(lz.Top[i], multx, multy, deltax, deltay);
            }

            multx = 1.0 / C.BottomMultSlopeX;
            multy = 1.0 / C.BottomMultSlopeY;
            deltax = -C.BottomDeltaSlopeX;
            deltay = -C.BottomDeltaSlopeY;
            n = lz.Bottom.Length;
            for (i = 0; i < n; i++)
            {
                MIPEmulsionTrack.AdjustSlopes(lz.Bottom[i], multx, multy, deltax, deltay);
            }
        }

        private static long DumpZone(string tlgpath, SySal.Scanning.Plate.IO.OPERA.LinkedZone lz, long db_brick_id, long db_plate_id, long db_procop_id, long series, string rawdatapath, DateTime starttime, DateTime endtime, SySal.OperaDb.OperaDbConnection conn, SySal.OperaDb.OperaDbTransaction trans)
        {
            try
            {
                long db_id_zone = 0;

                int s, i, n;
                double dz, basez;
                SySal.Scanning.Plate.Side side;

                SySal.DAQSystem.Scanning.IntercalibrationInfo transform = lz.Transform;
                double TDX = transform.TX - transform.MXX * transform.RX - transform.MXY * transform.RY;
                double TDY = transform.TY - transform.MYX * transform.RX - transform.MYY * transform.RY;

                //zone
                db_id_zone = SySal.OperaDb.Schema.TB_ZONES.Insert(db_brick_id, db_plate_id, db_procop_id, db_id_zone,
                    lz.Extents.MinX, lz.Extents.MaxX, lz.Extents.MinY, lz.Extents.MaxY,
                    tlgpath, starttime, endtime, series,
                    transform.MXX, transform.MXY, transform.MYX, transform.MYY, TDX, TDY);

                if (FullZoneDump == false)
                    return db_id_zone;

                //views
                for (s = 0; s < 2; s++)
                {
                    if (s == 0)
                    {
                        side = lz.Top;
                        basez = lz.Top.BottomZ;
                    }
                    else
                    {
                        side = lz.Bottom;
                        basez = lz.Bottom.TopZ;
                    }
                    n = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)side).ViewCount;
                    for (i = 0; i < n; i++)
                    {
                        SySal.Scanning.Plate.IO.OPERA.LinkedZone.View vw = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.Side)side).View(i);
                        SySal.OperaDb.Schema.TB_VIEWS.Insert(db_brick_id, db_id_zone, s + 1, 
                            //i + 1,
                            vw.Id + 1,
                            vw.TopZ, vw.BottomZ, vw.Position.X, vw.Position.Y);
                    }
                }
                SySal.OperaDb.Schema.TB_VIEWS.Flush();

                int TrackId = 0;
                int UpTrackId = 0;
                int DownTrackId = 0;

                SySal.Tracking.MIPEmulsionTrackInfo info = null;
                SySal.Tracking.MIPEmulsionTrackInfo tinfo = null;
                SySal.Tracking.MIPEmulsionTrackInfo binfo = null;

                //Basetracks
                for (i = 0; i < lz.Length; i++)
                {
                    if (lz[i].Info.Sigma >= 0)
                    {
                        info = lz[i].Info;
                        tinfo = lz[i].Top.Info;
                        binfo = lz[i].Bottom.Info;
                    }
                    else continue;

                    DownTrackId++;
                    basez = lz.Top.BottomZ;
                    dz = (basez - tinfo.Intercept.Z);
                    SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(db_brick_id, db_id_zone,
                        1, DownTrackId,
                        tinfo.Intercept.X + tinfo.Slope.X * dz,
                        tinfo.Intercept.Y + tinfo.Slope.Y * dz,
                        tinfo.Slope.X,
                        tinfo.Slope.Y,
                        tinfo.Count,
                        tinfo.AreaSum,
                        System.DBNull.Value,
                        tinfo.Sigma, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(lz.Top[i])).View.Id + 1);

                    UpTrackId++;
                    basez = lz.Bottom.TopZ;
                    dz = (basez - binfo.Intercept.Z);
                    SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(db_brick_id, db_id_zone,
                        2, UpTrackId,
                        binfo.Intercept.X + binfo.Slope.X * dz,
                        binfo.Intercept.Y + binfo.Slope.Y * dz,
                        binfo.Slope.X,
                        binfo.Slope.Y,
                        binfo.Count,
                        binfo.AreaSum,
                        System.DBNull.Value,
                        binfo.Sigma, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(lz.Bottom[i])).View.Id + 1);

                    TrackId++;
                    basez = ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(lz[i].Top)).View.BottomZ;
                    dz = 0; //TODO (basez - info.Intercept.Z);
                    SySal.OperaDb.Schema.TB_MIPBASETRACKS.Insert(db_brick_id, db_id_zone,
                        TrackId,
                        info.Intercept.X + info.Slope.X * dz,
                        info.Intercept.Y + info.Slope.Y * dz,
                        info.Slope.X,
                        info.Slope.Y,
                        info.Count,
                        info.AreaSum, System.DBNull.Value, info.Sigma,
                        1, DownTrackId, 2, UpTrackId);
                }

                //Microtracks
                for (i = 0; i < lz.Length; i++)
                {
                    if (lz[i].Info.Sigma >= 0)
                        continue;
                    else if (lz[i].Top.Info.Sigma >= 0)
                    {
                        tinfo = lz[i].Top.Info;

                        DownTrackId++;
                        basez = lz.Top.BottomZ;
                        dz = (basez - tinfo.Intercept.Z);
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(db_brick_id, db_id_zone,
                            1, DownTrackId,
                            tinfo.Intercept.X + tinfo.Slope.X * dz,
                            tinfo.Intercept.Y + tinfo.Slope.Y * dz,
                            tinfo.Slope.X,
                            tinfo.Slope.Y,
                            tinfo.Count,
                            tinfo.AreaSum,
                            System.DBNull.Value,
                            tinfo.Sigma, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(lz.Top[i])).View.Id + 1);
                    }
                    else if (lz[i].Bottom.Info.Sigma >= 0)
                    {
                        binfo = lz[i].Bottom.Info;
                        UpTrackId++;
                        basez = lz.Bottom.TopZ;
                        dz = (basez - binfo.Intercept.Z);
                        SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Insert(db_brick_id, db_id_zone,
                            2, UpTrackId,
                            binfo.Intercept.X + binfo.Slope.X * dz,
                            binfo.Intercept.Y + binfo.Slope.Y * dz,
                            binfo.Slope.X,
                            binfo.Slope.Y,
                            binfo.Count,
                            binfo.AreaSum,
                            System.DBNull.Value,
                            binfo.Sigma, ((SySal.Scanning.Plate.IO.OPERA.LinkedZone.MIPIndexedEmulsionTrack)(lz.Bottom[i])).View.Id + 1);

                    }
                }

                SySal.OperaDb.Schema.TB_MIPMICROTRACKS.Flush();
                SySal.OperaDb.Schema.TB_MIPBASETRACKS.Flush();

                return db_id_zone;
            }
            catch (Exception x)
            {
                throw x;
            }
        }
    }
}
