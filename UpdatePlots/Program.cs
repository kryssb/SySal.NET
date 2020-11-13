using System;
using System.Collections.Generic;
using System.Text;
using System.Xml;
using System.Xml.Serialization;
using ZoneStatus;

namespace UpdatePlots
{
    class Program
    {
        #region InternalParameters
        private static double MINX = -10000;
        private static double MINY = -10000;
        private static double MAXX = 130000;
        private static double MAXY = 100000;
        private static double NSIGMA = 3;
        #endregion

        private static SySal.OperaDb.OperaDbConnection Conn = null;
        
        private static SySal.DAQSystem.Drivers.ScanningStartupInfo StartupInfo = null;

        private static SySal.DAQSystem.Drivers.TaskProgressInfo ProgressInfo = null;

        private static System.Drawing.Bitmap gIm = null;

        private static System.Drawing.Graphics gMon = null;

        private static NumericalTools.Plot gPlot = null;

        private static int TotalStrips;

        private static int TotalCompletedStrips;

        private static StripMonitorArrayClass StripMonitorArray = null;

        private static StripStatusArrayClass StripStatusArray = null;

        static void Main(string[] args)
        {
            if (args.Length != 2)
            {
                Console.WriteLine("usage: UpdatePlots <StartupFile path> <ProgressFile path>");
                return;
            }

            string startupfile = args[0];
            string progressfile = args[1];
            if (System.IO.File.Exists(startupfile) == false)
            {
                Console.WriteLine("Startupfile not exists");
                return;
            }

            if (System.IO.File.Exists(progressfile) == false)
            {
                Console.WriteLine("Progressfile not exists");
                return;
            }

            try
            {
                gIm = new System.Drawing.Bitmap(500, 375);
                gMon = System.Drawing.Graphics.FromImage(gIm);
                gPlot = new NumericalTools.Plot();

                System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.ScanningStartupInfo));
                StartupInfo = (SySal.DAQSystem.Drivers.ScanningStartupInfo)xmls.Deserialize(new System.IO.StreamReader(startupfile));
                xmls = null;

                Conn = new SySal.OperaDb.OperaDbConnection(StartupInfo.DBServers, StartupInfo.DBUserName, StartupInfo.DBPassword);
                Conn.Open();
                long parentopid = System.Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("SELECT /*+INDEX (TB_PROC_OPERATIONS PK_PROC_OPERATIONS) */ ID_PARENT_OPERATION FROM TB_PROC_OPERATIONS WHERE ID = " + StartupInfo.ProcessOperationId, Conn, null).ExecuteScalar());
                if (StartupInfo.RawDataPath.IndexOf(parentopid.ToString()) < 0)
                {
                    StartupInfo.RawDataPath = StartupInfo.RawDataPath + "\\cssd_" + StartupInfo.Plate.BrickId + "_" + parentopid;
                }
                Conn.Close();

                xmls = new System.Xml.Serialization.XmlSerializer(typeof(SySal.DAQSystem.Drivers.TaskProgressInfo));
                ProgressInfo = (SySal.DAQSystem.Drivers.TaskProgressInfo)xmls.Deserialize(new System.IO.StreamReader(progressfile));

                XmlDocument xmldoc = new XmlDocument();
                xmldoc.LoadXml(ProgressInfo.CustomInfo.Replace('[', '<').Replace(']', '>'));
                System.Xml.XmlNode xmlprog = xmldoc.FirstChild;

                System.Xml.XmlNode xmlelem = xmlprog["ZoneInfos"];

                TotalStrips = xmlelem.ChildNodes.Count;
                StripStatusArray = new StripStatusArrayClass(TotalStrips, 0);
                StripMonitorArray = new StripMonitorArrayClass(TotalStrips);

                XmlNode xn = xmlelem.FirstChild;
                int Id;
                StripMonitor stripMonitor = null;

                while (xn != null)
                {
                    Id = System.Convert.ToInt32(xn["ID"].InnerText);
                    StripStatusArray[Id - 1].Id = Id;
                    StripStatusArray[Id - 1].MaxTrials = System.Convert.ToUInt32(xn["MaxTrials"].InnerText);
                    StripStatusArray[Id - 1].Scanned = System.Convert.ToBoolean(xn["Scanned"].InnerText);
                    StripStatusArray[Id - 1].Monitored = System.Convert.ToBoolean(xn["Monitored"].InnerText);
                    StripStatusArray[Id - 1].MonitoringFile = System.Convert.ToString(xn["MonitoringFile"].InnerText);
                    StripStatusArray[Id - 1].Completed = System.Convert.ToBoolean(xn["Completed"].InnerText);

                    try
                    {
                        if (System.IO.File.Exists(StripStatusArray[Id - 1].MonitoringFile) == true)
                        {
                            stripMonitor = new StripMonitor();
                            stripMonitor.Read(StripStatusArray[Id - 1].MonitoringFile);
                            StripMonitorArray[Id - 1] = stripMonitor;
                        }
                    }
                    catch (Exception x)
                    {
                        if (stripMonitor != null) stripMonitor = null;
                        Console.WriteLine(x.Message);
                    }
                    xn = xn.NextSibling;
                }
                ProgressInfo.ExitException = null;

                TotalCompletedStrips = StripMonitorArray.Count;

                if (TotalCompletedStrips > 0)
                    UpdatePlots();

            }
            catch (Exception x)
            {
                Console.WriteLine(x.Message);
                throw x;
            }
        }

        private static void UpdateArrayExtremals(long index, ref double max, ref double min, double value, double[] array)
        {
            array[index] = value;
            if (array[index] > max)
                max = array[index];
            else if (array[index] < min)
                min = array[index];
        }

        private static void UpdatePlots()
        {
            int i, j;
            //View plot edges 
            double mincount=0, maxcount=0;
            double maxtoptracks = 0, mintoptracks = 0;
            double maxbottracks = 0, minbottracks = 0;
            double maxzlevel = 0, minzlevel = 0;
            double minxpos = 0, maxxpos = 0;
            double minypos = 0, maxypos = 0;
            //Strip plot edges 
            double mincountstrips = 0, maxcountstrips = 0;
            double mintopthickness = 0, maxtopthickness = 0;
            double minbotthickness = 0, maxbotthickness = 0;
            double minbasethickness = 0, maxbasethickness = 0;
            double minnumbasetracks = 0, maxnumbasetracks = 0;
            double mintimespan = 0, maxtimespan = 0;

            System.TimeSpan timeSpan = new TimeSpan();
            double x, y;
            double topLowerLimit = 0;
            double topEmptyViews = 0;
            double botLowerLimit = 0;
            double botEmptyViews = 0;
            
            double topGaussianArea = 0;
            double botGaussianArea = 0;

            StripMonitor strip = null;

            uint TotalNumberOfZone = 0;
            uint TotalNumberOfView = 0;
            for (j = 0; j < TotalStrips; j++)
            {
                strip = (StripMonitor)StripMonitorArray[j];
                if (strip == null) continue;
                try
                {
                    if (System.IO.File.Exists(StripStatusArray[strip.Id - 1].MonitoringFile) == false)
                        continue;
                }
                catch
                {
                    continue;
                }
                TotalNumberOfZone++;
                TotalNumberOfView += strip.Length;
            }

            //Initialize
            if (TotalCompletedStrips != 0)
            {
#if true
                int ii = 0;
                while (strip == null && ii < TotalStrips)
                {
                    strip = (StripMonitor)StripMonitorArray[ii];
                    ii++;
                }

                mincountstrips = maxcountstrips = strip.Id; //TODO
                minnumbasetracks = maxnumbasetracks = strip.Nbasetracks;
                timeSpan = strip.EndTime - strip.StartTime;
                mintimespan = maxtimespan = timeSpan.TotalSeconds;

                mincount = maxcount = 1;
                mintopthickness = maxtopthickness = strip[0].TopLayerThickness;
                minbotthickness = maxbotthickness = strip[0].BottomLayerThickness;
                minbasethickness = maxbasethickness = strip[0].PlasticBaseThickness;

                maxtoptracks = mintoptracks = strip[0].TopTracks;
                maxbottracks = minbottracks = strip[0].BotTracks;
                maxzlevel = minzlevel = strip[0].Z;

                x = (strip[0].X < MINX || strip[0].X > MAXX) ? MINX : strip[0].X;
                y = (strip[0].Y < MINY || strip[0].Y > MAXY) ? MINY : strip[0].Y;

                minxpos = maxxpos = x;
                minypos = maxypos = y;
#else
                mincountstrips = maxcountstrips = 0;
                minnumbasetracks = maxnumbasetracks = 0;

                mincount = maxcount = 1;
                mintopthickness = maxtopthickness = 0;
                minbotthickness = maxbotthickness = 0;
                minbasethickness = maxbasethickness = 0;

                maxtoptracks = mintoptracks = 0;
                maxbottracks = minbottracks = 0;
                maxzlevel = minzlevel = 0;
                minxpos = maxxpos = 0;
                minypos = maxypos = 0;
#endif
            }

            //View containers
            double[] counts = new double[TotalNumberOfView];
            double[] toptracks = new double[TotalNumberOfView];
            double[] bottracks = new double[TotalNumberOfView];
            double[] zlevel = new double[TotalNumberOfView];
            double[] xpos = new double[TotalNumberOfView];
            double[] ypos = new double[TotalNumberOfView];
            double[] topthickness = new double[TotalNumberOfView];
            double[] bottomthickness = new double[TotalNumberOfView];
            double[] basethickness = new double[TotalNumberOfView];

            //Strip containers
            double[] countstrips = new double[TotalNumberOfZone];
            double[] numbasetracks = new double[TotalNumberOfZone];
            double[] timespans = new double[TotalNumberOfZone];

            int zoneIndex = 0;
            long viewIndex = 0;
            for (j = 0; j < TotalStrips; j++)
            {
                strip = (StripMonitor)StripMonitorArray[j];
                if (strip == null) continue;

                timeSpan = strip.EndTime - strip.StartTime;
                if (StripStatusArray[strip.Id - 1].Scanned == false) continue;
                try
                {
                    if (System.IO.File.Exists(StripStatusArray[strip.Id - 1].MonitoringFile) == false)
                        continue;
                }
                catch
                {
                    continue;
                }

                UpdateArrayExtremals(zoneIndex, ref maxcountstrips, ref mincountstrips, strip.Id, countstrips);
                UpdateArrayExtremals(zoneIndex, ref maxnumbasetracks, ref minnumbasetracks, strip.Nbasetracks, numbasetracks);
                UpdateArrayExtremals(zoneIndex, ref maxtimespan, ref mintimespan, timeSpan.TotalSeconds, timespans);

                zoneIndex++;

                //View
                for (i = 0; i < strip.Length; i++)
                {
                    if (strip[i] == null) continue;
                    UpdateArrayExtremals(viewIndex, ref maxcount, ref mincount, viewIndex, counts); //TODO: time index
                    UpdateArrayExtremals(viewIndex, ref maxtoptracks, ref mintoptracks, strip[i].TopTracks, toptracks);
                    UpdateArrayExtremals(viewIndex, ref maxbottracks, ref minbottracks, strip[i].BotTracks, bottracks);
                    UpdateArrayExtremals(viewIndex, ref maxzlevel, ref minzlevel, strip[i].Z, zlevel);

                    x = (strip[i].X < MINX || strip[i].X > MAXX) ? MINX : strip[i].X;
                    UpdateArrayExtremals(viewIndex, ref maxxpos, ref minxpos, x, xpos);

                    y = (strip[i].Y < MINY || strip[i].Y > MAXY) ? MINY : strip[i].Y;
                    UpdateArrayExtremals(viewIndex, ref maxypos, ref minypos, y, ypos);

                    UpdateArrayExtremals(viewIndex, ref maxtopthickness, ref mintopthickness, strip[i].TopLayerThickness, topthickness);
                    UpdateArrayExtremals(viewIndex, ref maxbotthickness, ref minbotthickness, strip[i].BottomLayerThickness, bottomthickness);
                    UpdateArrayExtremals(viewIndex, ref maxbasethickness, ref minbasethickness, strip[i].PlasticBaseThickness, basethickness);

                    viewIndex++;
                }
            }

            //TopMicro distribution
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = toptracks;
                gPlot.SetXDefaultLimits = false;
                gPlot.MinX = mintoptracks - 1.0;
                gPlot.MaxX = maxtoptracks + 1.0;
                gPlot.DX = (float)((gPlot.MaxX - gPlot.MinX) * 0.01);
                gPlot.XTitle = "Toptracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 0.0;
                gPlot.Histo(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_00_dist.png", System.Drawing.Imaging.ImageFormat.Png);

                double[] par = gPlot.FitPar;
                topLowerLimit = par[2] - NSIGMA * par[3];

                for (int iv = 0; iv < TotalNumberOfView; iv++)
                    if (toptracks[iv] < topLowerLimit) topEmptyViews++;

                topGaussianArea = Math.Sqrt(Math.PI) * par[3] * TotalNumberOfView;

            }
            catch (Exception) { }

            //BotMicro distribution
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = bottracks;
                gPlot.SetXDefaultLimits = false;
                gPlot.MinX = minbottracks - 1.0;
                gPlot.MaxX = maxbottracks + 1.0;
                gPlot.DX = (float)((gPlot.MaxX - gPlot.MinX) * 0.01);
                gPlot.XTitle = "Bottracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 0.0;
                gPlot.Histo(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_01_dist.png", System.Drawing.Imaging.ImageFormat.Png);

                double[] par = gPlot.FitPar;
                botLowerLimit = par[2] - NSIGMA * par[3];

                for (int iv = 0; iv < TotalNumberOfView; iv++)
                    if (bottracks[iv] < botLowerLimit) botEmptyViews++;

                botGaussianArea = Math.Sqrt(Math.PI) * par[3] * TotalNumberOfView;
            }
            catch (Exception) { }

            //TopMicro vs ViewID
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num #";
                gPlot.VecY = toptracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = mintoptracks - 1.0;
                gPlot.MaxY = maxtoptracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Toptracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_00.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //TopMicro vs X
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = xpos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minxpos - 1.0;
                gPlot.MaxX = maxxpos + 1.0;
                gPlot.XTitle = "X View (micron)";
                gPlot.VecY = toptracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = mintoptracks - 1.0;
                gPlot.MaxY = maxtoptracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Toptracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_00_vs_X.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //TopMicro vs Y
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = ypos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minypos - 1.0;
                gPlot.MaxX = maxypos + 1.0;
                gPlot.XTitle = "Y View (micron)";
                gPlot.VecY = toptracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = mintoptracks - 1.0;
                gPlot.MaxY = maxtoptracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Toptracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_00_vs_Y.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //BotMicro vs ViewID
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num ";
                gPlot.VecY = bottracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minbottracks - 1.0;
                gPlot.MaxY = maxbottracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Bottracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_01.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //BotMicro vs X
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = xpos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minxpos - 1.0;
                gPlot.MaxX = maxxpos + 1.0;
                gPlot.XTitle = "X (micron)";
                gPlot.VecY = bottracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minbottracks - 1.0;
                gPlot.MaxY = maxbottracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Bottracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_01_vs_X.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //BotMicro vs Y
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = ypos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minypos - 1.0;
                gPlot.MaxX = maxypos + 1.0;
                gPlot.XTitle = "Y (micron)";
                gPlot.VecY = bottracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minbottracks - 1.0;
                gPlot.MaxY = maxbottracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Bottracks";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_01_vs_Y.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //Z level
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num #";
                gPlot.VecY = zlevel;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minzlevel - 1.0;
                gPlot.MaxY = maxzlevel + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Z (micron)";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_04.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //Base thickness
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num #";
                gPlot.VecY = basethickness;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minbasethickness - 1.0;
                gPlot.MaxY = maxbasethickness + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Base Thickness";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_05.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //Top layer thickness
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num #";
                gPlot.VecY = topthickness;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = mintopthickness - 1.0;
                gPlot.MaxY = maxtopthickness + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "TOP Thickness";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_06.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //Bot layer thickness
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = counts;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincount - 1.0;
                gPlot.MaxX = maxcount + 1.0;
                gPlot.XTitle = "View num #";
                gPlot.VecY = bottomthickness;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minbotthickness - 1.0;
                gPlot.MaxY = maxbotthickness + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Bot Thickness";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_07.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = countstrips;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincountstrips - 1.0;
                gPlot.MaxX = maxcountstrips + 1.0;
                gPlot.XTitle = "Strip num #";
                gPlot.VecY = numbasetracks;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minnumbasetracks - 1.0;
                gPlot.MaxY = maxnumbasetracks + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Basetrack number";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_08.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //weighted TopMicro scatter plot
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.Palette = NumericalTools.Plot.PaletteType.RGBContinuous;
                gPlot.VecX = xpos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minxpos - 1.0;
                gPlot.MaxX = maxxpos + 1.0;
                gPlot.XTitle = "X View (micron)";
                gPlot.VecY = ypos;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minypos - 1.0;
                gPlot.MaxY = maxypos + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "TopTracks Y View (micron)";
                gPlot.VecZ = toptracks;
                gPlot.MinZ = mintoptracks - 1.0;
                gPlot.MaxZ = maxtoptracks + 1.0;
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.ScatterHue(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_02.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            //weighted BottomMicro scatter plot
            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.Palette = NumericalTools.Plot.PaletteType.RGBContinuous;
                gPlot.VecX = xpos;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = minxpos - 1.0;
                gPlot.MaxX = maxxpos + 1.0;
                gPlot.XTitle = "X View (micron)";
                gPlot.VecY = ypos;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = minypos - 1.0;
                gPlot.MaxY = maxypos + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Bottracks Y View (micron)";
                gPlot.VecZ = bottracks;
                gPlot.MinZ = minbottracks - 1.0;
                gPlot.MaxZ = maxbottracks + 1.0;
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.ScatterHue(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_03.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = countstrips;
                gPlot.SetXDefaultLimits = false;
                gPlot.DX = 1.0f;
                gPlot.MinX = mincountstrips - 1.0;
                gPlot.MaxX = maxcountstrips + 1.0;
                gPlot.XTitle = "Strip num #";
                gPlot.VecY = timespans;
                gPlot.SetYDefaultLimits = false;
                gPlot.MinY = mintimespan - 1.0;
                gPlot.MaxY = maxtimespan + 1.0;
                gPlot.DY = (float)((gPlot.MaxY - gPlot.MinY) * 0.1);
                gPlot.YTitle = "Zone time duration (second)";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Scatter(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_09.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            try
            {
                gMon.Clear(System.Drawing.Color.White);
                gPlot.VecX = timespans;
                gPlot.SetXDefaultLimits = false;
                gPlot.MinX = mintimespan - 1.0;
                gPlot.MaxX = maxtimespan + 1.0;
                gPlot.DX = (float)((gPlot.MaxY - gPlot.MinY) * 0.01);
                gPlot.XTitle = "Zone time duration (second)";
                gPlot.PanelFormat = "F0";
                gPlot.PanelX = 1.0;
                gPlot.PanelY = 1.0;
                gPlot.Histo(gMon, gIm.Width, gIm.Height);
                gIm.Save(StartupInfo.RawDataPath + "\\" + StartupInfo.ProcessOperationId + "_10.png", System.Drawing.Imaging.ImageFormat.Png);
            }
            catch (Exception) { }

            System.IO.StreamWriter w = null;
            try
            {
                double percent = (double)TotalNumberOfZone / (double)TotalStrips * 100;
                double percentTopEmptyViews = (double)topEmptyViews / (double)TotalNumberOfView * 100;
                double percentBotEmptyViews = (double)botEmptyViews / (double)TotalNumberOfView * 100;

                double percentTopOutOfGaussian = (double)(TotalNumberOfView - topGaussianArea) / (double)TotalNumberOfView * 100;
                double percentBotOutOfGaussian = (double)(TotalNumberOfView - botGaussianArea) / (double)TotalNumberOfView * 100;

                double elapsedTime = 0;
                for (int it = 0; it < timespans.Length; it++)
                    elapsedTime += timespans[it];                
                double timeToComplete = (timespans.Length != 0) ? (double)(elapsedTime / timespans.Length) * (double)(TotalStrips - timespans.Length) : 0;

                System.TimeSpan elapsedTimeSpan = new TimeSpan(0, 0, (int)elapsedTime);
                System.TimeSpan timeStanToComplete = new TimeSpan(0, 0, (int)timeToComplete);

                w = new System.IO.StreamWriter(StartupInfo.RawDataPath + "\\Monitoring_" + StartupInfo.Plate.BrickId + "_" + StartupInfo.ProcessOperationId + "_progress.htm");
                w.WriteLine(
                    "<html><head>" + ((TotalNumberOfZone < TotalStrips) ? "<meta http-equiv=\"REFRESH\" content=\"60\">" : "") + "<title>WASD Monitor</title></head><body>\r\n" +
                    "<div align=center><p><font face=\"Arial, Helvetica\" size=4 color=4444ff>WASD Brick #" + StartupInfo.Plate.BrickId + ", Plate #" + StartupInfo.Plate.PlateId + "<br>Operation ID = " + StartupInfo.ProcessOperationId + "</font><hr></p></div>\r\n" +
                    "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Total = " + TotalStrips + "<br>Completed = " + TotalNumberOfZone + " (" + percent.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +
                    "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Top views below " + NSIGMA + " sigma from fit mean value = " + topEmptyViews + "<br>Completed = " + TotalNumberOfView + " (" + percentTopEmptyViews.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +
                    //"<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Top views Gaussian area = " + topGaussianArea.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) + " (" + percentTopOutOfGaussian.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +

                    "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Bot views below " + NSIGMA + " sigma from fit mean value = " + botEmptyViews + "<br>Completed = " + TotalNumberOfView + " (" + percentBotEmptyViews.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +
                    //"<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Bot views Gaussian area = " + botGaussianArea.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) + " (" + percentBotOutOfGaussian.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "%)</font></p>\r\n" +

                    "<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Effective elapsed time = " + (elapsedTime/60).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " minute<br>Effective time to complete = " + (timeToComplete/60).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " minute</font></p>\r\n" +
                    //"<div align=center><p><font face = \"Arial, Helvetica\" size=2 color=0000cc>Effective elapsed time = " + elapsedTimeSpan.ToString() + "<br>Effective time to complete = " + timeStanToComplete.ToString() + "</font></p>\r\n" +
                     "<table border=1 align=center>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_00_dist.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_01_dist.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_00.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_01.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_00_vs_X.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_00_vs_Y.png\" border=0></td></tr>\r\n" +
                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_01_vs_X.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_01_vs_Y.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_02.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_03.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_04.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_05.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_06.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_07.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_08.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + "_09.png\" border=0></td></tr>\r\n" +

                    "<tr><td><img src=\"" + StartupInfo.ProcessOperationId + "_09.png\" border=0></td>\r\n" +
                    "<td><img src=\"" + StartupInfo.ProcessOperationId + "_10.png\" border=0></td></tr>\r\n" +

                    "</table></div></body></html>"
                    );
                w.Flush();
            }
            catch (Exception) { }
            finally
            {
                if (w != null) w.Close();
            }
        }
    }
}
