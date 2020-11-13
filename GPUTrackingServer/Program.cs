using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.GPUTrackingServer
{
    [Serializable]
    public class PostProcessingInfo
    {
        public string Name;
        public string Settings;
    }

    [Serializable]
    public class Zone
    {
        public struct ProgressInfo
        {
            public uint BottomStripsReady;
            public uint TopStripsReady;
        };

        [NonSerialized]
        public ProgressInfo Progress;

        public uint Strips;
        public uint Views;
        public bool HasTop;
        public bool HasBottom;
        public SySal.BasicTypes.Rectangle ScanRectangle;
        public SySal.BasicTypes.Rectangle TransformedScanRectangle;
        public SySal.DAQSystem.Scanning.IntercalibrationInfo PlateMap;
        public SySal.BasicTypes.Vector ImageDelta;
        public SySal.BasicTypes.Vector2 ViewDelta;
        public SySal.BasicTypes.Vector2 StripDelta;
        public string ScanSettings;
        public string FileNameTemplate;
        public string OutputRWDFileName = "";

        public PostProcessingInfo[] PostProcessingSettings;

        static System.Xml.Serialization.XmlSerializer s_xmls = new System.Xml.Serialization.XmlSerializer(typeof(Zone));

        public Zone() { }

        public static Zone FromFile(string s)
        {            
            Zone z = (Zone)s_xmls.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(s)));
            return z;
        }

        public string GetClusterFileName(uint strip, bool istop, uint view)
        {
            return FileNameTemplate.Replace(SequenceString, view.ToString()).Replace(SideString, istop ? "T" : "B").Replace(TypeString, "cls").Replace(StripString, strip.ToString()).Replace(ImageString, "");
        }

        static System.Text.RegularExpressions.Regex rx_RWDremove = new System.Text.RegularExpressions.Regex(@"_rwdremove\([^\)]*\)");

        public string GetRWDFileName(uint index, string outdir)
        {
            if (OutputRWDFileName != null && OutputRWDFileName.Length > 0) return OutputRWDFileName + ".rwd." + index.ToString("X08");
            string s = FileNameTemplate.Replace(SequenceString, "").Replace(SideString, "A").Replace(TypeString, "rwd." + index.ToString("X08")).Replace(StripString, "").Replace(ImageString, "");
            var m = rx_RWDremove.Match(s.ToLower());
            if (m.Success)            
                s = s.Replace(m.ToString(), "");            
            if (outdir.EndsWith("\\") == false) outdir += "\\";
            return outdir + s.Substring(s.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
        }

        public const string SequenceString = "$SEQ$";
        public const string ImageString = "$IMAGE$";
        public const string TypeString = "$TYPE$";
        public const string SideString = "$SIDE$";
        public const string StripString = "$STRIP$";
    }

    class RawDataFragment : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment, SySal.GPU.IRawDataViewSideConsumer
    {
        class GPUView : SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View
        {
            public GPUView(int ix, int iy)
            {
                this.m_Tile.X = ix;
                this.m_Tile.Y = iy;                
            }

            public void SetSideInfo(SySal.Scanning.Plate.IO.OPERA.RawData.Fragment.View.Side sd, bool istop)
            {
                if (istop) m_Top = sd;
                else m_Bottom = sd;
            }
        }

        public RawDataFragment(Zone z, uint strip, uint index, uint firstview, uint lastview)
        {
            this.m_CodingMode = FragmentCoding.GrainSuppression;
            this.m_Id.Part0 = 0;
            this.m_Id.Part1 = 0;
            this.m_Id.Part2 = 0;
            this.m_Id.Part3 = 0;
            this.m_Index = index;
            this.m_StartView = strip * z.Views + firstview;
            this.m_Views = new GPUView[lastview - firstview + 1];
            for (int i = 0; i < m_Views.Length; i++)
                m_Views[i] = new GPUView(i + (int)firstview, (int)strip);
        }

        public void ConsumeData(int n, bool istop, GPU.RawDataViewSide rwdvs)
        {
            ((GPUView)m_Views[n]).SetSideInfo(rwdvs, istop);
        }
    }

    class Job
    {
        public System.Guid Id = new Guid();
        public System.DateTime ExpirationTime;
        public string ExitException = "";
        public string ID { get { return Id.ToString().Replace("-", "").ToUpper(); } }
    }

    class RWDJob : Job
    {
        public string ZoneFile;
        public uint StripId;
        public int FragmentIndex = -1;
        public string OutputDir;
        public int FirstView = -1;
        public int LastView = -1;
    }

    public class Program : SySal.Web.IWebApplication
    {
        [Serializable]
        public class Config
        {
            public string NotifyAliveAddress = "127.0.0.1";

            public int MaxGPUs = -1;

            public string DebugDumpDir = "";

            public int LogLevel = 0;

            private static System.Xml.Serialization.XmlSerializer s_xmls = new XmlSerializer(typeof(Config));
            
            private static string FileName
            {
                get { return System.Diagnostics.Process.GetCurrentProcess().MainModule.FileName + ".xml"; }
            }

            public static Config Current
            {
                get
                {
                    Config C = null;
                    try
                    {
                        string cfgstr = System.IO.File.ReadAllText(FileName);
                        C = (Config)s_xmls.Deserialize(new System.IO.StringReader(cfgstr));
                    }
                    catch (Exception)
                    {
                        C = new Config();
                    }
                    return C;
                }
            }

            public void Save()
            {
                try
                {
                    System.IO.StringWriter swr = new System.IO.StringWriter();
                    s_xmls.Serialize(swr, this);
                    System.IO.File.WriteAllText(FileName, swr.ToString());
                }
                catch (Exception x)
                {
                    Console.WriteLine(x.ToString());
                }
            }
        }

        static Config C = Config.Current;

        static string ComputerName = System.Environment.MachineName;

        static string ProcessInstance = System.Guid.NewGuid().ToString().Replace("-", "").ToUpper();

        static System.IO.TextWriter Out = Console.Out;

        const string ScanExtension = ".scan";

        static System.Threading.AutoResetEvent JobsWaiting = new System.Threading.AutoResetEvent(false);

        static System.Collections.Generic.Queue<string> ProfilingQueue = new Queue<string>();

        static System.Collections.Generic.Dictionary<string, int> ProfilingInfo = new Dictionary<string, int>();

        class TrackerInfo
        {
            public SySal.GPU.MapTracker Tracker;
            public System.Collections.Generic.List<Job> Jobs = new List<Job>();
            public Job Current;
            public System.Collections.Generic.List<Job> Completed = new List<Job>();
            public bool Terminate = false;
            public System.Threading.Thread ExecThread;

            public void Process()
            {
                SySal.GPU.MapTracker mptk = Tracker;
                while (Terminate == false)
                {
                    Job o = null;
                    lock (Jobs)
                        if (Jobs.Count > 0)
                        {
                            o = Jobs.First();
                            Jobs.Remove(o);
                            Current = o;
                        }
                    if (o == null)
                    {
                        JobsWaiting.WaitOne(1000);
                        continue;
                    }
                    else if (o is RWDJob)
                    {
                        RWDJob rwdjo = o as RWDJob;
                        System.IO.FileStream outf = null;
                        try
                        {
                            if (C.LogLevel > 0) Out.WriteLine("Starting processing for " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                            Zone z = Zone.FromFile(rwdjo.ZoneFile);
                            string imgcorrcfg = null;
                            string clstchcfg = null;
                            string trkcfg = null;
                            string debugdumptemplate = null;
                            foreach (PostProcessingInfo ppi in z.PostProcessingSettings)
                                switch (ppi.Name.ToLower().Trim())
                                {
                                    case "imagecorrection":                                        
                                        imgcorrcfg = ppi.Settings;
                                        if (C.LogLevel >= 3) Out.WriteLine("ImageCorrection: " + imgcorrcfg);
                                        break;

                                    case "clusterchainer":
                                        clstchcfg = ppi.Settings;
                                        if (C.LogLevel >= 3) Out.WriteLine("ClusterChainer: " + clstchcfg);
                                        break;

                                    case "tracker":
                                        trkcfg = ppi.Settings;
                                        if (C.LogLevel >= 3) Out.WriteLine("Tracker: " + trkcfg);
                                        break;

                                    case "debugdumptemplate":
                                        debugdumptemplate = ppi.Settings;
                                        if (C.LogLevel >= 3) Out.WriteLine("DebugDumpTemplate: " + debugdumptemplate);
                                        break;
                                }
                            if (C.LogLevel >= 3) Out.WriteLine("Setting configuration for " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                            if (string.IsNullOrWhiteSpace(C.DebugDumpDir) || string.IsNullOrWhiteSpace(debugdumptemplate)) mptk.SetDebugDumpTemplate(null);
                            else
                            {
                                string debd = C.DebugDumpDir + System.IO.Path.DirectorySeparatorChar + debugdumptemplate;
                                mptk.SetDebugDumpTemplate(debd);
                                if (C.LogLevel >= 3) Out.WriteLine("Debug Dump Template is " + debd);
                            }
                            mptk.SetVerbosity(C.LogLevel);
                            mptk.SetImageCorrection(imgcorrcfg);
                            mptk.SetClusterChainerConfig(clstchcfg);
                            mptk.SetTrackerConfig(trkcfg);
                            float basethickness = -1.0f;
                            try
                            {
                                var xmld = new XmlDocument();
                                xmld.LoadXml(z.ScanSettings);
                                basethickness = float.Parse(xmld.ChildNodes[1]["BaseThickness"].InnerText, System.Globalization.CultureInfo.InvariantCulture);
                            }   
                            catch (Exception x)
                            {
                                if (C.LogLevel >= 1) Out.WriteLine("Can't find base thickness in zone info:" + Environment.NewLine + z.ScanSettings + Environment.NewLine + ", defaulting to " + basethickness.ToString(System.Globalization.CultureInfo.InvariantCulture) + " (" + x.Message + ")");
                            }
                            if (basethickness < 0)
                                throw new Exception("Base thickness is not positive, can't continue.");
                            if (C.LogLevel >= 1) Out.WriteLine("Using base thickness = " + basethickness);
                            if (rwdjo.FirstView < 0 && rwdjo.LastView < 0)
                            {
                                rwdjo.FirstView = 0;
                                rwdjo.LastView = (int)z.Views - 1;
                            }
                            if (rwdjo.FragmentIndex < 0) rwdjo.FragmentIndex = (int)rwdjo.StripId + 1;
                            RawDataFragment rwdf = new RawDataFragment(z, rwdjo.StripId, (uint)rwdjo.FragmentIndex, (uint)rwdjo.FirstView, (uint)rwdjo.LastView);
                            mptk.SetRawDataViewSideConsumer(rwdf);
                            string[] files = new string[z.Views];
                            if (z.HasTop)
                            {
                                if (C.LogLevel >= 1) Out.WriteLine("Finding tracks on top for " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                                int i;
                                for (i = 0; i < z.Views; i++)
                                    files[i] = z.GetClusterFileName(rwdjo.StripId, true, (uint)i);
                                mptk.SetDebugMark(rwdjo.StripId * 16);
                                mptk.FindTracks(files, true, 0.0f, z.PlateMap, (uint)rwdjo.FirstView, (uint)rwdjo.LastView);
                            }
                            if (z.HasBottom)
                            {
                                if (C.LogLevel >= 1) Out.WriteLine("Finding tracks on bottom for " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                                int i;
                                for (i = 0; i < z.Views; i++)
                                    files[i] = z.GetClusterFileName(rwdjo.StripId, false, (uint)i);
                                mptk.SetDebugMark(rwdjo.StripId * 16 + 1);
                                mptk.FindTracks(files, false, -basethickness, z.PlateMap, (uint)rwdjo.FirstView, (uint)rwdjo.LastView);
                            }
                            if (C.LogLevel >= 1) Out.WriteLine("Writing tracks for " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                            string outname = z.GetRWDFileName((uint)rwdjo.FragmentIndex, rwdjo.OutputDir);
                            string workname = outname + ".twr";
                            outf = new System.IO.FileStream(workname, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                            rwdf.Save(outf);
                            outf.Flush();
                            outf.Close();
                            outf = null;
                            if (C.LogLevel >= 1) Out.WriteLine("Done " + rwdjo.ZoneFile + " " + rwdjo.StripId);
                            /* The RWD must be finalized by the PostProcessManager, 
                            otherwise clashes may occur if the communication is lost while the job is completed. */
                            //System.IO.File.Move(workname, outname); 
                        }
                        catch (Exception x)
                        {                            
                            o.ExitException = x.ToString();
                            Out.WriteLine("\r\nError for " + rwdjo.ZoneFile + " " + rwdjo.StripId + ":\r\n" + x.ToString());
                            /*if (x.Message.IndexOf("GPU-level") >= 0)
                                Environment.Exit(-1);*/
                            if (x is System.Runtime.InteropServices.SEHException)
                                Out.WriteLine("SEHException:\r\n" + (x as System.Runtime.InteropServices.SEHException).ErrorCode);
                        }
                        finally
                        {
                            if (outf != null) outf.Close();
                            mptk.SetRawDataViewSideConsumer(null);                            
                        }
                    }
                    else
                    {
                        o.ExitException = "Unsupported job type.";
                    }
                    System.DateTime n = System.DateTime.Now;
                    o.ExpirationTime = n.AddHours(1);
                    lock (Completed)
                    {
                        Completed.Add(o);
                        Current = null;                        
                        System.Collections.Generic.List<Job> ncompl = new List<Job>();
                        ncompl.AddRange(Completed.Where(x => x.ExpirationTime >= n));
                        Completed = ncompl;
                    }
                }
            }
        }

        static TrackerInfo[] Trackers;        

        static System.Text.RegularExpressions.Regex rx_Exit = new System.Text.RegularExpressions.Regex(@"\s*exit\s*");

        static System.Text.RegularExpressions.Regex rx_Help = new System.Text.RegularExpressions.Regex(@"\s*help\s*");

        static System.Text.RegularExpressions.Regex rx_AliveAddress = new System.Text.RegularExpressions.Regex(@"\s*alive\s+(\S+)\s*");

        static System.Text.RegularExpressions.Regex rx_Log = new System.Text.RegularExpressions.Regex(@"\s*log\s+(0|1|2|3|on|off)\s*");

        static System.Text.RegularExpressions.Regex rx_DebugDumpDir = new System.Text.RegularExpressions.Regex(@"\s*debdumpdir\s+(\S+)\s*");

        static System.Text.RegularExpressions.Regex rx_NoDebugDump = new System.Text.RegularExpressions.Regex(@"\s*nodebdump\s*");

        static void Main(string[] args)
        {
            int gpus = SySal.GPU.Utilities.GetAvailableGPUs();
            if (C.MaxGPUs >= 0 && gpus > C.MaxGPUs) gpus = C.MaxGPUs;
            Trackers = new TrackerInfo[gpus];
            Out.WriteLine("GPUs found: " + gpus);
            int g;
            for (g = 0; g < gpus; g++)
            {
                Trackers[g] = new TrackerInfo();
                Trackers[g].Tracker = new GPU.MapTracker();
                Trackers[g].Tracker.SetGPU(g);
                Trackers[g].ExecThread = new System.Threading.Thread(new System.Threading.ThreadStart(Trackers[g].Process));
                Trackers[g].ExecThread.Start();
            }
            Out.WriteLine("Initialization done, starting HTTP server");
            SySal.Web.WebServer ws = new Web.WebServer(1783, new Program());
            Out.WriteLine("Notifying startup.");
            SendAliveNotification(true);
            Out.WriteLine("Starting \"alive\" notifier.");
            System.Timers.Timer alivetimer = new System.Timers.Timer(60000);
            alivetimer.Elapsed += new System.Timers.ElapsedEventHandler(alivetimer_Elapsed);
            alivetimer.Start();
            Out.WriteLine("Starting profiler.");
            System.Timers.Timer profiletimer = new System.Timers.Timer(100);
            profiletimer.Elapsed += new System.Timers.ElapsedEventHandler(profiletimer_Elapsed);
            profiletimer.Start();
            Out.WriteLine("Service started, type exit to terminate or help to get the list of commands.");
            string line;
            while ((line = Console.ReadLine()) != null)
            {
                System.Text.RegularExpressions.Match m;
                if ((m = rx_Exit.Match(line)).Success) break;
                else if ((m = rx_AliveAddress.Match(line)).Success)
                {
                    C.NotifyAliveAddress = m.Groups[1].Value;
                    C.Save();
                }
                else if ((m = rx_Help.Match(line)).Success)
                {
                    Out.WriteLine();
                    Out.WriteLine("exit -> Stop service.");
                    Out.WriteLine("alive <addr> -> Set <addr> as the address of the computer to notify that this server is working.");
                    Out.WriteLine("debdumpdir <directory> -> Set <directory> as location for debug dump files.");
                    Out.WriteLine("nodebdump -> Disable debug dump.");
                    Out.WriteLine("log <0|1|2|3|on|off> -> Enable|disable logging (on|off) or set verbosity level (0=off;1=on;2,3 for debugging).");
                    Out.WriteLine("help -> Show this help.");
                    Out.WriteLine();
                }
                else if ((m = rx_Log.Match(line)).Success)
                {
                    switch (m.Groups[1].Value)
                    {
                        case "0": C.LogLevel = 0; break;
                        case "1": C.LogLevel = 1; break;
                        case "2": C.LogLevel = 2; break;
                        case "3": C.LogLevel = 3; break;
                        case "off": C.LogLevel = 0; break;
                        case "on": C.LogLevel = 1; break;
                    }                    
                    C.Save();
                }
                else if ((m = rx_DebugDumpDir.Match(line)).Success)
                {
                    C.DebugDumpDir = m.Groups[1].Value;
                    if (C.DebugDumpDir.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString()))
                        C.DebugDumpDir = C.DebugDumpDir.Substring(0, C.DebugDumpDir.Length - 1);
                    C.Save();
                }
                else if ((m = rx_NoDebugDump.Match(line)).Success)
                {
                    C.DebugDumpDir = "";
                    C.Save();
                }
            }
            Out.WriteLine("Sending stop signal to trackers.");
            foreach (TrackerInfo t in Trackers)
                t.Terminate = true;
            Out.WriteLine("Stopping profiler.");
            profiletimer.Stop();
            Out.WriteLine("Stopping \"alive\" notifier.");
            alivetimer.Stop();
            Out.WriteLine("Notifying shutdown.");
            SendAliveNotification(false);
            Out.WriteLine("Waiting for trackers to stop - it may take long if they are working.");
            foreach (TrackerInfo t in Trackers)           
                t.ExecThread.Join();            
            Out.WriteLine("Terminated.");
        }

        static void profiletimer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            if (ProfilingQueue.Count >= 100 * Trackers.Length)
                lock (ProfilingInfo)
                    foreach (TrackerInfo t in Trackers)
                    {
                        string id = ProfilingQueue.Dequeue();
                        ProfilingInfo[id] = ProfilingInfo[id] - 1;
                    }
            lock (ProfilingInfo)
                foreach (TrackerInfo t in Trackers)
                {
                    string id = t.Tracker.GetCurrentActivity();
                    ProfilingQueue.Enqueue(id);
                    if (ProfilingInfo.Keys.Contains(id) == false) ProfilingInfo[id] = 1;
                    else ProfilingInfo[id] = ProfilingInfo[id] + 1;
                }
        }

        static void SendAliveNotification(bool alive)
        {
            System.IO.Stream rstr = null;
            using (System.Net.WebClient wcl = new System.Net.WebClient())
            {
                try
                {
                    rstr = wcl.OpenRead("http://" + C.NotifyAliveAddress + ":1784/trksrv_alive?alive=" + alive);
                    while (rstr.ReadByte() >= 0) ;
                }
                catch (Exception x) { Out.WriteLine("\n" + System.DateTime.Now + "\n" + x.ToString() + "\n"); }
                finally
                {
                    if (rstr != null) rstr.Close();
                }
            }
        }

        static void alivetimer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            SendAliveNotification(true);
        }

        public string ApplicationName
        {
            get { return "GPUTrackingServer Process " + ProcessInstance + " at " + ComputerName; }
        }

        const string a_rwdz = "rwdz";
        const string a_rwdi = "rwdi";
        const string a_rwds = "rwds";
        const string a_rwdd = "rwdd";
        const string a_rwdfv = "rwdfv";
        const string a_rwdlv = "rwdlv";
        const string a_gpu = "queue";
        const string a_jobid = "j";

        public Web.ChunkedResponse HttpGet(Web.Session sess, string page, params string[] queryget)
        {
            if (String.Compare(page, "/qn") == 0) return new SySal.Web.HTMLResponse(Trackers.Length.ToString());
            int g = -1;
            RWDJob rwdj = new RWDJob();
            bool rwdj_zone_set = false;
            bool rwdj_strip_set = false;
            bool rwdj_outdir_set = false;
            bool rwdj_firstview_set = false;
            bool rwdj_lastview_set = false;
            bool rwdj_index_set = false;
            string jobid = null;
            foreach (string s in queryget)
            {
                if (s.StartsWith(a_gpu + "="))
                {
                    g = int.Parse(s.Substring(a_gpu.Length + 1));
                    if (g < 0 || g >= Trackers.Length) return new SySal.Web.HTMLResponse("ERROR: INVALID QUEUE SPECIFIED");
                }
                else if (s.StartsWith(a_rwdz + "="))
                {
                    rwdj.ZoneFile = SySal.Web.WebServer.URLDecode(s.Substring(a_rwdz.Length + 1));
                    rwdj_zone_set = true;
                }
                else if (s.StartsWith(a_rwds + "="))
                {
                    rwdj.StripId = uint.Parse(s.Substring(a_rwds.Length + 1));
                    rwdj_strip_set = true;
                }
                else if (s.StartsWith(a_rwdi + "="))
                {
                    rwdj.FragmentIndex = (int)uint.Parse(s.Substring(a_rwdi.Length + 1));
                    rwdj_index_set = true;
                }
                else if (s.StartsWith(a_rwdfv + "="))
                {
                    rwdj.FirstView = (int)uint.Parse(s.Substring(a_rwdfv.Length + 1));
                    rwdj_firstview_set = true;
                }
                else if (s.StartsWith(a_rwdlv + "="))
                {
                    rwdj.LastView = (int)uint.Parse(s.Substring(a_rwdlv.Length + 1));
                    rwdj_lastview_set = true;
                }
                else if (s.StartsWith(a_rwdd + "="))
                {
                    rwdj.OutputDir = SySal.Web.WebServer.URLDecode(s.Substring(a_rwdd.Length + 1));
                    rwdj_outdir_set = true;
                }
                else if (s.StartsWith(a_jobid + "="))
                {
                    jobid = s.Substring(a_jobid.Length + 1);
                }
            }            
            if (g >= 0)
            {
                if (String.Compare(page, "/queryjob") == 0)
                {
                    IEnumerable<Job> res = null;
                    lock (Trackers[g].Jobs)
                        res = Trackers[g].Jobs.Where(x => String.Compare(x.ID, jobid) == 0);
                    if (res.Count() == 1)
                        return new SySal.Web.HTMLResponse("WAITING");
                    res = null;
                    lock (Trackers[g].Completed)
                    {
                        Job jc = Trackers[g].Current;
                        if (jc != null && String.Compare(jc.ID, jobid) == 0)
                            return new SySal.Web.HTMLResponse("RUNNING");
                        res = Trackers[g].Completed.Where(x => String.Compare(x.ID, jobid) == 0);
                    }
                    if (res.Count() == 1)
                    {
                        Job jj = res.First();
                        return new SySal.Web.HTMLResponse((rwdj.ExitException.Length > 0) ? ("FAILED\r\n" + jj.ExitException) : "DONE");
                    }
                    return new SySal.Web.HTMLResponse("UNKNOWN");
                }
                else if (String.Compare(page, "/killjob") == 0)
                {
                    bool killed = false;
                    IEnumerable<Job> res = null;
                    lock (Trackers[g].Jobs)
                    {
                        res = Trackers[g].Jobs.Where(x => String.Compare(x.ID, jobid) == 0);
                        if (res.Count() == 1)
                        {
                            Trackers[g].Jobs.Remove(res.First());
                            killed = true;
                        }
                    }
                    return new SySal.Web.HTMLResponse(killed ? "KILLED" : "CANNOT KILL");
                }
                else if (String.Compare(page, "/addrwdj") == 0 && rwdj_zone_set && rwdj_strip_set && rwdj_outdir_set && (rwdj_firstview_set == rwdj_lastview_set && rwdj_index_set == rwdj_firstview_set))
                {
                    rwdj.Id = System.Guid.NewGuid();
                    lock (Trackers[g].Jobs)
                        if (Trackers[g].Jobs.Count > 0 || Trackers[g].Current != null)
                            return new SySal.Web.HTMLResponse("REFUSED-BUSY");
                        else Trackers[g].Jobs.Add(rwdj);
                    JobsWaiting.Set();
                    return new SySal.Web.HTMLResponse("CREATED " + rwdj.ID);
                }
            }
            System.IO.StringWriter swr = new System.IO.StringWriter();
            swr.WriteLine("<html><head><title>GPUTrackingServer " + ProcessInstance + " at " + ComputerName + "</title>" +
                "  <style type=\"text/css\"> " +
                "H1 " +
                "{ " +
                " color : Navy; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 24px; opacity: 1; line-height: 24px; " +
                "} " +
                "H6 " +
                "{ " +
                " color : Navy; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 12px; opacity: 1; line-height: 24px; " +
                "} " +
                "DIV " +
                "{ " +
                " color : Blue; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 12px; opacity: 1; line-height: 12px; " +
                "} " +
                "TD " +
                "{ " +
                " color : Navy; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 12px; opacity: 1; line-height: 12px; " +
                "} " +
                "TH " +
                "{ " +
                " color : Blue; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 12px; opacity: 1; line-height: 12px; " +
                "} " +
                "  </style> " +
                "</head><body>");
            swr.WriteLine("<h1>GPUTrackingServer " + ComputerName + " running <b>" + Trackers.Length + "</b> GPU(s)</h1><h6>Process Instance " + ProcessInstance + " </h6><hr />");
            swr.WriteLine("<div><form action=\"addrwdj\" metod=\"get\"><table align=\"left\" border=\"0\" width=\"100%\">"  +
                "<tr><th colspan=\"2\" align=\"left\">Add RWD job</th><td width=\"100%\">&nbsp;</td></tr>" +
                "<tr><td>GPU</td><td><select name=\"" + a_gpu + "\" id=\"" + a_gpu + "\">" + GPUHTMLSelect + "</td></tr>" +
                "<tr><td>Zone path</td><td><input type=\"text\" name=\"" + a_rwdz + "\" id=\"" + a_rwdz + "\" size=\"100\" /></td></tr>" +
                "<tr><td>Strip</td><td><input type=\"text\" name=\"" + a_rwds + "\" id=\"" + a_rwds + "\" /></td><td>Index</td><td><input type=\"text\" name=\"" + a_rwdi + "\" id=\"" + a_rwdi + "\" /></td></tr>" +
                "<tr><td>First View</td><td><input type=\"text\" name=\"" + a_rwdfv + "\" id=\"" + a_rwdfv + "\" /></td><td>Last View</td><td><input type=\"text\" name=\"" + a_rwdlv + "\" id=\"" + a_rwdlv + "\" /></td></tr>" +
                "<tr><td>Output dir</td><td><input type=\"text\" name=\"" + a_rwdd + "\" id=\"" + a_rwdd + "\" size=\"100\" /></td></tr>" +
                "<tr><td colspan=\"2\"><input type=\"submit\" value=\"Submit job\" /></td></tr>" +
                "</table></form></div><br />");
            for (g = 0; g < Trackers.Length; g++)
            {
                Job current = Trackers[g].Current;
                swr.WriteLine("<hr /><br /><div><table align=\"left\" border=\"1\" width=\"100%\"><tr><th>Queue " + g + " current " + ((current == null) ? "" : current.ID) + "</th></tr>");
                string[] jstr = null;
                lock (Trackers[g].Jobs)
                {
                    jstr = new string[Trackers[g].Jobs.Count];
                    int i = 0;
                    foreach (Job jj in Trackers[g].Jobs)
                        jstr[i++] = jj.ID;
                }
                foreach (string ss in jstr)
                    swr.WriteLine("<tr><td>" + ss + "</td></tr>");
                swr.WriteLine("</table></div><br />");
            }
            swr.WriteLine("<hr /><br /><div><table align=\"left\"><tr><th colspan=\"2\">Profiling info</th></tr>");
            lock (ProfilingInfo)
            {
                int total = 0;
                foreach (int t in ProfilingInfo.Values) total += t;
                if (total > 0)
                {
                    foreach (string s in ProfilingInfo.Keys)
                        swr.WriteLine("<tr><td>" + s + "</td><td>" + (ProfilingInfo[s] * 100 / total).ToString() + "</td></tr>");
                }
            }
            swr.WriteLine("</table></div><br />");
            swr.WriteLine("</body></html>\r\n\r\n");
            return new SySal.Web.HTMLResponse(swr.ToString());
        }

        public Web.ChunkedResponse HttpPost(Web.Session sess, string page, params string[] postfields)
        {
            return HttpGet(sess, page, postfields);
        }

        public bool ShowExceptions
        {
            get { return true; }
        }

        public string GPUHTMLSelect
        {
            get { string s = ""; int i; for (i = 0; i < Trackers.Length; i++) s += "<option value=\"" + i + "\">" + i + "</option>"; return s; }
        }
    }
}
