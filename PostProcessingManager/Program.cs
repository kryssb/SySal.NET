using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using System.Xml.Serialization;

namespace SySal.Executables.PostProcessingManager
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

        [NonSerialized]
        public bool Killed = false;

        public PostProcessingInfo[] PostProcessingSettings = new PostProcessingInfo[0];

        public bool AutoDelete = true;

        [NonSerialized]
        public string StartFileName;

        [NonSerialized]
        private RawDataCatalog m_RWC;
#if ENABLE_TESTS
        internal RawDataCatalog GetRWC() { return m_RWC; }
#endif
        

        [NonSerialized]
        public string ProgressFileName;

        static System.Xml.Serialization.XmlSerializer s_xmls = new System.Xml.Serialization.XmlSerializer(typeof(Zone));

        public Zone() { }

        internal uint RegisterFragmentIndex(Job j)
        {
            lock (this)
            {
                if (m_RWC == null)
                    m_RWC = new RawDataCatalog(this);                
                var ret = m_RWC.RegisterFragmentIndex(j.StripId, j.FirstView, j.LastView);
                WriteCatalog();
                return ret;
            }
        }

        public bool AttemptImportCatalog()
        {
            if (m_RWC != null) return false;
            try
            {
                using (System.IO.FileStream rstr = new System.IO.FileStream(CatalogName, System.IO.FileMode.Open, System.IO.FileAccess.Read))
                {
                    var rwc = new SySal.Scanning.Plate.IO.OPERA.RawData.Catalog(rstr);
                    m_RWC = new RawDataCatalog(this);
                    return m_RWC.AttemptImportCatalog(rwc);
                }
            }
            catch (Exception)
            {
                return false;                
            }
        }

        public string OutDir = null;

        string CatalogName
        {
            get
            {
                string s;
                if (OutputRWDFileName != null && OutputRWDFileName.Length > 0)
                    s = OutputRWDFileName + ".rwc";
                else
                {
                    s = GetRWCFileName(OutDir);
                }
                return s;
            }
        }

        public void WriteCatalog()
        {
            if (m_RWC == null) m_RWC = new RawDataCatalog(this);
            System.IO.FileStream fstr = null;
            try
            {
                string s;
                fstr = new System.IO.FileStream(CatalogName, System.IO.FileMode.Create, System.IO.FileAccess.ReadWrite);
                m_RWC.Save(fstr);
                fstr.Flush();
                fstr.Close();
                fstr = null;
            }
            finally
            {
                if (fstr != null)
                    fstr.Close();
            }
        }

        static char[] dirseparators = new char[] { '\\', '/' };

        public void ReplaceDirectory(string newpath)
        {
            FileNameTemplate = newpath.TrimEnd(new char[] { '\\', '/' }) + System.IO.Path.DirectorySeparatorChar + FileNameTemplate.Substring(FileNameTemplate.LastIndexOfAny(dirseparators) + 1);            
            ProgressFileName = FileNameTemplate.Replace(SequenceString, "").Replace(SideString, "A").Replace(TypeString, "progress").Replace(StripString, "").Replace(ImageString, "");
            StartFileName = FileNameTemplate.Replace(SequenceString, "").Replace(SideString, "A").Replace(TypeString, "scan").Replace(StripString, "").Replace(ImageString, "");
        }

        public static Zone FromFile(string s, bool allowimportcatalog = true)
        {
            Zone z = (Zone)s_xmls.Deserialize(new System.IO.StringReader(System.IO.File.ReadAllText(s)));
            z.StartFileName = s;
            try
            {
                var workdir = s.Substring(0, s.LastIndexOfAny(dirseparators) + 1);
                if (z.FileNameTemplate.StartsWith(workdir) == false)
                {                    
                    z.FileNameTemplate = workdir + z.FileNameTemplate.Substring(z.FileNameTemplate.LastIndexOfAny(dirseparators) + 1);
                    System.IO.StringWriter swr = new System.IO.StringWriter();
                    s_xmls.Serialize(swr, z);
                    System.IO.File.WriteAllText(s, swr.ToString());
                    Console.WriteLine("File name template found inconsistent and corrected to " + z.FileNameTemplate);
                }
            }
            catch (Exception x)
            {
                Console.WriteLine("Error checking consistency of start file and file template: " + x.Message);
            }
            if (allowimportcatalog)
                try
                {
                    Console.WriteLine("Attempting import of catalog file.");
                    var res = z.AttemptImportCatalog();
                    Console.WriteLine(res ? "FAILED" : "OK");
                }
                catch (Exception x)
                {
                    Console.WriteLine("Error trying to import catalog file: " + x.Message);
                }
            z.ProgressFileName = s.Substring(0, s.LastIndexOf('.')) + ".progress";
            z.m_AutoUpdateTimer = new System.Timers.Timer(5000);
            z.m_AutoUpdateTimer.AutoReset = true;
            z.m_AutoUpdateTimer.Elapsed += new System.Timers.ElapsedEventHandler(z.AutoUpdateTimer_Elapsed);
            z.m_AutoUpdateTimer.Start();            
            return z;
        }

        void AutoUpdateTimer_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            Update();
        }

        public string GetClusterFileName(uint strip, bool istop, uint view)
        {
            return FileNameTemplate.Replace(SequenceString, view.ToString()).Replace(SideString, istop ? "T" : "B").Replace(TypeString, "cls").Replace(StripString, strip.ToString()).Replace(ImageString, "");
        }

        static System.Text.RegularExpressions.Regex rx_RWDremove = new System.Text.RegularExpressions.Regex(@"_rwdremove\([^\)]*\)");

        internal string GetRWCFileName(string outdir)
        {
            if (OutputRWDFileName != null && OutputRWDFileName.Length > 0) return OutputRWDFileName + ".rwc" ;
            string s = FileNameTemplate.Replace(SequenceString, "").Replace(SideString, "A").Replace(TypeString, "rwc").Replace(StripString, "").Replace(ImageString, "");
            var m = rx_RWDremove.Match(s.ToLower());
            if (m.Success)
                s = s.Replace(m.ToString(), "");
            if (outdir.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString()) == false) outdir += System.IO.Path.DirectorySeparatorChar;
            return outdir + s.Substring(s.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
        }

        internal string GetRWDFileName(Job j, string outdir)
        {
            if (OutputRWDFileName != null && OutputRWDFileName.Length > 0) return OutputRWDFileName + ".rwd." + j.FragmentIndex.ToString("X08");
            string s = FileNameTemplate.Replace(SequenceString, "").Replace(SideString, "A").Replace(TypeString, "rwd." + j.FragmentIndex.ToString("X08")).Replace(StripString, "").Replace(ImageString, "");
            var m = rx_RWDremove.Match(s.ToLower());
            if (m.Success)
                s = s.Replace(m.ToString(), "");
            if (outdir.EndsWith(System.IO.Path.DirectorySeparatorChar.ToString()) == false) outdir += System.IO.Path.DirectorySeparatorChar;
            return outdir + s.Substring(s.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
        }

        static System.Text.RegularExpressions.Regex s_Progress = new System.Text.RegularExpressions.Regex(@"\s(\d+)\s(\d+)\s");

        public void Update()
        {
            try
            {
                string text = System.IO.File.ReadAllText(ProgressFileName);
                System.Text.RegularExpressions.Match m = s_Progress.Match(text);
                if (m.Success == false || m.Length != text.Length) return;
                uint topstripsready = uint.Parse(m.Groups[1].Value);
                uint bottomstripsready = uint.Parse(m.Groups[2].Value);
                bool notify = topstripsready > this.Progress.TopStripsReady || bottomstripsready > this.Progress.BottomStripsReady;
                this.Progress.TopStripsReady = topstripsready;
                this.Progress.BottomStripsReady = bottomstripsready;
                if (notify) Program.NotifyProgress(this);
            }
            catch (Exception)
            {
            }
        }

        public void UpdateWrite()
        {
            while (true)
                try
                {
                    System.IO.File.WriteAllText(ProgressFileName, " " + this.Progress.TopStripsReady + " " + this.Progress.BottomStripsReady + " ");
                    break;
                }
                catch (Exception)
                {
                    System.Threading.Thread.Sleep(5000);
                }

        }

        [NonSerialized]
        System.Timers.Timer m_AutoUpdateTimer = null;

        public const string SequenceString = "$SEQ$";
        public const string ImageString = "$IMAGE$";
        public const string TypeString = "$TYPE$";
        public const string SideString = "$SIDE$";
        public const string StripString = "$STRIP$";

        public void DeleteStripFiles(uint _strip)
        {
            uint _view;
            char _side;
            for (_view = 0; _view < Views; _view++)
                for (_side = 'T'; _side == 'T' || _side == 'B'; _side = (_side == 'T') ? 'B' : 'A')
                    try
                    {
                        System.IO.File.Delete(FileNameTemplate.Replace(SequenceString, _view.ToString()).Replace(SideString, _side.ToString()).Replace(TypeString, "cls").Replace(StripString, _strip.ToString()).Replace(ImageString, ""));
                    }
                    catch (Exception) { }
        }

        ~Zone()
        {
            if (AutoDelete)
            {
                try
                {
                    System.IO.File.Delete(ProgressFileName);
                }
                catch (Exception) { }
                uint _strip;
                uint _view;
                char _side;
                for (_strip = 0; _strip < Strips; _strip++)
                   for (_view = 0; _view < Views; _view++)
                       for (_side = 'T'; _side == 'T' || _side == 'B'; _side = (_side == 'T') ? 'B' : 'A')
                           try
                           {
                               System.IO.File.Delete(FileNameTemplate.Replace(SequenceString, _view.ToString()).Replace(SideString, _side.ToString()).Replace(TypeString, "cls").Replace(StripString, _strip.ToString()).Replace(ImageString, ""));
                           }
                           catch (Exception) { }
                try
                {
                    System.IO.File.Delete(this.StartFileName);
                }
                catch (Exception) { }
                try
                {
                    string filedir = this.StartFileName.Substring(0, this.StartFileName.LastIndexOfAny(new char[] {'/', '\\'}) - 1);
                    if (filedir.EndsWith(".scan")) System.IO.Directory.Delete(filedir);
                }
                catch (Exception) {}
            }
        }
    }

    class RawDataCatalog : SySal.Scanning.Plate.IO.OPERA.RawData.Catalog
    {
        class SetupRep : SySal.Scanning.Plate.IO.OPERA.RawData.Catalog.SetupStringRepresentation
        {
            public SetupRep(Zone z)
            {
                this.m_Name = "Scanning Setup";
                System.Xml.XmlDocument xmld = new XmlDocument();
                xmld.LoadXml(z.ScanSettings);
                System.Collections.ArrayList arr = new System.Collections.ArrayList();
                try
                {
                    arr.Add(new MyCfg("Default", "ScanSettings", z.ScanSettings));
                }
                catch (Exception) { }
                foreach (PostProcessingInfo pi in z.PostProcessingSettings)
                    try
                    {
                        arr.Add(new MyCfg("Default", pi.Name, pi.Settings));
                    }
                    catch (Exception) { }
                this.m_Configs = (ConfigStringRepresentation[])arr.ToArray(typeof(ConfigStringRepresentation));
            }

            class MyKeyEntry : KeyStringRepresentation
            {
                public MyKeyEntry(string name, string value)
                {
                    this.m_Name = name;
                    this.m_Value = value;
                }
            }

            class MyCfg : ConfigStringRepresentation
            {
                public MyCfg(string name, string classname, string xmldoc)
                {
                    this.m_Name = name;
                    this.m_ClassName = classname;
                    System.Collections.ArrayList arr = new System.Collections.ArrayList();
                    System.Xml.XmlDocument xmld = new XmlDocument();
                    xmld.LoadXml(xmldoc);
                    FromXmlElement(xmld.FirstChild, "", arr);
                    this.m_Keys = (KeyStringRepresentation[])arr.ToArray(typeof(KeyStringRepresentation));
                }
            }

            static void FromXmlElement(System.Xml.XmlNode xe, string prefix, System.Collections.ArrayList arr)
            {
                if (xe.HasChildNodes == false)
                {
                    arr.Add(new MyKeyEntry(prefix, xe.Value));
                }
                else
                {
                    string prefix1 = prefix;
                    if (prefix1.Length > 0)
                        prefix1 += ".";
                    foreach (System.Xml.XmlNode xn in xe.ChildNodes)
                        FromXmlElement(xn, prefix1 + xn.Name, arr);
                }
            }
        }

        public RawDataCatalog(Zone z)
        {
            this.m_Extents = z.TransformedScanRectangle;
            this.m_Id.Part0 = 0;
            this.m_Id.Part1 = 0;
            this.m_Id.Part2 = 0;
            this.m_Id.Part3 = 0;
            this.m_Steps.X = Math.Max(Math.Abs(z.ViewDelta.X), Math.Abs(z.StripDelta.X));
            this.m_Steps.Y = Math.Max(Math.Abs(z.ViewDelta.Y), Math.Abs(z.StripDelta.Y));
            this.m_Fragments = 0;
            this.FragmentIndices = new uint[z.Strips, z.Views];
            int iy, ix;
            for (iy = 0; iy < z.Strips; iy++)
                for (ix = 0; ix < z.Views; ix++)
                    this.FragmentIndices[iy, ix] = 0;
            this.m_SetupInfo = new SetupRep(z);
        }

        public uint RegisterFragmentIndex(uint strip, uint firstview, uint lastview)
        {
            ++this.m_Fragments;
            for (uint i = firstview; i <= lastview; i++)
                this.FragmentIndices[strip, i] = this.m_Fragments;
            return this.m_Fragments;
        }

        public bool AttemptImportCatalog(SySal.Scanning.Plate.IO.OPERA.RawData.Catalog rwc)
        {
            if (rwc.XSize == XSize && rwc.YSize == YSize)
            {
                for (int iy = 0; iy < rwc.YSize; iy++)
                    for (int ix = 0; ix < rwc.XSize; ix++)
                        FragmentIndices[iy, ix] = rwc[iy, ix];
                m_Fragments = rwc.Fragments;
                return true;
            }
            return false;
        }

#if ENABLE_TESTS
        internal uint [,] GetFragmentIndices()
        {
            return this.FragmentIndices;
        }
#endif
    }

    enum StripStatus
    {
        Waiting,
        Locked,
        Processing,
        Done
    }

    class Job
    {
        public Zone OwnerZone;
        private uint _FragmentIndex = 0;
        public uint FragmentIndex
        {
            get { if (_FragmentIndex == 0) _FragmentIndex = OwnerZone.RegisterFragmentIndex(this); return _FragmentIndex; }
        } 
        public uint StripId;
        public bool Split = false;
        public int[] SplitCounter = new int[] { 1 };
        public uint FirstView;
        public uint LastView;
        public string JobName;
        public bool Completed = false;
        private int Busy = 0;
        public bool Lock()
        {
            if (System.Threading.Interlocked.Increment(ref Busy) > 1)
            {
                System.Threading.Interlocked.Decrement(ref Busy);
                return false;
            }
            return true;
        }
        public void Unlock()
        {
            System.Threading.Interlocked.Decrement(ref Busy);
        }

        public override string ToString()
        {
            return OwnerZone.FileNameTemplate + " " + StripId + (Split ? ("(" + FirstView + "-" + LastView + ")") : "") + " [" + FragmentIndex + "]";
        }

        public IEnumerable<Job> TrySplitJob(uint trackers)
        {
            if (Split || trackers < 2) return new Job[] { this };
            var ol = new List<Job>();
            if (trackers > this.OwnerZone.Views) trackers = this.OwnerZone.Views;
            uint step = (this.OwnerZone.Views + (trackers - 1)) / trackers;
            var retj = Enumerable.Range(0, (int)trackers).
                Select(i => new { first = i * step, last = Math.Min(Math.Max(i * step + 1, (i + 1) * step - 1), (int)this.OwnerZone.Views - 1) }).
                Where(j => j.first <= j.last && (j.first != j.last || j.last != this.OwnerZone.Views - 1)).
                Select(j => (j.first == 0) ? this : new Job() { Split = true, OwnerZone = this.OwnerZone, StripId = this.StripId, FirstView = (uint)j.first, LastView = (uint)j.last, JobName = this.JobName }).ToArray();
            this.Split = true;
            if (retj.Length > 1)
                this.LastView = retj[1].FirstView;
            var spc = new int[] { retj.Length };
            foreach (var rj in retj)
                rj.SplitCounter = spc;
            return retj;
        }
#if ENABLE_TESTS
        public void ForceFragmentIndex(uint ind)
        {
            _FragmentIndex = ind;
        }
#endif
    }

    [Serializable]
    public class Config
    {
        public string InputDir = System.Environment.GetEnvironmentVariable("TEMP");
        public string OutputDir = System.Environment.GetEnvironmentVariable("TEMP");
        public uint FSWBufferSize = 40960;
        public bool EnableLog = false;
        public string CommandFile = "";
        public bool EnableJobSplitting = false;

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
            catch (Exception)
            { }
        }
    }

    partial class Program : SySal.Web.IWebApplication
    {
        static void Log(string msg) { Out.WriteLine(msg); }

        static Config C = Config.Current;

        static string ComputerName = System.Environment.MachineName;

        static string ProcessInstance = System.Guid.NewGuid().ToString().Replace("-", "").ToUpper();

        static System.IO.TextWriter Out = Console.Out;

        const string ScanExtension = ".scan";

        const string ProgressExtension = ".progress";        

        static System.Collections.Generic.LinkedList<Job> ActiveJobs = new LinkedList<Job>();

        static System.Collections.Generic.LinkedList<Job> WaitingJobs = new LinkedList<Job>();

        static System.Collections.Generic.Dictionary<string, Job []> TrackServers = new Dictionary<string, Job []>();

        static uint CountTrackers()
        {
            uint trks = 0;
            foreach (Job[] sps in TrackServers.Values)
                trks += (uint)sps.Length;
            return trks;
        }

        public static void NotifyProgress(Zone z)
        {
            System.Collections.Generic.List<Job> strlist = new List<Job>();
            lock (ActiveJobs)
            {
                foreach (Job s in WaitingJobs)
                    if (s.OwnerZone == z && z.Progress.TopStripsReady > s.StripId && z.Progress.BottomStripsReady > s.StripId)
                        strlist.Add(s);
                foreach (Job s in strlist)
                {
                    WaitingJobs.Remove(s);
                    ActiveJobs.AddLast(s);
                }
            }
        }

        public static void KillJobs(string fname, int id = -1)
        {
            System.Collections.Generic.List<Job> strlist = new List<Job>();
            lock (ActiveJobs)
            {
                foreach (Job s in WaitingJobs)
                    if (String.Compare(s.OwnerZone.FileNameTemplate, fname, true) == 0 && (id < 0 || s.StripId == id))
                    {
                        if (id < 0) s.OwnerZone.Killed = true;
                        strlist.Add(s);
                    }
                foreach (Job s in ActiveJobs)
                    if (String.Compare(s.OwnerZone.FileNameTemplate, fname, true) == 0 && (id < 0 || s.StripId == id))
                    {
                        if (id < 0) s.OwnerZone.Killed = true;
                        strlist.Add(s);
                    }
                foreach (Job[] ss in TrackServers.Values)
                    if (ss != null)
                    {
                        int i;
                        for (i = 0; i < ss.Length; i++)
                        {
                            Job s = ss[i];
                            if (s != null && String.Compare(s.OwnerZone.FileNameTemplate, fname, true) == 0 && (id < 0 || s.StripId == id))
                            {
                                if (id < 0)
                                    s.OwnerZone.Killed = true;
                                ss[i] = null;
                            }
                        }
                    }
                foreach (Job s in strlist)
                    try
                    {
                        WaitingJobs.Remove(s);
                    }
                    catch (Exception x)
                    {
                        Console.WriteLine("WaitingJob Removal Error: " + x.Message);
                    }
                foreach (Job s in strlist)
                    try
                    {
                        ActiveJobs.Remove(s);
                    }
                    catch (Exception x)
                    {
                        Console.WriteLine("ActiveJob Removal Error: " + x.Message);
                    }
            }
        }

        public static void AddJob(string fname)
        {
            string indir = C.InputDir;
            if (indir.EndsWith("\\") == false && indir.EndsWith("/") == false) indir += "\\";
            if (System.IO.File.Exists(indir + fname))
            {
                System.IO.FileSystemEventArgs e = new System.IO.FileSystemEventArgs(System.IO.WatcherChangeTypes.Created, indir.Substring(0, indir.Length - 1), fname);
                new dCreateFile(onCreated).BeginInvoke(e, null, null);
            }
            else Program.Log("Can't find the specified file in input directory \"" + C.InputDir + "\" .");
        }

        public static string ListJobs()
        {
            System.IO.StringWriter wr = new System.IO.StringWriter();
            lock (ActiveJobs)
            {
                wr.WriteLine("WAITING:");
                foreach (Job s in WaitingJobs)
                    wr.WriteLine(s.ToString());
                wr.WriteLine("ACTIVE:");
                foreach (Job s in ActiveJobs)
                    wr.WriteLine(s.ToString());
                wr.WriteLine("RUNNING:");
                foreach (Job[] sps in TrackServers.Values)
                    foreach (Job s in sps)
                        if (s != null)
                            wr.WriteLine(s.ToString());
                wr.Flush();
            }
            return wr.ToString();
        }

        internal static IEnumerable<Job> GetJobs()
        {
#if ENABLE_TESTS
            var lj = new List<Job>();
            lock (ActiveJobs)
            {
                lj.AddRange(WaitingJobs);
                lj.AddRange(ActiveJobs);
                foreach (Job[] sps in TrackServers.Values)
                    foreach (Job s in sps)
                        if (s != null)
                            lj.Add(s);
            }
            return lj;
#endif
        }

        class Command
        {
            public delegate void dExec(System.Text.RegularExpressions.Match m);

            public string Description;
            public System.Text.RegularExpressions.Regex RX;
            public dExec Executor;

            public Command(string desc, System.Text.RegularExpressions.Regex rx, dExec xc)
            {
                Description = desc;
                RX = rx;
                Executor = xc;
            }

            public bool Execute(string line)
            {
                System.Text.RegularExpressions.Match m = RX.Match(line);
                if (m.Success)
                    Executor(m);
                return m.Success;
            }
        }

        static Command[] s_Commands = new Command[]
        {
            new Command("help -> show this help", 
                new System.Text.RegularExpressions.Regex(@"\s*help\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m) 
                {
                    foreach (Command c in s_Commands)
                        Console.WriteLine(c.Description);
                }),
            new Command("exit -> shuts down the service", 
                new System.Text.RegularExpressions.Regex(@"\s*exit\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Terminate = true;
                }),
            new Command("show inputdir -> shows the current input directory", 
                new System.Text.RegularExpressions.Regex(@"\s*show\s+inputdir\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Console.WriteLine(C.InputDir);
                }),
            new Command("show outputdir -> shows the current output directory", 
                new System.Text.RegularExpressions.Regex(@"\s*show\s+outputdir\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Console.WriteLine(C.OutputDir);
                }),
            new Command("show cmdfile -> shows the current command file", 
                new System.Text.RegularExpressions.Regex(@"\s*show\s+cmdfile\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Console.WriteLine("\"" + C.CommandFile + "\"");
                }),
            new Command("show fswbuffersize -> shows the current filesystemwatcher buffer size", 
                new System.Text.RegularExpressions.Regex(@"\s*show\s+fswbuffersize\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Console.WriteLine(C.FSWBufferSize);
                }),            
            new Command("set inputdir = <parameter> -> sets the input directory", 
                new System.Text.RegularExpressions.Regex(@"\s*set\s+inputdir\s*=\s*(.*)"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    string dir = m.Groups[1].Value.Trim();
                    if (dir.Length > 0)
                    {
                        C.InputDir = dir;
                        C.Save();
                    }
                    else Console.WriteLine("Empty directory name is invalid");
                    CheckConfig();
                }),
            new Command("set outputdir = <parameter> -> sets the output directory", 
                new System.Text.RegularExpressions.Regex(@"\s*set\s+outputdir\s*=\s*(.*)"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    string dir = m.Groups[1].Value.Trim();
                    if (dir.Length > 0)
                    {
                        C.OutputDir = dir;
                        C.Save();
                    }
                    else Console.WriteLine("Empty directory name is invalid");
                    CheckConfig();
                }),
            new Command("set cmdfile = <parameter> -> sets the command file", 
                new System.Text.RegularExpressions.Regex(@"\s*set\s+cmdfile\s*=\s*(.*)"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    string path = m.Groups[1].Value.Trim();
                    if (path.Length > 0)
                    {
                        C.CommandFile = path;
                        C.Save();
                    }
                    else Console.WriteLine("Empty file name is invalid");
                    CheckConfig();
                }),
            new Command("set fswbuffersize = <parameter> -> sets the size of the filesystemwatcher buffer", 
                new System.Text.RegularExpressions.Regex(@"\s*set\s+fswbuffersize\s*=\s*(\d+)\s*"), 
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    try
                    {
                        uint ub = uint.Parse(m.Groups[1].Value);
                        fsw.InternalBufferSize = (int)ub;
                        C.FSWBufferSize = ub;
                        C.Save();
                    }
                    catch (Exception x) { Console.WriteLine(x.Message); }
                }),
            new Command("process zone [autodelete=true|false] <parameter> -> requests processing for zone in the specified directory, optionally setting autodelete",
                new System.Text.RegularExpressions.Regex(@"\s*process\s+zone\s+((?:(?:autodelete)=[^=\s]+\s+)*)(.*)"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    bool? autodelete = null;
                    if (m.Groups.Count == 3)
                    {
                        System.Text.RegularExpressions.Regex trx = new System.Text.RegularExpressions.Regex(@"([^=]+)=(\S+)");
                        var groups = m.Groups[1].Value.Split(new char [] {' ', '\t'}, StringSplitOptions.RemoveEmptyEntries);
                        foreach (var g in groups)
                        {
                            var m1 = trx.Match(g);
                            if (m1.Success)
                            {
                                switch (m1.Groups[1].Value)
                                {
                                    case "autodelete": autodelete = bool.Parse(m1.Groups[2].Value); break;
                                    default: throw new Exception("Unsupported parameter \"" + m1.Groups[1].Value + "\".");
                                }
                            }
                        }
                    }
                    string newdir = m.Groups[m.Groups.Count - 1].Value.Trim();
                    if (newdir.EndsWith("/") || newdir.EndsWith("\\")) newdir = newdir.Substring(0, newdir.Length - 1);
                    string parentdir = newdir.Substring(0, newdir.LastIndexOfAny(new char [] { '\\', '/' }));
                    if (parentdir.Length == newdir.Length) 
                    {
                        Console.WriteLine("The directory must have a parent.");
                        return;
                    }
                    Console.WriteLine("WARNING: all *.CLS files will be deleted after processing. If you want to keep them, make them read-only.");
                    newdir = newdir.Substring(parentdir.Length + 1);
                    System.IO.FileSystemEventArgs earg = new System.IO.FileSystemEventArgs(System.IO.WatcherChangeTypes.Created, parentdir, newdir);
                    new dCreateFileWithParameters(onCreatedWithParameters).BeginInvoke(earg, autodelete, null, null);
                }),
            new Command("list jobs -> shows the list and status of jobs",
                new System.Text.RegularExpressions.Regex(@"\s*list\s+jobs\s*"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    Console.WriteLine(ListJobs());
                }),
            new Command("kill jobs <parameter> -> kills a job",
                new System.Text.RegularExpressions.Regex(@"\s*kill\s+jobs\s+(.*)"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    KillJobs(m.Groups[1].Value.Trim());
                }),
            new Command("skip job <strip> <parameter> -> skip",
                new System.Text.RegularExpressions.Regex(@"\s*skip\s+job\s+(\d+)\s+(.*)"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    KillJobs(m.Groups[2].Value.Trim(), int.Parse(m.Groups[1].Value));
                }),
            new Command("skip jobs <strip1>-<stripn> <parameter> -> skip",
                new System.Text.RegularExpressions.Regex(@"\s*skip\s+jobs\s+(\d+)\s*\-\s*(\d+)\s+(.*)"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    int start = int.Parse(m.Groups[1].Value);
                    int end = int.Parse(m.Groups[2].Value);
                    int i;
                    for (i = start; i <= end; i++)
                        KillJobs(m.Groups[3].Value.Trim(), i);
                }),
            new Command("enable job splitting -> enables job splitting",
                new System.Text.RegularExpressions.Regex(@"\s*enable\s+job\s+splitting\s*"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    C.EnableJobSplitting = true;
                    C.Save();
                }),
            new Command("disable job splitting -> disables job splitting",
                new System.Text.RegularExpressions.Regex(@"\s*disable\s+job\s+splitting\s*"),
                (Command.dExec) delegate (System.Text.RegularExpressions.Match m)
                {
                    C.EnableJobSplitting = false;
                    C.Save();
                })
        };

        static System.IO.FileSystemWatcher fsw = null;

        static void CheckConfig()
        {
            var fg = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            int pos;
            if (String.IsNullOrEmpty(C.InputDir))            
                Console.WriteLine("CONFIG ERROR: null input directory.");
            else if ((pos = C.InputDir.IndexOf(':')) >= 0)
            {
                if (pos == 1)
                    Console.WriteLine("CONFIG WARNING: input directory contains a drive letter specification. You MUST make sure that all GPUTrackingServers have the same drive letter mapping, otherwise the input files cannot be accessed." + 
                        System.Environment.NewLine + "An UNC path should always be preferred.");
                else
                    Console.WriteLine("CONFIG ERROR: ':' is not an allowed character in an input directory specification.");
            }
            else if (C.InputDir.StartsWith("\\\\") == false)
                Console.WriteLine("CONFIG ERROR: Input directory should start with a UNC path specification, e.g. \\\\192.168.0.1\\myshare");
            if (String.IsNullOrEmpty(C.OutputDir))
                Console.WriteLine("CONFIG ERROR: null output directory.");
            else if ((pos = C.OutputDir.IndexOf(':')) >= 0)
            {
                if (pos == 1)
                    Console.WriteLine("CONFIG WARNING: output directory contains a drive letter specification. You MUST make sure that all GPUTrackingServers have the same drive letter mapping, otherwise the output files cannot be produced." + 
                        System.Environment.NewLine + "An UNC path should always be preferred.");
                else
                    Console.WriteLine("CONFIG ERROR: ':' is not an allowed character in an output directory specification.");
            }
            else if (C.OutputDir.StartsWith("\\\\") == false)
                Console.WriteLine("CONFIG ERROR: Output directory should start with a UNC path specification, e.g. \\\\192.168.0.1\\myshare");            
            Console.ForegroundColor = fg;
        }

        static void Main(string[] args)
        {
#if ENABLE_TESTS
            if (args.Length > 0 && string.Compare(args[0], "/tests", true) == 0)
            {
                C.EnableLog = true;
                C.EnableJobSplitting = true;
                CodeTest.TestJobSplit(args[1]);
                C.EnableJobSplitting = false;
                CodeTest.TestAddRemoveTracker(args[1]);
                CodeTest.TestProcessZone(args[1]);
                C.EnableJobSplitting = true;
                CodeTest.TestProcessZone(args[1]);
                return;
            }
#endif
            if (MainFeedData(false, args)) return;
            if (args.Length > 0 && string.Compare(args[0], "/help", true) == 0)
            {
                Console.WriteLine("Usage:");
                Console.WriteLine("No arguments -> run as PostProcessingManager server");
                MainFeedData(true, args);
                return;
            }
            string line;            
            fsw = new System.IO.FileSystemWatcher(C.InputDir);
            fsw.InternalBufferSize = (int)C.FSWBufferSize;
            fsw.Created += new System.IO.FileSystemEventHandler(fsw_Created);
            fsw.Filter = "*" + ScanExtension;
            fsw.IncludeSubdirectories = false;
            fsw.EnableRaisingEvents = true;            
            Out.WriteLine("Starting tracker/job supervisor.");
            System.Timers.Timer supv = new System.Timers.Timer(1000);
            supv.Elapsed += new System.Timers.ElapsedEventHandler(supv_Elapsed);
            supv.Start();
            Out.WriteLine("Starting command file polling.");
            System.Timers.Timer cmdv = new System.Timers.Timer(1000);
            cmdv.Elapsed += new System.Timers.ElapsedEventHandler(cmdfile_Elapsed);
            cmdv.Start();
            Out.WriteLine("Starting Web server.");
            SySal.Web.WebServer ws = new SySal.Web.WebServer(1784, new Program());                        
            CheckConfig();
            Out.WriteLine("Service started, type exit to terminate; type help to get the list of allowed commands.");
            while ((line = Console.ReadLine()) != null && Terminate == false)
            {
                bool processed = false;
                System.Text.RegularExpressions.Match m;
                lock (s_Commands)                
                    foreach (Command c in s_Commands)
                    {
                        m = c.RX.Match(line);
                        if (m.Success)
                        {
                            processed = true;
                            c.Executor(m);
                            break;
                        }
                    }
                if (processed == false) Console.WriteLine("Unknown command or bad syntax.");
            }
            Terminate = true;
            Out.WriteLine("Stopping supervisor.");
            supv.Stop();
            Out.WriteLine("Stopping command polling");
            cmdv.Stop();
            Out.WriteLine("Terminating.");
        }

        static void cmdfile_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            string[] lines = new string[0];
            try
            {
                lines = System.IO.File.ReadAllLines(C.CommandFile);                
            }
            catch (Exception) { }
            try
            {
                System.IO.File.Delete(C.CommandFile);
            }
            catch (Exception) { }
            lock (s_Commands)
                foreach (var s in lines)                
                    foreach (Command c in s_Commands)
                    {
                        var m = c.RX.Match(s);
                        if (m.Success)
                        {
                            Out.WriteLine("--> " + s);
                            c.Executor(m);
                            break;
                        }
                    }
        }

        static bool supv_Elapsed_Running = false;

        internal static void supv_Elapsed(object sender, System.Timers.ElapsedEventArgs e)
        {
            if (supv_Elapsed_Running) return;
            try
            {
                supv_Elapsed_Running = true;
                lock (ActiveJobs)
                {
                    string[] sj = new string[WaitingJobs.Count];
                    int j = 0;
                    foreach (Job s in WaitingJobs)
                        sj[j++] = s.ToString();
                    MonInfo.WaitingJobs = sj;
                    sj = new string[ActiveJobs.Count];
                    j = 0;
                    foreach (Job s in ActiveJobs)
                        sj[j++] = s.ToString();
                    MonInfo.ActiveJobs = sj;
                    System.Collections.ArrayList ar = new System.Collections.ArrayList();
                    foreach (string k in TrackServers.Keys)
                    {
                        Job[] sps1 = TrackServers[k];
                        int sps1i;
                        for (sps1i = 0; sps1i < sps1.Length; sps1i++)
                        {
                            Job s = sps1[sps1i];
                            if (s != null)
                                ar.Add(s.ToString() + " (" + k + "/" + sps1i + ")");
                        }
                    }
                    MonInfo.RunningJobs = (string[])ar.ToArray(typeof(string));
                    var ks = TrackServers.Keys.ToArray();
                    foreach (string k in ks)
                        CheckTrackerJobs(k);
                }
            }
            finally
            {
                supv_Elapsed_Running = false;
            }
        }

        delegate void dCheckTrackerJobs(string addr);

        static void CheckTrackerJobs(string addr)
        {
            int queues =
#if ENABLE_TESTS
                ((string.Compare(addr, CodeTest.TestServer, true) == 0) ? CodeTest.HTTPCheckTrackerNumber(addr) :
#endif
                HTTPCheckTrackerNumber(addr)
#if ENABLE_TESTS
            )
#endif
            ;

            if (queues < 0)
            {
                AddRemoveTracker(addr, false);
                return;
            }
            Job[] srv = null;
            lock (ActiveJobs)
            {
                if (TrackServers.Keys.Contains(addr) == false)
                {
                    TrackServers.Add(addr, new Job[queues]);
                    int tks = 0;
                    foreach (Job[] sps in TrackServers.Values)
                        tks += sps.Length;
                    MonInfo.TotalTrackers = tks;
                }
                else
                {
                    srv = UpdateTrackerQueuesAndGet(addr, queues);
                    ScheduleFirstActiveJob(addr, srv);
                    CheckJobs(addr, srv);
                }
            }

        }

        internal static Job[] UpdateTrackerQueuesAndGet(string addr, int queues)
        {
            var srv = TrackServers[addr];
            if (srv.Length != queues)
            {
                Program.Log("Changing queues to " + addr + ": " + queues);
                AddRemoveTracker(addr, false);
                AddRemoveTracker(addr, true);
                srv = TrackServers[addr] = new Job[queues];
                int tks = 0;
                foreach (Job[] sps in TrackServers.Values)
                    tks += sps.Length;
                MonInfo.TotalTrackers = tks;            
            }
            return srv;
        }

        internal static void ScheduleFirstActiveJob(string hostname, Job [] srv)
        {
            int isi;
            for (isi = 0; isi < srv.Length; isi++)
                if (srv[isi] == null)
                {
                    Job sp = null;
                    lock (ActiveJobs)
                        if (ActiveJobs.Count > 0)
                        {
                            sp = ActiveJobs.First();
                            if (sp != null)
                            {
                                if (C.EnableJobSplitting && sp.Split == false)
                                {                                    
                                    uint trackers = Math.Max(1, CountTrackers());
                                    var spj = sp.TrySplitJob(trackers);
                                    if (C.EnableLog) Program.Log("Attempting splitting job to " + trackers + " tracker(s): " + spj.Count() + " jobs.");
                                    ActiveJobs.RemoveFirst();
                                    foreach (var sp1 in spj)                                        
                                        ActiveJobs.AddFirst(sp1);
                                    sp = ActiveJobs.First();
                                }
                                ScheduleJob(hostname, srv, isi, sp);
                            }
                        }
                }
        }

        internal static void CheckJobs(string addr, Job [] srv)
        {
            int s = 0;
            for (s = 0; s < srv.Length; s++)
                if (srv[s] != null && srv[s].JobName != null)
                {
                    if (srv[s].OwnerZone.Killed) srv[s] = null;
                    else CheckJob(addr, s, srv[s]);
                }

        }

        internal static void ScheduleJob(string addr, Job[] srv, int queuen, Job sp)
        {
            if (sp.Lock())
                try
                {
                    string outname = C.EnableJobSplitting ?
#if ENABLE_TESTS
                        ((string.Compare(addr, CodeTest.TestServer, true) == 0) ? CodeTest.HTTPScheduleJob(addr, queuen, sp.OwnerZone.GetRWDFileName(sp, C.OutputDir), (int)sp.StripId, (int)sp.FirstView, (int)sp.LastView, (int)sp.FragmentIndex) :
#endif
                        HTTPScheduleJob(addr, queuen, sp.OwnerZone.StartFileName, (int)sp.StripId, (int)sp.FirstView, (int)sp.LastView, (int)sp.FragmentIndex)
#if ENABLE_TESTS
                        )
#endif
                        :
#if ENABLE_TESTS
                        ((string.Compare(addr, CodeTest.TestServer, true) == 0) ? CodeTest.HTTPScheduleJob(addr, queuen, sp.OwnerZone.GetRWDFileName(sp, C.OutputDir), (int)sp.StripId) :
#endif
                        HTTPScheduleJob(addr, queuen, sp.OwnerZone.StartFileName, (int)sp.FragmentIndex)
#if ENABLE_TESTS
                        )
#endif
                        ;
                    if (outname != null)
                    {
                        sp.JobName = outname;
                        srv[queuen] = sp;
                        lock (ActiveJobs)
                            if (ActiveJobs.Contains(sp))
                                ActiveJobs.Remove(sp);
                        if (C.EnableLog) Program.Log("Scheduled job \"" + sp.JobName + "\" on " + addr + " " + queuen + ".");
                    }
                    else if (C.EnableLog) Program.Log("Failed to schedule job \"" + sp.OwnerZone.FileNameTemplate + " " + sp.StripId + "\" on " + addr + " " + queuen + ".");
                }
                finally
                {
                    sp.Unlock();
                }
        }

        static void CheckJob(string addr, int q, Job sp)
        {
            if (sp.Lock())
                try
                {
                    bool resOK = false;
                    bool reschedule = false;
                    int trials;
                    if (sp.Completed) return;
                    else
                        switch (
#if ENABLE_TESTS
                            ((string.Compare(addr, CodeTest.TestServer, true) == 0) ? CodeTest.HTTPCheckJob(addr, q, sp.JobName) :
#endif
                            HTTPCheckJob(addr, q, sp.JobName)
#if ENABLE_TESTS
                            )
#endif
                            )
                        {
                            case JobOutcome.Failed: resOK = false; reschedule = false; break;
                            case JobOutcome.Done: resOK = true; reschedule = false; break;
                            case JobOutcome.Waiting: return;
                            default: resOK = false; reschedule = true; break;
                        }
                    lock (ActiveJobs)
                    {
                        if (TrackServers.Keys.Contains(addr))
                        {
                            Job[] splist = TrackServers[addr];
                            int i;
                            for (i = 0; i < splist.Length; i++)
                                if (splist[i] == sp)
                                    splist[i] = null;
                        }
                        if (resOK)
                        {
                            if (sp.Completed) return;
                            string outname = sp.OwnerZone.GetRWDFileName(sp, C.OutputDir);
                            string workname = outname + ".twr";
                            for (trials = 3; trials >= 0; trials--)
                            {
                                try
                                {
                                    if (System.IO.File.Exists(outname))
                                    {
                                        if (C.EnableLog) Program.Log("Deleting duplicated file " + workname + ": " + outname + " exists.");
                                        System.IO.File.Delete(workname);
                                        break;
                                    }
                                    else
                                    {
                                        if (C.EnableLog) Program.Log("Renaming " + workname + " to " + outname);
                                        System.IO.File.Move(workname, outname);
                                        break;
                                    }
                                }
                                catch (Exception x0)
                                {
                                    if (C.EnableLog) Program.Log("ERROR managing job file: " + x0.Message);
                                }
                            }
                            sp.Completed = true;
                            if (sp.OwnerZone.AutoDelete)
                            {
                                if (sp.Split == false || System.Threading.Interlocked.Decrement(ref sp.SplitCounter[0]) == 0)
                                    sp.OwnerZone.DeleteStripFiles(sp.StripId);
                            }
                            return;
                        }
                        if (reschedule && sp.Completed == false)
                            RestoreActiveJob(sp);
                    }
                }
                finally
                {
                    sp.Unlock();
                }
        }

        delegate void dCreateFile(System.IO.FileSystemEventArgs e);

        delegate void dCreateFileWithParameters(System.IO.FileSystemEventArgs e, bool? autodelete);

        static void fsw_Created(object sender, System.IO.FileSystemEventArgs e)
        {
            new dCreateFile(onCreated).BeginInvoke(e, null, null);
        }

        static void onCreated(System.IO.FileSystemEventArgs e)
        {
            CreatedWithOptions(e, null);
        }

        static void onCreatedWithParameters(System.IO.FileSystemEventArgs e, bool? autodelete)
        {
            CreatedWithOptions(e, autodelete);
        }

        internal static void TestCreatedWithOptions(string zonedir, bool? autodelete)
        {
#if ENABLE_TESTS
            CreatedWithOptions(new System.IO.FileSystemEventArgs(System.IO.WatcherChangeTypes.Created, zonedir.Substring(0, zonedir.LastIndexOf(System.IO.Path.DirectorySeparatorChar)), zonedir.Substring(zonedir.LastIndexOf(System.IO.Path.DirectorySeparatorChar) + 1)), autodelete, false);
#endif
        }

        static void CreatedWithOptions(System.IO.FileSystemEventArgs e, bool? autodelete, bool allowimportcatalog = true)
        {
            Program.Log("Detected " + e.FullPath);
            try
            {
                int retryreaddir;
                Zone z = null;
                for (retryreaddir = 10; retryreaddir > 0; retryreaddir--)
                    try
                    {
                        System.Threading.Thread.Sleep(1000);
                        string[] scanfiles = System.IO.Directory.GetFiles(e.FullPath, "*" + ScanExtension);
                        if (scanfiles.Length == 1)
                        {
                            z = Zone.FromFile(scanfiles[0], allowimportcatalog);
                            break;
                        }
                    }
                    catch (Exception) { }
                if (retryreaddir <= 0) throw new Exception("Can't find an unambiguous " + ScanExtension + " file in the specified directory.");
                z.OutDir = C.OutputDir;
                z.WriteCatalog();
                if (autodelete != null) z.AutoDelete = (bool)autodelete;
                uint s;
                uint newjobs = 0;
                List<Job> towaiting = new List<Job>();
                for (s = 0; s < z.Strips; s++)
                {
                    Job str = new Job();
                    str.Split = false;
                    str.StripId = s;
                    str.FirstView = 0;
                    str.LastView = z.Views - 1;
                    str.OwnerZone = z;
                    towaiting.Add(str);
                    newjobs++;
                }
                if (towaiting.Count >= 0)
                    lock (ActiveJobs)
                    {
                        foreach (Job sps in towaiting)
                            WaitingJobs.AddLast(sps);
                    }
                Program.Log("Added " + newjobs + " new job(s)");
            }
            catch (Exception x)
            {
                Program.Log(x.ToString());
            }
        }

        static bool Terminate = false;

        public string ApplicationName
        {
            get { return "SySal QSS PostProcessingManager"; }
        }

        const string a_alive = "alive=";

        internal class MonitoringInfo
        {
            public int TotalTrackers;

            public int TotalTrackServers;

            public string[] WaitingJobs = new string[0];

            public string[] ActiveJobs = new string[0];

            public string[] RunningJobs = new string[0];

            public override string ToString()
            {
                return
                    "TotalTrackers: " + TotalTrackers + System.Environment.NewLine +
                    "TotalTrackServers: " + TotalTrackServers + System.Environment.NewLine +
                    "WaitingJobs: " + WaitingJobs.Length + System.Environment.NewLine +
                    "ActiveJobs: " + ActiveJobs.Length + System.Environment.NewLine +
                    "RunningJobs: " + RunningJobs.Length;
            }
        }

        internal static MonitoringInfo MonInfo = new MonitoringInfo();

        const string autorefresh_str = "autorefresh=";

        public Web.ChunkedResponse HttpGet(Web.Session sess, string page, params string[] queryget)
        {
            uint autorefresh = 600;
            switch (page)
            {
                case "/trksrv_alive":
                    {
                        foreach (string s in queryget)
                            if (s.StartsWith(a_alive))
                                try
                                {
                                    bool isalive = bool.Parse(s.Substring(a_alive.Length));
                                    string addr = sess.ClientAddress.ToString();
                                    new dSrvOp(AddRemoveTracker).BeginInvoke(addr, isalive, null, null);
                                    return new SySal.Web.HTMLResponse("ACKNOWLEDGED");
                                }
                                catch (Exception) 
                                {
                                    return new SySal.Web.HTMLResponse("WRONG SYNTAX");
                                }
                    }
                    return new SySal.Web.HTMLResponse("MISSING PARAMETER bool alive");                    
            }

            foreach (var ss in queryget)
                if (ss.StartsWith(autorefresh_str))
                    try
                    {
                        autorefresh = uint.Parse(ss.Substring(autorefresh_str.Length));                    
                    }
                    catch (Exception)
                    {

                    }
                    finally
                    {
                        if (autorefresh <= 0) autorefresh = 600;
                    }

            System.IO.StringWriter swr = new System.IO.StringWriter();
            swr.WriteLine("<html><head><title>PostProcessingManager " + ProcessInstance + " at " + ComputerName + "</title>" +
                " <meta http-equiv=\"refresh\" content=\"" + autorefresh + "\">" +
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
                "p " +
                "{ " +
                " color : Blue; font-family : Segoe UI, Trebuchet MS, Verdana, Helvetica, Arial; font-size : 10px; opacity: 1; line-height: 10px; " +
                "} " +
                "  </style> " +
                "</head><body>");
            swr.WriteLine("<h1>QSS PostProcessingManager " + ComputerName + "</h1><h6>Process Instance " + ProcessInstance + " </h6>");
            swr.WriteLine("<p>Set refresh time(s): <a href=\"?autorefresh=10\">10</a> <a href=\"?autorefresh=30\">30</a> <a href=\"?autorefresh=60\">60</a> <a href=\"?autorefresh=600\">600</a></p>");
            swr.WriteLine("<h6>Trackers: " + MonInfo.TotalTrackers + "</h6>");
            swr.WriteLine("<h6>TrackServers: " + MonInfo.TotalTrackServers + "</h6>");            
            swr.WriteLine("<table align=\"left\"><tr><td><table align=\"left\">");
            swr.WriteLine("<tr><th>Waiting Jobs</th></tr>");
            foreach (string sj in MonInfo.WaitingJobs)
                swr.WriteLine("<tr><td>" + SySal.Web.WebServer.HtmlFormat(sj) + "</td></tr>");
            swr.WriteLine("</table></td></tr>");
            swr.WriteLine("<tr><td><table align=\"left\">");
            swr.WriteLine("<tr><th>Active Jobs</th></tr>");
            foreach (string sj in MonInfo.ActiveJobs)
                swr.WriteLine("<tr><td>" + SySal.Web.WebServer.HtmlFormat(sj) + "</td></tr>");
            swr.WriteLine("</table></td></tr>");
            swr.WriteLine("<tr><td><table align=\"left\">");
            swr.WriteLine("<tr><th>Running Jobs</th></tr>");
            foreach (string sj in MonInfo.RunningJobs)
                swr.WriteLine("<tr><td>" + SySal.Web.WebServer.HtmlFormat(sj) + "</td></tr>");
            swr.WriteLine("</td></tr></table>");
            swr.WriteLine("</body></html>\r\n\r\n");
            return new SySal.Web.HTMLResponse(swr.ToString());

        }

        delegate void dSrvOp(string addr, bool alive);

        internal static void AddRemoveTracker(string addr, bool alive)
        {
            Job[] stripsremoved = null;
            lock (ActiveJobs)
            {
                if (TrackServers.Keys.Contains(addr))
                {
                    if (alive == false)
                    {
                        stripsremoved = TrackServers[addr];
                        TrackServers.Remove(addr);
                        if (C.EnableLog)
                            Program.Log("Removed tracker \"" + addr + "\" with " + stripsremoved.Length + " job slots.");
                    }
                }
                else
                {
                    if (alive == true)
                    {
                        TrackServers.Add(addr, new Job[0]);
                        if (C.EnableLog)
                            Program.Log("Added tracker \"" + addr + "\".");
                    }
                }
                MonInfo.TotalTrackServers = TrackServers.Keys.Count;
                int trks = 0;
                foreach (Job[] sps in TrackServers.Values)
                    trks += sps.Length;
                MonInfo.TotalTrackers = trks;
                if (stripsremoved != null)
                    foreach (Job s in stripsremoved)
                        if (s != null)
                            RestoreActiveJob(s);
            }
        }

        public static bool RestoreActiveJob(Job s)
        {
            if (ActiveJobs.Contains(s) == false)
            {
                ActiveJobs.AddFirst(s);
                if (C.EnableLog)
                    Program.Log("Added active job " + s.ToString() + " .");
                return true;
            }
            else
            {
                if (C.EnableLog)
                    Program.Log("Prevented rescheduling of active job " + s.ToString() + " .");
                return false;
            }
        }

        public Web.ChunkedResponse HttpPost(Web.Session sess, string page, params string[] postfields)
        {
            return HttpGet(sess, page, postfields);
        }

        public bool ShowExceptions
        {
            get { return false; }
        }
    }
}
