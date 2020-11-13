using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;
using System.Runtime.InteropServices;
using SySal.Executables.NExTScanner;
using SySal.DAQSystem;
using System.Linq;

namespace SySal.Executables.NExTScanner
{
    [Serializable]
    public class ScannerSettings : SySal.Management.Configuration
    {        
        public string StageLibrary = "";
        public string GrabberLibrary = "";
        public string GPULibrary = "";
        public string LogDirectory = "c:\\SySal.NET\\Logs";
        public string DataDirectory = "c:\\SySal.NET\\Data";
        public string ScanServerDataDirectory = "\\\\mysrv.scan.com\\data";
        public string ConfigDirectory = "c:\\SySal.NET\\Configs";
        public bool[] EnabledGPUs = new bool[0];

        public ScannerSettings() : base("") { }
        public ScannerSettings(string name) : base(name) { }        

        internal static ScannerSettings Default
        {
            get
            {
                ScannerSettings c = SySal.Management.MachineSettings.GetSettings(typeof(ScannerSettings)) as ScannerSettings;
                if (c == null) c = new ScannerSettings("Default");
                return c;
            }
        }

        public override object Clone()
        {
            ScannerSettings c = new ScannerSettings(Name);            
            c.StageLibrary = StageLibrary;
            c.GrabberLibrary = GrabberLibrary;
            c.GPULibrary = GPULibrary;
            c.LogDirectory = LogDirectory;
            c.DataDirectory = DataDirectory;
            c.ScanServerDataDirectory = ScanServerDataDirectory;
            c.ConfigDirectory = ConfigDirectory;
            c.EnabledGPUs = (bool [])EnabledGPUs.Clone();
            return c;
        }
    }
    
    public class Scanner : SySal.Management.IMachineSettingsEditor, IDisposable, ISySalLog, ISySalCameraDisplay, IMapProvider
    {
        [DllImport("kernel32", SetLastError = true)]
        private static extern bool FlushFileBuffers(IntPtr handle);

        protected ScannerSettings S = ScannerSettings.Default;

        protected ImagingConfiguration IC = ImagingConfiguration.Default;

        protected QuasiStaticAcqSettings QC = QuasiStaticAcqSettings.Default;

        protected FullSpeedAcqSettings FSC = FullSpeedAcqSettings.Default;

        protected SySal.DAQSystem.ScanServer NSS = null;

        protected bool m_NSS_connected = false;

        public bool NSS_connected
        {
            get { return m_NSS_connected; }
        }

        internal class SImageProcessor
        {
            public SySal.Imaging.IImageProcessor IProc;
            public bool Enabled;

            public SImageProcessor(SySal.Imaging.IImageProcessor iproc)
            {
                IProc = iproc;
                Enabled = false;
            }

            public override string ToString()
            {
                return IProc.ToString() + "\r\nEnabled: " + Enabled.ToString();
            }
        }

        public System.Threading.ManualResetEvent m_TerminateEvent = new System.Threading.ManualResetEvent(false);

        public System.Threading.Mutex m_DisplayMutex = new System.Threading.Mutex(false);

        public System.Threading.Thread m_DisplayThread;
        
        public Scanner(IScannerDataDisplay isd)
        {
            S = ScannerSettings.Default;
            {
                System.DateTime lfn = System.DateTime.Now;
                m_LogFileName = S.LogDirectory;
                if (m_LogFileName.EndsWith("\\") == false && m_LogFileName.EndsWith("/") == false) m_LogFileName += "\\";
                m_LogFileName += lfn.Year.ToString("D04") + lfn.Month.ToString("D02") + lfn.Day.ToString("D02") + lfn.Hour.ToString("D02") + lfn.Minute.ToString("D02") + lfn.Second.ToString("D02") + ".log";
            }
            m_GeneralTimeSource = new System.Diagnostics.Stopwatch();
            m_GeneralTimeSource.Reset();
            m_GeneralTimeSource.Start();
            m_CameraDisplay = new CameraDisplay();            
            m_CameraDisplay.m_DefaultDirectory = S.DataDirectory;
            m_CameraDisplay.iMap = this;
            m_CameraDisplay.Show();
            m_DisplayThread = new System.Threading.Thread(new System.Threading.ThreadStart(CameraDisplayRefresh));
            m_EnableDisplay = false;
            m_DisplayMutex.WaitOne();
            Log(StartStop, "Scanner started");
            m_ScanDataDisplay = isd;
            ApplyMachineSettings();            
            EnableAutoRefresh = true;            
            m_DisplayThread.Start();
            System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new System.Runtime.Remoting.Channels.Tcp.TcpChannel((int)SySal.DAQSystem.OperaPort.ScanServer));
            NSS = new SySal.DAQSystem.ScanServer();
            NSS.m_Owner = this;
        }        

        const string StartStop = "StartStop";

        protected System.Diagnostics.Stopwatch m_GeneralTimeSource = null;

        protected IScannerDataDisplay m_ScanDataDisplay = null;

        public IScannerDataDisplay ScanDataDisplay
        {
            set { m_ScanDataDisplay = value; }
        }

        public static Type FindCompatibleType(string libraryfile, Type neededinterface)
        {
            System.Reflection.Assembly ass = null;
            try
            {
                ass = System.Reflection.Assembly.LoadFile(libraryfile);
                Type[] types = ass.GetExportedTypes();
                foreach (Type t in types)
                    foreach (Type ti in t.GetInterfaces())
                        if (ti == neededinterface)
                            return t;
            }
            catch (Exception x)
            {
                ass = null;
                MessageBox.Show(x.ToString(), "Error browsing types.", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            return null;
        }

        internal static string GPUMonitorString(int device)
        {
            return "GPU #" + device + " info";
        }

        public void ApplyMachineSettings()
        {
            const string _thismethod_ = "ApplyMachineSettings";
            bool displayenabled = EnableAutoRefresh;
            EnableAutoRefresh = false;
            try
            {
                m_EnableDisplay = false;
                //lock (m_DisplayLock)
                {
                    int i;
                    for (i = 0; i < iGPU.Length; i++)
                    {
                        Log(_thismethod_, "Closing GPU " + i);
                        if (m_ScanDataDisplay != null) m_ScanDataDisplay.CloseMonitor(GPUMonitorString(i));
                        iGPU[i].IProc.Dispose();
                        Log(_thismethod_, "Closed GPU " + i);
                    }
                    iGPU = new SImageProcessor[0];
                    if (iGrab != null)
                    {
                        Log(_thismethod_, "Disposing Grabber");
                        iGrab.Dispose(); iGrab = null;
                        Log(_thismethod_, "Disposed Grabber");
                    }
                    if (iStage != null)
                    {
                        Log(_thismethod_, "Disposing Stage");
                        iStage.Dispose(); iStage = null;
                        Log(_thismethod_, "Disposed Stage");
                    }
                    Type t = null;
                    if (S.GrabberLibrary.Trim().Length != 0)
                    {
                        if ((t = FindCompatibleType(S.GrabberLibrary, typeof(SySal.Imaging.IImageGrabber))) == null)
                            MessageBox.Show("Can't use the specified library for image grabbing.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        else
                            try
                            {
                                Log(_thismethod_, "Creating Grabber");
                                iGrab = (SySal.Imaging.IImageGrabber)System.Activator.CreateInstance(t);
                                iGrab.TimeSource = m_GeneralTimeSource;
                                iGrab.SequenceSize = 1;
                                Log(_thismethod_, "Created Grabber");
                            }
                            catch (Exception x)
                            {
                                MessageBox.Show(x.ToString(), "Can't create instance of \"" + t + "\".", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            }
                    }
                    if (S.GPULibrary.Trim().Length != 0)
                    {
                        if ((t = FindCompatibleType(S.GPULibrary, typeof(SySal.Imaging.IImageProcessor))) == null)
                            MessageBox.Show("Can't use the specified library for image processing.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        else
                        {
                            System.Collections.ArrayList igpu = new System.Collections.ArrayList();
                            while (true)
                                try
                                {
                                    Log(_thismethod_, "Creating GPU " + igpu.Count);
                                    igpu.Add(new SImageProcessor((SySal.Imaging.IImageProcessor)System.Activator.CreateInstance(t, new object[] { igpu.Count })));
                                    Log(_thismethod_, "Created GPU " + (igpu.Count - 1));
                                }
                                catch (Exception)
                                {
                                    break;
                                }
                            iGPU = (SImageProcessor[])igpu.ToArray(typeof(SImageProcessor));
                            for (i = 0; i < S.EnabledGPUs.Length && i < iGPU.Length; i++) iGPU[i].Enabled = S.EnabledGPUs[i];
                            S.EnabledGPUs = new bool[iGPU.Length];
                            for (i = 0; i < S.EnabledGPUs.Length; i++) S.EnabledGPUs[i] = iGPU[i].Enabled;
                            Log(_thismethod_, "Applying GPU configuration");
                            ApplyGPUConfig();
                            Log(_thismethod_, "Applied GPU configuration");
                        }
                    }
                    if (S.StageLibrary.Trim().Length != 0)
                    {
                        if ((t = FindCompatibleType(S.StageLibrary, typeof(SySal.StageControl.IStage))) == null)
                            MessageBox.Show("Can't use the specified library for stage control.", "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        else
                            try
                            {
                                Log(_thismethod_, "Creating stage");
                                iStage = (SySal.StageControl.IStage)System.Activator.CreateInstance(t);
                                Log(_thismethod_, "Created stage");
                                iStage.TimeSource = m_GeneralTimeSource;
                                Log(_thismethod_, "Synchronized stage");
                                m_CameraDisplay.Stage = iStage;
                                Log(_thismethod_, "Set stage to CameraDisplay");
                            }
                            catch (Exception x)
                            {
                                MessageBox.Show(x.ToString(), "Can't create instance of \"" + t + "\".", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            }
                    }
                    Log(_thismethod_, "Refreshing info");
                    if (m_ScanDataDisplay != null)
                    {
                        for (i = 0; i < iGPU.Length; i++)
                            m_ScanDataDisplay.DisplayMonitor(GPUMonitorString(i), iGPU[i]);
                        m_ScanDataDisplay.DisplayMonitor("Grabber", iGrab);
                    }
                    Log(_thismethod_, "Refreshed info");
                }               
            }
            finally
            {
                EnableAutoRefresh = displayenabled;
            }
        }

        class DatedDCTInterpolationImage
        {
            public DateTime RequestedTime;
            public SySal.Imaging.DCTInterpolationImage Image;
        }

        static Dictionary<string, DatedDCTInterpolationImage> DCTInterpolationImageCache = new Dictionary<string, DatedDCTInterpolationImage>();

        public static SySal.Imaging.DCTInterpolationImage DCTInterpolationImageFromString(string dctstr)
        {
            lock (DCTInterpolationImageCache)
            {
                string tr = dctstr.Trim();
                if (DCTInterpolationImageCache.ContainsKey(tr))
                {
                    var dca = DCTInterpolationImageCache[tr];
                    dca.RequestedTime = DateTime.Now;
                    return dca.Image;
                }
                var dci = SySal.Imaging.DCTInterpolationImage.FromDCTString(tr);
                DCTInterpolationImageCache.Add(tr, new DatedDCTInterpolationImage() { Image = dci, RequestedTime = DateTime.Now });
                if (DCTInterpolationImageCache.Count > 10)
                {
                    var mintime = DCTInterpolationImageCache.Values.Select(x => x.RequestedTime).Min();
                    DCTInterpolationImageCache = DCTInterpolationImageCache.TakeWhile(kv => kv.Value.RequestedTime > mintime).ToDictionary(kv => kv.Key, kv => kv.Value);
                }
                return dci;
            }
        }

        void ApplyGPUConfig()
        {
            int i;
            SySal.Imaging.Image eqim = (IC.EmptyImage != null && IC.EmptyImage.Length > 0) ? SySal.Imaging.Base64ImageEncoding.ImageFromBase64(IC.EmptyImage) : null;
            SySal.Imaging.DCTInterpolationImage dctim = (IC.ThresholdImage != null && IC.ThresholdImage.Length > 0) ? DCTInterpolationImageFromString(IC.ThresholdImage) : null;
            for (i = 0; i < iGPU.Length; i++)
                try
                {
                    iGPU[i].IProc.ImageFormat = iGrab.ImageFormat;
                    iGPU[i].IProc.MaxSegmentsPerScanLine = IC.MaxSegmentsPerLine;
                    iGPU[i].IProc.MaxClustersPerImage = IC.MaxClusters;
                    iGPU[i].IProc.OutputFeatures = SySal.Imaging.ImageProcessingFeatures.Cluster2ndMomenta | SySal.Imaging.ImageProcessingFeatures.BinarizedImage;
                    iGPU[i].IProc.EqGreyLevelTargetMedian = (byte)IC.GreyTargetMedian;
                    iGPU[i].IProc.EmptyImage = eqim;
                    iGPU[i].IProc.ThresholdImage = dctim;
                }
                catch (Exception x)
                {
                    Log("ApplyGPUConfig", x.ToString());
                }

        }

        internal SySal.StageControl.IStage iStage = null;
        internal SySal.Imaging.IImageGrabber iGrab = null;
        internal SImageProcessor[] iGPU = new SImageProcessor[0];
        CameraDisplay m_CameraDisplay = null;
        bool m_EnableDisplay = false;

        internal bool CanScan
        {
            get
            {
                if (iStage == null) return false;
                if (iGrab == null) return false;
                foreach (SImageProcessor sip in iGPU)
                    if (sip.IProc != null && sip.Enabled) 
                        return true;
                return false;
            }
        }
            

        string m_LogFileName = null;

        public void Log(string error, string details)
        {
            try
            {
                System.IO.File.AppendAllText(m_LogFileName, System.DateTime.Now + "\t$\t" + error + "\t$\t" + details + "\r\n");                
            }
            catch (Exception) { }
        }

        const string DisplayLog = "Display";

        object m_DisplayLock = new object();

        private void CameraDisplayRefresh()
        {
            System.Threading.WaitHandle[] waiths = new System.Threading.WaitHandle[] { m_TerminateEvent, m_DisplayMutex };
            int waitresult = -1;
            while ((waitresult = System.Threading.WaitHandle.WaitAny(waiths)) == 1)
                try
                {                    
                    try
                    {
                        SySal.Imaging.IImageProcessor iproc = null;
                        foreach (SImageProcessor ipr in iGPU)
                            if (ipr.IProc != null)
                            {
                                iproc = ipr.IProc;
                                break;
                            }
                        if (m_CameraDisplay.ImageProcessor != iproc)
                            m_CameraDisplay.ImageProcessor = iproc;
                    }
                    catch (Exception x)
                    {
                        Log(DisplayLog, x.ToString());
                    }
                    object seq = null;
                    SySal.Imaging.LinearMemoryImage im = null;                    
                    try
                    {
                        System.DateTime start = System.DateTime.Now;
                        seq = iGrab.GrabSequence();
                        im = (SySal.Imaging.LinearMemoryImage)iGrab.MapSequenceToSingleImage(seq);
                        System.DateTime end1 = System.DateTime.Now;
                        m_CameraDisplay.ImageShown = im;
                        System.DateTime end2 = System.DateTime.Now;
                        iGrab.ClearMappedImage(im);
                        System.DateTime end3 = System.DateTime.Now;
                        iGrab.ClearGrabSequence(seq);
                        seq = null;
                        System.DateTime end4 = System.DateTime.Now;
                        //Log(DisplayLog, (end1 - start).TotalMilliseconds.ToString() + " " + (end2 - end1).TotalMilliseconds.ToString() + " " + (end3 - end2).TotalMilliseconds.ToString() + " " + (end4 - end3).TotalMilliseconds.ToString());
                        im = null;
                    }
                    catch (Exception x)
                    {
                        Log(DisplayLog, x.ToString());
                    }
                    finally
                    {
                        if (im != null)
                            try
                            {
                                iGrab.ClearMappedImage(im);
                                im = null;
                            }
                            catch (Exception x)
                            {
                                Log(DisplayLog, x.ToString());
                            }
                        if (seq != null)
                            try
                            {
                                iGrab.ClearGrabSequence(seq);
                                seq = null;
                            }
                            catch (Exception x)
                            {
                                Log(DisplayLog, x.ToString());
                            }
                    }
                }
                catch (Exception xc)
                {
                    Log("OnCameraDisplayTimer_Elapsed", xc.ToString());
                }
                finally
                {
                    try { m_DisplayMutex.ReleaseMutex(); }
                    catch (Exception) { };
                }
        }


        #region IMachineSettingsEditor Members

        public bool EditMachineSettings(Type t)
        {
            ScannerSettingsForm frm = new ScannerSettingsForm();
            frm.C = S;
            if (frm.ShowDialog() == DialogResult.OK)
            {
                S = frm.C;
                try
                {                                     
                    SySal.Management.MachineSettings.SetSettings(typeof(ScannerSettings), S);
                    MessageBox.Show("Configuration for Scanner saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                    ApplyMachineSettings();
                    return true;
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error setting configuration for Scanner", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return false;
                }
            }
            return false;
        }

        #endregion

        public void EditGPUSettings()
        {
            GPUSettingsForm gpuf = new GPUSettingsForm();

            gpuf.iGPUs = iGPU;

            if (gpuf.ShowDialog() == DialogResult.OK)
            {
                S.EnabledGPUs = new bool[iGPU.Length];
                int i;
                for (i = 0; i < S.EnabledGPUs.Length; i++) S.EnabledGPUs[i] = iGPU[i].Enabled;
                try
                {
                    ApplyMachineSettings();
                    SySal.Management.MachineSettings.SetSettings(typeof(ScannerSettings), S);
                    MessageBox.Show("Configuration for Scanner saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "Error setting configuration for Scanner", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
            }
        }

        const string ScanSrvMon = "ScanServer command queue";

        public void StartScanServer()
        {
            if (m_NSS_connected == false)
            {
                System.Runtime.Remoting.RemotingServices.Marshal(NSS, "ScanServer.rem");
                m_NSS_connected = true;
                m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " START.");
            }
        }

        public void StopScanServer()
        {
            if (m_NSS_connected)
            {
                System.Runtime.Remoting.RemotingServices.Disconnect(NSS);
                m_NSS_connected = false;
                m_ScanDataDisplay.CloseMonitor(ScanSrvMon);
            }
        }

        public void RunImagingWizard()
        {
            ImagingConfigurationForm icf = new ImagingConfigurationForm();
            icf.iCamDisp = this;
            icf.iLog = this;
            icf.ConfigDir = S.ConfigDirectory;
            icf.DataDir = S.DataDirectory;
            icf.CurrentConfig = IC;
            int enabledgpus = 0;
            foreach (SImageProcessor s in iGPU)
                if (s.Enabled)
                    enabledgpus++;
            SySal.Imaging.IImageProcessor[] igpu = new SySal.Imaging.IImageProcessor[enabledgpus];
            int i;
            for (i = enabledgpus = 0; i < iGPU.Length; i++)
                if (iGPU[i].Enabled)
                    igpu[enabledgpus++] = iGPU[i].IProc;
            icf.iGPU = igpu;
            icf.m_Display = m_ScanDataDisplay;
            icf.ShowDialog();
            IC = icf.CurrentConfig;
            SySal.Management.MachineSettings.SetSettings(typeof(ImagingConfiguration), IC);            
            ApplyGPUConfig();
            m_CameraDisplay.ImagingConfiguration = IC;
            EnableAutoRefresh = true;
        }

        public void RunQuasiStaticAcquisition()
        {
            QuasiStaticAcquisitionForm qaf = new QuasiStaticAcquisitionForm();
            qaf.ConfigDir = S.ConfigDirectory;
            qaf.DataDir = S.DataDirectory;
            qaf.iLog = this;
            qaf.iMap = this;
            qaf.ImagingConfig = IC;
            qaf.CurrentConfig = QC;
            int enabledgpus = 0;
            foreach (SImageProcessor s in iGPU)
                if (s.Enabled)
                    enabledgpus++;
            SySal.Imaging.IImageProcessor[] igpu = new SySal.Imaging.IImageProcessor[enabledgpus];
            int i;
            for (i = enabledgpus = 0; i < iGPU.Length; i++)
                if (iGPU[i].Enabled)
                    igpu[enabledgpus++] = iGPU[i].IProc;
            qaf.iGPU = igpu;
            qaf.iGrab = iGrab;
            qaf.iStage = iStage;
            qaf.iCamDisp = this;
            qaf.ShowDialog();
            QC = qaf.CurrentConfig;
        }

        public bool RunFullSpeedAcquisition(ScanServerSettings settings = null, SySal.DAQSystem.Scanning.ZoneDesc zone = null)
        {
            FullSpeedAcquisitionForm fsaf = new FullSpeedAcquisitionForm();
            fsaf.ConfigDir = S.ConfigDirectory;
            fsaf.DataDir = S.DataDirectory;
            fsaf.iLog = this;
            fsaf.iMap = this;
            fsaf.ImagingConfig = IC;
            if (settings == null)
                fsaf.CurrentConfig = FSC;
            else
            {
                if (zone != null)                
                    fsaf.DataDir = S.ScanServerDataDirectory;
                fsaf.SetScanSettings(settings.Scan, settings.Process, zone);
                if (zone != null)
                    m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " SCAN " + zone.MinX + "/" + zone.MaxX + "/" + zone.MinY + "/" + zone.MaxY + " \"" + zone.Outname + "\".");                
            }
            fsaf.GeneralTimeSource = m_GeneralTimeSource;
            int enabledgpus = 0;
            foreach (SImageProcessor s in iGPU)
                if (s.Enabled)
                    enabledgpus++;
            SySal.Imaging.IImageProcessor[] igpu = new SySal.Imaging.IImageProcessor[enabledgpus];
            int i;
            for (i = enabledgpus = 0; i < iGPU.Length; i++)
                if (iGPU[i].Enabled)
                    igpu[enabledgpus++] = iGPU[i].IProc;
            fsaf.iGPU = igpu;
            fsaf.iGrab = iGrab;
            fsaf.iStage = iStage;
            fsaf.iCamDisp = this;
            fsaf.iScanDataDisplay = m_ScanDataDisplay;
            fsaf.ShowDialog();
            if (zone != null) m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " SCAN completed (" + fsaf.DialogResult + ").");
            FSC = fsaf.CurrentConfig;
            return fsaf.DialogResult == DialogResult.OK;
        }

        public bool AcquireMarks(SySal.DAQSystem.Scanning.MountPlateDesc platedesc = null)
        {
            MarkAcquisitionForm maf = new MarkAcquisitionForm();
            maf.iStage = iStage;
            maf.iCamDisp = this;
            maf.iScanDataDisplay = m_ScanDataDisplay;
            if (platedesc != null)
            {
                maf.PlateDesc = platedesc;
                if (NSS_connected)
                    m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " MARKS " + platedesc.BrickId + "/" + platedesc.PlateId + ".");
            }
            if (maf.ShowDialog() == DialogResult.OK)
            {
                m_Map = maf.Map;
                m_ScanDataDisplay.DisplayMonitor("Current Plate-To-Stage Map",
                    "MXX = " + m_Map.MXX.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MXY = " + m_Map.MXY.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MYX = " + m_Map.MYX.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MYY = " + m_Map.MYY.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "RX  = " + m_Map.RX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "RY  = " + m_Map.RY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "TX  = " + m_Map.TX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "TY  = " + m_Map.TY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "Taken at " + System.DateTime.Now.ToUniversalTime()
                    );
                if (NSS_connected)
                    m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " MARKS completed (OK).");
                return true;
            }
            if (NSS_connected)
                m_ScanDataDisplay.DisplayStringAppend(ScanSrvMon, System.DateTime.Now + " MARKS completed (Cancel).");
            return false;
        }
        
        public void Dispose()
        {
            GC.SuppressFinalize(this);
            Log(StartStop, "Stopping");
            Log(StartStop, "Stopping display.");
            m_TerminateEvent.Set();
            try
            {
                m_DisplayMutex.ReleaseMutex();
            }
            catch (Exception) { }
            if (iStage != null)
            {
                Log(StartStop, "Deleting stage.");
                iStage.Dispose();
                iStage = null;
            }
            if (iGrab != null)
            {
                Log(StartStop, "Deleting grabber.");
                iGrab.Dispose();
                iGrab = null;
            }
            if (iGPU != null)
            {
                Log(StartStop, "Deleting GPUs.");
                foreach (SImageProcessor iproc in iGPU)
                    if (iproc.IProc != null)
                        iproc.IProc.Dispose();
                iGPU = null;
            }
            Log(StartStop, "Stopped.");
        }

        ~Scanner()
        {
            Dispose();
        }

        public bool EnableCross
        {
            get { return m_CameraDisplay.EnableCross; }
            set { m_CameraDisplay.EnableCross = value; }
        }

        public bool EnableAutoRefresh
        {
            get
            {
                return m_EnableDisplay;
            }
            [MTAThread]
            set
            {
                try
                {
                    if (m_EnableDisplay == false && value == true)
                    {
                        Log("EnableAutoRefresh", "Applying config and releasing mutex");
                        ApplyGPUConfig();
                        iGrab.SequenceSize = 1;
                        m_DisplayMutex.ReleaseMutex();
                        Log("EnableAutoRefresh", "Mutex released");
                    }
                    if (m_EnableDisplay == true && value == false)
                    {
                        Log("EnableAutoRefresh", "Getting mutex");
                        bool waitres = m_DisplayMutex.WaitOne();
                        Log("EnableAutoRefresh", "Mutex " + waitres);
                    }
                    m_EnableDisplay = value;
                    if (m_CameraDisplay.InvokeRequired) m_CameraDisplay.Invoke(new dVoid(delegate()
                    {
                        m_CameraDisplay.Visible = value;
                    }   ));
                    else m_CameraDisplay.Visible = value;
                }
                catch (Exception xc)
                {
                    Log("EnableAutoRefresh", xc.ToString());
                }
            }
        }

        private delegate void dVoid();

        public Imaging.LinearMemoryImage ImageShown
        {
            set
            {
                try
                {
                    EnableAutoRefresh = false;
                    m_CameraDisplay.SetNonreusableImage(value);
                }
                catch (Exception xc)
                {
                    Log("ImageShown", xc.ToString());
                } 
            }
        }

        SySal.DAQSystem.Scanning.IntercalibrationInfo m_Map = IdMap;

        static SySal.DAQSystem.Scanning.IntercalibrationInfo IdMap
        {
            get
            {
                SySal.DAQSystem.Scanning.IntercalibrationInfo info = new DAQSystem.Scanning.IntercalibrationInfo();
                info.MXX = info.MYY = 1.0;
                info.MXY = info.MYX = 0.0;
                info.RX = info.RY = 0.0;
                info.TX = info.TY = info.TZ = 0.0;
                return info;
            }
        }

        public DAQSystem.Scanning.IntercalibrationInfo PlateMap
        {
            get 
            {
                return m_Map;            
            }
        }

        public DAQSystem.Scanning.IntercalibrationInfo InversePlateMap
        {
            get
            {
                SySal.DAQSystem.Scanning.IntercalibrationInfo inv = new DAQSystem.Scanning.IntercalibrationInfo();
                inv.TZ = -m_Map.TZ;
                inv.RX = m_Map.RX + m_Map.TX;
                inv.RY = m_Map.RY + m_Map.TY;
                double D = 1.0 / (m_Map.MXX * m_Map.MYY - m_Map.MXY * m_Map.MYX);
                inv.TX = -m_Map.TX;
                inv.TY = -m_Map.TY;
                inv.MXX = m_Map.MYY * D;
                inv.MXY = -m_Map.MXY * D;
                inv.MYX = -m_Map.MYX * D;
                inv.MYY = m_Map.MXX * D;
                return inv;
            }
            set
            {
                m_Map = new DAQSystem.Scanning.IntercalibrationInfo();
                m_Map.TZ = -value.TZ;
                m_Map.RX = value.RX + value.TX;
                m_Map.RY = value.RY + value.TY;
                double D = 1.0 / (value.MXX * value.MYY - value.MXY * value.MYX);
                m_Map.TX = -value.TX;
                m_Map.TY = -value.TY;
                m_Map.MXX = value.MYY * D;
                m_Map.MXY = -value.MXY * D;
                m_Map.MYX = -value.MYX * D;
                m_Map.MYY = value.MXX * D;
                m_ScanDataDisplay.DisplayMonitor("Current Plate-To-Stage Map",
                    "MXX = " + m_Map.MXX.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MXY = " + m_Map.MXY.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MYX = " + m_Map.MYX.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "MYY = " + m_Map.MYY.ToString("F5", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "RX  = " + m_Map.RX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "RY  = " + m_Map.RY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "TX  = " + m_Map.TX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "TY  = " + m_Map.TY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\r\n" +
                    "Computed from inverse file"
                    );
            }
        }
    }

    public interface IScannerDataDisplay
    {
        void DisplayStringAppend(string infotitle, string content);
        void Display(string infotitle, object content);
        void DisplayMonitor(string infotitle, object monitoredobj);
        void CloseMonitor(string infotitle);
    }
}



namespace SySal.DAQSystem
{
    [Serializable]
    public class ScanServerSettings
    {
        public FullSpeedAcqSettings Scan;
        public QuasiStaticAcquisition.PostProcessingInfo[] Process;

        static System.Xml.Serialization.XmlSerializer s_xmls = new System.Xml.Serialization.XmlSerializer(typeof(ScanServerSettings));

        internal static ScanServerSettings FromXML(string xmlsettings)
        {
            return (ScanServerSettings)s_xmls.Deserialize(new System.IO.StringReader(xmlsettings));
        }
    }        

    public class ScanServer : MarshalByRefObject
    {
        public ScanServer() { }

        internal Scanner m_Owner = null;

        internal SySal.DAQSystem.Scanning.MountPlateDesc m_CurrentPlate = null;

        internal SySal.DAQSystem.Scanning.ZoneDesc m_CurrentZone = null;

        internal ScanServerSettings m_ScanSettings = null;

        internal bool Busy = false;

        public override object InitializeLifetimeService()
        {
            return null;
        }

        delegate bool dRunFullSpeedAcquisition(ScanServerSettings scansettings, SySal.DAQSystem.Scanning.ZoneDesc zone);

        public bool Scan(SySal.DAQSystem.Scanning.ZoneDesc zone)
        {
            lock (m_Owner)
            {
                if (Busy) return false;
                m_CurrentZone = zone;
                Busy = true;
            }
            bool ret = (bool)SySalMainForm.TheMainForm.Invoke(new dRunFullSpeedAcquisition(m_Owner.RunFullSpeedAcquisition), new object [] { m_ScanSettings, m_CurrentZone});
            lock (m_Owner)
            {
                m_CurrentZone = null;
                Busy = false;
            }
            return ret;
        }

        public bool ScanAndMoveToNext(SySal.DAQSystem.Scanning.ZoneDesc zone, SySal.BasicTypes.Rectangle nextzone)
        {
            return Scan(zone);
        }

        delegate DialogResult dMessageBox(string text, string caption);

        delegate bool dAcquireMarks(SySal.DAQSystem.Scanning.MountPlateDesc plate);

        public bool LoadPlate(SySal.DAQSystem.Scanning.MountPlateDesc plate)
        {
            lock (m_Owner)
            {                
                if (Busy) return false;                
                if (UnloadPlate() == false) return false;                
                Busy = true;
            }
            if ((DialogResult)SySalMainForm.TheMainForm.Invoke(new dMessageBox(MessageBox.Show), new object[] { "Please load plate " + plate.BrickId + "/" + plate.PlateId + " " + plate.TextDesc, "Load plate" }) != DialogResult.OK)
                lock (m_Owner)
                {
                    Busy = false;
                    return false;
                }
            if ((bool)SySalMainForm.TheMainForm.Invoke(new dAcquireMarks(m_Owner.AcquireMarks), new object [] { plate }))
                lock (m_Owner)
                {
                    m_CurrentPlate = plate;
                    Busy = false;
                    return true;
                }            
            lock (m_Owner)
                Busy = false;
            return false;
        }

        public bool UnloadPlate()
        {
            lock (m_Owner)
            {
                if (Busy) return false;
            }
            if (m_CurrentPlate != null)
                if ((DialogResult)SySalMainForm.TheMainForm.Invoke(new dMessageBox(MessageBox.Show), new object[] { "Please unload plate", "Load plate" }) != DialogResult.OK)
                    lock (m_Owner)
                    {
                        return false;
                    }
            lock (m_Owner)
            {
                m_CurrentPlate = null;
                return true;
            }
        }

        public bool TestComm(int h)
        {
            return (h == 0);
        }

        public bool SetSingleParameter(string objectname, string parametername, string parametervalue)
        {
            throw new System.NotImplementedException("This method is not implemented for NExTScanner.");
        }

        public bool SetObjectConfiguration(string objectname, string xmlconfig)
        {
            throw new System.NotImplementedException("This method is not implemented for NExTScanner.");
        }

        public bool SetScanLayout(string xmllayout)
        {
            m_ScanSettings = ScanServerSettings.FromXML(xmllayout);
            return true;
        }

        public bool IsBusy
        {
            get { lock (m_Owner) return Busy; }
        }

        public bool IsLoaded
        {
            get { lock (m_Owner) return m_CurrentPlate != null; }
        }

        public long CurrentZone
        {
            get { lock (m_Owner) return (m_CurrentZone == null) ? 0 : m_CurrentZone.Series; }
        }

        public SySal.DAQSystem.Scanning.MountPlateDesc CurrentPlate
        {
            get { lock (m_Owner) return m_CurrentPlate; }
        }

        public SySal.DAQSystem.Scanning.ManualCheck.OutputBaseTrack RequireManualCheck(SySal.DAQSystem.Scanning.ManualCheck.InputBaseTrack inputbasetrack)
        {
            throw new System.NotImplementedException("RequireManualCheck method is not implemented for NExTScanner.");
        }

        public SySal.DAQSystem.Scanning.PlateQuality.FogThicknessSet GetFogAndThickness()
        {
            throw new System.NotImplementedException("GetFogAndThickness method is not implemented for NExTScanner.");
        }

        public bool ImageDump(SySal.DAQSystem.Scanning.ImageDumpRequest imdumpreq)
        {
            throw new System.NotImplementedException("ImageDump method is not implemented for NExTScanner.");
        }

    }
}