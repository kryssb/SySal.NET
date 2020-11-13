using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class FullSpeedAcquisitionForm : SySal.SySalNExTControls.SySalDialog
    {
        public class FullSpeedAcquisitionFormSettings : SySal.Management.Configuration
        {
            public QuasiStaticAcquisition.PostProcessingInfo[] PostProcessSteps = new QuasiStaticAcquisition.PostProcessingInfo[0];

            public FullSpeedAcquisitionFormSettings()
                : base("")
            {
            }

            public override object Clone()
            {
                FullSpeedAcquisitionFormSettings s = new FullSpeedAcquisitionFormSettings();
                s.Name = Name;
                s.PostProcessSteps = (QuasiStaticAcquisition.PostProcessingInfo [])PostProcessSteps.Clone();
                return s;
            }

            internal static FullSpeedAcquisitionFormSettings Stored
            {
                get
                {
                    FullSpeedAcquisitionFormSettings c = SySal.Management.MachineSettings.GetSettings(typeof(FullSpeedAcquisitionFormSettings)) as FullSpeedAcquisitionFormSettings;
                    if (c == null) c = new FullSpeedAcquisitionFormSettings();
                    return c;
                }
            }
        }

        private bool AutoScan = false;

        private FullSpeedAcquisitionFormSettings S = FullSpeedAcquisitionFormSettings.Stored;

        private string OutputRWDFileName = "";

        public void SetScanSettings(FullSpeedAcqSettings scan, QuasiStaticAcquisition.PostProcessingInfo [] process, SySal.DAQSystem.Scanning.ZoneDesc zone)
        {            
            CurrentConfig = scan;
            S.PostProcessSteps = process;
            if (zone != null)
            {
                AutoScan = true;
                txtMinX.Text = zone.MinX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                OnMinXLeave(this, null);
                txtMinY.Text = zone.MinY.ToString(System.Globalization.CultureInfo.InvariantCulture);
                OnMinYLeave(this, null);
                txtMaxX.Text = zone.MaxX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                OnMaxXLeave(this, null);
                txtMaxY.Text = zone.MaxY.ToString(System.Globalization.CultureInfo.InvariantCulture);
                OnMaxYLeave(this, null);
                string zonename = zone.Outname.Substring(zone.Outname.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
                txtSummaryFile.Text = (DataDir.EndsWith("\\") ? DataDir : (DataDir + "\\")) + zonename + "." + QuasiStaticAcquisition.TypeStringScan + "\\" + zonename +
                    "_rwdremove(" + QuasiStaticAcquisition.SideString + "_" + QuasiStaticAcquisition.StripString + "_" + 
                    QuasiStaticAcquisition.SequenceString + "_" + QuasiStaticAcquisition.ImageString + 
                    ")." + QuasiStaticAcquisition.TypeString;                
                OutputRWDFileName = zone.Outname;
            }
        }

        struct FocusInfo
        {
            public SySal.BasicTypes.Vector2 Pos;
            public double TopZ, BottomZ;
            public System.DateTime MeasureTime;
            public bool Valid;
        }

        class FocusMap
        {
            public SySal.BasicTypes.Rectangle Extents;
            public SySal.BasicTypes.Vector2 Step;
            public FocusInfo[,] TopInfo;
            public FocusInfo[,] BottomInfo;
            public System.TimeSpan Duration;

            public FocusMap(SySal.BasicTypes.Rectangle extents, SySal.BasicTypes.Vector2 step, System.TimeSpan duration, FocusMap importmap)
            {
                Extents = extents;
                if (importmap != null)
                {
                    if (importmap.Extents.MinX < Extents.MinX) Extents.MinX = importmap.Extents.MinX;
                    if (importmap.Extents.MaxX > Extents.MaxX) Extents.MaxX = importmap.Extents.MaxX;
                    if (importmap.Extents.MinY < Extents.MinY) Extents.MinY = importmap.Extents.MinY;
                    if (importmap.Extents.MaxY > Extents.MaxY) Extents.MaxY = importmap.Extents.MaxY;
                }
                int xw = (int)Math.Ceiling((extents.MaxX - extents.MinX) / step.X + 2);
                int yw = (int)Math.Ceiling((extents.MaxY - extents.MinY) / step.Y + 2);
                Extents.MinX -= step.X;
                Extents.MaxX += step.X;
                Extents.MinY -= step.Y;
                Extents.MaxY += step.Y;
                Step = step;
                Duration = duration;
                TopInfo = new FocusInfo[xw, yw];
                BottomInfo = new FocusInfo[xw, yw];
                Clear(); 
                if (importmap != null)
                {
                    foreach (FocusInfo f in importmap.TopInfo) Write(f, true);
                    foreach (FocusInfo f in importmap.BottomInfo) Write(f, false);
                }
            }

            public void Clear()
            {
                int xi, yi;
                int xw = TopInfo.GetLength(0);
                int yw = TopInfo.GetLength(1);
                for (xi = 0; xi < xw; xi++)
                    for (yi = 0; yi < yw; yi++)
                    {
                        TopInfo[xi, yi].Valid = BottomInfo[xi, yi].Valid = false;
                    }
            }

            public bool Write(FocusInfo info, bool istop)
            {
                int xi = (int)((info.Pos.X - Extents.MinX) / Step.X);
                if (xi < 0 || xi >= TopInfo.GetLength(0)) return false;
                int yi = (int)((info.Pos.Y - Extents.MinY) / Step.Y);
                if (yi < 0 || yi >= TopInfo.GetLength(1)) return false;
                if (info.Valid == false) return false;
                if (istop)
                {
                    if (TopInfo[xi, yi].Valid == false || TopInfo[xi, yi].MeasureTime < info.MeasureTime)
                        TopInfo[xi, yi] = info;
                }
                else
                {
                    if (BottomInfo[xi, yi].Valid == false || BottomInfo[xi, yi].MeasureTime < info.MeasureTime)
                        BottomInfo[xi, yi] = info;
                }
                return true;
            }

            static void BestFocusInfo(FocusInfo[,] sk, System.DateTime sktime, SySal.BasicTypes.Vector2 p, int xi, int yi, ref double bestd, ref int bestxi, ref int bestyi)
            {
                if (sk[xi, yi].Valid && sk[xi, yi].MeasureTime >= sktime)
                {
                    double dx = p.X - sk[xi, yi].Pos.X;
                    double dy = p.Y - sk[xi, yi].Pos.Y;
                    double d = dx * dx + dy * dy;
                    if (bestd < 0.0 || bestd > d)
                    {
                        bestxi = xi;
                        bestyi = yi;
                        bestd = d;

                    }
                }
            }

            public bool GetFocusInfo(ref FocusInfo fi, SySal.BasicTypes.Vector2 pos, bool istop, bool hintXstrips, bool hintYstrips)
            {
                System.DateTime sktime = System.DateTime.Now - Duration;
                int xi = (int)((pos.X - Extents.MinX) / Step.X);
                if (xi < 0 || xi >= TopInfo.GetLength(0)) return false;
                int yi = (int)((pos.Y - Extents.MinY) / Step.Y);
                if (yi < 0 || yi >= TopInfo.GetLength(1)) return false;
                FocusInfo[,] sk = istop ? TopInfo : BottomInfo;
                if (sk[xi, yi].Valid && sk[xi, yi].MeasureTime >= sktime)
                {
                    fi = sk[xi, yi];
                    return true;
                }
                double bestd = -1.0;
                int bestxi = -1, bestyi = -1;
                int xw = sk.GetLength(0);
                int yw = sk.GetLength(1);
                int iloop;
                if (hintXstrips)
                {
                    if (xi > 0) BestFocusInfo(sk, sktime, pos, xi - 1, yi, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1) BestFocusInfo(sk, sktime, pos, xi + 1, yi, ref bestd, ref bestxi, ref bestyi);
                    if (yi > 0) BestFocusInfo(sk, sktime, pos, xi, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (bestd >= 0.0)
                    {
                        fi = sk[bestxi, bestyi];
                        return true;
                    }
                    if (xi > 0 && yi > 0) BestFocusInfo(sk, sktime, pos, xi - 1, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1 && yi > 0) BestFocusInfo(sk, sktime, pos, xi + 1, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi > 0 && yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi - 1, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1 && yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi + 1, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (bestd >= 0.0)
                    {
                        fi = sk[bestxi, bestyi];
                        return true;
                    }
                }
                if (hintYstrips)
                {
                    if (yi > 0) BestFocusInfo(sk, sktime, pos, xi, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi > 0) BestFocusInfo(sk, sktime, pos, xi - 1, yi, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1) BestFocusInfo(sk, sktime, pos, xi + 1, yi, ref bestd, ref bestxi, ref bestyi);
                    if (bestd >= 0.0)
                    {
                        fi = sk[bestxi, bestyi];
                        return true;
                    }
                    if (xi > 0 && yi > 0) BestFocusInfo(sk, sktime, pos, xi - 1, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi > 0 && yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi - 1, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1 && yi < yw - 1) BestFocusInfo(sk, sktime, pos, xi + 1, yi + 1, ref bestd, ref bestxi, ref bestyi);
                    if (xi < xw - 1 && yi > 0) BestFocusInfo(sk, sktime, pos, xi + 1, yi - 1, ref bestd, ref bestxi, ref bestyi);
                    if (bestd >= 0.0)
                    {
                        fi = sk[bestxi, bestyi];
                        return true;
                    }
                }
                for (iloop = 1; iloop < Math.Max(xw, yw); iloop++)
                {
                    int xxi, yyi;
                    yyi = yi - iloop;
                    if (yyi >= 0)
                        for (xxi = Math.Max(0, xi - iloop); xxi <= xi + iloop && xxi < xw; xxi++)
                            BestFocusInfo(sk, sktime, pos, xxi, yyi, ref bestd, ref bestxi, ref bestyi);
                    yyi = yi + iloop;
                    if (yyi < yw)
                        for (xxi = Math.Max(0, xi - iloop); xxi <= xi + iloop && xxi < xw; xxi++)
                            BestFocusInfo(sk, sktime, pos, xxi, yyi, ref bestd, ref bestxi, ref bestyi);
                    xxi = xi - iloop;
                    if (xxi >= 0)
                        for (yyi = Math.Max(0, yi - iloop + 1); yyi <= yi + iloop - 1 && yyi < yw; yyi++)
                            BestFocusInfo(sk, sktime, pos, xxi, yyi, ref bestd, ref bestxi, ref bestyi);
                    xxi = xi + iloop;
                    if (xxi < xw)
                        for (yyi = Math.Max(0, yi - iloop + 1); yyi <= yi + iloop - 1 && yyi < yw; yyi++)
                            BestFocusInfo(sk, sktime, pos, xxi, yyi, ref bestd, ref bestxi, ref bestyi);
                    if (bestd >= 0.0)
                    {
                        fi = sk[bestxi, bestyi];
                        return true;
                    }
                }
                return false;
            }

            public override string ToString()
            {
                string ret = "MinX: " + Extents.MinX.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nMaxX: " + Extents.MaxX.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nMinY: " + Extents.MinY.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nMaxY: " + Extents.MaxY.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nStepX: " + Step.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nStepY: " + Step.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nXCells: " + TopInfo.GetLength(0) +
                    "\r\nYCells: " + TopInfo.GetLength(1) + 
                    "\r\nXI\tYI\tSIDE\tX\tY\tZTop\tZBottom\tValid\tAge"; 
                System.DateTime baseage = System.DateTime.Now;
                for (FocusInfo[,] sk = TopInfo; sk != null; sk = (sk == BottomInfo) ? (FocusInfo [,])null : BottomInfo)
                {
                    int xi, yi;
                    for (xi = 0; xi < sk.GetLength(0); xi++)
                        for (yi = 0; yi < sk.GetLength(1); yi++)
                            ret += "\r\n" + xi + "\t" + yi + "\t" + ((sk == TopInfo) ? "0" : "1") +
                                "\t" + (sk[xi, yi].Valid ? sk[xi, yi].Pos.X.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) : "0") +
                                "\t" + (sk[xi, yi].Valid ? sk[xi, yi].Pos.Y.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) : "0") +
                                "\t" + (sk[xi, yi].Valid ? sk[xi, yi].TopZ.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) : "0") +
                                "\t" + (sk[xi, yi].Valid ? sk[xi, yi].BottomZ.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) : "0") +
                                "\t" + (sk[xi, yi].Valid ? "1" : "0") +
                                "\t" + (sk[xi, yi].Valid ? (baseage - sk[xi, yi].MeasureTime).TotalMinutes.ToString("F0", System.Globalization.CultureInfo.InvariantCulture) : "0");
                }
                return ret;
            }
        }

        public string ConfigDir = "";

        public string DataDir = "";

        public ISySalLog iLog;

        public IMapProvider iMap;

        public ISySalCameraDisplay iCamDisp;

        public IScannerDataDisplay iScanDataDisplay;

        public SySal.Imaging.IImageGrabber iGrab;

        public SySal.Imaging.IImageProcessor[] iGPU;

        public SySal.StageControl.IStage iStage;

        public SySal.Executables.NExTScanner.ImagingConfiguration ImagingConfig;

        internal SySal.Executables.NExTScanner.FullSpeedAcqSettings CurrentConfig;

        private SySal.Executables.NExTScanner.FullSpeedAcqSettings WorkConfig;

        static System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(FullSpeedAcqSettings));        

        void LoadConfigurationList()
        {
            lbConfigurations.Items.Clear();
            string[] files = System.IO.Directory.GetFiles(ConfigDir, "*." + FullSpeedAcqSettings.FileExtension, System.IO.SearchOption.TopDirectoryOnly);
            foreach (string f in files)
            {
                string s = f.Substring(f.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
                lbConfigurations.Items.Add(s.Substring(0, s.Length - FullSpeedAcqSettings.FileExtension.Length - 1));
            }
        }

        public FullSpeedAcquisitionForm()
        {
            InitializeComponent();

            m_ScanThread = null;
        }        

        private void OnLoad(object sender, EventArgs e)
        {
            LoadConfigurationList();
            if (WorkConfig == null) WorkConfig = (FullSpeedAcqSettings)CurrentConfig.Clone();
            ShowWorkConfig();
            if (AutoScan == false)
            {
                string summaryfile = this.DataDir;
                if (summaryfile.EndsWith("/") == false && summaryfile.EndsWith("\\") == false) summaryfile += "/";
                summaryfile += QuasiStaticAcquisition.IndexedPatternTemplate;
                txtSummaryFile.Text = summaryfile;
            }
            foreach (ListViewItem lvi in lvPostProcessing.Items)            
                foreach (QuasiStaticAcquisition.PostProcessingInfo pi in S.PostProcessSteps)
                    if (String.Compare(pi.Name.Trim(), lvi.SubItems[0].Text, true) == 0)
                        lvi.SubItems[1].Text = pi.Settings; 
            EnableControls(!AutoScan);
            if (AutoScan)
            {                
                btnStart_Click(sender, e);
            }
        }

        void ShowWorkConfig()
        {
            txtLayers.Text = WorkConfig.Layers.ToString();
            txtPitch.Text = WorkConfig.Pitch.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtZSweep.Text = WorkConfig.ZSweep.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtContinuousMotionFraction.Text = WorkConfig.ContinuousMotionDutyFraction.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtClusterThreshold.Text = WorkConfig.ClusterThreshold.ToString();
            txtMinValidLayers.Text = WorkConfig.MinValidLayers.ToString();
            txtFocusSweep.Text = WorkConfig.FocusSweep.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            chkTop.Checked = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both);
            chkBottom.Checked = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both);
            txtBaseThickness.Text = WorkConfig.BaseThickness.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtEmuThickness.Text = WorkConfig.EmulsionThickness.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtFOVSize.Text = Math.Abs(ImagingConfig.ImageWidth * ImagingConfig.Pixel2Micron.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\xD7" + Math.Abs(ImagingConfig.ImageHeight * ImagingConfig.Pixel2Micron.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtViewOverlap.Text = WorkConfig.ViewOverlap.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            rdMoveX.Checked = (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X);
            rdMoveY.Checked = (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.Y);
            txtXYSpeed.Text = WorkConfig.XYSpeed.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtZSpeed.Text = WorkConfig.ZSpeed.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtXYAcceleration.Text = WorkConfig.XYAcceleration.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtZAcceleration.Text = WorkConfig.ZAcceleration.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtSlowdownTime.Text = WorkConfig.SlowdownTimeMS.ToString();
            txtPositionTolerance.Text = WorkConfig.PositionTolerance.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtFPS.Text = WorkConfig.FramesPerSecond.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtMotionLatency.Text = WorkConfig.MotionLatencyMS.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtZDataExpirationMin.Text = WorkConfig.ZDataExpirationMin.ToString();
            txtMaxViewsStrip.Text = WorkConfig.MaxViewsPerStrip.ToString();
        }        

        private void btnNew_Click(object sender, EventArgs e)
        {
            txtNewName.Text = txtNewName.Text.Trim();
            if (txtNewName.Text.Length == 0)
            {
                MessageBox.Show("The configuration name cannot be empty.", "Missing input", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            System.IO.StreamWriter w = null;
            try
            {
                string f = ConfigDir;
                if (f.EndsWith("\\") == false && f.EndsWith("/") == false) f += "/";
                f += txtNewName.Text + "." + FullSpeedAcqSettings.FileExtension;
                w = new System.IO.StreamWriter(f);
                xmls.Serialize(w, WorkConfig);
                w.Flush();
                w.Close();
                w = null;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (w != null) w.Close();
                LoadConfigurationList();
            }
        }

        private void btnDuplicate_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            txtNewName.Text = txtNewName.Text.Trim();
            if (txtNewName.Text.Length == 0)
            {
                MessageBox.Show("The configuration name cannot be empty.", "Missing input", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            try
            {
                string f = ConfigDir;
                if (f.EndsWith("\\") == false && f.EndsWith("/") == false) f += "/";
                System.IO.File.Copy(f + lbConfigurations.SelectedItem.ToString() + "." + ImagingConfiguration.FileExtension, f + txtNewName.Text + "." + ImagingConfiguration.FileExtension);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                LoadConfigurationList();
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + FullSpeedAcqSettings.FileExtension)[0]);
                WorkConfig = (FullSpeedAcqSettings)xmls.Deserialize(r);
                r.Close();
                r = null;
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (r != null) r.Close();
                ShowWorkConfig();
            }
        }

        private void btnMakeCurrent_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + FullSpeedAcqSettings.FileExtension)[0]);
                WorkConfig = (FullSpeedAcqSettings)xmls.Deserialize(r);
                r.Close();
                r = null;
                CurrentConfig.Copy(WorkConfig);
                MessageBox.Show("The selected configuration is now the current configuration.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (r != null) r.Close();
                ShowWorkConfig();
            }
        }

        private void btnDel_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            if (MessageBox.Show("Are you sure you want to delete the selected configuration?", "Confirmation", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                try
                {
                    System.IO.File.Delete(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + FullSpeedAcqSettings.FileExtension)[0]);
                }
                catch (Exception x)
                {
                    MessageBox.Show(x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
                finally
                {
                    LoadConfigurationList();
                }
        }

        private void OnLayersLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.Layers = uint.Parse(txtLayers.Text);
                txtZSweep.Text = WorkConfig.ZSweep.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnPitchLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.Pitch = double.Parse(txtPitch.Text, System.Globalization.CultureInfo.InvariantCulture);
                txtZSweep.Text = WorkConfig.ZSweep.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnClusterThresholdLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ClusterThreshold = uint.Parse(txtClusterThreshold.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMinValidLayersLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MinValidLayers = uint.Parse(txtMinValidLayers.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnFocusSweepLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.FocusSweep = double.Parse(txtFocusSweep.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnTopChecked(object sender, EventArgs e)
        {
            if (chkTop.Checked)
            {
                if (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Both;
                else WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Top;
            }
            else WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Bottom;
        }

        private void OnBottomChecked(object sender, EventArgs e)
        {
            if (chkBottom.Checked)
            {
                if (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Both;
                else WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Bottom;
            }
            else WorkConfig.Sides = FullSpeedAcqSettings.ScanMode.Top;
        }

        private void OnBaseThicknessLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.BaseThickness = double.Parse(txtBaseThickness.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnEmuThicknessLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.EmulsionThickness = double.Parse(txtEmuThickness.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnViewOverlapLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ViewOverlap = double.Parse(txtViewOverlap.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMoveXChecked(object sender, EventArgs e)
        {
            if (rdMoveX.Checked) WorkConfig.AxisToMove = FullSpeedAcqSettings.MoveAxisForScan.X;
        }

        private void OnMoveYChecked(object sender, EventArgs e)
        {
            if (rdMoveY.Checked) WorkConfig.AxisToMove = FullSpeedAcqSettings.MoveAxisForScan.Y;
        }

        private void OnXYSpeedLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.XYSpeed = double.Parse(txtXYSpeed.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnXYAccelLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.XYAcceleration = double.Parse(txtXYAcceleration.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnZSpeedLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ZSpeed = double.Parse(txtZSpeed.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnZAccelLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ZAcceleration = double.Parse(txtZAcceleration.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnSlowdownTimeLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.SlowdownTimeMS = uint.Parse(txtSlowdownTime.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnPosTolLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.PositionTolerance = double.Parse(txtPositionTolerance.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }
        


        private void OnMotionLatencyLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MotionLatencyMS = double.Parse(txtMotionLatency.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }


        private void OnContinuousMotionFractionLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ContinuousMotionDutyFraction = double.Parse(txtContinuousMotionFraction.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnLeaveFPS(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.FramesPerSecond = double.Parse(txtFPS.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private SySal.BasicTypes.Rectangle TransformedScanRectangle;
        private SySal.BasicTypes.Rectangle ScanRectangle;

        private void btnFromHere_Click(object sender, EventArgs e)
        {
            txtMinX.Text = (ScanRectangle.MinX = iStage.GetPos(StageControl.Axis.X)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtMinY.Text = (ScanRectangle.MinY = iStage.GetPos(StageControl.Axis.Y)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
        }

        private void btnToHere_Click(object sender, EventArgs e)
        {
            txtMaxX.Text = (ScanRectangle.MaxX = iStage.GetPos(StageControl.Axis.X)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtMaxY.Text = (ScanRectangle.MaxY = iStage.GetPos(StageControl.Axis.Y)).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
        }

        private void OnMinXLeave(object sender, EventArgs e)
        {
            try
            {
                txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
                ScanRectangle.MinX = double.Parse(txtMinX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtMinX.Text = ScanRectangle.MinX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void OnMinYLeave(object sender, EventArgs e)
        {
            try
            {
                txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
                ScanRectangle.MinY = double.Parse(txtMinY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtMinY.Text = ScanRectangle.MinY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void OnMaxXLeave(object sender, EventArgs e)
        {
            try
            {
                txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
                ScanRectangle.MaxX = double.Parse(txtMaxX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtMaxX.Text = ScanRectangle.MaxX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        private void OnMaxYLeave(object sender, EventArgs e)
        {
            try
            {
                txtTransfMinX.Text = txtTransfMaxX.Text = txtTransfMinY.Text = txtTransfMaxY.Text = "";
                ScanRectangle.MaxY = double.Parse(txtMaxY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                txtMaxY.Text = ScanRectangle.MaxY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        bool ShouldStop = true;

        System.Threading.Thread m_ScanThread = null;

        bool DumpFocusInfo = false;

        string m_QAPattern;

        string m_FocusDumpFileName = "";

        private void btnStart_Click(object sender, EventArgs e)
        {            
            if (ShouldStop == false) return;            
            if (ScanRectangle.MinX >= ScanRectangle.MaxX || ScanRectangle.MinY >= ScanRectangle.MaxY)
            {
                MessageBox.Show("Null area defined.", "Nothing to do", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            SySal.DAQSystem.Scanning.IntercalibrationInfo stagemap = iMap.PlateMap;
            {
                SySal.BasicTypes.Vector2 v2 = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 v2t = new BasicTypes.Vector2();
                v2.X = ScanRectangle.MinX;
                v2.Y = ScanRectangle.MinY;
                v2t = stagemap.Transform(v2);
                TransformedScanRectangle.MinX = TransformedScanRectangle.MaxX = v2t.X;
                TransformedScanRectangle.MinY = TransformedScanRectangle.MaxY = v2t.Y;
                v2.X = ScanRectangle.MinX;
                v2.Y = ScanRectangle.MaxY;
                v2t = stagemap.Transform(v2);
                TransformedScanRectangle.MinX = Math.Min(TransformedScanRectangle.MinX, v2t.X);
                TransformedScanRectangle.MaxX = Math.Max(TransformedScanRectangle.MaxX, v2t.X);
                TransformedScanRectangle.MinY = Math.Min(TransformedScanRectangle.MinY, v2t.Y);
                TransformedScanRectangle.MaxY = Math.Max(TransformedScanRectangle.MaxY, v2t.Y);
                v2.X = ScanRectangle.MaxX;
                v2.Y = ScanRectangle.MinY;
                v2t = stagemap.Transform(v2);
                TransformedScanRectangle.MinX = Math.Min(TransformedScanRectangle.MinX, v2t.X);
                TransformedScanRectangle.MaxX = Math.Max(TransformedScanRectangle.MaxX, v2t.X);
                TransformedScanRectangle.MinY = Math.Min(TransformedScanRectangle.MinY, v2t.Y);
                TransformedScanRectangle.MaxY = Math.Max(TransformedScanRectangle.MaxY, v2t.Y);
                v2.X = ScanRectangle.MaxX;
                v2.Y = ScanRectangle.MaxY;
                v2t = stagemap.Transform(v2);
                TransformedScanRectangle.MinX = Math.Min(TransformedScanRectangle.MinX, v2t.X);
                TransformedScanRectangle.MaxX = Math.Max(TransformedScanRectangle.MaxX, v2t.X);
                TransformedScanRectangle.MinY = Math.Min(TransformedScanRectangle.MinY, v2t.Y);
                TransformedScanRectangle.MaxY = Math.Max(TransformedScanRectangle.MaxY, v2t.Y);
                txtTransfMinX.Text = TransformedScanRectangle.MinX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                txtTransfMinY.Text = TransformedScanRectangle.MinY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                txtTransfMaxX.Text = TransformedScanRectangle.MaxX.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
                txtTransfMaxY.Text = TransformedScanRectangle.MaxY.ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            }
            {
                int icpi, ispi;
                icpi = 0;
                foreach (ListViewItem lvi in lvPostProcessing.Items)
                    if (lvi.SubItems[1].Text.Trim().Length > 0)
                        icpi++;
                m_PostProcessingSettings = new QuasiStaticAcquisition.PostProcessingInfo[icpi + 1];
                QuasiStaticAcquisition.PostProcessingInfo im1 = new QuasiStaticAcquisition.PostProcessingInfo();
                im1.Name = "ImageCorrection";
                im1.Settings = this.ImagingConfig.ToString();
                m_PostProcessingSettings[0] = im1;
                ispi = 1;
                for (icpi = 0; icpi < lvPostProcessing.Items.Count; icpi++)
                {
                    ListViewItem lvi = lvPostProcessing.Items[icpi];
                    if (lvi.SubItems[1].Text.Trim().Length > 0)
                    {
                        QuasiStaticAcquisition.PostProcessingInfo imp = new QuasiStaticAcquisition.PostProcessingInfo();
                        imp.Name = lvi.SubItems[0].Text.Trim();
                        imp.Settings = lvi.SubItems[1].Text.Trim();
                        m_PostProcessingSettings[ispi++] = imp;
                    }
                }
            }
            //CameraDisplay.Enabled = false;
            CheckGPU = chkEnableGPUTest.Checked;
            WriteClusterFiles = chkWriteClusterFiles.Checked;
            DumpFocusInfo = chkDumpFocusInfo.Checked;
            ShouldStop = false;
            EnableControls(false);
            StageSimulationOnly = chkStageSimOnly.Checked;
            RecordedTrajectorySamples = null;
            try
            {
                m_QAPattern = txtSummaryFile.Text;
                (m_ScanThread = new System.Threading.Thread(new System.Threading.ThreadStart(Scan))).Start();
            }
            catch (Exception xc)
            {
                iLog.Log("btnStart_Click", xc.ToString());
                ShouldStop = true;
                EnableControls(true);
                //CameraDisplay.Enabled = true;
            }
        }

        QuasiStaticAcquisition.PostProcessingInfo [] m_PostProcessingSettings = new QuasiStaticAcquisition.PostProcessingInfo[0];

        private void btnStop_Click(object sender, EventArgs e)
        {
            ShouldStop = true;        
        }

        delegate void dEnableCtls(bool enable);

        void EnableControls(bool enable)
        {
            if (this.InvokeRequired)
            {
                this.Invoke(new dEnableCtls(EnableControls), new object[] { enable });
                return;
            }
            txtBaseThickness.Enabled = !AutoScan && enable;
            txtContinuousMotionFraction.Enabled = !AutoScan && enable;
            txtClusterThreshold.Enabled = !AutoScan && enable;
            txtEmuThickness.Enabled = !AutoScan && enable;
            txtFocusSweep.Enabled = !AutoScan && enable;
            txtLayers.Enabled = !AutoScan && enable;
            txtMaxX.Enabled = !AutoScan && enable;
            txtMaxY.Enabled = !AutoScan && enable;
            txtMinValidLayers.Enabled = !AutoScan && enable;
            txtMinX.Enabled = !AutoScan && enable;
            txtMinY.Enabled = !AutoScan && enable;
            txtPitch.Enabled = !AutoScan && enable;
            txtPositionTolerance.Enabled = !AutoScan && enable;
            txtSlowdownTime.Enabled = !AutoScan && enable;
            txtSummaryFile.Enabled = !AutoScan && enable;
            txtViewOverlap.Enabled = !AutoScan && enable;
            txtXYAcceleration.Enabled = !AutoScan && enable;
            txtXYSpeed.Enabled = !AutoScan && enable;
            txtZAcceleration.Enabled = !AutoScan && enable;
            txtZSpeed.Enabled = !AutoScan && enable;
            txtZSweep.Enabled = !AutoScan && enable;
            txtMotionLatency.Enabled = !AutoScan && enable;
            chkBottom.Enabled = !AutoScan && enable;
            chkTop.Enabled = !AutoScan && enable;
            rdMoveX.Enabled = !AutoScan && enable;
            rdMoveY.Enabled = !AutoScan && enable;
            btnDel.Enabled = !AutoScan && enable;
            btnDuplicate.Enabled = !AutoScan && enable;
            btnFromHere.Enabled = !AutoScan && enable;
            btnLoad.Enabled = !AutoScan && enable;
            btnMakeCurrent.Enabled = !AutoScan && enable;
            btnNew.Enabled = !AutoScan && enable;
            btnStart.Enabled = !AutoScan && enable;
            btnRecover.Enabled = !AutoScan && enable;
            btnStop.Enabled = !enable;
            btnToHere.Enabled = !AutoScan && enable;
            btnExit.Enabled = enable;
            btnEmptyFocusMap.Enabled = !AutoScan && enable;
            btnDumpZMap.Enabled = !AutoScan && enable;
            btnStepPaste.Enabled = !AutoScan && enable;
            btnStepReset.Enabled = !AutoScan && enable;
            btnStepSave.Enabled = !AutoScan && enable;
            lvPostProcessing.Enabled = !AutoScan && enable;
            chkStageSimOnly.Enabled = !AutoScan && enable;
            chkEnableGPUTest.Enabled = !AutoScan && enable;
            chkDumpFocusInfo.Enabled = !AutoScan && enable;
            chkWriteClusterFiles.Enabled = !AutoScan && enable;
            txtZDataExpirationMin.Enabled = !AutoScan && enable;
            txtMaxViewsStrip.Enabled = !AutoScan && enable;            
            if (!AutoScan && RecordedTrajectorySamples != null)
            {
                iScanDataDisplay.Display("Recorded Trajectory Samples", TrajectoryDataToString(RecordedTrajectorySamples));
                RecordedTrajectorySamples = null;
            }
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        class SyncZData
        {
            private uint StartSignature = 0;
            private bool Valid = false;
            private double ZCenter = 0.0;
            private uint EndSignature = 0;
            private uint Signature = 0;
            public void WriteZData(bool v, double zc)
            {
                ++Signature;
                StartSignature = Signature;
                Valid = v;
                ZCenter = zc;
                EndSignature = Signature;
            }
            public void ReadZData(ref bool v, ref double zc)
            {
                uint s;
                do
                {
                    s = StartSignature;
                    v = Valid;
                    zc = ZCenter;
                }
                while (EndSignature != s);
            }
        }

        class GrabData
        {
            public int strip;
            public int side;
            public int view;
            public object GrabSeq;
            public SySal.StageControl.TrajectorySample[] StageInfo;
            public double[] TimeInfo;            
            public int GPUBank;
        };

        class ProcOutputData
        {
            public SySal.StageControl.TrajectorySample[] ImagePositionInfo;            
            public SySal.Imaging.Fast.IClusterSequenceContainer Clusters;
            public SySal.Imaging.Fast.IImageProcessorFast ClusterContainerDeallocator;
            public SySal.Imaging.ImageProcessingException[] Exceptions;
            public SySal.StageControl.TrajectorySample[] StageInfo;
            public string OutputFilename;
            public bool EmulsionValid;
            public bool EmulsionBoundarySeen;
            public string DumpStr;            
        };

        public System.Diagnostics.Stopwatch GeneralTimeSource;

        bool StageSimulationOnly;
        SySal.StageControl.TrajectorySample[] RecordedTrajectorySamples = null;
        
        int[] GrabReadySignal = new int[0];
        bool[] ProcReadySignal = new bool[0];
        GrabData[] GrabDataSlots = new GrabData[0];
        ProcOutputData[] ProcDataSlots = new ProcOutputData[0];

        bool FocusInterrupt;

        static void GPUProcess(SySal.Imaging.Fast.IImageProcessorFast igpu, int bank, SySal.Imaging.LinearMemoryImage lmi, ProcOutputData po)
        {
            lock (igpu)
            {
                igpu.CurrentBank = bank;
                igpu.Input = lmi;
                po.ClusterContainerDeallocator = igpu;
                po.Clusters = po.ClusterContainerDeallocator.ClusterSequence;
                po.Exceptions = igpu.Warnings;
            }
        }

        private void ImageProcessingThread(object oid)
        {
            FocusInfo fi = new FocusInfo();
            GrabData next_gd = null;
            GrabData gd = null;
            SySal.Imaging.LinearMemoryImage lmi = null;
            SySal.Imaging.LinearMemoryImage next_lmi = null;
            try
            {
                int id = (int)oid;
                int signid = id + 1;
                while (TerminateSignal.WaitOne(0) == false)
                {
                    System.Threading.Thread mapthread = new System.Threading.Thread((System.Threading.ThreadStart)
                            delegate()
                            {
                                int i = -1;
                                next_gd = null;
                                next_lmi = null;
                                for (i = GrabDataSlots.Length - 1; i >= 0 && (GrabDataSlots[i] == null || GrabReadySignal[i] != 0); i--) ;
                                if (i >= 0)
                                {
                                    next_gd = GrabDataSlots[i];
                                    if (next_gd != null)
                                        lock (next_gd)
                                        {
                                            if (GrabReadySignal[i] != 0)
                                            {
                                                next_gd = null;
                                                return;
                                            }
                                            GrabReadySignal[i] = signid;
                                            GrabDataSlots[i] = null;

                                            if (next_gd.GrabSeq == null) return;
                                            next_lmi = (SySal.Imaging.LinearMemoryImage)iGrab.MapSequenceToSingleImage(next_gd.GrabSeq);
                                            iGrab.ClearGrabSequence(next_gd.GrabSeq);
                                            next_gd.GrabSeq = null;
                                            GrabReadySignal[i] = 0;
                                            System.Threading.Interlocked.Decrement(ref LockedGrabSequences);
                                            iLog.Log("GPUProc " + id, "Sequences locked = " + LockedGrabSequences);
                                        }
                                }

                            });
                    mapthread.Start();

                    if (gd != null)
                    {
                        TimeSpan procstart = GeneralTimeSource.Elapsed;
                        ProcOutputData po = new ProcOutputData();
                        try
                        {
                            SetQueueLength();
                            int im;
                            try
                            {
                                GPUProcess((SySal.Imaging.Fast.IImageProcessorFast)iGPU[id], gd.GPUBank, lmi, po);
                            }
                            catch (SySal.Imaging.Fast.TemporaryMemoryException x)
                            {
                                iLog.Log("ImageProcessingThread " + id, x.ToString());
                                bool _mustend;
                                while ((_mustend = TerminateSignal.WaitOne(1000)) == false)
                                    try
                                    {
                                        GPUProcess((SySal.Imaging.Fast.IImageProcessorFast)iGPU[id], gd.GPUBank, lmi, po);
                                        break;
                                    }
                                    catch (SySal.Imaging.Fast.TemporaryMemoryException) { };
                                if (_mustend)
                                    throw new Exception("Aborting image processing wait.");
                            }

                            if (gd.TimeInfo != null)
                                po.ImagePositionInfo = ComputePosFromTime(gd.StageInfo, gd.TimeInfo);
                            else
                                po.ImagePositionInfo = gd.StageInfo;
                            po.StageInfo = gd.StageInfo;
                            iGrab.ClearMappedImage(lmi);
                            po.Clusters.PixelToMicron = ImagingConfig.Pixel2Micron;
                            lmi = null;
                            int firstlayer = po.Clusters.Images - 1;
                            int lastlayer = 0;
                            for (im = 0; im < po.Clusters.Images; im++)
                            {
                                po.Clusters.SetImagePosition(im, po.ImagePositionInfo[im].Position);
                                if (po.Clusters.ClustersInImage(im) >= WorkConfig.ClusterThreshold)
                                {
                                    if (im < firstlayer) firstlayer = im;
                                    if (im > lastlayer) lastlayer = im;
                                };
                            }
                            if (DumpFocusInfo)
                            {
                                try
                                {
                                    po.DumpStr = "";
                                    for (im = 0; im < po.Clusters.Images; im++)
                                        po.DumpStr += "\n" + gd.strip + "\t" + gd.view + "\t" + gd.side + "\t" + im + "\t" +
                                                po.ImagePositionInfo[im].TimeMS.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                                po.ImagePositionInfo[im].Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                                po.ImagePositionInfo[im].Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                                po.ImagePositionInfo[im].Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\t" +
                                                po.Clusters.ClustersInImage(im);
                                }
                                catch (Exception x)
                                {
                                    iLog.Log("ImageProcessingThread " + id, x.ToString());
                                }
                            }
                            if (lastlayer - firstlayer + 1 >= WorkConfig.MinValidLayers)
                            {
                                po.EmulsionValid = true;
                                po.EmulsionBoundarySeen = ((firstlayer > 0) && (lastlayer < WorkConfig.Layers - 1));
                                double newz = (po.ImagePositionInfo[firstlayer].Position.Z + po.ImagePositionInfo[lastlayer].Position.Z) * 0.5;
                                iLog.Log("ImageProcessingThread " + id, "View " + gd.view + " sets Z to " + newz + " firstlayer " + firstlayer + " lastlayer " + lastlayer + " BoundarySeen " + po.EmulsionBoundarySeen
                                    /*+ " " + Enumerable.Range(firstlayer, lastlayer - firstlayer + 1).Select(xx => po.ImagePositionInfo[xx].Position.Z.ToString()).Aggregate((a,b) => a + " " + b)*/);
                                ZData[gd.side].WriteZData(true, newz);
                                if (firstlayer == 0 && lastlayer < po.Clusters.Images - 1)
                                {
                                    fi.Pos.X = po.ImagePositionInfo[lastlayer].Position.X;
                                    fi.Pos.Y = po.ImagePositionInfo[lastlayer].Position.Y;
                                    fi.BottomZ = po.ImagePositionInfo[lastlayer].Position.Z;
                                    fi.TopZ = fi.BottomZ + WorkConfig.EmulsionThickness/*m_Thickness[gd.side]*/;
                                    fi.Valid = true;
                                }
                                else if (lastlayer == po.Clusters.Images - 1 && firstlayer > 0)
                                {
                                    fi.Pos.X = po.ImagePositionInfo[firstlayer].Position.X;
                                    fi.Pos.Y = po.ImagePositionInfo[firstlayer].Position.Y;
                                    fi.TopZ = po.ImagePositionInfo[firstlayer].Position.Z;
                                    fi.BottomZ = fi.TopZ - WorkConfig.EmulsionThickness/*m_Thickness[gd.side]*/;
                                    fi.Valid = true;
                                }
                                else
                                {
                                    fi.Pos.X = 0.5 * (po.ImagePositionInfo[firstlayer].Position.X + po.ImagePositionInfo[lastlayer].Position.X);
                                    fi.Pos.Y = 0.5 * (po.ImagePositionInfo[firstlayer].Position.Y + po.ImagePositionInfo[lastlayer].Position.Y);
                                    fi.TopZ = po.ImagePositionInfo[firstlayer].Position.Z;
                                    fi.BottomZ = po.ImagePositionInfo[lastlayer].Position.Z;
                                    if (firstlayer == 0 && lastlayer == po.Clusters.Images - 1)
                                    {
                                        fi.Valid = false;
                                        iLog.Log("ImageProcessingThread " + id, "Anomalous view " + gd.view + "\nBEGIN DUMP\nZ-samples" + ((po.StageInfo == null) ? "GD-STAGEINFO-NULL" :
                                            gd.StageInfo.Select((x, i) => "\n" + i + " " + x.TimeMS.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " " + x.Position.X.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " + x.Position.Y.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " " + x.Position.Z.ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + " Clusters " + po.Clusters.ClustersInImage(i)).DefaultIfEmpty("").Aggregate((a, b) => a + b)) +
                                            "\nEND DUMP");
                                    }
                                    else fi.Valid = true;
                                }
                                fi.MeasureTime = System.DateTime.Now;
                                iLog.Log("ImageProcessingThread " + id, "View " + gd.view + " FocusMap.Write " + gd.side + " X " + fi.Pos.X + " Y " + fi.Pos.Y + " TopZ " + fi.TopZ + " BottomZ " + fi.BottomZ + " Valid " + fi.Valid + " firstlayer " + firstlayer + " lastlayer " + lastlayer + " Images " + po.Clusters.Images + " newz " + newz + " firstZ " + po.ImagePositionInfo[firstlayer].Position.Z + " lastZ " + po.ImagePositionInfo[lastlayer].Position.Z);
                                m_FocusMap.Write(fi, gd.side == 0);
                            }
                            else
                            {
                                po.EmulsionValid = false;
                                po.EmulsionBoundarySeen = false;
                                iLog.Log("ImageProcessingThread " + id, "View " + gd.view + " not enough layers: " + (lastlayer - firstlayer + 1));
                                //ZData[gd.side].WriteZData(false, 0.0);
                                double newz = (po.ImagePositionInfo[firstlayer].Position.Z + po.ImagePositionInfo[lastlayer].Position.Z) * 0.5;
                                if (firstlayer > WorkConfig.MinValidLayers / 2)
                                {
                                    newz = po.ImagePositionInfo[firstlayer].Position.Z - WorkConfig.MinValidLayers * 0.5 * WorkConfig.Pitch;
                                    //ZData[gd.side].WriteZData(true, newz);
                                }
                                else if (lastlayer < WorkConfig.MinValidLayers / 2)
                                {
                                    newz = po.ImagePositionInfo[lastlayer].Position.Z + WorkConfig.MinValidLayers * 0.5 * WorkConfig.Pitch;
                                    // ZData[gd.side].WriteZData(true, newz);
                                }
                            }
                        }
                        catch (Exception xc)
                        {
                            iLog.Log("ImageProcessingThread " + id, "Strip: " + gd.strip + " Side: " + gd.side + " View: " + gd.view + "\r\n" + xc.ToString());
                            po.EmulsionValid = false;
                            po.EmulsionBoundarySeen = false;
                            po.Clusters = null;
                            po.StageInfo = new StageControl.TrajectorySample[0];
                            po.ImagePositionInfo = new StageControl.TrajectorySample[0];
                            po.Exceptions = new Imaging.ImageProcessingException[0];
                        }/*
                        if (po.EmulsionValid == false)
                        {
                            FocusInterrupt = true;
                        }*/

                        //po.OutputFilename = I_Acq.Sequences[gd.view].ClusterMapFileName;
                        po.OutputFilename = I_Acq.Sequences[gd.view].ClusterMapFileNameWithInversion((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both && I_Side == MaxSide), (uint)Views);
                        lock (WriteQueue)
                            WriteQueue.Enqueue(po);
                        System.Threading.Interlocked.Increment(ref I_ViewsProcessed);
                        TimeSpan procend = GeneralTimeSource.Elapsed;
                        iLog.Log("GPUProc " + id, "GPU time " + (procend - procstart).TotalMilliseconds.ToString() + " logtime " + GeneralTimeSource.ElapsedMilliseconds);
                        /*
                        lock (ProcReadySignal)
                        {
                            ProcDataSlots[gd.view] = po;
                            ProcReadySignal[gd.view] = true;
                        }
                         */
                        //iLog.Log("ImageProcessingThread " + id, "Flagged ProcessReadySignal " + gd.view + " check " + ProcReadySignal[gd.view]);                        

                        if (lmi != null)
                        {
                            try
                            {
                                iGrab.ClearMappedImage(lmi);
                            }
                            catch (Exception) { }
                            lmi = null;
                        }
                        if (gd.GrabSeq != null)
                        {
                            try
                            {
                                iGrab.ClearGrabSequence(gd.GrabSeq);
                            }
                            catch (Exception) { }
                            gd.GrabSeq = null;
                        }
                    }
                    if (mapthread != null)
                    {
                        mapthread.Join();
                        gd = next_gd;
                        lmi = next_lmi;
                    }
                }
            }
            catch (Exception xc1)
            {
                iLog.Log("ImageProcessingThread " + oid.ToString(), xc1.ToString());
            }
            if (next_lmi != null)
            {
                try
                {
                    iGrab.ClearMappedImage(next_lmi);
                }
                catch (Exception) { }
                next_lmi = null;
            }
            if (next_gd != null)
                if (next_gd.GrabSeq != null)
                {
                    try
                    {
                        iGrab.ClearGrabSequence(next_gd.GrabSeq);
                    }
                    catch (Exception) { }
                    next_gd.GrabSeq = null;
                }
        }

        System.Threading.Thread[] ImgProcThreads = new System.Threading.Thread[0];        
        
        double GrabTriggerZ = 0.0;
        int I_Strip = 0;
        int I_Side = 0;
        int I_View = 0;
        int I_ViewsGrabbed = 0;
        int I_ViewsProcessed = 0;
        int I_ViewsWritten = 0;
        QuasiStaticAcquisition I_Acq = null;

        System.Threading.AutoResetEvent ArmGrabbing = new System.Threading.AutoResetEvent(false);
        System.Threading.AutoResetEvent GrabDone = new System.Threading.AutoResetEvent(false);
        bool m_GrabDone = false;

        System.Threading.ManualResetEvent TerminateSignal = new System.Threading.ManualResetEvent(false);

        System.Threading.ManualResetEvent WriteTerminateSignal = new System.Threading.ManualResetEvent(false);

        System.Collections.Stack ViewStack = new System.Collections.Stack();

        void GrabberThreadProc()
        {
            System.Threading.WaitHandle [] evs = new System.Threading.WaitHandle[] { TerminateSignal, ArmGrabbing };            
            while (System.Threading.WaitHandle.WaitAny(evs) != 0)
            {
                while (iStage.GetPos(StageControl.Axis.Z) > GrabTriggerZ) ;
                try
                {
                    iStage.StartRecording(1.0, WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000.0);
                    GrabData gd = new GrabData();
                    gd.strip = I_Strip;
                    gd.side = I_Side;
                    gd.view = I_View;
                    gd.GrabSeq = iGrab.GrabSequence();
                    //GrabDone.Set();
                    m_GrabDone = true;
                    gd.TimeInfo = iGrab.GetImageTimesMS(gd.GrabSeq);
                    gd.StageInfo = iStage.Trajectory;
                    lock (ViewStack)
                    {
                        ViewStack.Push(gd);
                    }
                    //iLog.Log("GrabberThread", "Grab View " + I_View);
                }
                catch (Exception xc)
                {
                    iLog.Log("GrabberThread", xc.ToString());
                }
                System.Threading.Thread.Yield();
            }
        }

        SyncZData[] ZData = new SyncZData[] { new SyncZData(), new SyncZData() };

        System.Collections.Queue WriteQueue = new System.Collections.Queue();

        bool CheckGPU = false;
        bool WriteClusterFiles = false;
        int MinSide, MaxSide;

        FocusMap m_FocusMap = null;
        double [] m_Thickness = new double[2] {0,0};

        int Views, Strips;

        uint RecoverStart = 0;

        string m_DebugString = "";

        string m_WriteQueueDebugString = "";

        int LockedGrabSequences = 0;

        private void Scan()
        {            
            System.Collections.ArrayList traj_arr = new System.Collections.ArrayList();
            TerminateSignal.Reset();
            WriteTerminateSignal.Reset();
            GrabDone.Reset();
            RecordedTrajectorySamples = null;
            long timems = 0;
            long dtimems;
            long lastms;
            long currms;
            int igpu;
            int gpubank = 1;
            int totalviews = -1;
            int viewsok = 0;
            StageControl.IStageWithTimer iStageWT = null;
            if (iStage is StageControl.IStageWithTimer) iStageWT = (StageControl.IStageWithTimer)iStage;
            StageControl.IStageWithDirectTrajectoryData iStageDT = null;
            if (iStage is StageControl.IStageWithDirectTrajectoryData) iStageDT = (StageControl.IStageWithDirectTrajectoryData)iStage;
            else
            {
                iLog.Log("Scan setup", "IStageWithDirectTrajectoryData not supported.");
                ShouldStop = true;
                EnableControls(true);
                return;
            }
            Imaging.IImageGrabberWithTimer iGrabWT = null;
            if (iGrab is Imaging.IImageGrabberWithTimer) iGrabWT = (Imaging.IImageGrabberWithTimer)iGrab;
            bool use_timed_scan_grab = (iStageWT != null) && (iGrabWT != null);
            iLog.Log("Scan setup", "Using timed scan: " + use_timed_scan_grab + " (Stage support for timer = " + (iStageWT != null) + ", Grabber support for timer = " + (iGrabWT != null) + ")");

            double stripsidestartx, stripsidestarty, stripsidefinishx, stripsidefinishy, stripsideviewdeltax, stripsideviewdeltay;
            ViewStack.Clear();
            System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Normal;
            System.Threading.Thread grabthread = null;
            WriteQueue.Clear();
            System.Threading.Thread writetthread = new System.Threading.Thread(new System.Threading.ThreadStart(
                delegate()
                {
                    m_WriteQueueDebugString = "A";
                    while (WriteTerminateSignal.WaitOne(0) == false)
                    {
                        m_WriteQueueDebugString = "B";
                        SetWriterQueueLength();
                        m_WriteQueueDebugString = "C";
                        QuasiStaticAcquisition acq = null;                        
                        ProcOutputData po = null;
                        m_WriteQueueDebugString = "D";
                        lock (WriteQueue)
                        {
                            m_WriteQueueDebugString = "E";
                            if (WriteQueue.Count > 0)
                            {
                                //acq = (QuasiStaticAcquisition)WriteQueue.Dequeue();
                                m_WriteQueueDebugString = "F";
                                po = (ProcOutputData)WriteQueue.Dequeue();
                                m_WriteQueueDebugString = "G";
                            }
                        }
                        m_WriteQueueDebugString = "H";
                        if (po != null)
                        {
                            if (po.Clusters != null)
                            {
                                m_WriteQueueDebugString = "I";
                                System.Exception wxc = null;
                                while (WriteTerminateSignal.WaitOne(0) == false)
                                    try
                                    {
                                        wxc = null;
                                        m_WriteQueueDebugString = "J";
                                        if (WriteClusterFiles) po.Clusters.WriteToFile(po.OutputFilename);
                                        m_WriteQueueDebugString = "L";
                                        break;
                                    }
                                    catch (Exception xc)
                                    {
                                        wxc = xc;
                                    }
                                m_WriteQueueDebugString = "M";
                                if (wxc != null) iLog.Log("WriteThread", wxc.ToString());
                                m_WriteQueueDebugString = "N";
                                try
                                {
                                    m_WriteQueueDebugString = "O";
                                    lock (po.ClusterContainerDeallocator)
                                        po.ClusterContainerDeallocator.ReleaseClusterSequence(po.Clusters);
                                }
                                catch (Exception xc)
                                {
                                    m_WriteQueueDebugString = "P";
                                    iLog.Log("WriteThread", xc.ToString());
                                }
                                try
                                {
                                    m_WriteQueueDebugString = "Q";
                                    if (po.DumpStr != null)
                                        System.IO.File.AppendAllText(m_FocusDumpFileName, po.DumpStr);
                                }
                                catch (Exception xc)
                                {
                                    m_WriteQueueDebugString = "R";
                                    iLog.Log("WriteThread", xc.ToString());
                                }
                            }
                            m_WriteQueueDebugString = "S";
                            System.Threading.Interlocked.Increment(ref I_ViewsWritten);
                            m_WriteQueueDebugString = "T";
                            SetQueueLength();
                            m_WriteQueueDebugString = "U";
                        }
                        else
                        {
                            m_WriteQueueDebugString = "V";
                            System.Threading.Thread.Sleep(1000);
                        }
                        m_WriteQueueDebugString = "W";
                    }
                }
                ));
            writetthread.Priority = System.Threading.ThreadPriority.Normal;
            writetthread.Start();
            
            QuasiStaticAcquisition.Zone Zone = null;
            try
            {
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Normal;
                iCamDisp.EnableAutoRefresh = false;
                iGrab.SequenceSize = (int)WorkConfig.Layers;
                double fovwidth = Math.Abs(ImagingConfig.ImageWidth * ImagingConfig.Pixel2Micron.X);
                double fovheight = Math.Abs(ImagingConfig.ImageHeight * ImagingConfig.Pixel2Micron.Y);
                double stepwidth = fovwidth - WorkConfig.ViewOverlap;
                double stepheight = fovheight - WorkConfig.ViewOverlap;
                double stepxspeed = WorkConfig.ContinuousMotionDutyFraction * stepwidth * WorkConfig.FramesPerSecond / WorkConfig.Layers;
                double stepyspeed = WorkConfig.ContinuousMotionDutyFraction * stepheight * WorkConfig.FramesPerSecond / WorkConfig.Layers;
                double lowestz = iStage.GetNamedReferencePosition("LowestZ");
                double[] expectedzcenters = new double[]
                {
                    lowestz + WorkConfig.EmulsionThickness * 1.5 + WorkConfig.BaseThickness, lowestz + WorkConfig.EmulsionThickness * 0.5
                };
                ZData[0].WriteZData(StageSimulationOnly, expectedzcenters[0]);
                ZData[1].WriteZData(StageSimulationOnly, expectedzcenters[1]);
                if (stepheight <= 0.0 || stepwidth <= 0.0) throw new Exception("Too much overlap, or null image or pixel/micron factor defined.");
                SySal.BasicTypes.Vector2 StripDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 ViewDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 FocusDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector ImageDelta = new BasicTypes.Vector();
                ImageDelta.Z = -WorkConfig.Pitch;                
                if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                {
                    ImageDelta.Y = 0.0;
                    ImageDelta.X = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepwidth / (WorkConfig.Layers - 1));                    
                    ViewDelta.X = stepwidth;
                    ViewDelta.Y = 0.0;
                    Views = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxX - TransformedScanRectangle.MinX) / stepwidth));
                    StripDelta.X = 0.0;
                    StripDelta.Y = stepheight;
                    Strips = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxY - TransformedScanRectangle.MinY) / stepheight));
                }
                else
                {
                    ImageDelta.X = 0.0;
                    ImageDelta.Y = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepheight / (WorkConfig.Layers - 1));
                    ViewDelta.X = 0.0;
                    ViewDelta.Y = stepheight;
                    Views = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxY - TransformedScanRectangle.MinY) / stepheight));
                    StripDelta.X = stepwidth;
                    StripDelta.Y = 0.0;
                    Strips = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxX - TransformedScanRectangle.MinX) / stepwidth));
                }
                FocusDelta.X = stepwidth;
                FocusDelta.Y = stepheight;
                Zone = new QuasiStaticAcquisition.Zone(m_QAPattern);
                Zone.Views = (uint)Views;
                Zone.Strips = (uint)Strips;
                Zone.HasTop = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both);
                Zone.HasBottom = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both);
                Zone.ScanRectangle = ScanRectangle;
                Zone.TransformedScanRectangle = TransformedScanRectangle;
                Zone.PlateMap = iMap.InversePlateMap;
                Zone.ImageDelta = ImageDelta;
                Zone.ViewDelta = ViewDelta;
                Zone.StripDelta = StripDelta;
                Zone.PostProcessingSettings = m_PostProcessingSettings;
                Zone.OutputRWDFileName = (OutputRWDFileName == null) ? "" : OutputRWDFileName;
                m_FocusDumpFileName = QuasiStaticAcquisition.GetFocusDumpPattern(m_QAPattern);
                if (m_FocusDumpFileName != null)
                    try
                    {
                        System.IO.File.WriteAllText(m_FocusDumpFileName, "STRIP\tVIEW\tSIDE\tIMAGE\tTimeMS\tX\tY\tZ\tCLUSTERS");
                    }
                    catch (Exception x)
                    {
                        iLog.Log("Scan", "Can't initialize focus dump file.\r\n" + x.ToString());
                    }

                {
                    System.IO.StringWriter swr = new System.IO.StringWriter();
                    FullSpeedAcqSettings.s_XmlSerializer.Serialize(swr, WorkConfig);
                    Zone.ScanSettings = swr.ToString();
                    Zone.Update();
                }                
                m_FocusMap = new FocusMap(TransformedScanRectangle, FocusDelta, System.TimeSpan.FromMinutes((double)WorkConfig.ZDataExpirationMin), m_FocusMap);
                int i_strip, i_view, i_side, i_image;
                totalviews = Strips * Views * ((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) ? 2 : 1);
                SetProgressValue(0);
                SetProgressMax(totalviews);
                SetQueueLengthMax(Math.Min(Views, /*iGrab.MappedSequences*/ iGrab.Sequences));
                SetWriterQueueLengthMax(Views);
                viewsok = 0;
                int[] clustercounts = new int[WorkConfig.Layers];
                GrabReadySignal = new int[Views];
                ProcReadySignal = new bool[Views];
                GrabDataSlots = new GrabData[Views];
                ProcDataSlots = new ProcOutputData[Views];
                TerminateSignal.Reset();
                SetQueueLength();
                SetWriterQueueLength();
                FocusInfo fi = new FocusInfo();
                long deadtimestart = GeneralTimeSource.ElapsedMilliseconds;
                long deadtimeend;
                iLog.Log("Scan", "Grab sequences available = " + iGrab.Sequences + " seqsize = " + iGrab.SequenceSize + " mapped = " + iGrab.MappedSequences);
                if (CheckGPU)
                {
                    /* GPU TESTING */
                    SetStatus("Checking GPU functions");
                    int seqsize;
                    for (seqsize = 1; seqsize <= WorkConfig.Layers; seqsize++)
                    {
                        string gpustr = "";
                        //SetStatus("Checking GPU functions: step " + seqsize + " of " + WorkConfig.Layers);
                        iStage.Stop(SySal.StageControl.Axis.Z);
                        iLog.Log("GPU TESTING", "Step 0 SeqSize " + seqsize);
                        iGrab.SequenceSize = seqsize;
                        iLog.Log("GPU TESTING", "Step 1");
                        for (igpu = 0; igpu < iGPU.Length; igpu++)
                        {
                            iLog.Log("GPU TESTING", "Step 2");
                            object test_gseq = iGrab.GrabSequence();
                            iLog.Log("GPU TESTING", "Step 3");
                            SySal.Imaging.LinearMemoryImage test_lmi = (SySal.Imaging.LinearMemoryImage)iGrab.MapSequenceToSingleImage(test_gseq);
                            iLog.Log("GPU TESTING", "Step 4");
                            iGrab.ClearGrabSequence(test_gseq);
                            iLog.Log("GPU TESTING", "Step 5 - GPU " + igpu);
                            long start = GeneralTimeSource.ElapsedMilliseconds;
                            iGPU[igpu].Input = test_lmi;
                            int cls = 0;
                            SySal.Imaging.Cluster[][] clsplanes = iGPU[igpu].Clusters;
                            long end = GeneralTimeSource.ElapsedMilliseconds;
                            foreach (SySal.Imaging.Cluster[] clsp in clsplanes)
                                cls += clsp.Length;
                            iLog.Log("GPU TESTING", "Step 6 - GPU " + igpu + " clusters " + cls + " MS " + (end - start));
                            gpustr += " " + ((end - start) / seqsize);
                            iLog.Log("GPU TESTING", "Step 7");
                            iGrab.ClearMappedImage(test_lmi);
                            iLog.Log("GPU TESTING", "Step 8 SeqSize " + seqsize);
                        }
                        SetStatus("GPU teststep " + seqsize + " of " + WorkConfig.Layers + " res (ms/image): " + gpustr);
                    }
                    iGrab.SequenceSize = (int)WorkConfig.Layers;
                    SetStatus("GPU check passed.");
                }                
                ImgProcThreads = new System.Threading.Thread[iGPU.Length];
                for (igpu = 0; igpu < iGPU.Length; igpu++)
                {
                    (ImgProcThreads[igpu] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(ImageProcessingThread))).Start(igpu);
                    ImgProcThreads[igpu].Priority = System.Threading.ThreadPriority.Normal;
                }
                //(grabthread = new System.Threading.Thread(new System.Threading.ThreadStart(GrabberThreadProc))).Start();
                //grabthread.Priority = System.Threading.ThreadPriority.AboveNormal;
                GeneralTimeSource.Reset();
                GeneralTimeSource.Restart();
                iGrab.TimeSource = GeneralTimeSource;
                iStage.TimeSource = GeneralTimeSource;
                ShouldStop = CheckGPU;
                if (false)
                {
                    SetStatus("Checking stage recording functions.");
                    SySal.StageControl.TrajectorySample[] zerosamples = null;
                    int trials = 0;
                    const int maxtrials = 100;
                    do
                    {
                        trials++;
                        iStage.StartRecording(1, 100.0);
                        zerosamples = iStage.Trajectory;
                    }
                    while (zerosamples[0].TimeMS == 0.0 || zerosamples[0].Position.Z == 0.0 && trials < maxtrials);
                    if (trials == maxtrials) MessageBox.Show("Trials: " + trials, "Stage trajectory testing failure");
                    SetStatus("Stage recording check passed.");
                }
                long time1 = GeneralTimeSource.ElapsedMilliseconds;
                long time2 = time1;
                MinSide = (((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top)) ? 0 : 1);
                MaxSide = (((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) || (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom)) ? 1 : 0);
                System.Threading.Thread.CurrentThread.Priority = System.Threading.ThreadPriority.Highest;
                System.DateTime starttime = System.DateTime.Now;
                if (StageSimulationOnly) iStage.StartRecording(1.0, 100000.0);
                if (RecoverStart > 0)
                {
                    iLog.Log("Scan restarting in recover mode", "RecoverStart = " + RecoverStart);
                    if (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top)
                        Zone.Progress.TopStripsReady = RecoverStart;
                    if (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both || WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom)
                        Zone.Progress.BottomStripsReady = RecoverStart;
                    viewsok = (int)(RecoverStart * Views * ((WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) ? 2 : 1));
                    Zone.Update();
                }

                double ttz = 0.0;
                m_DebugString = "Start";
                SySal.StageControl.TrajectorySample [][] posamples = new StageControl.TrajectorySample[Views][];
                for (i_view = 0; i_view < Views; i_view++)
                    posamples[i_view] = new StageControl.TrajectorySample[WorkConfig.Layers];
                for (i_strip = (int)RecoverStart; ShouldStop == false && i_strip != Strips; i_strip++)
                /*
                                for (i_side = minside;
                                    i_side <= maxside && ShouldStop == false;
                                    i_side++)
                 */
                {
                    //for (i_strip = (i_side == minside) ? 0 : (strips - 1); ShouldStop == false && i_strip != ((i_side == minside) ? strips : -1); i_strip += ((i_side == minside) ? 1 : -1))
                    for (i_side = MinSide;
                        i_side <= MaxSide && ShouldStop == false;
                        i_side++)
                    {
                        //bool evenstripside = ((((i_side - minside) * strips + i_strip) % 2) == 0);
                        m_DebugString = "A";
                        bool evenstripside = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both ? (i_side == MinSide) : (i_strip % 2 == 0));
                        if (evenstripside)
                        {
                            stripsidestartx = TransformedScanRectangle.MinX + StripDelta.X * i_strip;
                            stripsidestarty = TransformedScanRectangle.MinY + StripDelta.Y * i_strip;
                            if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                            {
                                stripsidefinishx = TransformedScanRectangle.MaxX;
                                stripsidefinishy = stripsidestarty;
                            }
                            else
                            {
                                stripsidefinishx = stripsidestartx;
                                stripsidefinishy = TransformedScanRectangle.MaxY;
                            }
                            stripsideviewdeltax = ViewDelta.X;
                            stripsideviewdeltay = ViewDelta.Y;
                        }
                        else
                        {
                            stripsidefinishx = TransformedScanRectangle.MinX + StripDelta.X * i_strip;
                            stripsidefinishy = TransformedScanRectangle.MinY + StripDelta.Y * i_strip;
                            if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                            {
                                stripsidestartx = TransformedScanRectangle.MaxX;
                                stripsidestarty = stripsidefinishy;
                            }
                            else
                            {
                                stripsidestartx = stripsidefinishx;
                                stripsidestarty = TransformedScanRectangle.MaxY;
                            }
                            stripsideviewdeltax = -ViewDelta.X;
                            stripsideviewdeltay = -ViewDelta.Y;
                        }
                        /*
                        gpubank = 1 - gpubank;
                        for (igpu = 0; igpu < iGPU.Length; igpu++)
                        {
                            while (((SySal.Imaging.Fast.IImageProcessorFast)iGPU[igpu]).IsBankFree(gpubank) == false)
                                System.Threading.Thread.Sleep(100);
                            ((SySal.Imaging.Fast.IImageProcessorFast)iGPU[igpu]).CurrentBank = gpubank;
                        }
                         */
                        //bool evenstripside = (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) ? (i_side == 0) : (i_strip % 2 == 0);
                        iLog.Log("Scan", "Strip " + i_strip + " Side " + i_side + " WorkConfig.Sides " + WorkConfig.Sides + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Both) + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Top) + " " + (WorkConfig.Sides == FullSpeedAcqSettings.ScanMode.Bottom) + " Evenstripside " + evenstripside);
                        QuasiStaticAcquisition qa = new QuasiStaticAcquisition();
                        qa.FilePattern = QuasiStaticAcquisition.GetFilePattern(m_QAPattern, i_side == 0, (uint)i_strip);
                        qa.Sequences = new QuasiStaticAcquisition.Sequence[Views];
                        bool firstview = true;// !StageSimulationOnly;
                        deadtimeend = GeneralTimeSource.ElapsedMilliseconds;
                        SetStatus("Dead time: " + (deadtimeend - deadtimestart) + " ms");
                        int focusretry = 2;
                        i_view = -1;
                        m_DebugString = "B";
                        while (focusretry-- >= 0)
                        {
                            m_DebugString = "C";
                            FocusInterrupt = false;
                            lock (GrabReadySignal)
                            {
                                int i_v;
                                for (i_v = 0; i_v < Views; i_v++)
                                {
                                    GrabReadySignal[i_v] = 0;
                                    ProcReadySignal[i_v] = false;
                                    GrabDataSlots[i_v] = null;
                                    ProcDataSlots[i_v] = null;
                                }
                            }
                            I_ViewsGrabbed = 0;
                            I_ViewsProcessed = 0;
                            I_ViewsWritten = 0;
                            bool grabfailedretry = false;
                            m_DebugString = "D";
                            if (use_timed_scan_grab)
                            {
                                m_DebugString = "D--Idle";                                
                                iStageWT.Idle();
                                iGrabWT.Idle();
                            }
                            
                            System.Runtime.GCSettings.LatencyMode = System.Runtime.GCLatencyMode.SustainedLowLatency;
                            bool reliablepostime = false;
                            double pos = 0.0;
                            for (i_view = 0; i_view < Views && ShouldStop == false && (FocusInterrupt == false || focusretry == 0); i_view++)
                                if (use_timed_scan_grab)
                                {
#region TIME-DRIVEN-SCANNING
                                    const double zgotol = 1.0;
                                    m_DebugString = "D--1";
                                    if (grabfailedretry == false)
                                    {
                                        if ((i_view % WorkConfig.MaxViewsPerStrip) == 0)
                                        {
                                            gpubank = 1 - gpubank;
                                            for (igpu = 0; igpu < iGPU.Length; igpu++)
                                            {
                                                while (((SySal.Imaging.Fast.IImageProcessorFast)iGPU[igpu]).IsBankFree(gpubank) == false)
                                                    /*System.Threading.Thread.Sleep(100)*/
                                                        ;
                                            }
                                            iLog.Log("Scan", "GPU bank switched to " + gpubank);
                                        }
                                    }
                                    else grabfailedretry = false;
                                    m_DebugString = "E";
                                    QuasiStaticAcquisition.Sequence seq = qa.Sequences[i_view] = new QuasiStaticAcquisition.Sequence();
                                    seq.Owner = qa;
                                    seq.Id = (uint)i_view;
                                    seq.Layers = new QuasiStaticAcquisition.Sequence.Layer[WorkConfig.Layers];
                                    SetProgressValue(viewsok);
                                    for (i_image = 0; i_image < clustercounts.Length; i_image++) clustercounts[i_image] = -1;
                                    double tx = stripsidestartx + stripsideviewdeltax * i_view;
                                    double ty = stripsidestarty + stripsideviewdeltay * i_view;
                                    bool ok = false;
                                    double tz = ttz;
                                    fi.Pos.X = tx;
                                    fi.Pos.Y = ty;
                                    m_DebugString = "F";
                                    if (StageSimulationOnly)
                                    {
                                        fi.Valid = true;
                                        fi.MeasureTime = System.DateTime.Now.AddYears(1);
                                        fi.TopZ = iStage.GetNamedReferencePosition("LowestZ") + ((i_side == 0) ? (WorkConfig.BaseThickness + 2.0 * WorkConfig.EmulsionThickness) : (WorkConfig.EmulsionThickness));
                                        fi.BottomZ = fi.TopZ - WorkConfig.EmulsionThickness;
                                    }
                                    m_DebugString = "G";
                                    ok = StageSimulationOnly || m_FocusMap.GetFocusInfo(ref fi, fi.Pos, i_side == 0, WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X, WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.Y);
                                    iLog.Log("Scan Debug", "Focus info X " + fi.Pos.X + " Y " + fi.Pos.Y + " TopZ " + fi.TopZ + " BottomZ " + fi.BottomZ + " Valid " + fi.Valid);
                                    if (ok) tz = ttz = 0.5 * (fi.TopZ + fi.BottomZ);
                                    iLog.Log("Scan Debug", "OK " + ok + " FirstView " + firstview + " TZ " + tz + " TTZ " + ttz);
                                    m_DebugString = "H";
                                    if (ok == false)
                                    {
                                        m_DebugString = "I";
                                        reliablepostime = false;
                                        iStage.Stop(StageControl.Axis.X);
                                        iStage.Stop(StageControl.Axis.Y);
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "J";
                                        if (GoToPos(tx, ty, expectedzcenters[i_side], true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        double topz = 0.0, bottomz = 0.0;
                                        ok = FindEmulsionZs(expectedzcenters[i_side], ref topz, ref bottomz);
                                        iStage.Stop(StageControl.Axis.X);
                                        iStage.Stop(StageControl.Axis.Y);
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "K";
                                        if (ok == false)
                                        {
                                            m_DebugString = "L";
                                            /* make empty view */
                                            ProcOutputData po = new ProcOutputData();
                                            po.Clusters = null;// new SySal.Imaging.Cluster[0][];
                                            po.Exceptions = new Imaging.ImageProcessingException[0];
                                            po.ImagePositionInfo = new StageControl.TrajectorySample[0];
                                            po.StageInfo = new StageControl.TrajectorySample[0];
                                            ProcDataSlots[i_view] = po;
                                            ProcReadySignal[i_view] = true;
                                            lock (WriteQueue)
                                                WriteQueue.Enqueue(po);
                                            System.Threading.Interlocked.Increment(ref I_ViewsGrabbed);
                                            System.Threading.Interlocked.Increment(ref I_ViewsProcessed);
                                            continue;
                                        }
                                        m_DebugString = "M";
                                        m_Thickness[i_side] = topz - bottomz;
                                        fi.Pos.X = tx;
                                        fi.Pos.Y = ty;
                                        fi.TopZ = topz;
                                        fi.BottomZ = bottomz;
                                        fi.MeasureTime = System.DateTime.Now;
                                        fi.Valid = true;
                                        m_FocusMap.Write(fi, i_side == 0);
                                        tz = ttz = 0.5 * (topz + bottomz);
                                        SetStatus("FindEmulsionZs: " + ok + " top " + topz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " bottom " + bottomz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " thickness " + (topz - bottomz).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                                        while (GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, expectedzcenters[i_side], false) == false)
                                        {
                                            iStage.Stop(StageControl.Axis.X);
                                            iStage.Stop(StageControl.Axis.Y);
                                            iStage.Stop(StageControl.Axis.Z);
                                            SetStatus("Cannot reach scan start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        }
                                        m_DebugString = "N";
                                        SetStatus("FindEmulsionZs: " + ok + " top " + topz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " bottom " + bottomz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " thickness " + (topz - bottomz).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                                        tz += WorkConfig.ZSweep * 0.5;
                                        if (GoToPosZ(tz + /*WorkConfig.PositionTolerance*/ zgotol, true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, ty, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        else
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, tx, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "O";
                                    }
                                    else if (firstview)
                                    {
                                        m_DebugString = "P";
                                        if (GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, /*expectedzcenters[i_side]*/tz, true) == false) throw new Exception("Cannot reach scan start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        m_DebugString = "Q";
                                        if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, ty, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        else
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, tx, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        tz += WorkConfig.ZSweep * 0.5;
                                        if (GoToPosZ(tz + zgotol /*WorkConfig.PositionTolerance*/, true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        System.Threading.Thread.Sleep((int)(WorkConfig.Layers * 500.0 / WorkConfig.FramesPerSecond)); /* wait to get constant speed: use sweep time / 2 */
                                        m_DebugString = "R";
                                    }
                                    else tz += WorkConfig.ZSweep * 0.5;
                                    m_DebugString = "S";
                                    bool axiserror = false;
                                    long time3 = GeneralTimeSource.ElapsedMilliseconds;
                                    m_DebugString = "T";
                                    //if (GoToPosZ(tz + WorkConfig.PositionTolerance, true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                    m_DebugString = "U";

                                    bool overtravel = false;
                                    m_DebugString = "V";
                                    if (true /*reliablepostime == false*/)
                                    {
                                        lastms = GeneralTimeSource.ElapsedMilliseconds;
                                        pos = iStage.GetPos((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? StageControl.Axis.X : StageControl.Axis.Y);
                                        currms = GeneralTimeSource.ElapsedMilliseconds;
                                        timems = (lastms + currms) >> 1;
                                    }
                                    else
                                    {
                                        long ms = GeneralTimeSource.ElapsedMilliseconds;
                                        pos += (ms - timems) * ((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? stepxspeed : stepyspeed) * 1e-3;
                                        timems = ms;
                                    }
                                    double delta = (((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? tx : ty) - pos) * (evenstripside ? 1.0 : -1.0);
                                    dtimems = (long)(delta / ((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? stepxspeed : stepyspeed) * 1000.0);
                                    iLog.Log("Scan Debug", "tz " + tz + " ttz " + ttz + " tx " + tx + " ty " + ty + " pos " + pos + " currms " + currms + " lastms " + lastms + " timems " + timems + " dtimems " + dtimems);
                                    if (/*dtimems <= 0*/ delta < -WorkConfig.PositionTolerance)
                                    {
                                        m_DebugString = "W";
                                        iLog.Log("Scan", "Overtravel at view " + i_view + " deltax " + delta + " dtimems " + dtimems);
                                        overtravel = true;
                                    }

                                    if (overtravel)
                                    {
                                        int trials;
                                        iStage.Stop(StageControl.Axis.X);
                                        iStage.Stop(StageControl.Axis.Y);
                                        iStage.Stop(StageControl.Axis.Z);
                                        for (trials = 0; trials < 2 && GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, tz + zgotol /* WorkConfig.PositionTolerance */, true) == false; trials++)
                                        {
                                            iLog.Log("Scan", "Resync");
                                        }
                                        if (trials < 2)
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            System.Threading.Thread.Sleep((int)(WorkConfig.Layers * 1000.0 / WorkConfig.FramesPerSecond));
                                            lastms = GeneralTimeSource.ElapsedMilliseconds;
                                            pos = iStage.GetPos((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? StageControl.Axis.X : StageControl.Axis.Y);
                                            currms = GeneralTimeSource.ElapsedMilliseconds;
                                            timems = (lastms + currms) >> 1;
                                            delta = (((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? tx : ty) - pos) * (evenstripside ? 1.0 : -1.0);
                                            dtimems = (long)(delta / ((WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? stepxspeed : stepyspeed) * 1000.0);
                                        }
                                        else
                                        {
                                            axiserror = true;
                                            if (axiserror) throw new Exception("Axis error on travelling axis.");
                                        }
                                    }

                                    //iStage.Stop(StageControl.Axis.Z);

                                    long d2timems = (long)(WorkConfig.Layers * 1000 / WorkConfig.FramesPerSecond);                                    
                                    iStageWT.AtTimeMoveProfile(
                                        timems + dtimems - (long)WorkConfig.MotionLatencyMS, 
                                        StageControl.Axis.Z, 
                                        new bool[] { false, true }, 
                                        new double[] { 0.0, tz + zgotol }, 
                                        new double[] { -WorkConfig.ZSweepSpeed, WorkConfig.ZSpeed },
                                        new long[] { d2timems, 0 }, 
                                        WorkConfig.ZAcceleration, 
                                        WorkConfig.ZAcceleration);

                                    long zresettimems = timems + dtimems;
                                    iLog.Log("Scan Debug", "timems " + timems + " dtimems " + dtimems + " d2timems " + d2timems);
                                                                             

                                    m_DebugString = "A1";
                                    if (axiserror) throw new Exception("Axis error on travelling axis.");
                                    I_Strip = i_strip;
                                    I_Side = i_side;
                                    I_View = i_view;
                                    I_Acq = qa;

                                    if (StageSimulationOnly == false)
                                    {
                                        m_DebugString = "D1";
                                        iStage.StartRecording(1.0, d2timems + dtimems + (long)(2.0 * WorkConfig.MotionLatencyMS));
                                        GrabData gd = new GrabData();
                                        gd.strip = I_Strip;
                                        gd.side = I_Side;
                                        gd.view = I_View;
                                        try
                                        {
                                            m_DebugString = "G1";
                                            gd.GrabSeq = iGrabWT.GrabSequenceAtTime(dtimems + timems);
                                            iStage.CancelRecording();
                                            if (gd.GrabSeq != null)
                                                System.Threading.Interlocked.Increment(ref LockedGrabSequences);
                                            m_GrabDone = true;
                                            gd.GPUBank = gpubank;
                                            gd.StageInfo = posamples[I_View];
                                            var timeinfo = iGrab.GetImageTimesMS(gd.GrabSeq);
                                            ComputePosFromTime(gd.StageInfo, iStageDT, timeinfo);
                                            /*
                                            for (int _i_ = 4; _i_ < gd.StageInfo.Length; _i_++)
                                                if (gd.StageInfo[_i_].Position.Z >= gd.StageInfo[_i_ - 4].Position.Z)
                                                {
                                                    iLog.Log("SYNC-ERROR", gd.StageInfo.Select((x, i) => i + " " + x.TimeMS + " " + x.Position.Z).Aggregate((a, b) => a + "\r\n" + b));
                                                    System.IO.StringWriter sswr = new System.IO.StringWriter();
                                                    SySal.StageControl.TrajectorySample ts = new StageControl.TrajectorySample();
                                                    for (uint __j = 0; iStageDT.GetTrajectoryData(__j, ref ts); __j++)
                                                        sswr.WriteLine(__j + " " + ts.TimeMS + " " + ts.Position.X + " " + ts.Position.Y + " " + ts.Position.Z);
                                                    iLog.Log("SYNC-ERROR-INFO", sswr.ToString());
                                                    iLog.Log("SYNC-ERROR-ADDITIONAL-INFO", " timeinfo[0] " + timeinfo[0] + " grabseqattime " + (dtimems + timems) + " attimemoveprofile " + (timems + dtimems - (long)WorkConfig.MotionLatencyMS));                                                    
                                                    throw new Exception("SYNC ERROR - see above");
                                                }
                                            */
                                            reliablepostime = true;
                                            pos = (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X) ? gd.StageInfo[gd.StageInfo.Length - 1].Position.X : gd.StageInfo[gd.StageInfo.Length - 1].Position.Y;
                                            timems = (long)gd.StageInfo[gd.StageInfo.Length - 1].TimeMS;                                                                                        
                                            GrabDataSlots[I_View] = gd;
                                            GrabReadySignal[I_View] = 0;
                                            System.Threading.Interlocked.Increment(ref I_ViewsGrabbed);
                                            iLog.Log("VIEWTIME", i_view + " " + timeinfo[0] + " currtime " + GeneralTimeSource.ElapsedMilliseconds);
                                            m_DebugString = "H1";
                                        }
                                        catch (Exception xc)
                                        {
                                            iLog.Log("Scan", "Error: " + xc.ToString());
                                            if (gd != null && gd.GrabSeq != null)
                                            {
                                                iGrab.ClearGrabSequence(gd.GrabSeq);
                                                System.Threading.Interlocked.Decrement(ref LockedGrabSequences);
                                            }
                                            grabfailedretry = true;
                                            iStage.Stop(StageControl.Axis.X);
                                            iStage.Stop(StageControl.Axis.Y);
                                            iStage.Stop(StageControl.Axis.Z);
                                            i_view--;
                                            GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, tz + zgotol, false);
                                            while (ShouldStop == false && LockedGrabSequences > 0) System.Threading.Thread.Yield();
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            reliablepostime = false;
                                            System.Threading.Thread.Sleep((int)WorkConfig.MotionLatencyMS + (int)(d2timems / 2));
                                            continue;
                                        }
                                    }
                                    else
                                    {
                                        var __gd = iGrabWT.GrabSequenceAtTime(dtimems + timems);
                                        var __timeinfo = iGrab.GetImageTimesMS(__gd);
                                        iGrab.ClearGrabSequence(__gd);
                                    }
                                    while (GeneralTimeSource.ElapsedMilliseconds < zresettimems) ;
                                    iLog.Log("Scan Debug 2", "Elapsed " + GeneralTimeSource.ElapsedMilliseconds);
                                    m_DebugString = "I1";
                                    viewsok++;
                                    firstview = false;
                                    SetQueueLength();
                                    SetWriterQueueLength();
                                    m_DebugString = "I2";
#endregion
                                }
                                else
                                {
#region POSITION-DRIVEN-SCANNING
                                    m_DebugString = "D--1";
                                    if (grabfailedretry == false)
                                    {
                                        if ((i_view % WorkConfig.MaxViewsPerStrip) == 0)
                                        {
                                            gpubank = 1 - gpubank;
                                            for (igpu = 0; igpu < iGPU.Length; igpu++)
                                            {
                                                while (((SySal.Imaging.Fast.IImageProcessorFast)iGPU[igpu]).IsBankFree(gpubank) == false)
                                                    System.Threading.Thread.Sleep(100);
                                            }
                                            /*
                                            for (igpu = 0; igpu < iGPU.Length; igpu++)
                                                lock (iGPU[igpu])
                                                    ((SySal.Imaging.Fast.IImageProcessorFast)iGPU[igpu]).CurrentBank = gpubank;
                                             */
                                            iLog.Log("Scan", "GPU bank switched to " + gpubank);
                                        }
                                    }
                                    else grabfailedretry = false;
                                    m_DebugString = "E";
                                    //iLog.Log("Scan", "View " + i_view);
                                    QuasiStaticAcquisition.Sequence seq = qa.Sequences[i_view] = new QuasiStaticAcquisition.Sequence();
                                    seq.Owner = qa;
                                    seq.Id = (uint)i_view;
                                    seq.Layers = new QuasiStaticAcquisition.Sequence.Layer[WorkConfig.Layers];
                                    SetProgressValue(viewsok);
                                    for (i_image = 0; i_image < clustercounts.Length; i_image++) clustercounts[i_image] = -1;
                                    double tx = stripsidestartx + stripsideviewdeltax * i_view;
                                    double ty = stripsidestarty + stripsideviewdeltay * i_view;
                                    bool ok = false;
                                    double tz = ttz;
                                    //ZData[i_side].ReadZData(ref ok, ref tz);
                                    fi.Pos.X = tx;
                                    fi.Pos.Y = ty;
                                    m_DebugString = "F";
                                    if (StageSimulationOnly)
                                    {
                                        fi.Valid = true;
                                        fi.MeasureTime = System.DateTime.Now.AddYears(1);
                                        fi.TopZ = iStage.GetNamedReferencePosition("LowestZ") + ((i_side == 0) ? (WorkConfig.BaseThickness + 2.0 * WorkConfig.EmulsionThickness) : (WorkConfig.EmulsionThickness));
                                        fi.BottomZ = fi.TopZ - WorkConfig.EmulsionThickness;
                                    }
                                    m_DebugString = "G";
                                    ok = StageSimulationOnly || m_FocusMap.GetFocusInfo(ref fi, fi.Pos, i_side == 0, WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X, WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.Y);
                                    if (ok) tz = ttz = 0.5 * (fi.TopZ + fi.BottomZ);
                                    m_DebugString = "H";
                                    if (ok == false)
                                    {
                                        m_DebugString = "I";
                                        iStage.Stop(StageControl.Axis.X);
                                        iStage.Stop(StageControl.Axis.Y);
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "J";
                                        if (GoToPos(tx, ty, expectedzcenters[i_side], true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        double topz = 0.0, bottomz = 0.0;
                                        ok = FindEmulsionZs(expectedzcenters[i_side], ref topz, ref bottomz);
                                        iStage.Stop(StageControl.Axis.X);
                                        iStage.Stop(StageControl.Axis.Y);
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "K";
                                        if (ok == false)
                                        {
                                            m_DebugString = "L";
                                            /* make empty view */
                                            ProcOutputData po = new ProcOutputData();
                                            po.Clusters = null;// new SySal.Imaging.Cluster[0][];
                                            po.Exceptions = new Imaging.ImageProcessingException[0];
                                            po.ImagePositionInfo = new StageControl.TrajectorySample[0];
                                            po.StageInfo = new StageControl.TrajectorySample[0];
                                            ProcDataSlots[i_view] = po;
                                            ProcReadySignal[i_view] = true;
                                            lock (WriteQueue)
                                                WriteQueue.Enqueue(po);
                                            System.Threading.Interlocked.Increment(ref I_ViewsGrabbed);
                                            System.Threading.Interlocked.Increment(ref I_ViewsProcessed);
                                            continue;
                                        }
                                        m_DebugString = "M";
                                        m_Thickness[i_side] = topz - bottomz;
                                        fi.Pos.X = tx;
                                        fi.Pos.Y = ty;
                                        fi.TopZ = topz;
                                        fi.BottomZ = bottomz;
                                        fi.MeasureTime = System.DateTime.Now;
                                        fi.Valid = true;
                                        m_FocusMap.Write(fi, i_side == 0);
                                        tz = 0.5 * (topz + bottomz);
                                        //ZData[i_side].WriteZData(ok, tz);
                                        SetStatus("FindEmulsionZs: " + ok + " top " + topz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " bottom " + bottomz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " thickness " + (topz - bottomz).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                                        while (GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, expectedzcenters[i_side], false) == false)
                                        {
                                            iStage.Stop(StageControl.Axis.X);
                                            iStage.Stop(StageControl.Axis.Y);
                                            iStage.Stop(StageControl.Axis.Z);
                                            SetStatus("Cannot reach scan start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        }
                                        m_DebugString = "N";
                                        SetStatus("FindEmulsionZs: " + ok + " top " + topz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " bottom " + bottomz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + " thickness " + (topz - bottomz).ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                                        if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, ty, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        else
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, tx, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        iStage.Stop(StageControl.Axis.Z);
                                        m_DebugString = "O";
                                    }
                                    else if (firstview)
                                    {
                                        m_DebugString = "P";
                                        if (GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, /*expectedzcenters[i_side]*/tz, true) == false) throw new Exception("Cannot reach scan start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                        m_DebugString = "Q";
                                        if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, ty, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        else
                                        {
                                            iStage.PosMove(SySal.StageControl.Axis.X, tx, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                            iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                        }
                                        m_DebugString = "R";
                                    }
                                    m_DebugString = "S";
                                    bool axiserror = false;
                                    tz += WorkConfig.ZSweep * 0.5;
                                    long time3 = GeneralTimeSource.ElapsedMilliseconds;
                                    double oldz = iStage.GetPos(StageControl.Axis.Z);
                                    m_DebugString = "T";
                                    if (GoToPosZ(tz + WorkConfig.PositionTolerance, true) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                                    m_DebugString = "U";

                                    time2 = GeneralTimeSource.ElapsedMilliseconds;
                                    if (WorkConfig.AxisToMove == FullSpeedAcqSettings.MoveAxisForScan.X)
                                    {
                                        m_DebugString = "V";
                                        lastms = GeneralTimeSource.ElapsedMilliseconds;                                        
                                        //iLog.Log("Scan", "X: " + iStage.GetPos(StageControl.Axis.X) + " Target: " + tx + " motionlatency " + WorkConfig.MotionLatencyMS + " evenstripside " + evenstripside + " viewdelta " + stripsideviewdeltax);
                                        while (ShouldStop == false)
                                        {
                                            m_DebugString = "W";
                                            axiserror = (iStage.GetStatus(StageControl.Axis.X) != StageControl.AxisStatus.OK) || (iStage.GetStatus(StageControl.Axis.Y) != StageControl.AxisStatus.OK);
                                            currms = GeneralTimeSource.ElapsedMilliseconds;
                                            if (axiserror) break;
                                            double deltax = (iStage.GetPos(StageControl.Axis.X) - tx + WorkConfig.MotionLatencyMS * 0.001 * stepxspeed * (evenstripside ? 1.0 : -1.0)) * (evenstripside ? 1.0 : -1.0);
                                            //iLog.Log("Scan", "View " + i_view + " X exp " + tx.ToString("F1") + " found " + iStage.GetPos(StageControl.Axis.X).ToString("F1") + " " + deltax.ToString("F1"));
                                            //if (Math.Abs(deltax) < WorkConfig.PositionTolerance) break;
                                            //else if (deltax > 0.0)
                                            if (deltax >= 0.0 && deltax < WorkConfig.PositionTolerance) break;
                                            else if (deltax > WorkConfig.PositionTolerance)
                                            {
                                                m_DebugString = "X";
                                                iLog.Log("Scan", "Overtravel at view " + i_view + " " + deltax + " lastdeltams " + (currms - lastms) + " Z move " + (time2 - time1) + " check " + (time2 - time3) + " oldZ " + oldz);
                                                if (deltax > WorkConfig.ViewOverlap - WorkConfig.PositionTolerance - WorkConfig.MotionLatencyMS * 0.001 * stepxspeed)
                                                {
                                                    m_DebugString = "Y";
                                                    int trials;
                                                    iStage.Stop(StageControl.Axis.X);
                                                    iStage.Stop(StageControl.Axis.Y);
                                                    iStage.Stop(StageControl.Axis.Z);
                                                    iStage.SpeedMove(StageControl.Axis.X, 0, WorkConfig.XYAcceleration);
                                                    iStage.SpeedMove(StageControl.Axis.Y, 0, WorkConfig.XYAcceleration);
                                                    for (trials = 0; trials < 2 && GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, tz + WorkConfig.PositionTolerance, true) == false; trials++)
                                                    {
                                                        iLog.Log("Scan", "Resync");
                                                    }
                                                    if (trials < 2)
                                                    {
                                                        iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                                        iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                                        continue;
                                                    }
                                                    axiserror = true;
                                                }

                                                break;
                                            }
                                            lastms = currms;
                                            m_DebugString = "Z";
                                        }
                                    }
                                    else
                                    {
                                        lastms = 0;                                        
                                        while (ShouldStop == false)
                                        {
                                            axiserror = (iStage.GetStatus(StageControl.Axis.X) != StageControl.AxisStatus.OK) || (iStage.GetStatus(StageControl.Axis.Y) != StageControl.AxisStatus.OK);
                                            currms = GeneralTimeSource.ElapsedMilliseconds;
                                            if (axiserror) { iLog.Log("Scan", "Axis error triggered: " + iStage.GetStatus(StageControl.Axis.X).ToString() + " " + iStage.GetStatus(StageControl.Axis.Y)); break; }
                                            double deltay = (iStage.GetPos(StageControl.Axis.Y) - ty - WorkConfig.MotionLatencyMS * 0.001 * stepyspeed * (evenstripside ? 1.0 : -1.0)) * (evenstripside ? 1.0 : -1.0);
                                            if (Math.Abs(deltay) < WorkConfig.PositionTolerance) break;
                                            else if (deltay > 0.0)
                                            {
                                                iLog.Log("Scan", "Overtravel at view " + i_view + " " + deltay + " lastdeltams " + (currms - lastms) + " Z move " + (time2 - time1) + " check " + (time2 - time3) + " oldZ " + oldz);
                                                if (deltay > WorkConfig.ViewOverlap - WorkConfig.PositionTolerance - WorkConfig.MotionLatencyMS * 0.001 * stepxspeed)
                                                {
                                                    int trials;
                                                    iStage.Stop(StageControl.Axis.X);
                                                    iStage.Stop(StageControl.Axis.Y);
                                                    iStage.Stop(StageControl.Axis.Z);
                                                    iStage.SpeedMove(StageControl.Axis.X, 0, WorkConfig.XYAcceleration);
                                                    iStage.SpeedMove(StageControl.Axis.Y, 0, WorkConfig.XYAcceleration);
                                                    for (trials = 0; trials < 2 && GoToPos(tx - stripsideviewdeltax, ty - stripsideviewdeltay, tz + WorkConfig.PositionTolerance, true) == false; trials++)
                                                    {
                                                        iLog.Log("Scan", "Resync");
                                                    }
                                                    if (trials < 2)
                                                    {
                                                        iStage.PosMove(SySal.StageControl.Axis.X, stripsidefinishx + stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                                        iStage.PosMove(SySal.StageControl.Axis.Y, stripsidefinishy + stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                                        continue;
                                                    }
                                                    axiserror = true;
                                                }
                                                break;
                                            }
                                            lastms = currms;
                                        }
                                    }
                                    m_DebugString = "A1";
                                    if (axiserror) throw new Exception("Axis error on travelling axis.");
                                    I_Strip = i_strip;
                                    I_Side = i_side;
                                    I_View = i_view;
                                    I_Acq = qa;
                                    GrabTriggerZ = tz;
                                    m_DebugString = "B1";
                                    if (StageSimulationOnly == false)
                                    {
                                        //iLog.Log("Scan", "Arming grabbing");
                                        m_GrabDone = false;
                                        ArmGrabbing.Set();
                                    }
                                    m_DebugString = "C1";
                                    iStage.Stop(StageControl.Axis.Z);
                                    if (StageSimulationOnly == false) iStage.StartRecording(1.0, WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000.0);
                                    m_DebugString = "D1";
                                    iStage.SawToothPosMove(StageControl.Axis.Z,
                                        tz - WorkConfig.ZSweep - WorkConfig.PositionTolerance, WorkConfig.ZSweepSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration,
                                        (tz - WorkConfig.ZSweepSpeed + WorkConfig.PositionTolerance),
                                        tz + WorkConfig.PositionTolerance, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration
                                        );

                                    //iStage.PosMove(SySal.StageControl.Axis.Z, tz - WorkConfig.ZSweep - WorkConfig.PositionTolerance, WorkConfig.ZSweepSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                                    if (StageSimulationOnly == false)
                                    {

                                        m_DebugString = "D1";
                                        //iStage.StartRecording(1.0, WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000.0);
                                        GrabData gd = new GrabData();
                                        gd.strip = I_Strip;
                                        gd.side = I_Side;
                                        gd.view = I_View;

                                        m_DebugString = "E1 " + GrabTriggerZ;
                                        while (iStage.GetPos(StageControl.Axis.Z) > GrabTriggerZ) ;

                                        try
                                        {
                                            m_DebugString = "G1";
                                            gd.GrabSeq = iGrab.GrabSequence();
                                            m_GrabDone = true;
                                            gd.GPUBank = gpubank;
                                            gd.TimeInfo = iGrab.GetImageTimesMS(gd.GrabSeq);
                                            gd.StageInfo = iStage.Trajectory;
                                            /*
                                            lock (ViewStack)
                                            {
                                                ViewStack.Push(gd);
                                            } 
                                             */
                                            GrabDataSlots[I_View] = gd;
                                            GrabReadySignal[I_View] = 0;
                                            System.Threading.Interlocked.Increment(ref I_ViewsGrabbed);
                                            iLog.Log("VIEWTIME", i_view + " " + gd.TimeInfo[0]);
                                            m_DebugString = "H1";
                                        }
                                        catch (Exception xc)
                                        {
                                            iLog.Log("Scan", "Grab error: " + xc.ToString());
                                            grabfailedretry = true;
                                            iStage.Stop(StageControl.Axis.X);
                                            iStage.Stop(StageControl.Axis.Y);
                                            i_view--;
                                            continue;
                                        }

                                    }
#if false
                                while (ShouldStop == false && iStage.GetPos(StageControl.Axis.Z) > (tz - WorkConfig.ZSweepSpeed + WorkConfig.PositionTolerance) && (axiserror = (iStage.GetStatus(StageControl.Axis.Z) != StageControl.AxisStatus.OK)))
                                    /*System.Threading.Thread.Yield()*/;
                                if (axiserror)
                                {
                                    iStage.Stop(StageControl.Axis.X);
                                    iStage.Stop(StageControl.Axis.Y);
                                    iStage.Stop(StageControl.Axis.Z);
                                    throw new Exception("Axis error detected!");
                                };
                                time1 = GeneralTimeSource.ElapsedMilliseconds;
                                //while (ArmGrabbing == true) System.Threading.Thread.Yield();

                                while (GeneralTimeSource.ElapsedMilliseconds < time1 + WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000 - WorkConfig.MotionLatencyMS)/* System.Threading.Thread.Yield()*/;
                                //while (m_GrabDone == false && GeneralTimeSource.ElapsedMilliseconds < time1 + WorkConfig.Layers / WorkConfig.FramesPerSecond * 1000 + 1000)/* System.Threading.Thread.Yield()*/;
                                if (StageSimulationOnly == false && /*GrabDone.WaitOne(1000) == false*/ m_GrabDone == false)
                                {
                                    /* grabbing failed by some reason! */
                                    iStage.Stop(StageControl.Axis.X);
                                    iStage.Stop(StageControl.Axis.Y);
                                    iStage.Stop(StageControl.Axis.Z);
                                    iStage.PosMove(SySal.StageControl.Axis.X, tx - stripsideviewdeltax, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                    iStage.PosMove(SySal.StageControl.Axis.Y, ty - stripsideviewdeltay, stepyspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                    iStage.PosMove(SySal.StageControl.Axis.Z, tz + WorkConfig.PositionTolerance, stepxspeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                    i_view--;
                                    iLog.Log("Scan", "View " + i_view + " lost, retrying");
                                    continue;
                                }
                                //iStage.PosMove(SySal.StageControl.Axis.Z, tz + WorkConfig.PositionTolerance, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
#endif
                                    m_DebugString = "I1";
                                    viewsok++;
                                    firstview = false;
                                    SetQueueLength();
                                    SetWriterQueueLength();
                                    m_DebugString = "I2";
#endregion
                                }
                            m_DebugString = "J1 W " + I_ViewsWritten + " G " + I_ViewsGrabbed + " P " + I_ViewsProcessed;
                            while (ShouldStop == false && (LockedGrabSequences > 0 || I_ViewsWritten < I_ViewsGrabbed || I_ViewsProcessed < I_ViewsGrabbed)) ;
                            m_DebugString = "K1";

                            System.Runtime.GCSettings.LatencyMode = System.Runtime.GCLatencyMode.Interactive;
                            deadtimestart = GeneralTimeSource.ElapsedMilliseconds;
#if false
                            if (i_side + 1 <= maxside)
                            {
                                //                            iStage.PosMove(StageControl.Axis.X, ScanRectangle.MinX + StripDelta.X * i_strip - ViewDelta.X, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                //                            iStage.PosMove(StageControl.Axis.Y, ScanRectangle.MinY + StripDelta.Y * i_strip - ViewDelta.Y, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                double z = 0.0;
                                bool ok = false;
                                ZData[i_side + 1].ReadZData(ref ok, ref z);
                                //if (ok) iStage.PosMove(StageControl.Axis.Z, z + WorkConfig.PositionTolerance, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                            }
                            else if (i_strip + 1 < strips)
                            {
                                //                            iStage.PosMove(StageControl.Axis.X, ScanRectangle.MinX + StripDelta.X * (i_strip + 1) - ViewDelta.X, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                //                            iStage.PosMove(StageControl.Axis.Y, ScanRectangle.MinY + StripDelta.Y * (i_strip + 1) - ViewDelta.Y, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                                double z = 0.0;
                                bool ok = false;
                                ZData[minside].ReadZData(ref ok, ref z);
                            }
                            if (FocusInterrupt)
                                iLog.Log("Scan", "FocusInterrupt = " + FocusInterrupt);
                            if (/*ShouldStop == false &&*/StageSimulationOnly == false)
                            {
                                int i_v;
                                for (/*i_view = views - 1*/ i_v = i_view - 1; i_v >= 0/*i_view >= 0*/; i_v--/*i_view--*/)
                                {
                                    bool completed = ProcReadySignal[i_v/*i_view*/];
                                    while (completed == false)
                                    {
                                        lock (ProcReadySignal)
                                            completed = ProcReadySignal[i_v/*i_view*/];
                                        if (completed == false)
                                        {
                                            for (igpu = 0; igpu < iGPU.Length; igpu++)
                                                if (ImgProcThreads[igpu].IsAlive == false)
                                                    throw new Exception("Image processing thread " + igpu + " dead!");
                                            iLog.Log("Scan", "Awaiting view " + i_v/*i_view*/);
                                            System.Threading.Thread.Sleep(100);
                                        }
                                    }
                                    ProcOutputData po = ProcDataSlots[i_v/*i_view*/];
                                }
                                if (FocusInterrupt == false || focusretry == 0)
                                    lock (WriteQueue)
                                    {
                                        for (i_v = 0; i_v < i_view; i_v++)
                                            WriteQueue.Enqueue(ProcDataSlots[i_v]);
                                    }
                                else
                                {
                                    for (i_v = 0; i_v < i_view; i_v++)
                                    {
                                        iLog.Log("Scan", "FocusInterrupt fired, freeing cluster sequence " + i_v);
                                        if (ProcDataSlots[i_v] != null && ProcDataSlots[i_v].ClusterContainerDeallocator != null)
                                            lock (ProcDataSlots[i_v].ClusterContainerDeallocator)
                                                ProcDataSlots[i_v].ClusterContainerDeallocator.ReleaseClusterSequence(ProcDataSlots[i_v].Clusters);
                                    }
                                    iLog.Log("Scan", "FocusInterrupt fired, freed all sequences.");
                                }

                            }
#endif
                            if (FocusInterrupt == false) break;
                            else iLog.Log("Scan", "FocusInterrupt fired, refocusing");
                        }

                        m_DebugString = "L1";
                        if (i_side == 0)
                            Zone.Progress.TopStripsReady = (uint)i_strip + 1;
                        else
                            Zone.Progress.BottomStripsReady = (uint)i_strip + 1;
                        Zone.Update();
                        m_DebugString = "M1";
                    }
                }
                if (StageSimulationOnly)
                {
                    RecordedTrajectorySamples = iStage.Trajectory;
                }
                m_DebugString = "ZZ";
                SetProgressValue(viewsok);
                TerminateSignal.Set();
                for (igpu = 0; igpu < iGPU.Length; igpu++)
                    ImgProcThreads[igpu].Join();
                ImgProcThreads = new System.Threading.Thread[0];
                //grabthread.Join();
                grabthread = null;
                while (true)
                {
                    lock (WriteQueue)
                        if (WriteQueue.Count == 0)
                            break;
                    System.Threading.Thread.Sleep(1000);
                }
                WriteTerminateSignal.Set();
                writetthread.Join();
                writetthread = null;
                System.DateTime endtime = System.DateTime.Now;
                for (I_View = 0; I_View < Views; I_View++)
                {
                    GrabReadySignal[I_View] = 0;
                    ProcReadySignal[I_View] = false;
                    GrabDataSlots[I_View] = null;
                    ProcDataSlots[I_View] = null;
                }
                SetQueueLength();
                SetWriterQueueLength();                
                iLog.Log("Scan", "Total time: " + (endtime - starttime));
            }
            catch (Exception xc)
            {
                iLog.Log("Scan error", xc.ToString());
                SetStatus("Scan error - check log.");
                if (Zone != null)
                    Zone.Abort();
            }
            finally
            {
                RecoverStart = 0;
                try
                {
                    TerminateSignal.Set();
                    WriteTerminateSignal.Set();
                    for (igpu = 0; igpu < ImgProcThreads.Length; igpu++)
                        ImgProcThreads[igpu].Join();
                    ImgProcThreads = new System.Threading.Thread[0];
                    if (grabthread != null)
                    {
                        grabthread.Join();
                        grabthread = null;
                    }
                    if (writetthread != null)
                    {
                        writetthread.Join();
                        writetthread = null;
                    }
                    if (AutoScan == false)
                    {
                        SetStatus("Dumping trajectory");
                        string tswfile = DataDir;
                        if (tswfile.EndsWith("\\") == false && tswfile.EndsWith("/") == false) tswfile += "/";
                        tswfile += "debug_trajdump.txt";
                        System.IO.StreamWriter tsw = new System.IO.StreamWriter(tswfile);
                        tsw.WriteLine("ID\tT\tX\tY\tZ");
                        int i;
                        foreach (ProcOutputData po in ProcDataSlots)
                            if (po != null)
                                traj_arr.AddRange(po.StageInfo);
                        for (i = 0; i < traj_arr.Count; i++)
                        {
                            SySal.StageControl.TrajectorySample ts = (SySal.StageControl.TrajectorySample)traj_arr[i];
                            tsw.WriteLine(i + "\t" + ts.TimeMS + "\t" + ts.Position.X + "\t" + ts.Position.Y + "\t" + ts.Position.Z);
                        }
                        tsw.Flush();
                        tsw.Close();
                    }
                }
                catch (Exception xcx)
                {
                    MessageBox.Show(xcx.ToString(), "Scan");
                }                
                SetStatus("Done");
/*
                foreach (Imaging.IImageProcessor igpu1 in iGPU)
                    MessageBox.Show("Bank 0 Free = " + ((SySal.Imaging.Fast.IImageProcessorFast)igpu1).IsBankFree(0) + " Bank 1 Free = " + ((SySal.Imaging.Fast.IImageProcessorFast)igpu1).IsBankFree(1));
 */                
                iGrab.SequenceSize = 1;
                iCamDisp.EnableAutoRefresh = true;
                if (AutoScan)
                {
                    ScanResult = (viewsok == totalviews);
                    this.Invoke(new dSetVoid(delegate() { this.AutoCheckAndClose(); }));                    
                }
                else
                {
                    ShouldStop = true;
                    EnableControls(true);
                }
            }
        }

        bool ScanResult = false;

        void AutoCheckAndClose()
        {
            DialogResult = ScanResult ? DialogResult.OK : DialogResult.Cancel;
            Close();
        }

        delegate DialogResult dShowMessageBox(string msg, string caption);

        delegate void dSetText(string txt);

        void SetStatus(string status)
        {
            if (txtStatus.InvokeRequired) txtStatus.Invoke(new dSetText(SetStatus), status);
            else txtStatus.Text = status;
        }

        delegate void dSetInt(int max);

        void SetProgressMax(int max)
        {
            if (pbScanProgress.InvokeRequired) pbScanProgress.Invoke(new dSetInt(SetProgressMax), max);
            else pbScanProgress.Maximum = max;
        }

        void SetProgressValue(int v)
        {
            if (pbScanProgress.InvokeRequired) pbScanProgress.Invoke(new dSetInt(SetProgressValue), v);
            else pbScanProgress.Value = v;
        }

        delegate void dSetVoid();

        void SetQueueLength()
        {
            if (pbQueueLength.InvokeRequired) pbQueueLength.Invoke(new dSetVoid(SetQueueLength));
            else
            {
                /*
                int len = 0;
                foreach (int v in GrabReadySignal)
                    if (v != 0)
                        len++;
                pbQueueLength.Value = len;
                 */
                pbQueueLength.Value = LockedGrabSequences;
            }
        }

        void SetWriterQueueLength()
        {
            if (pbWriteQueueLength.InvokeRequired) pbQueueLength.Invoke(new dSetVoid(SetWriterQueueLength));
            else
            {
                try
                {
                    pbWriteQueueLength.Value = WriteQueue.Count;
                }
                catch (Exception) { }
            }
        }

        void SetQueueLengthMax(int maxval)
        {
            if (pbQueueLength.InvokeRequired) pbQueueLength.Invoke(new dSetInt(SetQueueLengthMax), maxval);
            else pbQueueLength.Maximum = maxval;
        }

        void SetWriterQueueLengthMax(int maxval)
        {
            if (pbWriteQueueLength.InvokeRequired) pbWriteQueueLength.Invoke(new dSetInt(SetWriterQueueLengthMax), maxval);
            else pbWriteQueueLength.Maximum = maxval;
        }

        bool GoToPos(double tx, double ty, double tz, bool dontwait)
        {
            SetStatus("Goto: " + tx.ToString("F1") + " " + ty.ToString("F1") + " " + tz.ToString("F1"));
            double sx = iStage.GetPos(StageControl.Axis.X);
            double sy = iStage.GetPos(StageControl.Axis.Y);
            double sz = iStage.GetPos(StageControl.Axis.Z);
            double timelimit = 0.0;
            timelimit = Math.Max(timelimit, Math.Abs(tx - sx) / WorkConfig.XYSpeed + 2.0 * Math.Sqrt(2.0 * Math.Abs(tx - sx) / WorkConfig.XYAcceleration) + WorkConfig.SlowdownTimeMS * 0.001);
            timelimit = Math.Max(timelimit, Math.Abs(ty - sy) / WorkConfig.XYSpeed + 2.0 * Math.Sqrt(2.0 * Math.Abs(ty - sy) / WorkConfig.XYAcceleration) + WorkConfig.SlowdownTimeMS * 0.001);
            timelimit = Math.Max(timelimit, Math.Abs(tz - sz) / WorkConfig.ZSpeed + 2.0 * Math.Sqrt(2.0 * Math.Abs(tz - sz) / WorkConfig.ZAcceleration) + WorkConfig.SlowdownTimeMS * 0.001);
            iStage.PosMove(StageControl.Axis.X, tx, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
            iStage.PosMove(StageControl.Axis.Y, ty, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
            iStage.PosMove(StageControl.Axis.Z, tz, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
            double endtime = GeneralTimeSource.ElapsedMilliseconds + timelimit * 1000.0;
            double timenext = GeneralTimeSource.ElapsedMilliseconds + 500.0;
            if (dontwait)
            {
                while (ShouldStop == false /*&& GeneralTimeSource.ElapsedMilliseconds < endtime*/)
                {
                    if (Math.Abs(iStage.GetPos(StageControl.Axis.X) - tx) <= WorkConfig.PositionTolerance &&
                        Math.Abs(iStage.GetPos(StageControl.Axis.Y) - ty) <= WorkConfig.PositionTolerance &&
                        Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) <= WorkConfig.PositionTolerance)
                        return true;
                    if (GeneralTimeSource.ElapsedMilliseconds >= timenext)
                    {
                        timenext += 500.0;
                        iStage.PosMove(StageControl.Axis.X, tx, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                        iStage.PosMove(StageControl.Axis.Y, ty, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
                        iStage.PosMove(StageControl.Axis.Z, tz, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                    }
                }
                iLog.Log("GoToPos", "DX: " + Math.Abs(iStage.GetPos(StageControl.Axis.X) - tx) + " DY: " + Math.Abs(iStage.GetPos(StageControl.Axis.Y) - ty) + " DZ: " + Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz));
                return false;
            }
            else
            {
                while (ShouldStop == false && GeneralTimeSource.ElapsedMilliseconds < endtime) ;
                bool retval = 
                    Math.Abs(iStage.GetPos(StageControl.Axis.X) - tx) <= WorkConfig.PositionTolerance &&
                    Math.Abs(iStage.GetPos(StageControl.Axis.Y) - ty) <= WorkConfig.PositionTolerance &&
                    Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) <= WorkConfig.PositionTolerance;
                if (retval == false)
                    iLog.Log("GoToPos", "DX: " + Math.Abs(iStage.GetPos(StageControl.Axis.X) - tx) + " DY: " + Math.Abs(iStage.GetPos(StageControl.Axis.Y) - ty) + " DZ: " + Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz));
                return retval;
            }
        }

        bool GoToPosZ(double tz, bool dontwait)
        {
            SetStatus("GotoZ: " + tz.ToString("F1"));
            double sz = iStage.GetPos(StageControl.Axis.Z);
            double timelimit = 0.0;
            timelimit = Math.Abs(tz - sz) / WorkConfig.ZSpeed + Math.Sqrt(2.0 * Math.Abs(tz - sz) / WorkConfig.ZAcceleration) + WorkConfig.SlowdownTimeMS * 0.001;
            iStage.PosMove(StageControl.Axis.Z, tz, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
            double endtime = GeneralTimeSource.ElapsedMilliseconds + timelimit * 1000.0;
            if (dontwait)
            {
                while (ShouldStop == false && GeneralTimeSource.ElapsedMilliseconds < endtime)
                    if (Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) <= WorkConfig.PositionTolerance)
                        return true;
                iLog.Log("GoToPosZ", "DZ: " + Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) + " ShouldStop: " + ShouldStop + " Time: " + GeneralTimeSource.ElapsedMilliseconds + " EndTime: " + endtime);
                SetStatus("GotoZ: " + tz.ToString("F1") + " FAILED");
                return false;
            }
            else
            {
                while (ShouldStop == false && GeneralTimeSource.ElapsedMilliseconds < endtime) ;
                bool retval = Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) <= WorkConfig.PositionTolerance;
                if (retval == false)
                    iLog.Log("GoToPosZ", "DZ: " + Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) + " ShouldStop: " + ShouldStop + " Time: " + GeneralTimeSource.ElapsedMilliseconds + " EndTime: " + endtime);
                SetStatus("GotoZ: " + tz.ToString("F1") + " " + retval);
                return retval;
            }
        }

        bool FindEmulsionZs(double expectedzcenter, ref double topz, ref double bottomz)
        {   
            int i;
            for (i = 0; i < GrabDataSlots.Length; i++)
                while (GrabDataSlots[i] != null)
                    System.Threading.Thread.Sleep(0);
            int seqsize = iGrab.SequenceSize;
            SySal.Imaging.LinearMemoryImage lmi = null;
            object gseq = null;
            try
            {
                int focusseqsize = (int)(WorkConfig.FocusSweep / WorkConfig.Pitch) + 1;
                if (focusseqsize >= iGPU[0].MaxImages)
                {
                    focusseqsize = iGPU[0].MaxImages - 1;
                    iLog.Log("FindEmulsionZs", "Max capacity of first GPU exceeded: reducing focusing sequence size to " + focusseqsize + " images (" + (WorkConfig.Pitch * (focusseqsize - 1)) + " micron).");
                }
                while (focusseqsize > 0)
                    try
                    {
                        iGrab.SequenceSize = focusseqsize;
                        break;
                    }
                    catch (Exception xc)
                    {
                        iLog.Log("FindEmulsionZs", "Can't set the focusing sequence size to " + focusseqsize + " images (" + (WorkConfig.Pitch * (focusseqsize - 1)) + " micron).\r\n" + xc.ToString());
                        focusseqsize--;
                    };
                double currx = iStage.GetPos(StageControl.Axis.X);
                double curry = iStage.GetPos(StageControl.Axis.Y);
                double focussweep = (focusseqsize - 1) * WorkConfig.Pitch;
                double currz;
                int firstlayer, lastlayer;
                bool axiserror = false;
                int trial = 0;
                lock (iGPU[0])
                {
                    SySal.StageControl.TrajectorySample[] psamples = null;                    
                    do
                    {
                        currz = expectedzcenter + focussweep * 0.5;
                        if (GoToPos(currx, curry, currz/* + WorkConfig.PositionTolerance*/, false) == false) return false;
                        iStage.Stop(StageControl.Axis.Z);
                        iStage.PosMove(StageControl.Axis.Z, currz - focussweep/* - WorkConfig.PositionTolerance*/, WorkConfig.ZSweepSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                        //double checkz1 = iStage.GetPos(StageControl.Axis.Z);
                        //double time1 = (double)this.GeneralTimeSource.ElapsedMilliseconds;
                        /*
                        while (ShouldStop == false && iStage.GetPos(StageControl.Axis.Z) > currz)
                            if (axiserror = (iStage.GetStatus(StageControl.Axis.Z) != StageControl.AxisStatus.OK))
                                break;                         
                        if (axiserror) throw new Exception("Axis error on Z while reaching the focus scan start position");
                         */
                        iStage.StartRecording(1.0, focusseqsize / WorkConfig.FramesPerSecond * 1000.0);
                        //double time3 = (double)this.GeneralTimeSource.ElapsedMilliseconds;
                        var wtime = this.GeneralTimeSource.ElapsedMilliseconds + WorkConfig.MotionLatencyMS;
                        while (this.GeneralTimeSource.ElapsedMilliseconds < wtime) ;
                        gseq = iGrab.GrabSequence();
                        //double checkz2 = iStage.GetPos(StageControl.Axis.Z);
                        //double time2 = (double)this.GeneralTimeSource.ElapsedMilliseconds;
                        double[] times = iGrab.GetImageTimesMS(gseq);
                        psamples = iStage.Trajectory;
                        lmi = iGrab.MapSequenceToSingleImage(gseq) as SySal.Imaging.LinearMemoryImage;
                        iGrab.ClearGrabSequence(gseq);
                        gseq = null;
                        try
                        {
                            iGPU[0].Input = lmi;
                            SySal.Imaging.Cluster[][] cls = iGPU[0].Clusters;
                            iGrab.ClearMappedImage(lmi);
                            lmi = null;
                            psamples = ComputePosFromTime(psamples, times);
                            int im;
                            firstlayer = psamples.Length - 1;
                            lastlayer = 0;
                            for (im = 0; im < cls.Length; im++)
                                if (cls[im].Length >= WorkConfig.ClusterThreshold)
                                {
                                    if (firstlayer > im) firstlayer = im;
                                    if (lastlayer < im) lastlayer = im;
                                };
                            {
                                System.IO.StringWriter strw = new System.IO.StringWriter();
                                strw.WriteLine("IM CLUSTERS TIME Z");
                                for (im = 0; im < cls.Length; im++)
                                    strw.WriteLine(im + " " + cls[im].Length + " " + times[im] + " " + psamples[im].Position.Z);
                                iLog.Log("FindEmulsionZs", strw.ToString());
                            }
                        }
                        catch (Exception xc)
                        {
                            iLog.Log("FindEmulsionZs", xc.ToString());
                            return false;
                        }
                        finally
                        {
                            if (lmi != null)
                            {
                                iGrab.ClearMappedImage(lmi);
                                lmi = null;
                            }
                            if (gseq != null)
                            {
                                iGrab.ClearGrabSequence(gseq);
                                gseq = null;
                            }
                        }
                        if (firstlayer < lastlayer)
                        {
                            if ((firstlayer == 0 && lastlayer == psamples.Length - 1) || (firstlayer > 0 && lastlayer < psamples.Length - 1))
                                expectedzcenter = 0.5 * (psamples[firstlayer].Position.Z + psamples[lastlayer].Position.Z);
                            else if (firstlayer == 0)                            
                                expectedzcenter = psamples[lastlayer].Position.Z + 0.5 * WorkConfig.EmulsionThickness;
                            else if (lastlayer == psamples.Length - 1)
                                expectedzcenter = psamples[firstlayer].Position.Z - 0.5 * WorkConfig.EmulsionThickness;
                            iLog.Log("FindEmulsionZs", "Suggesting emulsion center at " + expectedzcenter + " firstlayer " + firstlayer + " lastlayer " + lastlayer);
                        }
                    }
                    while (++trial < 10 && (firstlayer >= lastlayer || firstlayer == 0 || lastlayer == focusseqsize - 1));
                    if (lastlayer - firstlayer + 1 >= WorkConfig.MinValidLayers/* && firstlayer > 0 && lastlayer < focusseqsize - 1*/)
                    {
                       /*
                        topz = psamples[firstlayer].Position.Z;
                        bottomz = psamples[lastlayer].Position.Z;
                        */
                        topz = expectedzcenter + WorkConfig.EmulsionThickness * 0.5;
                        bottomz = expectedzcenter - WorkConfig.EmulsionThickness * 0.5;
                        return true;
                    }
                    iLog.Log("FindEmulsionZs", "Focus not found\r\nFirstLayer: " + firstlayer + " Z: " + topz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nLastLayer: " + lastlayer + " Z: " + bottomz.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    return false;
                }
            }
            catch (Exception xc)
            {
                iLog.Log("FindEmulsionZs", xc.ToString());
                return false;
            }
            finally
            {
                try
                {
                    iGrab.SequenceSize = seqsize;
                }
                catch (Exception xc1)
                {
                    iLog.Log("FindEmulsionZs", "Cannot restore sequence size to " + seqsize + "\r\n" + xc1.ToString());
                }
            }
        }

        bool FindEmulsionZsSlow(double expectedzcenter, ref double topz, ref double bottomz)
        {
            int seqsize = iGrab.SequenceSize;
            try
            {
                iGrab.SequenceSize = 1;
                double currx = iStage.GetPos(StageControl.Axis.X);
                double curry = iStage.GetPos(StageControl.Axis.Y);
                double currz = expectedzcenter + WorkConfig.FocusSweep * 0.5;
                double lastz = currz - WorkConfig.FocusSweep;
                double enterz = 0.0;
                double exitz = 0.0;
                bool entered = false;
                bool exited = false;
                lock (iGPU[0])
                    do
                    {
                        if (GoToPos(currx, curry, currz, false) == false) return false;
                        object gseq = iGrab.GrabSequence();
                        SySal.Imaging.LinearMemoryImage lmi = iGrab.MapSequenceToSingleImage(gseq) as SySal.Imaging.LinearMemoryImage;
                        iCamDisp.ImageShown = lmi;
                        try
                        {
                            iGPU[0].Input = lmi;
                            if (iGPU[0].Clusters[0].Length >= WorkConfig.ClusterThreshold)
                            {
                                if (entered == false)
                                {
                                    enterz = currz;
                                    entered = true;
                                }
                                else
                                {
                                    exitz = currz;
                                }
                            }
                            else
                            {
                                if (entered)
                                {
                                    exited = true;
                                    break;
                                }
                            }
                        }
                        catch (Exception xc)
                        {
                            iLog.Log("FindEmulsionZs", xc.ToString());
                            return false;
                        }
                        finally
                        {
                            iGrab.ClearMappedImage(lmi);
                            iGrab.ClearGrabSequence(gseq);
                        }
                    }
                    while ((currz -= WorkConfig.Pitch) >= lastz);
                bool ok = entered && exited && (enterz - exitz >= WorkConfig.Pitch * (WorkConfig.MinValidLayers - 1));
                if (ok)
                {
                    topz = enterz;
                    bottomz = exitz;
                }
                return ok;
            }
            catch (Exception xc)
            {
                iLog.Log("FindEmulsionZs", xc.ToString());
                return false;
            }
            finally
            {
                try
                {
                    iGrab.SequenceSize = seqsize;
                }
                catch (Exception xc1)
                {
                    iLog.Log("FindEmulsionZs", "Cannot restore sequence size to " + seqsize + "\r\n" + xc1.ToString());
                }
            }
        }

        private void SetFramesPerSecond()
        {
            if (WorkConfig.Layers < 2) return;
            object seq = null;
            try
            {
                if (m_TimingOutFile != null) System.IO.File.WriteAllText(m_TimingOutFile, "SEQ\tID\tDELTAMS");
                iCamDisp.EnableAutoRefresh = false;
                int i;
                iGrab.SequenceSize = (int)WorkConfig.Layers;
                double[] fps = new double[21];
                for (i = 0; i < fps.Length; i++)
                {
                    seq = iGrab.GrabSequence();                    
                    double[] times = iGrab.GetImageTimesMS(seq);                    
                    iGrab.ClearGrabSequence(seq);
                    if (m_TimingOutFile != null)
                        for (int j = 0; j < times.Length; j++)
                            System.IO.File.AppendAllText(m_TimingOutFile, "\r\n" + i + "\t" + j + "\t" + ((j == 0) ? 0.0 : (times[j] - times[j - 1])));
                    seq = null;
                    iLog.Log("SetFramesPerSecond", "Sequence " + i + " of " + fps.Length + " (" + WorkConfig.Layers + " frames)");
                    System.IO.StringWriter strw = new System.IO.StringWriter();
                    foreach (double t in times) strw.Write(" " + t);
                    iLog.Log("SetFramesPerSecond", strw.ToString());
                    fps[i] = WorkConfig.FramesPerSecond = (times.Length - 1) / (times[times.Length - 1] - times[0]) * 1000.0;
                    iLog.Log("SetFramesPerSecond", "FPS " + fps[i]);
                    this.Invoke(new dSetVoid(ShowWorkConfig));
                    System.Threading.Thread.Sleep(1000);
                }
                WorkConfig.FramesPerSecond = NumericalTools.Fitting.Quantiles(fps, new double[] { 0.5 })[0];
                this.Invoke(new dSetVoid(ShowWorkConfig));
            }
            catch (Exception xc)
            {
                MessageBox.Show("Measurement/timing error:\r\n" + xc.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            finally
            {
                if (seq != null) iGrab.ClearGrabSequence(seq);
                iGrab.SequenceSize = 1;
                iCamDisp.EnableAutoRefresh = true;
                this.Invoke(new dEnableCtls(EnableControls), new object[] { true });
            }
        }


        string m_TimingOutFile = null;

        private void btnMeasureFramesPerSecond_Click(object sender, EventArgs e)
        {
            if (WorkConfig.Layers < 2)
            {
                MessageBox.Show("At least 2 layers must be acquired to measure frame rate.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            SaveFileDialog s1dlg = new SaveFileDialog();
            s1dlg.Title = "Select file to dump timing data.";
            s1dlg.Filter = "Text files (*.txt)|*.txt";
            m_TimingOutFile = null;
            if (s1dlg.ShowDialog() == System.Windows.Forms.DialogResult.OK)
                m_TimingOutFile = s1dlg.FileName;
            EnableControls(false);
            new System.Threading.Thread(new System.Threading.ThreadStart(SetFramesPerSecond)).Start();
        }

        static SySal.StageControl.TrajectorySample ts0 = new StageControl.TrajectorySample();
        static SySal.StageControl.TrajectorySample ts1 = new StageControl.TrajectorySample();

        static void ComputePosFromTime(SySal.StageControl.TrajectorySample[] outsamples, StageControl.IStageWithDirectTrajectoryData iSTD, double[] times)
        {
            if (outsamples.Length != times.Length)
                throw new Exception("Programming error: outsamples length (" + outsamples.Length + ") is different from times length (" + times.Length + ") in calling ComputePosFromTime");
            int i = 0;
            uint j = 1;
            double lambda;
            ts0.TimeMS = 0.0;
            ts1.TimeMS = 0.0;
            bool intraj;
            if ((intraj = iSTD.GetTrajectoryData(0, ref ts0)) == false)
                throw new Exception("No trajectory samples available.");
            ts1 = ts0;
            for (i = 0; i < times.Length; i++)
            {
                outsamples[i].TimeMS = times[i];
                outsamples[i].Position = ts1.Position;
                if (times[i] >= ts0.TimeMS)
                    while (intraj)
                        if (ts1.TimeMS < times[i])
                        {
                            ts0 = ts1;
                            intraj = iSTD.GetTrajectoryData(j++, ref ts1);
                            outsamples[i].Position = ts1.Position;
                        }
                        else
                        {
                            lambda = ts1.TimeMS - ts0.TimeMS;
                            if (lambda > 0.0)
                            {
                                lambda = (times[i] - ts0.TimeMS) / lambda;
                                outsamples[i].Position.X = lambda * ts1.Position.X + (1.0 - lambda) * ts0.Position.X;
                                outsamples[i].Position.Y = lambda * ts1.Position.Y + (1.0 - lambda) * ts0.Position.Y;
                                outsamples[i].Position.Z = lambda * ts1.Position.Z + (1.0 - lambda) * ts0.Position.Z;
                            }
                            break;
                        }   
            }
        }

        static SySal.StageControl.TrajectorySample[] ComputePosFromTime(SySal.StageControl.TrajectorySample[] tj, double[] times)
        {
            SySal.StageControl.TrajectorySample[] poss = new SySal.StageControl.TrajectorySample[times.Length];
            int i = 0, j = 0;
            double lambda;
            for (i = j = 0; i < times.Length; i++)
            {                
                while (j < tj.Length && tj[j].TimeMS < times[i]) j++;
                if (j == 0) poss[i] = tj[0];
                else if (j >= tj.Length - 1) poss[i] = tj[tj.Length - 1];
                else
                {
                    lambda = (times[i] - tj[j - 1].TimeMS) / (tj[j].TimeMS - tj[j - 1].TimeMS);
                    poss[i].TimeMS = times[i];
                    poss[i].Position.X = lambda * tj[j].Position.X + (1.0 - lambda) * tj[j - 1].Position.X;
                    poss[i].Position.Y = lambda * tj[j].Position.Y + (1.0 - lambda) * tj[j - 1].Position.Y;
                    poss[i].Position.Z = lambda * tj[j].Position.Z + (1.0 - lambda) * tj[j - 1].Position.Z;
                }
            }
            return poss;
        }

        private void btnContinuousMotionFraction_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("About to test Z axis return motion to compute continuous motion\r\nProceed?", "Authorization", MessageBoxButtons.OKCancel, MessageBoxIcon.Warning) == DialogResult.OK)
            {   
                double ztop = iStage.GetNamedReferencePosition("LowestZ") + WorkConfig.BaseThickness + 2 * WorkConfig.EmulsionThickness + WorkConfig.ZSweep;
                double zbottom = ztop - WorkConfig.ZSweep;
                const int testnum = 10;
                SySal.StageControl.TrajectorySample [][] topdowntj = new SySal.StageControl.TrajectorySample[testnum][];
                SySal.StageControl.TrajectorySample [][] bottomuptj = new SySal.StageControl.TrajectorySample[testnum][];
                int i, j;
                ShouldStop = false;
                for (i = 0; i < topdowntj.Length; i++)
                {
                    if (GoToPosZ(ztop, false) == false)
                    {
                        MessageBox.Show("Axis error on Z.\r\nTest interrupted.", "Stage Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }                   
                    GeneralTimeSource.Reset();
                    GeneralTimeSource.Restart();
                    iGrab.TimeSource = GeneralTimeSource;
                    iStage.TimeSource = GeneralTimeSource;
                    iStage.StartRecording(1.0, 1000.0);
                    iStage.PosMove(StageControl.Axis.Z, zbottom - WorkConfig.PositionTolerance, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                    topdowntj[i] = iStage.Trajectory;
                }
                for (i = 0; i < bottomuptj.Length; i++)
                {
                    if (GoToPosZ(zbottom, false) == false)
                    {
                        MessageBox.Show("Axis error on Z.\rn\nTest interrupted.", "Stage Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;                        
                    }
                    GeneralTimeSource.Reset();
                    GeneralTimeSource.Restart();
                    iGrab.TimeSource = GeneralTimeSource;
                    iStage.TimeSource = GeneralTimeSource;
                    iStage.StartRecording(1.0, 1000.0);
                    iStage.PosMove(StageControl.Axis.Z, ztop + WorkConfig.PositionTolerance, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);
                    bottomuptj[i] = iStage.Trajectory;
                }
                ShouldStop = true;
                double [] topdowntimes = new double[topdowntj.Length];
                double [] bottomuptimes = new double[bottomuptj.Length];
                double[] topdownstarttimes = new double[topdowntj.Length];
                double[] bottomupstarttimes = new double[bottomuptj.Length];
                for (i = 0; i < topdowntj.Length; i++)
                {
                    for (j = topdowntj[i].Length - 1; j > 0 && Math.Abs(zbottom - WorkConfig.PositionTolerance - topdowntj[i][j].Position.Z) <= WorkConfig.PositionTolerance; j--);
                    topdowntimes[i] = topdowntj[i][j].TimeMS;
                    for (j = 0; j < topdowntj[i].Length - 1 && Math.Abs(ztop - topdowntj[i][j].Position.Z) <= WorkConfig.PositionTolerance; j++) ;
                    topdownstarttimes[i] = topdowntj[i][j].TimeMS;
                }                
                for (i = 0; i < bottomuptj.Length; i++)
                {
                    for (j = bottomuptj[i].Length - 1; j > 0 && Math.Abs(ztop + WorkConfig.PositionTolerance - bottomuptj[i][j].Position.Z) <= WorkConfig.PositionTolerance; j--)
                    bottomuptimes[i] = bottomuptj[i][j].TimeMS;
                    for (j = 0; j < bottomuptj[i].Length - 1 && Math.Abs(zbottom - bottomuptj[i][j].Position.Z) <= WorkConfig.PositionTolerance; j++) ;
                    bottomupstarttimes[i] = bottomuptj[i][j].TimeMS;
                }
                double dummy = 0.0;
                double topdownavg = 0.0, topdownrms = 0.0, topdownmax = 0.0;
                double bottomupavg = 0.0, bottomuprms = 0.0, bottomupmax = 0.0;
                double topdownstartavg = 0.0, topdownstartrms = 0.0, topdownstartmax = 0.0;
                double bottomupstartavg = 0.0, bottomupstartrms = 0.0, bottomupstartmax = 0.0;
                NumericalTools.Fitting.FindStatistics(topdowntimes, ref topdownmax, ref dummy, ref topdownavg, ref topdownrms);
                NumericalTools.Fitting.FindStatistics(bottomuptimes, ref bottomupmax, ref dummy, ref bottomupavg, ref bottomuprms);
                NumericalTools.Fitting.FindStatistics(topdownstarttimes, ref topdownstartmax, ref dummy, ref topdownstartavg, ref topdownrms);
                NumericalTools.Fitting.FindStatistics(bottomupstarttimes, ref bottomupstartmax, ref dummy, ref bottomupstartavg, ref bottomuprms);
                double topdowncycletime = (1000.0 * WorkConfig.Layers / WorkConfig.FramesPerSecond + topdownstartmax + topdownmax);
                double bottomupcycletime = (1000.0 * WorkConfig.Layers / WorkConfig.FramesPerSecond + bottomupstartmax + bottomupmax);
                MessageBox.Show(
                    "Top-Down results:\r\nMove Avg: " + topdownavg.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nMove RMS: " + topdownrms.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nMove Max: " + topdownmax.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nStart Avg: " + topdownstartavg.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nStart RMS: " + topdownstartrms.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nStart Max: " + topdownstartmax.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nCycle time: " + topdowncycletime.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nContinuous motion fraction: " + ((1000.0 * WorkConfig.Layers / WorkConfig.FramesPerSecond) / topdowncycletime).ToString("F3", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\n\r\nBottom-Up results:\r\nMove Avg: " + bottomupavg.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nMove RMS: " + bottomuprms.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nMove Max: " + bottomupmax.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nStart Avg: " + bottomupstartavg.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nStart RMS: " + bottomupstartrms.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nStart Max: " + bottomupstartmax.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\r\nCycle time: " + bottomupcycletime.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) + "\r\nContinuous motion fraction: " + ((1000.0 * WorkConfig.Layers / WorkConfig.FramesPerSecond) / bottomupcycletime).ToString("F3", System.Globalization.CultureInfo.InvariantCulture),
                    "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                if (sdlgZTest.ShowDialog() != DialogResult.OK) return;
                System.IO.StreamWriter w  = null;
                try
                {
                    w = new System.IO.StreamWriter(sdlgZTest.FileName);
                    w.WriteLine("DIRECTION\tRUN\tT\tZ");
                    for (i = 0; i < topdowntj.Length; i++)
                        for (j = 0; j < topdowntj[i].Length; j++)
                            w.WriteLine("0\t" + i + "\t" + topdowntj[i][j].TimeMS.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + "\t" + topdowntj[i][j].Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    for (i = 0; i < bottomuptj.Length; i++)
                        for (j = 0; j < bottomuptj[i].Length; j++)
                            w.WriteLine("1\t" + i + "\t" + bottomuptj[i][j].TimeMS.ToString("F3", System.Globalization.CultureInfo.InvariantCulture) + "\t" + bottomuptj[i][j].Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture));
                    w.Flush();
                    w.Close();
                    w = null;
                    MessageBox.Show("Done", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
                catch(Exception x)
                {

                }
                finally
                {
                    if (w != null) w.Close();               
                }
            }
        }

        static string TrajectoryDataToString(SySal.StageControl.TrajectorySample[] samples)
        {
            string str = "ID\tT\tX\tY\tZ";
            int i;
            for (i = 0; i < samples.Length; i++)
            {
                str += "\r\n" + i + "\t" + samples[i].TimeMS.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\t" + samples[i].Position.X.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\t" + samples[i].Position.Y.ToString("F2", System.Globalization.CultureInfo.InvariantCulture) +
                    "\t" + samples[i].Position.Z.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            }
            return str;
        }

        private void btnEmptyFocusMap_Click(object sender, EventArgs e)
        {
            if (m_FocusMap != null)
                m_FocusMap.Clear();
        }

        private void OnZDataExpirMinLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ZDataExpirationMin = uint.Parse(txtZDataExpirationMin.Text);     
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMaxViewsPerStripLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MaxViewsPerStrip = uint.Parse(txtMaxViewsStrip.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void btnDumpZMap_Click(object sender, EventArgs e)
        {
            if (m_FocusMap != null) ;
            if (sdlgZDump.ShowDialog() == DialogResult.OK)
                try
                {
                    System.IO.File.WriteAllText(sdlgZDump.FileName, m_FocusMap.ToString());
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't dump file!\r\n" + x.ToString(), "File Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }

        private void btnStepReset_Click(object sender, EventArgs e)
        {
            if (lvPostProcessing.SelectedItems.Count == 1)
                lvPostProcessing.SelectedItems[0].SubItems[1].Text = "";
        }


        private void btnStepCopy_Click(object sender, EventArgs e)
        {
            if (lvPostProcessing.SelectedItems.Count == 1 && lvPostProcessing.SelectedItems[0].SubItems[1].Text.Trim().Length > 0)
                try
                {
                    Clipboard.SetText(lvPostProcessing.SelectedItems[0].SubItems[1].Text.Trim(), TextDataFormat.Text);
                }
                catch (Exception)
                {
                    MessageBox.Show("Error copying text to the clipboard.", "Clipboard error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
        }

        private void btnStepPaste_Click(object sender, EventArgs e)
        {
            if (lvPostProcessing.SelectedItems.Count == 1)
                try
                {
                    string s = null;
                    if (Clipboard.ContainsText(TextDataFormat.Text)) s = Clipboard.GetText(TextDataFormat.Text);
                    else if (Clipboard.ContainsText(TextDataFormat.UnicodeText)) s = Clipboard.GetText(TextDataFormat.UnicodeText);
                    else if (Clipboard.ContainsText(TextDataFormat.Rtf)) s = Clipboard.GetText(TextDataFormat.Rtf);
                    else if (Clipboard.ContainsText(TextDataFormat.Html)) s = Clipboard.GetText(TextDataFormat.Html);                    
                    lvPostProcessing.SelectedItems[0].SubItems[1].Text = s;
                }
                catch (Exception) 
                {
                    MessageBox.Show("No text to paste.", "Clipboard error", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                }
        }

        private void btnStepSave_Click(object sender, EventArgs e)
        {
            try
            {
                int icpi;
                icpi = 0;
                foreach (ListViewItem lvi in lvPostProcessing.Items)
                    if (lvi.SubItems[1].Text.Trim().Length > 0)
                        icpi++;
                S.PostProcessSteps = new QuasiStaticAcquisition.PostProcessingInfo[icpi];
                icpi = 0;
                foreach (ListViewItem lvi in lvPostProcessing.Items)
                    if (lvi.SubItems[1].Text.Trim().Length > 0)
                    {
                        QuasiStaticAcquisition.PostProcessingInfo imp = new QuasiStaticAcquisition.PostProcessingInfo();
                        imp.Name = lvi.SubItems[0].Text.Trim();
                        imp.Settings = lvi.SubItems[1].Text.Trim();
                        S.PostProcessSteps[icpi++] = imp;
                    }

                SySal.Management.MachineSettings.SetSettings(typeof(FullSpeedAcquisitionFormSettings), S);
                MessageBox.Show("Settings saved.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }

        }

        private void btnRecover_Click(object sender, EventArgs e)
        {
            if (odlgRecoverZone.ShowDialog() == DialogResult.OK)
                try
                {
                    QuasiStaticAcquisition.Zone z = QuasiStaticAcquisition.Zone.RecoverFromFile(odlgRecoverZone.FileName);
                    txtMinX.Text = z.ScanRectangle.MinX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    OnMinXLeave(sender, e);
                    txtMinY.Text = z.ScanRectangle.MinY.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    OnMinYLeave(sender, e);
                    txtMaxX.Text = z.ScanRectangle.MaxX.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    OnMaxXLeave(sender, e);
                    txtMaxY.Text = z.ScanRectangle.MaxY.ToString(System.Globalization.CultureInfo.InvariantCulture);
                    OnMaxYLeave(sender, e);
                    txtSummaryFile.Text = z.FileNameTemplate;
                    iMap.InversePlateMap = z.PlateMap;
                    uint nexttop = 0;
                    uint nextbottom = 0;                    
                    uint recstart = 0;
                    if (z.ReadProgress(ref nexttop, ref nextbottom))
                    {
                        switch (WorkConfig.Sides)
                        {
                            case FullSpeedAcqSettings.ScanMode.Top: recstart = nexttop; break;
                            case FullSpeedAcqSettings.ScanMode.Bottom: recstart = nextbottom; break;
                            case FullSpeedAcqSettings.ScanMode.Both: recstart = Math.Min(nexttop, nextbottom); break;
                        }
                    }
                    if (recstart > 0 && MessageBox.Show("Recover from strip " + recstart + "?", "Confirmation needed", MessageBoxButtons.YesNo) == DialogResult.Yes)
                        RecoverStart = recstart;                        
                }
                catch (Exception x)
                {
                    MessageBox.Show("Can't recover zone:\r\n" + x.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                }
        }

        private void btnDebug_Click(object sender, EventArgs e)
        {
            txtStatus.Text = m_DebugString + " / " + m_WriteQueueDebugString + " lockseq " + LockedGrabSequences;
        }
    }
}
