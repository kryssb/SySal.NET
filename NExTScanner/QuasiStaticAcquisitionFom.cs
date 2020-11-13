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
    public partial class QuasiStaticAcquisitionForm : SySal.SySalNExTControls.SySalDialog
    {
        public string ConfigDir = "";

        public string DataDir = "";

        public ISySalLog iLog;

        public IMapProvider iMap;

        public ISySalCameraDisplay iCamDisp;

        public SySal.Imaging.IImageGrabber iGrab;

        public SySal.Imaging.IImageProcessor[] iGPU;

        public SySal.StageControl.IStage iStage;

        public SySal.Executables.NExTScanner.ImagingConfiguration ImagingConfig;

        internal SySal.Executables.NExTScanner.QuasiStaticAcqSettings CurrentConfig;

        private SySal.Executables.NExTScanner.QuasiStaticAcqSettings WorkConfig;

        static System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(QuasiStaticAcqSettings));        

        void LoadConfigurationList()
        {
            lbConfigurations.Items.Clear();
            string[] files = System.IO.Directory.GetFiles(ConfigDir, "*." + QuasiStaticAcqSettings.FileExtension, System.IO.SearchOption.TopDirectoryOnly);
            foreach (string f in files)
            {
                string s = f.Substring(f.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
                lbConfigurations.Items.Add(s.Substring(0, s.Length - QuasiStaticAcqSettings.FileExtension.Length - 1));
            }
        }

        public QuasiStaticAcquisitionForm()
        {
            InitializeComponent();

            m_ScanThread = null;
        }

        private void OnLoad(object sender, EventArgs e)
        {
            LoadConfigurationList();
            if (WorkConfig == null) WorkConfig = (QuasiStaticAcqSettings)CurrentConfig.Clone();
            ShowWorkConfig();
            string summaryfile = this.DataDir;
            if (summaryfile.EndsWith("/") == false && summaryfile.EndsWith("\\") == false) summaryfile += "/";
            summaryfile += QuasiStaticAcquisition.IndexedPatternTemplate;
            txtSummaryFile.Text = summaryfile;
            EnableControls(true);
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
            chkTop.Checked = (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Top) || (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both);
            chkBottom.Checked = (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Bottom) || (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both);
            txtBaseThickness.Text = WorkConfig.BaseThickness.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtEmuThickness.Text = WorkConfig.EmulsionThickness.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtFOVSize.Text = Math.Abs(ImagingConfig.ImageWidth * ImagingConfig.Pixel2Micron.X).ToString("F1", System.Globalization.CultureInfo.InvariantCulture) + "\xD7" + Math.Abs(ImagingConfig.ImageHeight * ImagingConfig.Pixel2Micron.Y).ToString("F1", System.Globalization.CultureInfo.InvariantCulture);
            txtViewOverlap.Text = WorkConfig.ViewOverlap.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            rdMoveX.Checked = (WorkConfig.AxisToMove == QuasiStaticAcqSettings.MoveAxisForScan.X);
            rdMoveY.Checked = (WorkConfig.AxisToMove == QuasiStaticAcqSettings.MoveAxisForScan.Y);
            txtXYSpeed.Text = WorkConfig.XYSpeed.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtZSpeed.Text = WorkConfig.ZSpeed.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtXYAcceleration.Text = WorkConfig.XYAcceleration.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtZAcceleration.Text = WorkConfig.ZAcceleration.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
            txtSlowdownTime.Text = WorkConfig.SlowdownTimeMS.ToString();
            txtPositionTolerance.Text = WorkConfig.PositionTolerance.ToString("F2", System.Globalization.CultureInfo.InvariantCulture);
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
                f += txtNewName.Text + "." + QuasiStaticAcqSettings.FileExtension;
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
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + QuasiStaticAcqSettings.FileExtension)[0]);
                WorkConfig = (QuasiStaticAcqSettings)xmls.Deserialize(r);
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
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + QuasiStaticAcqSettings.FileExtension)[0]);
                WorkConfig = (QuasiStaticAcqSettings)xmls.Deserialize(r);
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
                    System.IO.File.Delete(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + QuasiStaticAcqSettings.FileExtension)[0]);
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
                if (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Bottom || WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both) WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Both;
                else WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Top;
            }
            else WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Bottom;
        }

        private void OnBottomChecked(object sender, EventArgs e)
        {
            if (chkBottom.Checked)
            {
                if (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Top || WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both) WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Both;
                else WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Bottom;
            }
            else WorkConfig.Sides = QuasiStaticAcqSettings.ScanMode.Top;
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
            if (rdMoveX.Checked) WorkConfig.AxisToMove = QuasiStaticAcqSettings.MoveAxisForScan.X;
        }

        private void OnMoveYChecked(object sender, EventArgs e)
        {
            if (rdMoveY.Checked) WorkConfig.AxisToMove = QuasiStaticAcqSettings.MoveAxisForScan.Y;
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

        string m_QAPattern;

        private void btnStart_Click(object sender, EventArgs e)
        {
            if (ShouldStop == false) return;
            if (ScanRectangle.MinX >= ScanRectangle.MaxX || ScanRectangle.MinY >= ScanRectangle.MaxY)
            {
                MessageBox.Show("Null area defined.", "Nothing to do", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            //CameraDisplay.Enabled = false;
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
            ShouldStop = false;
            EnableControls(false);
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
            txtBaseThickness.Enabled = enable;
            txtContinuousMotionFraction.Enabled = enable;
            txtClusterThreshold.Enabled = enable;
            txtEmuThickness.Enabled = enable;
            txtFocusSweep.Enabled = enable;
            txtLayers.Enabled = enable;
            txtMaxX.Enabled = enable;
            txtMaxY.Enabled = enable;
            txtMinValidLayers.Enabled = enable;
            txtMinX.Enabled = enable;
            txtMinY.Enabled = enable;
            txtPitch.Enabled = enable;
            txtPositionTolerance.Enabled = enable;
            txtSlowdownTime.Enabled = enable;
            txtSummaryFile.Enabled = enable;
            txtViewOverlap.Enabled = enable;
            txtXYAcceleration.Enabled = enable;
            txtXYSpeed.Enabled = enable;
            txtZAcceleration.Enabled = enable;
            txtZSpeed.Enabled = enable;
            txtZSweep.Enabled = enable;
            chkBottom.Enabled = enable;            
            chkTop.Enabled = enable;
            rdMoveX.Enabled = enable;
            rdMoveY.Enabled = enable;
            btnDel.Enabled = enable;
            btnDuplicate.Enabled = enable;
            btnFromHere.Enabled = enable;
            btnLoad.Enabled = enable;
            btnMakeCurrent.Enabled = enable;
            btnNew.Enabled = enable;
            btnStart.Enabled = enable;
            btnStop.Enabled = !enable;
            btnToHere.Enabled = enable;
            btnExit.Enabled = enable;
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void Scan()
        {
            QuasiStaticAcquisition.Zone Zone = null;
            try
            {                
                iCamDisp.EnableAutoRefresh = false;
                iGrab.SequenceSize = 1;
                double fovwidth = Math.Abs(ImagingConfig.ImageWidth * ImagingConfig.Pixel2Micron.X);
                double fovheight = Math.Abs(ImagingConfig.ImageHeight * ImagingConfig.Pixel2Micron.Y);
                double stepwidth = fovwidth - WorkConfig.ViewOverlap;
                double stepheight = fovheight - WorkConfig.ViewOverlap;
                double lowestz = iStage.GetNamedReferencePosition("LowestZ");
                double[] expectedzcenters = new double[]
                {
                    lowestz + WorkConfig.EmulsionThickness * 1.5 + WorkConfig.BaseThickness, lowestz + WorkConfig.EmulsionThickness * 0.5
                };
                bool[] validzcenters = new bool[]
                {
                    false, false
                };
                double[] currentzcenters = new double[]
                {
                    expectedzcenters[0], expectedzcenters[1]
                };
                if (stepheight <= 0.0 || stepwidth <= 0.0) throw new Exception("Too much overlap, or null image or pixel/micron factor defined.");
                SySal.BasicTypes.Vector2 StripDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector2 ViewDelta = new BasicTypes.Vector2();
                SySal.BasicTypes.Vector ImageDelta = new BasicTypes.Vector();
                ImageDelta.Z = -WorkConfig.Pitch;
                int views, strips;
                if (WorkConfig.AxisToMove == QuasiStaticAcqSettings.MoveAxisForScan.X)
                {
                    ImageDelta.Y = 0.0;
                    ImageDelta.X = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepwidth / (WorkConfig.Layers - 1));
                    ViewDelta.X = stepwidth;
                    ViewDelta.Y = 0.0;
                    views = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxX - TransformedScanRectangle.MinX) / stepwidth));
                    StripDelta.X = 0.0;
                    StripDelta.Y = stepheight;
                    strips = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxY - TransformedScanRectangle.MinY) / stepheight));
                }
                else
                {
                    ImageDelta.X = 0.0;
                    ImageDelta.Y = (WorkConfig.Layers == 0) ? 0.0 : (WorkConfig.ContinuousMotionDutyFraction * stepheight / (WorkConfig.Layers - 1));
                    ViewDelta.X = 0.0;
                    ViewDelta.Y = stepheight;
                    views = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxY - TransformedScanRectangle.MinY) / stepheight));
                    StripDelta.X = stepwidth;
                    StripDelta.Y = 0.0;
                    strips = Math.Max(1, (int)Math.Ceiling((TransformedScanRectangle.MaxX - TransformedScanRectangle.MinX) / stepwidth));
                }
                Zone = new QuasiStaticAcquisition.Zone(m_QAPattern);
                Zone.Views = (uint)views;
                Zone.Strips = (uint)strips;
                Zone.HasTop = (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Top || WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both);
                Zone.HasBottom = (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Bottom || WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both);
                Zone.ScanRectangle = ScanRectangle;
                Zone.TransformedScanRectangle = TransformedScanRectangle;
                Zone.PlateMap = iMap.InversePlateMap;
                Zone.ImageDelta = ImageDelta;
                Zone.ViewDelta = ViewDelta;
                Zone.StripDelta = StripDelta;
                {
                    System.IO.StringWriter swr = new System.IO.StringWriter();
                    QuasiStaticAcqSettings.s_XmlSerializer.Serialize(swr, WorkConfig);                    
                    Zone.ScanSettings = swr.ToString();
                }
                Zone.PostProcessingSettings = new QuasiStaticAcquisition.PostProcessingInfo[0];

                SetProgressValue(0);
                SetProgressMax(views * strips);
                int i_strip, i_view, i_side, i_image;
                int firstlayer, lastlayer;
                int[] clustercounts = new int[WorkConfig.Layers];
                for (i_strip = 0; i_strip < strips && ShouldStop == false; i_strip++)
                {
                    for (i_side = (((WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both) || (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Top)) ? 0 : 1);
                        i_side <= (((WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both) || (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Bottom)) ? 1 : 0) && ShouldStop == false;
                        i_side++)
                    {
                        iLog.Log("Scan", "Side " + i_side + " WorkConfig.Sides " + WorkConfig.Sides + " " + (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Both) + " " + (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Top) + " " + (WorkConfig.Sides == QuasiStaticAcqSettings.ScanMode.Bottom));
                        QuasiStaticAcquisition qa = new QuasiStaticAcquisition();
                        qa.FilePattern = QuasiStaticAcquisition.GetFilePattern(m_QAPattern, i_side == 0, (uint)i_strip);
                        qa.Sequences = new QuasiStaticAcquisition.Sequence[views];
                        for (i_view = 0; i_view < views && ShouldStop == false; i_view++)
                        {
                            QuasiStaticAcquisition.Sequence seq = qa.Sequences[i_view] = new QuasiStaticAcquisition.Sequence();
                            seq.Owner = qa;
                            seq.Id = (uint)i_view;
                            seq.Layers = new QuasiStaticAcquisition.Sequence.Layer[WorkConfig.Layers];
                            SetProgressValue(i_strip * views + i_view);
                            for (i_image = 0; i_image < clustercounts.Length; i_image++) clustercounts[i_image] = -1;
                            firstlayer = (int)WorkConfig.Layers;
                            lastlayer = -1;
                            double tx = TransformedScanRectangle.MinX + ViewDelta.X * i_view + StripDelta.X * i_strip;
                            double ty = TransformedScanRectangle.MinY + ViewDelta.Y * i_view + StripDelta.Y * i_strip;
                            double tz = currentzcenters[i_side] + WorkConfig.ZSweep * 0.5;
                            if (GoToPos(tx, ty, tz) == false) throw new Exception("Cannot reach sweep start position at strip " + i_strip + ", view " + i_view + " side " + i_side + ".");
                            if (validzcenters[i_side] == false)
                            {
                                double topz = 0.0, bottomz = 0.0;
                                bool ok = FindEmulsionZs(expectedzcenters[i_side], ref topz, ref bottomz);
                                SetStatus("FindEmulsionZs: " + ok + " top " + topz + " bottom " + bottomz + " thickness " + (topz - bottomz));
                                if (ok)
                                {
                                    currentzcenters[i_side] = 0.5 * (topz + bottomz);
                                    validzcenters[i_side] = true;
                                }
                                else
                                {
                                    validzcenters[i_side] = false;
                                    SetStatus("Focus lost on side " + (i_side == 0 ? "top" : "bottom"));
                                    continue;
                                }
                            }
                            for (i_image = 0; i_image < WorkConfig.Layers && ShouldStop == false; i_image++)
                            {
                                QuasiStaticAcquisition.Sequence.Layer lay = new QuasiStaticAcquisition.Sequence.Layer();
                                lay.Id = (uint)i_image;
                                lay.Owner = seq;
                                tx = TransformedScanRectangle.MinX + ViewDelta.X * i_view + StripDelta.X * i_strip + i_image * ImageDelta.X;
                                ty = TransformedScanRectangle.MinY + ViewDelta.Y * i_view + StripDelta.Y * i_strip + i_image * ImageDelta.Y;
                                tz = currentzcenters[i_side] + WorkConfig.ZSweep * 0.5 - i_image * WorkConfig.Pitch;
                                if (GoToPos(tx, ty, tz) == false) throw new Exception("Cannot reach image position at strip " + i_strip + ", view " + i_view + " side " + i_side + " image " + i_image + ".");
                                lay.Position.X = iStage.GetPos(StageControl.Axis.X);
                                lay.Position.Y = iStage.GetPos(StageControl.Axis.Y);
                                lay.Position.Z = iStage.GetPos(StageControl.Axis.Z);
                                object gseq = iGrab.GrabSequence();
                                SySal.Imaging.LinearMemoryImage lmi = iGrab.MapSequenceToSingleImage(gseq) as SySal.Imaging.LinearMemoryImage;
                                iCamDisp.ImageShown = lmi;
                                iGPU[0].Input = lmi;
                                lay.Clusters = (uint)(clustercounts[i_image] = iGPU[0].Clusters[0].Length);
                                if (clustercounts[i_image] >= WorkConfig.ClusterThreshold)
                                {
                                    if (firstlayer > i_image) firstlayer = i_image;
                                    if (lastlayer < i_image) lastlayer = i_image;
                                }
                                iGPU[0].ImageToFile(lmi, lay.ImageFileName);
                                lay.WriteSummary();
                                iGrab.ClearMappedImage(lmi);
                                iGrab.ClearGrabSequence(gseq);
                                SetStatus("Strip " + i_strip + "/" + strips + " View " + i_view + "/" + views + " Side " + ((i_side == 0) ? "T" : "B") + " Image " + i_image + "/" + WorkConfig.Layers + " Clusters " + clustercounts[i_image]);
                            }
                            if ((lastlayer - firstlayer + 1) >= WorkConfig.MinValidLayers)
                                currentzcenters[i_side] += 0.5 * (WorkConfig.Layers - lastlayer - firstlayer) * WorkConfig.Pitch;
                            else
                                validzcenters[i_side] = false;
                        }

                        if (i_side == 0)
                            Zone.Progress.TopStripsReady = (uint)i_strip;
                        else
                            Zone.Progress.BottomStripsReady = (uint)i_strip;
                        Zone.Update();
                    }
                }
                if (ShouldStop == false) SetProgressValue(strips * views);
            }
            catch (Exception xc)
            {
                iLog.Log("Scan error", xc.ToString());
                if (Zone != null)
                    Zone.Abort();
            }
            finally
            {
                ShouldStop = true;
                EnableControls(true);
                iGrab.SequenceSize = 1;
                iCamDisp.EnableAutoRefresh = true;
            }
        }

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

        bool GoToPos(double tx, double ty, double tz)
        {
            double sx = iStage.GetPos(StageControl.Axis.X);
            double sy = iStage.GetPos(StageControl.Axis.Y);
            double sz = iStage.GetPos(StageControl.Axis.Z);
            double timelimit = 0.0;
            timelimit = Math.Max(timelimit, Math.Abs(tx - sx) * (1.0 / WorkConfig.XYSpeed + 4.0 / WorkConfig.XYAcceleration) + WorkConfig.SlowdownTimeMS * 0.002);
            timelimit = Math.Max(timelimit, Math.Abs(ty - sy) * (1.0 / WorkConfig.XYSpeed + 4.0 / WorkConfig.XYAcceleration) + WorkConfig.SlowdownTimeMS * 0.002);
            timelimit = Math.Max(timelimit, Math.Abs(tz - sz) * (1.0 / WorkConfig.ZSpeed + 4.0 / WorkConfig.ZAcceleration) + WorkConfig.SlowdownTimeMS * 0.002);
            iStage.PosMove(StageControl.Axis.X, tx, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
            iStage.PosMove(StageControl.Axis.Y, ty, WorkConfig.XYSpeed, WorkConfig.XYAcceleration, WorkConfig.XYAcceleration);
            iStage.PosMove(StageControl.Axis.Z, tz, WorkConfig.ZSpeed, WorkConfig.ZAcceleration, WorkConfig.ZAcceleration);            
            System.DateTime endtime = System.DateTime.Now;
            endtime = endtime.AddSeconds(timelimit);
            while (ShouldStop == false && System.DateTime.Now < endtime);
            return 
                Math.Abs(iStage.GetPos(StageControl.Axis.X) - tx) <= WorkConfig.PositionTolerance &&
                Math.Abs(iStage.GetPos(StageControl.Axis.Y) - ty) <= WorkConfig.PositionTolerance &&
                Math.Abs(iStage.GetPos(StageControl.Axis.Z) - tz) <= WorkConfig.PositionTolerance;
        }

        bool FindEmulsionZs(double expectedzcenter, ref double topz, ref double bottomz)
        {
            try
            {
                double currx = iStage.GetPos(StageControl.Axis.X);
                double curry = iStage.GetPos(StageControl.Axis.Y);
                double currz = expectedzcenter + WorkConfig.FocusSweep * 0.5;
                double lastz = currz - WorkConfig.FocusSweep;
                double enterz = 0.0;
                double exitz = 0.0;
                bool entered = false;
                bool exited = false;
                do
                {
                    if (GoToPos(currx, curry, currz) == false) return false;
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
        }
    }
}
