using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class ImagingConfigurationForm : SySal.SySalNExTControls.SySalDialog
    {
        public ImagingConfigurationForm()
        {
            InitializeComponent();
        }

        public string ConfigDir = "";

        public string DataDir = "";

        public ISySalCameraDisplay iCamDisp;

        public ISySalLog iLog;

        public SySal.Imaging.IImageProcessor[] iGPU;

        public SySal.Executables.NExTScanner.IScannerDataDisplay m_Display;

        public SySal.Executables.NExTScanner.ImagingConfiguration CurrentConfig;

        public SySal.Executables.NExTScanner.ImagingConfiguration WorkConfig;

        static System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(ImagingConfiguration));        

        private void OnLoad(object sender, EventArgs e)
        {
            LoadConfigurationList();
            if (WorkConfig == null) WorkConfig = (ImagingConfiguration)CurrentConfig.Clone();
            ShowWorkConfig();
            string summaryfile = this.DataDir;
            if (summaryfile.EndsWith("/") == false && summaryfile.EndsWith("\\") == false) summaryfile += "/";
            summaryfile += QuasiStaticAcquisition.PatternTemplate;
            txtSummaryFile.Text = summaryfile;
        }

        private void btnExit_Click(object sender, EventArgs e)
        {
            Close();
        }

        void LoadConfigurationList()
        {
            lbConfigurations.Items.Clear();
            string[] files = System.IO.Directory.GetFiles(ConfigDir, "*." + ImagingConfiguration.FileExtension, System.IO.SearchOption.TopDirectoryOnly);
            foreach (string f in files)
            {
                string s = f.Substring(f.LastIndexOfAny(new char[] { '\\', '/' }) + 1);
                lbConfigurations.Items.Add(s.Substring(0, s.Length - ImagingConfiguration.FileExtension.Length - 1));
            }
        }

        void ShowWorkConfig()
        {
            txtWidth.Text = WorkConfig.ImageWidth.ToString();
            txtHeight.Text = WorkConfig.ImageHeight.ToString();
            txtGreyLevelTargetMedian.Text = WorkConfig.GreyTargetMedian.ToString();
            txtMaxSegsLine.Text = WorkConfig.MaxSegmentsPerLine.ToString();
            txtMaxClusters.Text = WorkConfig.MaxClusters.ToString();
            string imgstr = WorkConfig.EmptyImage;
            int pos = imgstr.IndexOfAny(new char [] { '\r', '\n' });
            txtEmptyImage.Text = (pos < 0) ? imgstr : imgstr.Substring(0, pos);
            imgstr = WorkConfig.ThresholdImage;
            pos = imgstr.IndexOfAny(new char[] { '\r', '\n' });
            txtThresholdImage.Text = (pos < 0) ? imgstr : imgstr.Substring(0, pos);
            txtPixMicronX.Text = WorkConfig.Pixel2Micron.X.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
            txtPixMicronY.Text = WorkConfig.Pixel2Micron.Y.ToString("F6", System.Globalization.CultureInfo.InvariantCulture);
            txtCMPosTol.Text = WorkConfig.ClusterMatchPositionTolerance.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);
            txtCMMaxOffset.Text = WorkConfig.ClusterMatchMaxOffset.ToString("F3", System.Globalization.CultureInfo.InvariantCulture);
            txtMinClusterArea.Text = WorkConfig.MinClusterArea.ToString();
            txtMinGrainVolume.Text = WorkConfig.MinGrainVolume.ToString();
            txtCMMinMatches.Text = WorkConfig.MinClusterMatchCount.ToString();
            txtXSlant.Text = WorkConfig.XSlant.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtYSlant.Text = WorkConfig.YSlant.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDmagDX.Text = WorkConfig.DMagDX.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDmagDY.Text = WorkConfig.DMagDY.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtDmagDZ.Text = WorkConfig.DMagDZ.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtZCurvature.Text = WorkConfig.ZCurvature.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtXYCurvature.Text = WorkConfig.XYCurvature.ToString(System.Globalization.CultureInfo.InvariantCulture);
            txtCameraRotation.Text = WorkConfig.CameraRotation.ToString(System.Globalization.CultureInfo.InvariantCulture);
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
                f += txtNewName.Text + "." + ImagingConfiguration.FileExtension;
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

        private void OnWidthLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ImageWidth = uint.Parse(txtWidth.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnHeightLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ImageHeight = uint.Parse(txtHeight.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnGreyLevelTargetMedianLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.GreyTargetMedian = uint.Parse(txtGreyLevelTargetMedian.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMaxSegsLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MaxSegmentsPerLine = uint.Parse(txtMaxSegsLine.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMaxClustersLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MaxClusters = uint.Parse(txtMaxClusters.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnPixMicronXLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.Pixel2Micron.X = double.Parse(txtPixMicronX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnPixMicronYLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.Pixel2Micron.Y = double.Parse(txtPixMicronY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnDmagDZLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.DMagDZ = double.Parse(txtDmagDZ.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnDmagDYLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.DMagDY = double.Parse(txtDmagDY.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnDmagDXLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.DMagDX = double.Parse(txtDmagDX.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnXSlantLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.XSlant = double.Parse(txtXSlant.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnYSlantLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.YSlant = double.Parse(txtYSlant.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnCameraRotationLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.CameraRotation = double.Parse(txtCameraRotation.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnZCurvLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ZCurvature = double.Parse(txtZCurvature.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnXYCurvLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.XYCurvature = double.Parse(txtXYCurvature.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void btnLoad_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + ImagingConfiguration.FileExtension)[0]);
                WorkConfig = (ImagingConfiguration)xmls.Deserialize(r);
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

        private void btnDel_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            if (MessageBox.Show("Are you sure you want to delete the selected configuration?", "Confirmation", MessageBoxButtons.YesNo, MessageBoxIcon.Question) == DialogResult.Yes)
                try
                {
                    System.IO.File.Delete(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + ImagingConfiguration.FileExtension)[0]);
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

        private void btnMakeCurrent_Click(object sender, EventArgs e)
        {
            if (lbConfigurations.SelectedItems.Count != 1) return;
            System.IO.StreamReader r = null;
            try
            {
                r = new System.IO.StreamReader(System.IO.Directory.GetFiles(ConfigDir, lbConfigurations.SelectedItem.ToString() + "." + ImagingConfiguration.FileExtension)[0]);
                WorkConfig = (ImagingConfiguration)xmls.Deserialize(r);
                r.Close();
                r = null;
                if (WorkConfig.EmptyImage.Length == 0 || WorkConfig.ThresholdImage.Length == 0)
                {
                    MessageBox.Show("The empty image and the threshold image must be non-null.\r\nCan't apply this configuration.\r\nYou can instead load it, correct it and then apply it.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Exclamation);
                    return;
                }
                CurrentConfig.Copy(WorkConfig);
                m_Display.DisplayMonitor("Empty image", SySal.Executables.NExTScanner.SySalImageFromImage.ToImage(SySal.Imaging.Base64ImageEncoding.ImageFromBase64(CurrentConfig.EmptyImage)));
                m_Display.DisplayMonitor("Threshold image", SySal.Executables.NExTScanner.SySalImageFromImage.ThresholdImage(Scanner.DCTInterpolationImageFromString(CurrentConfig.ThresholdImage)));
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

        private void btnEmptyImage_Click(object sender, EventArgs e)
        {
            if (OpenEmptyImagesDlg.ShowDialog() == DialogResult.OK)
            {
                if (OpenEmptyImagesDlg.FileNames.Length < 1)
                {
                    MessageBox.Show("At least one empty image file is needed.\r\nA void empty image will be set.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                    WorkConfig.EmptyImage = "";
                    ShowWorkConfig();
                    return;
                }
                SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
                int validimages = 0;
                LongOperationForm lf = new LongOperationForm();
                lf.DialogCaption = "Computing empty image";
                lf.m_Minimum = 0.0;
                lf.m_Maximum = (double)OpenEmptyImagesDlg.FileNames.Length;
                bool stop = false;
                lf.m_StopCallback = new LongOperationForm.dStopCallback(delegate() { stop = true; });                
                uint [] pixels = null;                
                int i;
                string [] filenames = OpenEmptyImagesDlg.FileNames;
                System.Threading.Thread execthread = new System.Threading.Thread(new System.Threading.ThreadStart(delegate()
                    {
                        double v = 0.0;
                        ushort x, y;
                        foreach (string f in filenames)
                        {
                            try
                            {
                                SySal.Imaging.Image im = iGPU[0].ImageFromFile(f);
                                if (stop) break;
                                if (pixels == null)
                                {
                                    info = im.Info;
                                    pixels = new uint[im.Info.Width * im.Info.Height];
                                }
                                else
                                {
                                    if (info.Width != im.Info.Width || info.Height != im.Info.Height) throw new Exception("All images must have the same size.");
                                }
                                for (y = 0; y < info.Height; y++)
                                    for (x = 0; x < info.Width; x++)
                                        pixels[y * info.Width + x] += (uint)im.Pixels[x, y, 0];                                
                                if (stop) break;
                                validimages++;                        
                            }
                            catch (Exception exc)
                            {
                                MessageBox.Show("Error on file \"" + f + "\":\r\n" + exc.ToString(), "File error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                            }
                            lf.InvokeSetValue(v += 1.0);
                            if (stop) break;
                        }
                        lf.InvokeClose();
                    }));
                execthread.Start();
                lf.ShowDialog();
                execthread.Join();                                   
                //if (stop) return;
                if (validimages <= 0)
                {
                    MessageBox.Show("Not enough valid images.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                    return;
                }
                uint maxval = 0;
                for (i = 0; i < pixels.Length; i++)
                {
                    pixels[i] = (uint)Math.Round((double)pixels[i] / (double)validimages);
                    if (pixels[i] >= maxval) maxval = pixels[i];
                }
                byte[] b = new byte[pixels.Length];
                for (i = 0; i < pixels.Length; i++)
                    b[i] = (byte)(maxval - pixels[i]);
                WorkConfig.ImageWidth = info.Width;
                WorkConfig.ImageHeight = info.Height;
                WorkConfig.EmptyImage = SySal.Imaging.Base64ImageEncoding.ImageToBase64(new SySal.Imaging.Image(info, new BufferPixels(info, b)));
                ShowWorkConfig();
            }
        }

        public class BufferPixels : SySal.Imaging.IImagePixels
        {
            protected int m_Width;
            byte[] m_Buffer;

            public BufferPixels(SySal.Imaging.ImageInfo info, byte[] buffer)
            {
                m_Width = info.Width;
                m_Buffer = buffer;
            }

            #region IImagePixels Members

            public ushort Channels
            {
                get { return 1; }
            }

            public byte this[ushort x, ushort y, ushort channel]
            {
                get
                {
                    return m_Buffer[y * m_Width + x];
                }
                set
                {
                    throw new Exception("This image is read-only.");
                }
            }

            public byte this[uint index]
            {
                get
                {
                    return m_Buffer[index];
                }
                set
                {
                    throw new Exception("This image is read-only.");
                }
            }

            #endregion
        }

        private void btnViewEmptyImage_Click(object sender, EventArgs e)
        {
            if (WorkConfig.EmptyImage.Length > 0 && m_Display != null)
            {
                System.Drawing.Image im = SySal.Executables.NExTScanner.SySalImageFromImage.ToImage(SySal.Imaging.Base64ImageEncoding.ImageFromBase64(WorkConfig.EmptyImage));
                InfoPanel ip = new InfoPanel();
                ip.AllowsRefreshContent = false;
                ip.AllowsClose = true;
                ip.AllowsExport = true;
                ip.SetContent("Empty image", im);
                ip.Parent = null;
                ip.TopLevel = true;
                ip.ShowDialog();
            }
        }

        private void btnThresholdImage_Click(object sender, EventArgs e)
        {
            if (WorkConfig.EmptyImage.Length <= 0)
            {
                MessageBox.Show("An empty image must be defined.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            ThresholdImageForm thf = new ThresholdImageForm();
            SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
            info.Width = (ushort)WorkConfig.ImageWidth;
            info.Height = (ushort)WorkConfig.ImageHeight;
            info.BitsPerPixel = 8;
            info.PixelFormat = SySal.Imaging.PixelFormatType.GrayScale8;
            thf.m_ImageFormat = info;
            int i;
            for (i = 0; i < iGPU.Length; i++)
            {
                iGPU[i].ImageFormat = info;
                iGPU[i].MaxSegmentsPerScanLine = WorkConfig.MaxSegmentsPerLine;
                iGPU[i].MaxClustersPerImage = WorkConfig.MaxClusters;
                iGPU[i].OutputFeatures = SySal.Imaging.ImageProcessingFeatures.Clusters;
                iGPU[i].EqGreyLevelTargetMedian = (byte)WorkConfig.GreyTargetMedian;
                iGPU[i].EmptyImage = (WorkConfig.EmptyImage.Length > 0) ? SySal.Imaging.Base64ImageEncoding.ImageFromBase64(WorkConfig.EmptyImage) : null;
            }
            thf.iGPU = iGPU;            
            if (thf.ShowDialog() == DialogResult.OK)
            {
                WorkConfig.ThresholdImage = thf.m_Result;
                ShowWorkConfig();
            }
        }

        private void btnViewThresholdImage_Click(object sender, EventArgs e)
        {
            if (WorkConfig.ThresholdImage.Length <= 0) return;
            try
            {
                SySal.Imaging.DCTInterpolationImage dct = Scanner.DCTInterpolationImageFromString(WorkConfig.ThresholdImage);
                InfoPanel panel = new InfoPanel();
                panel.SetContent("Threshold image", SySalImageFromImage.ThresholdImage(dct));
                panel.TopLevel = true;
                panel.ShowDialog();                
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Error creating interpolated image.", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        private void btnPixelToMicron_Click(object sender, EventArgs e)
        {
            try
            {
                Pixel2MicronComputeForm pxf = new Pixel2MicronComputeForm();
                pxf.iLog = iLog;
                string[] summaryfiles;
                if (txtSummaryFile.Text.ToUpper().EndsWith(".TXT"))
                {
                    try
                    {
                        summaryfiles = System.IO.File.ReadAllLines(txtSummaryFile.Text);
                    }
                    catch (Exception xc)
                    {
                        MessageBox.Show("Cannot interpret this string as a list of summary files.\r\n" + xc.ToString(), "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                }
                else summaryfiles = new string[] { txtSummaryFile.Text };
                pxf.SummaryFiles = summaryfiles;
                pxf.XConv = WorkConfig.Pixel2Micron.X;
                pxf.YConv = WorkConfig.Pixel2Micron.Y;
                if (pxf.ShowDialog() == DialogResult.OK)
                {
                    WorkConfig.Pixel2Micron.X = pxf.XConv;
                    WorkConfig.Pixel2Micron.Y = pxf.YConv;
                    ShowWorkConfig();
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.ToString(), "Computation error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        private void btnComputeClusterMaps_Click(object sender, EventArgs e)
        {
            if (WorkConfig.EmptyImage.Length <= 0 || WorkConfig.ThresholdImage.Length <= 0)
            {
                MessageBox.Show("An empty image and a threshold image must be defined.", "Input error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }            
            try
            {
                //iCamDisp.EnableAutoRefresh = false;
                SySal.Imaging.ImageInfo info = new SySal.Imaging.ImageInfo();
                info.Width = (ushort)WorkConfig.ImageWidth;
                info.Height = (ushort)WorkConfig.ImageHeight;
                info.BitsPerPixel = 8;
                info.PixelFormat = SySal.Imaging.PixelFormatType.GrayScale8;
                int i;
                for (i = 0; i < iGPU.Length; i++)
                {
                    iGPU[i].ImageFormat = info;
                    iGPU[i].MaxSegmentsPerScanLine = WorkConfig.MaxSegmentsPerLine;
                    iGPU[i].MaxClustersPerImage = WorkConfig.MaxClusters;
                    iGPU[i].OutputFeatures = SySal.Imaging.ImageProcessingFeatures.Cluster2ndMomenta | SySal.Imaging.ImageProcessingFeatures.BinarizedImage;
                    iGPU[i].EqGreyLevelTargetMedian = (byte)WorkConfig.GreyTargetMedian;
                    iGPU[i].EmptyImage = (WorkConfig.EmptyImage.Length > 0) ? SySal.Imaging.Base64ImageEncoding.ImageFromBase64(WorkConfig.EmptyImage) : null;
                    iGPU[i].ThresholdImage = (WorkConfig.ThresholdImage.Length > 0) ? Scanner.DCTInterpolationImageFromString(WorkConfig.ThresholdImage) : null;
                }                                
                string[] summaryfiles;                
                if (txtSummaryFile.Text.ToUpper().EndsWith(".TXT"))
                {
                    try
                    {
                        summaryfiles = System.IO.File.ReadAllLines(txtSummaryFile.Text);
                    }
                    catch (Exception xc)
                    {
                        MessageBox.Show("Cannot interpret this string as a list of summary files.\r\n" + xc.ToString(), "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                }
                else summaryfiles = new string[] { txtSummaryFile.Text };                
                int qsan = 0;
                for (qsan = 0; qsan < summaryfiles.Length; qsan++)
                {
                    if (summaryfiles[qsan].Trim().Length <= 0) continue;
                    QuasiStaticAcquisition qsa = new QuasiStaticAcquisition(summaryfiles[qsan]);
                    System.Collections.Queue allimages = new System.Collections.Queue();
                    foreach (QuasiStaticAcquisition.Sequence qs in qsa.Sequences)
                        foreach (QuasiStaticAcquisition.Sequence.Layer q in qs.Layers)
                            allimages.Enqueue(q);
                    LongOperationForm lf = new LongOperationForm();
                    lf.DialogCaption = "Processing images (acquisition " + (qsan + 1) + "/" + summaryfiles.Length + ")";
                    lf.m_Minimum = 0.0;
                    lf.m_Maximum = allimages.Count;
                    lf.Value = 0.0;
                    bool stop = false;
                    lf.m_StopCallback = new LongOperationForm.dStopCallback(delegate() { stop = true; });
                    System.Threading.Thread[] workerthreads = new System.Threading.Thread[iGPU.Length];
                    System.Exception workexception = null;
                    SySal.Imaging.Image inputimageexception = null;
                    SySal.Imaging.Image binarizedimageexception = null;
                    for (i = 0; i < workerthreads.Length; i++)
                        workerthreads[i] = new System.Threading.Thread(new System.Threading.ParameterizedThreadStart(delegate(object oigpu)
                            {
                                SySal.Imaging.IImageProcessor igpu = oigpu as SySal.Imaging.IImageProcessor;
                                iLog.Log("WorkerThread " + oigpu.GetHashCode() + "\r\n" + igpu.ToString(), "Entering");                                
                                while (stop == false)
                                {
                                    QuasiStaticAcquisition.Sequence.Layer q = null;
                                    lock (allimages)
                                        if (allimages.Count == 0)
                                        {
                                            lf.InvokeClose();
                                            break;
                                        }
                                        else
                                        {
                                            q = allimages.Dequeue() as QuasiStaticAcquisition.Sequence.Layer;
                                            lf.InvokeSetValue(lf.m_Maximum - allimages.Count);
                                        }
                                    SySal.Imaging.LinearMemoryImage inputimage = null;
                                    try
                                    {
                                        iLog.Log("WorkerThread " + oigpu.GetHashCode(), "LoadFile");
                                        inputimage = igpu.ImageFromFile(q.ImageFileName);
                                        iLog.Log("WorkerThread " + oigpu.GetHashCode(), "Process start");
                                        igpu.Input = inputimage;
                                        iLog.Log("WorkerThread " + oigpu.GetHashCode(), "Process end");
                                        inputimage.Dispose();
                                        iLog.Log("WorkerThread " + oigpu.GetHashCode(), "Dispose image");
                                        if (igpu.Warnings != null && igpu.Warnings.Length > 0) throw igpu.Warnings[0];
                                        q.WriteClusters(igpu.Clusters[0]);
                                    }
                                    catch (Exception xc)
                                    {                                        
                                        stop = true;
                                        lock (allimages)
                                        {
                                            workexception = new Exception("Error on file " + q.ImageFileName + "\r\nGPU: " + oigpu.GetHashCode() + "\r\n" + xc.ToString());
                                            try
                                            {
                                                inputimageexception = inputimage;
                                                binarizedimageexception = igpu.BinarizedImages;
                                            }
                                            catch (Exception) { }
                                        }
                                        lf.InvokeClose();
                                    }
                                }
                            }));
                    for (i = 0; i < workerthreads.Length; i++) workerthreads[i].Start(iGPU[i]);
                    lf.ShowDialog();
                    for (i = 0; i < workerthreads.Length; i++) workerthreads[i].Join();
                    if (workexception != null)
                    {
                        MessageBox.Show(workexception.ToString(), "Processing error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        if (inputimageexception != null)
                        {
                            InfoPanel panel = new InfoPanel();
                            panel.TopLevel = true;
                            panel.SetContent("Input image that caused a warning", SySalImageFromImage.ToImage(inputimageexception));
                            ((SySal.Imaging.LinearMemoryImage)inputimageexception).Dispose();
                            panel.ShowDialog();
                        }
                        if (binarizedimageexception != null)
                        {
                            InfoPanel panel = new InfoPanel();
                            panel.TopLevel = true;
                            panel.SetContent("Binarized image that caused a warning", SySalImageFromImage.ToImage(binarizedimageexception));
                            ((SySal.Imaging.LinearMemoryImage)binarizedimageexception).Dispose();
                            panel.ShowDialog();
                        }
                    }
                    if (stop) break;
                }
                MessageBox.Show("Images processed.", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception xc)
            {
                MessageBox.Show(xc.ToString(), "Setup error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            finally
            {
                //iCamDisp.EnableAutoRefresh = true;
            }
        }

        private void btnViewAcquisition_Click(object sender, EventArgs e)
        {
            try
            {
                QuasiStaticAcquisition qsa = new QuasiStaticAcquisition(txtSummaryFile.Text);
                NumericalTools.Plot pl = new NumericalTools.Plot();
                int w = qsa.Sequences.Length * 100 + 100;
                int h = 400;
                System.Drawing.Bitmap bmp = new System.Drawing.Bitmap(w, h);
                int total = 0;
                foreach (QuasiStaticAcquisition.Sequence qs in qsa.Sequences)
                    total += qs.Layers.Length;
                double[] x = new double[total];
                double[] y = new double[total];
                total = 0;
                foreach (QuasiStaticAcquisition.Sequence qs in qsa.Sequences)
                    foreach (QuasiStaticAcquisition.Sequence.Layer qsl in qs.Layers)
                    {
                        x[total] = qs.Id * 100 + (qsl.Id * 100.0 / qs.Layers.Length);
                        y[total] = (double)qsl.Clusters;
                        total++;
                    }
                pl.LabelFont = new System.Drawing.Font("Segoe UI", 12);
                pl.PanelX = 2.0;
                pl.PanelY = 0.0;
                pl.VecX = x;
                pl.VecY = y;
                pl.XTitle = "Sequence,Layer";
                pl.YTitle = "Cluster counts";
                pl.SetXDefaultLimits = true;
                pl.SetYDefaultLimits = true;
                pl.Scatter(System.Drawing.Graphics.FromImage(bmp), w, h);
                InfoPanel p = new InfoPanel();
                p.TopLevel = true;
                p.SetContent("Cluster counts", bmp);
                p.ShowDialog();
            }
            catch (Exception xc)
            {
                MessageBox.Show(xc.ToString(), "Data error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        private void btnMakeChains_Click(object sender, EventArgs e)
        {
            try
            {
                string[] summaryfiles;
                if (txtSummaryFile.Text.ToUpper().EndsWith(".TXT"))
                {
                    try
                    {
                        summaryfiles = System.IO.File.ReadAllLines(txtSummaryFile.Text);
                    }
                    catch (Exception xc)
                    {
                        MessageBox.Show("Cannot interpret this string as a list of summary files.\r\n" + xc.ToString(), "Input Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                        return;
                    }
                }
                else summaryfiles = new string[] { txtSummaryFile.Text };                
                BasicGrain3DMaker gm = new BasicGrain3DMaker();
                LongOperationForm lf = new LongOperationForm();
                System.Collections.ArrayList demags = new System.Collections.ArrayList();
                System.Collections.ArrayList matchdx = new System.Collections.ArrayList();
                System.Collections.ArrayList matchdy = new System.Collections.ArrayList();
                bool stop = false;
                lf.m_Minimum = 0.0;
                lf.m_Maximum = (double)100.0;
                lf.DialogCaption = "Making grains";
                lf.m_StopCallback = new LongOperationForm.dStopCallback(delegate() { stop = true; });
                gm.Config = WorkConfig;
                gm.OptimizeDemagCoefficient = chkOptimizeDMagDZ.Checked;
                int seq = 0;
                InfoPanel ip = null;
                System.Drawing.Bitmap bmp = new Bitmap(900, 400);
                NumericalTools.Plot plx = new NumericalTools.Plot();
                plx.LabelFont = plx.PanelFont = new System.Drawing.Font("Segoe UI", 10);
                plx.PanelX = 1.0;
                plx.PanelY = 1.0;
                plx.XTitle = "Match \x018AX";
                plx.YTitle = "Counts";
                plx.VecX = new double[1] { 0.0 };
                plx.SetXDefaultLimits = true;
                plx.SetYDefaultLimits = true;
                plx.HistoFit = -2;                
                plx.PanelFormat = "F3";
                NumericalTools.Plot ply = new NumericalTools.Plot();
                ply.LabelFont = ply.PanelFont = new System.Drawing.Font("Segoe UI", 10);
                ply.PanelX = 1.0;
                ply.PanelY = 1.0;
                ply.XTitle = "Match \x018AY";
                ply.YTitle = "Counts";
                ply.VecX = new double[1] { 0.0 };
                ply.SetXDefaultLimits = true;
                ply.SetYDefaultLimits = true;
                ply.HistoFit = -2;
                ply.PanelFormat = "F3";
                NumericalTools.Plot pl = new NumericalTools.Plot();
                pl.LabelFont = pl.PanelFont = new System.Drawing.Font("Segoe UI", 10);
                pl.PanelX = 1.0;
                pl.PanelY = 0.0;
                pl.VecX = new double[1] { 0.0 };                
                pl.XTitle = "DMagDfocus";
                pl.YTitle = "Counts";
                pl.SetXDefaultLimits = true;
                pl.SetYDefaultLimits = true;
                pl.HistoFit = -2;
                pl.PanelFormat = "F5";
                int sfi = 0;
                QuasiStaticAcquisition qsa = null;
                gm.Progress = new BasicGrain3DMaker.dProgress(delegate(double x) 
                    { 
                        lf.InvokeSetValue((qsa == null) ? 0.0 : (100.0 * (((x + seq) / qsa.Sequences.Length) + sfi) / summaryfiles.Length));
                        try
                        {
                            lf.Invoke(new BasicGrain3DMaker.dShouldStop(delegate()
                            {
                                if (ip == null)
                                {
                                    ip = new InfoPanel();
                                    ip.TopLevel = true;
                                    ip.TopMost = true;
                                    ip.AllowsClose = false;
                                    ip.SetContent("DMag/DZ", bmp);
                                    ip.Show();
                                }
                                System.Drawing.Graphics gr = System.Drawing.Graphics.FromImage(bmp);
                                gr.TextRenderingHint = System.Drawing.Text.TextRenderingHint.ClearTypeGridFit;
                                gr.Clear(Color.Transparent);
                                matchdx.AddRange(gm.MatchDX);
                                gr.SetClip(new Rectangle(0, 0, bmp.Width / 3, bmp.Height));
                                gr.TranslateTransform(0, 0);
                                plx.VecX = (double[])matchdx.ToArray(typeof(double));
                                plx.DX = (float)5e-3;
                                if (matchdx.Count > 10)
                                    plx.HistoSkyline(gr, bmp.Width / 3, bmp.Height);
                                matchdy.AddRange(gm.MatchDY);
                                gr.SetClip(new Rectangle(bmp.Width / 3, 0, bmp.Width / 3, bmp.Height));
                                gr.TranslateTransform(bmp.Width / 3, 0);
                                ply.VecX = (double[])matchdy.ToArray(typeof(double));
                                ply.DX = (float)5e-3;
                                if (matchdy.Count > 10) 
                                    ply.HistoSkyline(gr, bmp.Width / 3, bmp.Height);
                                string dmgc = "";
                                foreach (double cd in gm.DemagCoefficients) dmgc += "\r\n" + cd;
                                iLog.Log("btnMakeChains_Click", dmgc);
                                ip.SetContent("DMag/DZ", bmp);
                                if (x == 1.0)
                                {
                                    double[] dmag = (double[])demags.ToArray(typeof(double));
                                    if (dmag != null && dmag.Length > 1)
                                    {
                                        pl.VecX = dmag;
                                        double q, qerr, qmin, qmax;
                                        if (Utilities.SafeMeanAverage(dmag, 0.68, out q, out qmin, out qmax, out qerr) == NumericalTools.ComputationResult.OK)
                                        {
                                            q = Math.Round(q * 1e6) * 1e-6;
                                            qerr = Math.Round(qerr * 1e6) * 1e-6;
                                            gr.SetClip(new Rectangle(bmp.Width * 2 / 3, 0, bmp.Width / 3, bmp.Height));
                                            gr.TranslateTransform(bmp.Width * 2 / 3, 0);
                                            pl.DX = (float)Math.Min(1.0e-4, Math.Max(1.0e-5, (qmax - qmin) / Math.Sqrt(dmag.Length)));
                                            pl.SetXDefaultLimits = false;
                                            pl.MinX = qmin;
                                            pl.MaxX = qmax;
                                            pl.HistoSkyline(gr, bmp.Width / 3, bmp.Height);
                                            ip.SetContent("DMag/DZ = " + q.ToString("F6", System.Globalization.CultureInfo.InvariantCulture) + "\xB1" + qerr.ToString("F6", System.Globalization.CultureInfo.InvariantCulture), bmp);
                                        }                                        
                                    }
                                }                                
                                return false;
                            }));
                        }
                        catch (Exception) { }
                    });
                gm.ShouldStop = new BasicGrain3DMaker.dShouldStop(delegate() { return stop; });
                System.Threading.Thread workthread = new System.Threading.Thread(new System.Threading.ThreadStart(delegate()
                    {                        
                        for (sfi = 0; sfi < summaryfiles.Length; sfi++)
                        {
                            qsa = new QuasiStaticAcquisition(summaryfiles[sfi]);
                            for (seq = 0; seq < qsa.Sequences.Length; seq++)
                            {
                                SySal.Imaging.Cluster3D[][] cplanes = new SySal.Imaging.Cluster3D[qsa.Sequences[seq].Layers.Length][];
                                int i;
                                for (i = 0; i < cplanes.Length; i++)
                                {
                                    System.Collections.ArrayList carr = new System.Collections.ArrayList();
                                    SySal.Imaging.Cluster[] cpl = qsa.Sequences[seq].Layers[i].ReadClusters();
                                    foreach (SySal.Imaging.Cluster c1 in cpl)
                                        if (c1.Area >= WorkConfig.MinClusterArea)
                                        {
                                            SySal.Imaging.Cluster3D c3 = new SySal.Imaging.Cluster3D();
                                            c3.Cluster = c1;
                                            c3.Layer = (uint)i;
                                            c3.Z = qsa.Sequences[seq].Layers[i].Position.Z;
                                            carr.Add(c3);
                                        }
                                    cplanes[i] = (SySal.Imaging.Cluster3D[])carr.ToArray(typeof(SySal.Imaging.Cluster3D));
                                }
                                SySal.BasicTypes.Vector[] positions = new SySal.BasicTypes.Vector[qsa.Sequences[seq].Layers.Length];
                                for (i = 0; i < positions.Length; i++)
                                    positions[i] = qsa.Sequences[seq].Layers[i].Position;
                                if (stop) return;
                                SySal.Imaging.Grain3D[] grs = gm.MakeGrainsFromClusters(cplanes, positions);
                                if (grs != null) qsa.Sequences[seq].WriteGrains(grs);
                                matchdx.AddRange(gm.MatchDX);
                                matchdy.AddRange(gm.MatchDY);
                                demags.AddRange(gm.DemagCoefficients);
                                if (stop) return;
                            }
                            lf.InvokeClose();
                        }
                    }));
                workthread.Start();
                lf.ShowDialog();
                workthread.Join();
                if (ip != null) ip.Close();
            }
            catch (Exception xc)
            {
                MessageBox.Show(xc.ToString(), "Data processing error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
        }

        private void OnClusterMatchPosTolLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ClusterMatchPositionTolerance = double.Parse(txtCMPosTol.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnClusterMatchMaxOffsetLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ClusterMatchMaxOffset = double.Parse(txtCMMaxOffset.Text, System.Globalization.CultureInfo.InvariantCulture);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMinClusterAreaLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MinClusterArea = uint.Parse(txtMinClusterArea.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMinGrainVolumeLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MinGrainVolume = uint.Parse(txtMinGrainVolume.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnMinClusterMatchesLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.MinClusterMatchCount = uint.Parse(txtCMMinMatches.Text);
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnEmptyImageLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.EmptyImage = txtEmptyImage.Text;
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }

        private void OnThresholdImageLeave(object sender, EventArgs e)
        {
            try
            {
                WorkConfig.ThresholdImage = txtThresholdImage.Text;
            }
            catch (Exception)
            {
                ShowWorkConfig();
            }
        }
    }
}

