using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Executables.NExTScanner
{
    public partial class ScannerSettingsForm : SySal.SySalNExTControls.SySalDialog
    {
        public ScannerSettingsForm()
        {
            InitializeComponent();
        }

        public ScannerSettings C;

        private void btnOK_Click(object sender, EventArgs e)
        {
            C.LogDirectory = txtLogDir.Text;
            C.DataDirectory = txtDataDir.Text;
            C.ScanServerDataDirectory = txtScanSrvDataDir.Text;
            C.ConfigDirectory = txtConfigDir.Text;
            C.StageLibrary = txtStageLib.Text;
            C.GrabberLibrary = txtGrabberLib.Text;
            C.GPULibrary = txtGPULib.Text;
            DialogResult = DialogResult.OK;
            Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            txtLogDir.Text = C.LogDirectory;
            txtDataDir.Text = C.DataDirectory;
            txtScanSrvDataDir.Text = C.ScanServerDataDirectory;
            txtConfigDir.Text = C.ConfigDirectory;
            txtStageLib.Text = C.StageLibrary;
            txtGrabberLib.Text = C.GrabberLibrary;
            txtGPULib.Text = C.GPULibrary;
        }

        private void btnLogDirSel_Click(object sender, EventArgs e)
        {
            FolderBrowserDlg.SelectedPath = txtLogDir.Text;
            FolderBrowserDlg.Description = "Select a directory to store log files.";
            if (FolderBrowserDlg.ShowDialog() == DialogResult.OK)
                txtLogDir.Text = FolderBrowserDlg.SelectedPath;
        }

        private void btnDataDirSel_Click(object sender, EventArgs e)
        {
            FolderBrowserDlg.SelectedPath = txtDataDir.Text;
            FolderBrowserDlg.Description = "Select a directory to store data files.";
            if (FolderBrowserDlg.ShowDialog() == DialogResult.OK)
                txtDataDir.Text = FolderBrowserDlg.SelectedPath;
        }

        private void btnScanServerDataDir_Click(object sender, EventArgs e)
        {
            FolderBrowserDlg.SelectedPath = txtScanSrvDataDir.Text;
            FolderBrowserDlg.Description = "Select an output directory for ScanServer data.";
            if (FolderBrowserDlg.ShowDialog() == DialogResult.OK)
                txtScanSrvDataDir.Text = FolderBrowserDlg.SelectedPath;
        }

        private void btnConfigDirSel_Click(object sender, EventArgs e)
        {
            FolderBrowserDlg.SelectedPath = txtConfigDir.Text;
            FolderBrowserDlg.Description = "Select a directory to store configurations.";
            if (FolderBrowserDlg.ShowDialog() == DialogResult.OK)
                txtConfigDir.Text = FolderBrowserDlg.SelectedPath;
        }

        private void btnStageLibSel_Click(object sender, EventArgs e)
        {
            OpenFileDlg.Title = "Select the file that contains the stage control library.";
            OpenFileDlg.FileName = txtStageLib.Text;
            if (OpenFileDlg.ShowDialog() == DialogResult.OK)
                txtStageLib.Text = OpenFileDlg.FileName;
        }

        private void btnGrabberLibSel_Click(object sender, EventArgs e)
        {
            OpenFileDlg.Title = "Select the file that contains the frame grabber control library.";
            OpenFileDlg.FileName = txtGrabberLib.Text;
            if (OpenFileDlg.ShowDialog() == DialogResult.OK)
                txtGrabberLib.Text = OpenFileDlg.FileName;
        }

        private void btnGPULibSel_Click(object sender, EventArgs e)
        {
            OpenFileDlg.Title = "Select the file that contains the GPU control library.";
            OpenFileDlg.FileName = txtGPULib.Text;
            if (OpenFileDlg.ShowDialog() == DialogResult.OK)
                txtGPULib.Text = OpenFileDlg.FileName;
        }
    }
}

