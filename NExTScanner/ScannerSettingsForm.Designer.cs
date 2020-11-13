namespace SySal.Executables.NExTScanner
{
    partial class ScannerSettingsForm
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.components = new System.ComponentModel.Container();
            System.ComponentModel.ComponentResourceManager resources = new System.ComponentModel.ComponentResourceManager(typeof(ScannerSettingsForm));
            this.btnLogDirSel = new SySal.SySalNExTControls.SySalButton();
            this.txtLogDir = new System.Windows.Forms.TextBox();
            this.txtDataDir = new System.Windows.Forms.TextBox();
            this.btnDataDirSel = new SySal.SySalNExTControls.SySalButton();
            this.txtScanSrvDataDir = new System.Windows.Forms.TextBox();
            this.btnScanSrvDataDir = new SySal.SySalNExTControls.SySalButton();
            this.txtGPULib = new System.Windows.Forms.TextBox();
            this.btnGPULibSel = new SySal.SySalNExTControls.SySalButton();
            this.txtGrabberLib = new System.Windows.Forms.TextBox();
            this.btnGrabberLibSel = new SySal.SySalNExTControls.SySalButton();
            this.txtStageLib = new System.Windows.Forms.TextBox();
            this.btnStageLibSel = new SySal.SySalNExTControls.SySalButton();
            this.btnOK = new SySal.SySalNExTControls.SySalButton();
            this.btnCancel = new SySal.SySalNExTControls.SySalButton();
            this.FolderBrowserDlg = new System.Windows.Forms.FolderBrowserDialog();
            this.OpenFileDlg = new System.Windows.Forms.OpenFileDialog();
            this.MachineSettingsToolTip = new System.Windows.Forms.ToolTip(this.components);
            this.btnConfigDirSel = new SySal.SySalNExTControls.SySalButton();
            this.txtConfigDir = new System.Windows.Forms.TextBox();
            this.SuspendLayout();
            // 
            // btnLogDirSel
            // 
            this.btnLogDirSel.AutoSize = true;
            this.btnLogDirSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnLogDirSel.BackColor = System.Drawing.Color.Transparent;
            this.btnLogDirSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnLogDirSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnLogDirSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnLogDirSel.Location = new System.Drawing.Point(15, 34);
            this.btnLogDirSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnLogDirSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnLogDirSel.Name = "btnLogDirSel";
            this.btnLogDirSel.Size = new System.Drawing.Size(166, 25);
            this.btnLogDirSel.TabIndex = 1;
            this.btnLogDirSel.Text = "Directory for Log Files";
            this.MachineSettingsToolTip.SetToolTip(this.btnLogDirSel, "Click here to choose the directory where this Scanner writes log files.");
            this.btnLogDirSel.Click += new System.EventHandler(this.btnLogDirSel_Click);
            // 
            // txtLogDir
            // 
            this.txtLogDir.BackColor = System.Drawing.Color.GhostWhite;
            this.txtLogDir.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtLogDir.ForeColor = System.Drawing.Color.Navy;
            this.txtLogDir.Location = new System.Drawing.Point(227, 34);
            this.txtLogDir.Name = "txtLogDir";
            this.txtLogDir.Size = new System.Drawing.Size(284, 25);
            this.txtLogDir.TabIndex = 2;
            // 
            // txtDataDir
            // 
            this.txtDataDir.BackColor = System.Drawing.Color.GhostWhite;
            this.txtDataDir.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtDataDir.ForeColor = System.Drawing.Color.Navy;
            this.txtDataDir.Location = new System.Drawing.Point(227, 65);
            this.txtDataDir.Name = "txtDataDir";
            this.txtDataDir.Size = new System.Drawing.Size(284, 25);
            this.txtDataDir.TabIndex = 4;
            // 
            // btnDataDirSel
            // 
            this.btnDataDirSel.AutoSize = true;
            this.btnDataDirSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnDataDirSel.BackColor = System.Drawing.Color.Transparent;
            this.btnDataDirSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnDataDirSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnDataDirSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnDataDirSel.Location = new System.Drawing.Point(15, 65);
            this.btnDataDirSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnDataDirSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnDataDirSel.Name = "btnDataDirSel";
            this.btnDataDirSel.Size = new System.Drawing.Size(137, 25);
            this.btnDataDirSel.TabIndex = 3;
            this.btnDataDirSel.Text = "Directory for Data";
            this.MachineSettingsToolTip.SetToolTip(this.btnDataDirSel, "Click here to choose the directory where this Scanner writes data.\r\nThis applies " +
                    "only if the output path is not explicitly specified.");
            this.btnDataDirSel.Click += new System.EventHandler(this.btnDataDirSel_Click);
            // 
            // txtScanSrvDataDir
            // 
            this.txtScanSrvDataDir.BackColor = System.Drawing.Color.GhostWhite;
            this.txtScanSrvDataDir.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtScanSrvDataDir.ForeColor = System.Drawing.Color.Navy;
            this.txtScanSrvDataDir.Location = new System.Drawing.Point(227, 96);
            this.txtScanSrvDataDir.Name = "txtScanSrvDataDir";
            this.txtScanSrvDataDir.Size = new System.Drawing.Size(284, 25);
            this.txtScanSrvDataDir.TabIndex = 6;
            // 
            // btnScanSrvDataDir
            // 
            this.btnScanSrvDataDir.AutoSize = true;
            this.btnScanSrvDataDir.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnScanSrvDataDir.BackColor = System.Drawing.Color.Transparent;
            this.btnScanSrvDataDir.FocusedColor = System.Drawing.Color.Navy;
            this.btnScanSrvDataDir.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnScanSrvDataDir.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnScanSrvDataDir.Location = new System.Drawing.Point(15, 96);
            this.btnScanSrvDataDir.Margin = new System.Windows.Forms.Padding(6);
            this.btnScanSrvDataDir.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnScanSrvDataDir.Name = "btnScanSrvDataDir";
            this.btnScanSrvDataDir.Size = new System.Drawing.Size(206, 25);
            this.btnScanSrvDataDir.TabIndex = 5;
            this.btnScanSrvDataDir.Text = "Directory for ScanServer Data";
            this.MachineSettingsToolTip.SetToolTip(this.btnScanSrvDataDir, resources.GetString("btnScanSrvDataDir.ToolTip"));
            this.btnScanSrvDataDir.Click += new System.EventHandler(this.btnScanServerDataDir_Click);
            // 
            // txtGPULib
            // 
            this.txtGPULib.BackColor = System.Drawing.Color.GhostWhite;
            this.txtGPULib.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtGPULib.ForeColor = System.Drawing.Color.Navy;
            this.txtGPULib.Location = new System.Drawing.Point(226, 221);
            this.txtGPULib.Name = "txtGPULib";
            this.txtGPULib.Size = new System.Drawing.Size(284, 25);
            this.txtGPULib.TabIndex = 14;
            // 
            // btnGPULibSel
            // 
            this.btnGPULibSel.AutoSize = true;
            this.btnGPULibSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnGPULibSel.BackColor = System.Drawing.Color.Transparent;
            this.btnGPULibSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnGPULibSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnGPULibSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnGPULibSel.Location = new System.Drawing.Point(14, 221);
            this.btnGPULibSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnGPULibSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnGPULibSel.Name = "btnGPULibSel";
            this.btnGPULibSel.Size = new System.Drawing.Size(151, 25);
            this.btnGPULibSel.TabIndex = 13;
            this.btnGPULibSel.Text = "GPU Control Library";
            this.MachineSettingsToolTip.SetToolTip(this.btnGPULibSel, "Click here to choose the DLL that contains the library to be used to control the " +
                    "GPU\'s.\r\nThe same library will be used for all available GPU\'s.");
            this.btnGPULibSel.Click += new System.EventHandler(this.btnGPULibSel_Click);
            // 
            // txtGrabberLib
            // 
            this.txtGrabberLib.BackColor = System.Drawing.Color.GhostWhite;
            this.txtGrabberLib.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtGrabberLib.ForeColor = System.Drawing.Color.Navy;
            this.txtGrabberLib.Location = new System.Drawing.Point(226, 190);
            this.txtGrabberLib.Name = "txtGrabberLib";
            this.txtGrabberLib.Size = new System.Drawing.Size(284, 25);
            this.txtGrabberLib.TabIndex = 12;
            // 
            // btnGrabberLibSel
            // 
            this.btnGrabberLibSel.AutoSize = true;
            this.btnGrabberLibSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnGrabberLibSel.BackColor = System.Drawing.Color.Transparent;
            this.btnGrabberLibSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnGrabberLibSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnGrabberLibSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnGrabberLibSel.Location = new System.Drawing.Point(14, 190);
            this.btnGrabberLibSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnGrabberLibSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnGrabberLibSel.Name = "btnGrabberLibSel";
            this.btnGrabberLibSel.Size = new System.Drawing.Size(178, 25);
            this.btnGrabberLibSel.TabIndex = 11;
            this.btnGrabberLibSel.Text = "Grabber Control Library";
            this.MachineSettingsToolTip.SetToolTip(this.btnGrabberLibSel, "Click here to choose the DLL that contains the library that allows controlling im" +
                    "age grabbing.");
            this.btnGrabberLibSel.Click += new System.EventHandler(this.btnGrabberLibSel_Click);
            // 
            // txtStageLib
            // 
            this.txtStageLib.BackColor = System.Drawing.Color.GhostWhite;
            this.txtStageLib.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtStageLib.ForeColor = System.Drawing.Color.Navy;
            this.txtStageLib.Location = new System.Drawing.Point(226, 159);
            this.txtStageLib.Name = "txtStageLib";
            this.txtStageLib.Size = new System.Drawing.Size(284, 25);
            this.txtStageLib.TabIndex = 10;
            // 
            // btnStageLibSel
            // 
            this.btnStageLibSel.AutoSize = true;
            this.btnStageLibSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnStageLibSel.BackColor = System.Drawing.Color.Transparent;
            this.btnStageLibSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnStageLibSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnStageLibSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnStageLibSel.Location = new System.Drawing.Point(14, 159);
            this.btnStageLibSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnStageLibSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnStageLibSel.Name = "btnStageLibSel";
            this.btnStageLibSel.Size = new System.Drawing.Size(160, 25);
            this.btnStageLibSel.TabIndex = 9;
            this.btnStageLibSel.Text = "Stage Control Library";
            this.MachineSettingsToolTip.SetToolTip(this.btnStageLibSel, "Click here to choose the DLL that contains the stage control library.");
            this.btnStageLibSel.Click += new System.EventHandler(this.btnStageLibSel_Click);
            // 
            // btnOK
            // 
            this.btnOK.AutoSize = true;
            this.btnOK.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnOK.BackColor = System.Drawing.Color.Transparent;
            this.btnOK.FocusedColor = System.Drawing.Color.Navy;
            this.btnOK.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnOK.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnOK.Location = new System.Drawing.Point(16, 263);
            this.btnOK.Margin = new System.Windows.Forms.Padding(7);
            this.btnOK.MinimumSize = new System.Drawing.Size(5, 5);
            this.btnOK.Name = "btnOK";
            this.btnOK.Size = new System.Drawing.Size(34, 29);
            this.btnOK.TabIndex = 15;
            this.btnOK.Text = "OK";
            this.btnOK.Click += new System.EventHandler(this.btnOK_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.AutoSize = true;
            this.btnCancel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnCancel.BackColor = System.Drawing.Color.Transparent;
            this.btnCancel.FocusedColor = System.Drawing.Color.Navy;
            this.btnCancel.Font = new System.Drawing.Font("Segoe UI", 14.25F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnCancel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnCancel.Location = new System.Drawing.Point(446, 263);
            this.btnCancel.Margin = new System.Windows.Forms.Padding(7);
            this.btnCancel.MinimumSize = new System.Drawing.Size(5, 5);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(65, 29);
            this.btnCancel.TabIndex = 16;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // OpenFileDlg
            // 
            this.OpenFileDlg.FileName = "openFileDialog1";
            this.OpenFileDlg.Filter = "Dynamic Linking Libraries (*.dll)|*.dll|Executables (*.exe)|*.exe";
            // 
            // MachineSettingsToolTip
            // 
            this.MachineSettingsToolTip.IsBalloon = true;
            // 
            // btnConfigDirSel
            // 
            this.btnConfigDirSel.AutoSize = true;
            this.btnConfigDirSel.AutoSizeMode = System.Windows.Forms.AutoSizeMode.GrowAndShrink;
            this.btnConfigDirSel.BackColor = System.Drawing.Color.Transparent;
            this.btnConfigDirSel.FocusedColor = System.Drawing.Color.Navy;
            this.btnConfigDirSel.Font = new System.Drawing.Font("Segoe UI", 12F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.btnConfigDirSel.ForeColor = System.Drawing.Color.DodgerBlue;
            this.btnConfigDirSel.Location = new System.Drawing.Point(15, 127);
            this.btnConfigDirSel.Margin = new System.Windows.Forms.Padding(6);
            this.btnConfigDirSel.MinimumSize = new System.Drawing.Size(4, 4);
            this.btnConfigDirSel.Name = "btnConfigDirSel";
            this.btnConfigDirSel.Size = new System.Drawing.Size(210, 25);
            this.btnConfigDirSel.TabIndex = 7;
            this.btnConfigDirSel.Text = "Directory for Configurations";
            this.MachineSettingsToolTip.SetToolTip(this.btnConfigDirSel, "Click here to choose the directory where configurations are stored.");
            this.btnConfigDirSel.Click += new System.EventHandler(this.btnConfigDirSel_Click);
            // 
            // txtConfigDir
            // 
            this.txtConfigDir.BackColor = System.Drawing.Color.GhostWhite;
            this.txtConfigDir.Font = new System.Drawing.Font("Segoe UI", 9.75F, System.Drawing.FontStyle.Regular, System.Drawing.GraphicsUnit.Point, ((byte)(0)));
            this.txtConfigDir.ForeColor = System.Drawing.Color.Navy;
            this.txtConfigDir.Location = new System.Drawing.Point(227, 127);
            this.txtConfigDir.Name = "txtConfigDir";
            this.txtConfigDir.Size = new System.Drawing.Size(284, 25);
            this.txtConfigDir.TabIndex = 8;
            // 
            // ScannerSettingsForm
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(9F, 21F);
            this.ClientSize = new System.Drawing.Size(523, 308);
            this.Controls.Add(this.txtConfigDir);
            this.Controls.Add(this.btnConfigDirSel);
            this.Controls.Add(this.btnCancel);
            this.Controls.Add(this.btnOK);
            this.Controls.Add(this.txtGPULib);
            this.Controls.Add(this.btnGPULibSel);
            this.Controls.Add(this.txtGrabberLib);
            this.Controls.Add(this.btnGrabberLibSel);
            this.Controls.Add(this.txtStageLib);
            this.Controls.Add(this.btnStageLibSel);
            this.Controls.Add(this.txtScanSrvDataDir);
            this.Controls.Add(this.btnScanSrvDataDir);
            this.Controls.Add(this.txtDataDir);
            this.Controls.Add(this.btnDataDirSel);
            this.Controls.Add(this.txtLogDir);
            this.Controls.Add(this.btnLogDirSel);
            this.DialogCaption = "Machine settings for Scanner";
            this.ForeColor = System.Drawing.Color.DodgerBlue;
            this.Name = "ScannerSettingsForm";
            this.Load += new System.EventHandler(this.OnLoad);
            this.Controls.SetChildIndex(this.btnLogDirSel, 0);
            this.Controls.SetChildIndex(this.txtLogDir, 0);
            this.Controls.SetChildIndex(this.btnDataDirSel, 0);
            this.Controls.SetChildIndex(this.txtDataDir, 0);
            this.Controls.SetChildIndex(this.btnScanSrvDataDir, 0);
            this.Controls.SetChildIndex(this.txtScanSrvDataDir, 0);
            this.Controls.SetChildIndex(this.btnStageLibSel, 0);
            this.Controls.SetChildIndex(this.txtStageLib, 0);
            this.Controls.SetChildIndex(this.btnGrabberLibSel, 0);
            this.Controls.SetChildIndex(this.txtGrabberLib, 0);
            this.Controls.SetChildIndex(this.btnGPULibSel, 0);
            this.Controls.SetChildIndex(this.txtGPULib, 0);
            this.Controls.SetChildIndex(this.btnOK, 0);
            this.Controls.SetChildIndex(this.btnCancel, 0);
            this.Controls.SetChildIndex(this.btnConfigDirSel, 0);
            this.Controls.SetChildIndex(this.txtConfigDir, 0);
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private SySal.SySalNExTControls.SySalButton btnLogDirSel;
        private System.Windows.Forms.TextBox txtLogDir;
        private System.Windows.Forms.TextBox txtDataDir;
        private SySal.SySalNExTControls.SySalButton btnDataDirSel;
        private System.Windows.Forms.TextBox txtScanSrvDataDir;
        private SySal.SySalNExTControls.SySalButton btnScanSrvDataDir;
        private System.Windows.Forms.TextBox txtGPULib;
        private SySal.SySalNExTControls.SySalButton btnGPULibSel;
        private System.Windows.Forms.TextBox txtGrabberLib;
        private SySal.SySalNExTControls.SySalButton btnGrabberLibSel;
        private System.Windows.Forms.TextBox txtStageLib;
        private SySal.SySalNExTControls.SySalButton btnStageLibSel;
        private SySal.SySalNExTControls.SySalButton btnOK;
        private SySal.SySalNExTControls.SySalButton btnCancel;
        private System.Windows.Forms.FolderBrowserDialog FolderBrowserDlg;
        private System.Windows.Forms.OpenFileDialog OpenFileDlg;
        private System.Windows.Forms.ToolTip MachineSettingsToolTip;
        private System.Windows.Forms.TextBox txtConfigDir;
        private SySal.SySalNExTControls.SySalButton btnConfigDirSel;
    }
}
