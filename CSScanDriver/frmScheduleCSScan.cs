using System;
using System.Drawing;
using System.Collections;
using System.ComponentModel;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    /// <summary>
    /// Summary description for frmScheduleCSScanDriver.
    /// </summary>
    public class frmScheduleCSScanDriver : System.Windows.Forms.Form
    {
        private System.Windows.Forms.Label lblBrick;
        private System.Windows.Forms.Label lblBatchManager;
        private System.Windows.Forms.Label lblConfigName;
        private System.Windows.Forms.Button btnSchedule;
        private System.Windows.Forms.Button btnCancel;
        private System.Windows.Forms.ComboBox cmbMachine;
        private System.Windows.Forms.ComboBox cmbBrick;
        private System.Windows.Forms.TextBox txtBatchManager;
        private System.Windows.Forms.ComboBox cmbConfig;
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.Container components = null;

        private SySal.OperaDb.OperaDbConnection Conn;

        private string BatchManager;

        private string InterruptString = null;

        internal long _programsettings;
        internal long _machine;
        internal long _brick = 0;
        internal long _event = 0;

        internal bool InsertScanningArea = false;

        private System.Windows.Forms.Label lblScanServer;
        private Label label1;
        private Label label2;
        private TextBox OPERAUserNameText;
        private TextBox OPERAPasswordText;
        private TextBox textNotes;
        private Label labelText;
        private Button buttonInsertArea;

        private string _exeName = "CSScanDriver.exe";

        private bool IsFormFilled()
        {

            return (_brick != 0 & _machine != 0 & _programsettings != 0 & BatchManager.Trim() != "" & ((InsertScanningArea == false) || (InsertScanningArea == true && InterruptString != null)) );
        }

        public frmScheduleCSScanDriver()
        {
            //
            // Required for Windows Form Designer support
            //
            InitializeComponent();
            //
            // TODO: Add any constructor code after InitializeComponent call
            //
            btnSchedule.Enabled = false;

            Conn = SySal.OperaDb.OperaDbCredentials.CreateFromRecord().Connect();
            Conn.Open();
            BatchManager = txtBatchManager.Text = Convert.ToString(new SySal.OperaDb.OperaDbCommand("select address from tb_machines where ISBATCHSERVER=1 and id_site = (SELECT value  FROM lz_sitevars where name='ID_SITE')", Conn).ExecuteScalar());
            cmbConfig.Enabled = false;
            Utilities.FillComboBox(cmbMachine, @"SELECT ID, NAME FROM TB_MACHINES WHERE ISSCANNINGSERVER=1 and id_site = (SELECT value  FROM lz_sitevars where name='ID_SITE' ) ORDER BY NAME", Conn);
            Utilities.FillComboBox(cmbBrick, @"SELECT UNIQUE TO_CHAR(ID_CS_EVENTBRICK), ID_CS_EVENTBRICK FROM vw_local_predictions ORDER BY ID_cs_EVENTBRICK", Conn);
            btnSchedule.Enabled = IsFormFilled();
        }
        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                if (components != null)
                {
                    components.Dispose();
                }
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code
        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.lblBrick = new System.Windows.Forms.Label();
            this.lblBatchManager = new System.Windows.Forms.Label();
            this.lblConfigName = new System.Windows.Forms.Label();
            this.btnSchedule = new System.Windows.Forms.Button();
            this.btnCancel = new System.Windows.Forms.Button();
            this.cmbMachine = new System.Windows.Forms.ComboBox();
            this.cmbBrick = new System.Windows.Forms.ComboBox();
            this.txtBatchManager = new System.Windows.Forms.TextBox();
            this.cmbConfig = new System.Windows.Forms.ComboBox();
            this.lblScanServer = new System.Windows.Forms.Label();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.OPERAUserNameText = new System.Windows.Forms.TextBox();
            this.OPERAPasswordText = new System.Windows.Forms.TextBox();
            this.textNotes = new System.Windows.Forms.TextBox();
            this.labelText = new System.Windows.Forms.Label();
            this.buttonInsertArea = new System.Windows.Forms.Button();
            this.SuspendLayout();
            // 
            // lblBrick
            // 
            this.lblBrick.Location = new System.Drawing.Point(8, 162);
            this.lblBrick.Name = "lblBrick";
            this.lblBrick.Size = new System.Drawing.Size(134, 13);
            this.lblBrick.TabIndex = 80;
            this.lblBrick.Text = "Brick";
            // 
            // lblBatchManager
            // 
            this.lblBatchManager.Location = new System.Drawing.Point(8, 15);
            this.lblBatchManager.Name = "lblBatchManager";
            this.lblBatchManager.Size = new System.Drawing.Size(134, 13);
            this.lblBatchManager.TabIndex = 2;
            this.lblBatchManager.Text = "Batch Manager:";
            // 
            // lblConfigName
            // 
            this.lblConfigName.Location = new System.Drawing.Point(8, 85);
            this.lblConfigName.Name = "lblConfigName";
            this.lblConfigName.Size = new System.Drawing.Size(134, 13);
            this.lblConfigName.TabIndex = 0;
            this.lblConfigName.Text = "Configuration name:";
            // 
            // btnSchedule
            // 
            this.btnSchedule.Enabled = false;
            this.btnSchedule.Location = new System.Drawing.Point(8, 313);
            this.btnSchedule.Name = "btnSchedule";
            this.btnSchedule.Size = new System.Drawing.Size(89, 23);
            this.btnSchedule.TabIndex = 10;
            this.btnSchedule.Text = "Schedule";
            this.btnSchedule.Click += new System.EventHandler(this.btnSchedule_Click);
            // 
            // btnCancel
            // 
            this.btnCancel.Location = new System.Drawing.Point(145, 313);
            this.btnCancel.Name = "btnCancel";
            this.btnCancel.Size = new System.Drawing.Size(89, 23);
            this.btnCancel.TabIndex = 11;
            this.btnCancel.Text = "Cancel";
            this.btnCancel.Click += new System.EventHandler(this.btnCancel_Click);
            // 
            // cmbMachine
            // 
            this.cmbMachine.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbMachine.Location = new System.Drawing.Point(216, 36);
            this.cmbMachine.Name = "cmbMachine";
            this.cmbMachine.Size = new System.Drawing.Size(200, 21);
            this.cmbMachine.TabIndex = 4;
            this.cmbMachine.SelectedIndexChanged += new System.EventHandler(this.cmbMachine_SelectedIndexChanged);
            // 
            // cmbBrick
            // 
            this.cmbBrick.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbBrick.Location = new System.Drawing.Point(216, 159);
            this.cmbBrick.Name = "cmbBrick";
            this.cmbBrick.Size = new System.Drawing.Size(200, 21);
            this.cmbBrick.TabIndex = 6;
            this.cmbBrick.SelectedIndexChanged += new System.EventHandler(this.cmbBrick_SelectedIndexChanged);
            // 
            // txtBatchManager
            // 
            this.txtBatchManager.Location = new System.Drawing.Point(8, 37);
            this.txtBatchManager.Name = "txtBatchManager";
            this.txtBatchManager.Size = new System.Drawing.Size(200, 20);
            this.txtBatchManager.TabIndex = 3;
            this.txtBatchManager.Leave += new System.EventHandler(this.txtBatchManager_Leave);
            // 
            // cmbConfig
            // 
            this.cmbConfig.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.cmbConfig.Location = new System.Drawing.Point(8, 111);
            this.cmbConfig.Name = "cmbConfig";
            this.cmbConfig.Size = new System.Drawing.Size(408, 21);
            this.cmbConfig.TabIndex = 14;
            this.cmbConfig.SelectedIndexChanged += new System.EventHandler(this.cmbConfig_SelectedIndexChanged);
            // 
            // lblScanServer
            // 
            this.lblScanServer.Location = new System.Drawing.Point(216, 15);
            this.lblScanServer.Name = "lblScanServer";
            this.lblScanServer.Size = new System.Drawing.Size(134, 13);
            this.lblScanServer.TabIndex = 83;
            this.lblScanServer.Text = "Scan Server:";
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(8, 205);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(95, 13);
            this.label1.TabIndex = 86;
            this.label1.Text = "OPERA Username";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(216, 205);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(93, 13);
            this.label2.TabIndex = 87;
            this.label2.Text = "OPERA Password";
            // 
            // OPERAUserNameText
            // 
            this.OPERAUserNameText.Location = new System.Drawing.Point(8, 223);
            this.OPERAUserNameText.Name = "OPERAUserNameText";
            this.OPERAUserNameText.Size = new System.Drawing.Size(200, 20);
            this.OPERAUserNameText.TabIndex = 7;
            // 
            // OPERAPasswordText
            // 
            this.OPERAPasswordText.Location = new System.Drawing.Point(216, 223);
            this.OPERAPasswordText.Name = "OPERAPasswordText";
            this.OPERAPasswordText.Size = new System.Drawing.Size(200, 20);
            this.OPERAPasswordText.TabIndex = 8;
            this.OPERAPasswordText.UseSystemPasswordChar = true;
            // 
            // textNotes
            // 
            this.textNotes.Location = new System.Drawing.Point(8, 275);
            this.textNotes.Name = "textNotes";
            this.textNotes.Size = new System.Drawing.Size(408, 20);
            this.textNotes.TabIndex = 9;
            // 
            // labelText
            // 
            this.labelText.AutoSize = true;
            this.labelText.Location = new System.Drawing.Point(8, 257);
            this.labelText.Name = "labelText";
            this.labelText.Size = new System.Drawing.Size(35, 13);
            this.labelText.TabIndex = 5;
            this.labelText.Text = "Notes";
            // 
            // buttonInsertArea
            // 
            this.buttonInsertArea.Enabled = false;
            this.buttonInsertArea.Location = new System.Drawing.Point(284, 313);
            this.buttonInsertArea.Name = "buttonInsertArea";
            this.buttonInsertArea.Size = new System.Drawing.Size(132, 23);
            this.buttonInsertArea.TabIndex = 88;
            this.buttonInsertArea.Text = "Insert scanning area";
            this.buttonInsertArea.UseVisualStyleBackColor = true;
            this.buttonInsertArea.Click += new System.EventHandler(this.buttonInsertArea_Click);
            // 
            // frmScheduleCSScanDriver
            // 
            this.AutoScaleBaseSize = new System.Drawing.Size(5, 13);
            this.ClientSize = new System.Drawing.Size(420, 349);
            this.Controls.Add(this.buttonInsertArea);
            this.Controls.Add(this.cmbBrick);
            this.Controls.Add(this.textNotes);
            this.Controls.Add(this.OPERAPasswordText);
            this.Controls.Add(this.OPERAUserNameText);
            this.Controls.Add(this.cmbMachine);
            this.Controls.Add(this.txtBatchManager);
            this.Controls.Add(this.cmbConfig);
            this.Controls.Add(this.labelText);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.lblScanServer);
            this.Controls.Add(this.lblBrick);
            this.Controls.Add(this.lblBatchManager);
            this.Controls.Add(this.lblConfigName);
            this.Controls.Add(this.btnSchedule);
            this.Controls.Add(this.btnCancel);
            this.Name = "frmScheduleCSScanDriver";
            this.Text = "Schedule a Plate doublet scan";
            this.ResumeLayout(false);
            this.PerformLayout();

        }
        #endregion

        private void btnCancel_Click(object sender, System.EventArgs e)
        {
            Close();
        }

        private void cmbConfig_SelectedIndexChanged(object sender, System.EventArgs e)
        {
            _programsettings = ((Utilities.ConfigItem)cmbConfig.SelectedItem).Id;

            string settings = Convert.ToString(new SySal.OperaDb.OperaDbCommand("select settings from tb_programsettings where id = " + _programsettings, Conn).ExecuteScalar());
            System.Xml.Serialization.XmlSerializer xmls = new System.Xml.Serialization.XmlSerializer(typeof(CSScanDriverSettings));
            CSScanDriverSettings ProgSettings = (CSScanDriverSettings)xmls.Deserialize(new System.IO.StringReader(settings));
            xmls = null;

            InterruptString = null;
            if (ProgSettings.WaitForScanningArea == true)
            {
                InsertScanningArea = buttonInsertArea.Enabled = true;
            }
            else
            {
                InsertScanningArea = buttonInsertArea.Enabled = false;
            }

            btnSchedule.Enabled = IsFormFilled();
        }

        private void txtBatchManager_Leave(object sender, System.EventArgs e)
        {
            BatchManager = txtBatchManager.Text;
            btnSchedule.Enabled = IsFormFilled();
        }

        private void cmbBrick_SelectedIndexChanged(object sender, System.EventArgs e)
        {
            _brick = ((Utilities.ConfigItem)cmbBrick.SelectedItem).Id;
            btnSchedule.Enabled = IsFormFilled();
        }

        private void cmbMachine_SelectedIndexChanged(object sender, System.EventArgs e)
        {
            _machine = ((Utilities.ConfigItem)cmbMachine.SelectedItem).Id;

            string machinename = ((Utilities.ConfigItem)cmbMachine.SelectedItem).Name;
            cmbConfig.Enabled = true;
//TODO:            Utilities.FillComboBox(cmbConfig, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '" + _exeName + "' and description like '%" + machinename + "%' ORDER BY ID DESC", Conn);
            Utilities.FillComboBox(cmbConfig, @"SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS" + 
                " inner join lz_programsettings on (lz_programsettings.id_programsettings = tb_programsettings.id)" + 
                " WHERE EXECUTABLE LIKE '" + _exeName + "' and description like '%" + machinename + "%' ORDER BY ID DESC", Conn);
            
            btnSchedule.Enabled = IsFormFilled();
        }

        private void btnEditConfig_Click(object sender, System.EventArgs e)
        {
            long newConfig = frmConfig.GetConfig(_programsettings, Conn);
            if (newConfig != 0) _programsettings = newConfig;
            Utilities.FillComboBox(cmbConfig, "SELECT ID, DESCRIPTION FROM TB_PROGRAMSETTINGS WHERE EXECUTABLE LIKE '" + _exeName + "'", Conn);
            Utilities.SelectId(cmbConfig, _programsettings);
        }

        private void btnSchedule_Click(object sender, System.EventArgs e)
        {
            int nmarks = Convert.ToInt32(new SySal.OperaDb.OperaDbCommand("SELECT COUNT(*) FROM tb_templatemarksets WHERE id_eventbrick = " + _brick, Conn).ExecuteScalar());
            if (nmarks < 2)
            {
                MessageBox.Show("Scanning not allowed! Brick " + _brick + " has " + nmarks + " marks.", "Warning", MessageBoxButtons.OK, MessageBoxIcon.Hand);
                return;
            }

            System.Runtime.Remoting.Channels.Tcp.TcpChannel ch = new System.Runtime.Remoting.Channels.Tcp.TcpChannel();
            System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(ch, false);
            SySal.DAQSystem.BatchManager BM = (SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + BatchManager + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer).ToString() + "/BatchManager.rem");

            SySal.DAQSystem.Drivers.VolumeOperationInfo xinfo = new SySal.DAQSystem.Drivers.VolumeOperationInfo();
            xinfo.BrickId = _brick;

            SySal.DAQSystem.Drivers.TaskStartupInfo ts = new TaskStartupInfo();
            SySal.DAQSystem.Drivers.TaskStartupInfo tsinfo = xinfo;
            tsinfo.MachineId = _machine;
            tsinfo.OPERAUsername = OPERAUserNameText.Text;
            tsinfo.OPERAPassword = OPERAPasswordText.Text;
            tsinfo.ProgramSettingsId = _programsettings;
            tsinfo.MachineId = _machine;
            tsinfo.Notes = textNotes.Text + " " + InterruptString;

            System.Data.DataSet ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("select avg(pred_localx), avg(pred_localy) from vw_local_predictions where id_cs_eventbrick = " + _brick, Conn).Fill(ds);
            double x = Convert.ToSingle(ds.Tables[0].Rows[0][0]);
            double y = Convert.ToSingle(ds.Tables[0].Rows[0][1]);

            try
            {
                long parent_id = 0;
                try
                {
                    parent_id = Convert.ToInt64(new SySal.OperaDb.OperaDbCommand("select tb_proc_operations.id from tb_proc_operations inner join tb_programsettings on (tb_proc_operations.id_programsettings= tb_programsettings.id) where executable='CSHumanDriver.exe' and success='R' and id_eventbrick=" + _brick, Conn).ExecuteScalar());
                }
                catch { }

                long op = 0;
                if (MessageBox.Show("PUT OIL!\nArea centered around:\nX = " + x + "\nY = " + y, "Warning", MessageBoxButtons.YesNo, MessageBoxIcon.Warning, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
                    op = BM.Start(parent_id, tsinfo);

                if (InterruptString != null)
                    BM.Interrupt(op, tsinfo.OPERAUsername, tsinfo.OPERAPassword, InterruptString);
            }
            catch (Exception ex)
            {
                System.Runtime.Remoting.Channels.ChannelServices.UnregisterChannel(ch);
                MessageBox.Show(@"Failed to connect to the BatchManager... " + ex.Message, "", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            Close();
        }

        private void buttonInsertArea_Click(object sender, EventArgs e)
        {
            AreaForm form = new AreaForm();
            if (form.ShowDialog() == DialogResult.OK)
            {
                InterruptString = form.InterruptString;
            }

            btnSchedule.Enabled = IsFormFilled();
        }
    }
}
