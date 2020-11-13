using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using System.Runtime.Remoting;
using System.Runtime.Remoting.Channels;
using System.Runtime.Remoting.Channels.Tcp;

namespace SySal.DAQSystem.Drivers.CSScanDriver
{
    public partial class frmEasyInterrupt : Form
    {
        public frmEasyInterrupt()
        {
            InitializeComponent();
        }

        SySal.OperaDb.OperaDbConnection Conn = null;

        SySal.DAQSystem.BatchManager BM = null;

        System.Text.RegularExpressions.Regex ZoneEx = new System.Text.RegularExpressions.Regex(@"\s*(\S+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*");
        
        private void OnLoad(object sender, System.EventArgs e)
        {
            SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
            System.Runtime.Remoting.Channels.ChannelServices.RegisterChannel(new TcpChannel(), false);
            Conn = new SySal.OperaDb.OperaDbConnection(cred.DBServer, cred.DBUserName, cred.DBPassword);
            Conn.Open();
            System.Data.DataSet ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("select name from tb_machines where id_site = (select to_number(value) from opera.lz_sitevars where name = 'ID_SITE') and isbatchserver = 1", Conn, null).Fill(ds);
            foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                comboBatchManager.Items.Add(dr[0].ToString());
            new SySal.OperaDb.OperaDbCommand("alter session set nls_comp='LINGUISTIC'", Conn).ExecuteNonQuery();
            new SySal.OperaDb.OperaDbCommand("alter session set NLS_SORT='BINARY_CI'", Conn).ExecuteNonQuery();
        }

        private void OnBatchManagerSelected(object sender, System.EventArgs e)
        {
            comboProcOpId.Items.Clear();
            string addr = new SySal.OperaDb.OperaDbCommand("SELECT ADDRESS FROM TB_MACHINES WHERE NAME = '" + comboBatchManager.Text + "'", Conn, null).ExecuteScalar().ToString();
            BM = (SySal.DAQSystem.BatchManager)System.Runtime.Remoting.RemotingServices.Connect(typeof(SySal.DAQSystem.BatchManager), "tcp://" + addr + ":" + ((int)SySal.DAQSystem.OperaPort.BatchServer).ToString() + "/BatchManager.rem");
            long[] ids = BM.Operations;
            if (ids.Length == 0) return;
            string wherestr = ids[0].ToString();
            int i;
            for (i = 1; i < ids.Length; i++)
                wherestr += ", " + ids[i].ToString();
            System.Data.DataSet ds = new System.Data.DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("SELECT TB_PROC_OPERATIONS.ID FROM TB_PROC_OPERATIONS INNER JOIN TB_PROGRAMSETTINGS ON (TB_PROC_OPERATIONS.ID_PROGRAMSETTINGS = TB_PROGRAMSETTINGS.ID AND TB_PROGRAMSETTINGS.EXECUTABLE = 'CSScanDriver.exe') WHERE TB_PROC_OPERATIONS.ID IN (" + wherestr + ")", Conn, null).Fill(ds);
            foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                comboProcOpId.Items.Add(dr[0].ToString());
        }

        private void buttonCancel_Click(object sender, System.EventArgs e)
        {
            Close();
        }

        private void buttonSend_Click(object sender, System.EventArgs e)
        {
            string InterruptString = null;
            try
            {
                double MinX = Convert.ToDouble(textMinX.Text, System.Globalization.CultureInfo.InvariantCulture);
                double MaxX = Convert.ToDouble(textMaxX.Text, System.Globalization.CultureInfo.InvariantCulture);
                double MinY = Convert.ToDouble(textMinY.Text, System.Globalization.CultureInfo.InvariantCulture);
                double MaxY = Convert.ToDouble(textMaxY.Text, System.Globalization.CultureInfo.InvariantCulture);

                if ((MinX > MaxX) || (MinY > MaxY)) throw new Exception("area not valid");

                InterruptString = Convert.ToString("Zone " + MinX.ToString() + " " + MaxX.ToString() + " " + MinY.ToString() + " " + MaxY.ToString());
                System.Text.RegularExpressions.Match mz = ZoneEx.Match(InterruptString);
                if (mz.Success == false)
                {
                    InterruptString = null;
                    MessageBox.Show("area string not valid");
                    return;
                }
            }
            catch (Exception x)
            {
                InterruptString = null;
                MessageBox.Show(x.Message, "Error");
                return;

            }
            try
            {
                BM.Interrupt(SySal.OperaDb.Convert.ToInt64(comboProcOpId.Text), textOPERAUsername.Text, textOPERAPwd.Text, InterruptString);
                MessageBox.Show("Interrupt message:\r\n" + InterruptString, "Interrupt sent", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error sending interrupt", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void SendInterrupt(string interruptdata)
        {
            try
            {
                BM.Interrupt(SySal.OperaDb.Convert.ToInt64(comboProcOpId.Text), textOPERAUsername.Text, textOPERAPwd.Text, interruptdata);
                MessageBox.Show("Interrupt message:\r\n" + interruptdata, "Interrupt sent", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "Error sending interrupt", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

    }
}