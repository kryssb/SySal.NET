using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal;

namespace SySal.Executables.OperaLabManager
{
    public partial class PubForm : Form
    {
        string m_BrickId;

        OperaDb.OperaDbConnection Conn;

        OperaDb.OperaDbConnection OperaDbConn;

        public PubForm(OperaDb.OperaDbConnection conn, string brickid)
        {
            InitializeComponent();
            m_BrickId = brickid;
            Conn = conn;
        }

        private void OnLoad(object sender, EventArgs e)
        {
            try
            {
                System.Data.DataSet ds = new DataSet();
                new OperaDb.OperaDbDataAdapter("select id_processoperation, optype, notes from opera.xt_brick_tasks where id_eventbrick = " + m_BrickId + " and id_processoperation is not null order by id_processoperation", Conn).Fill(ds);
                foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
                {
                    ListViewItem lvi = lvProcOps.Items.Add(dr[0].ToString());
                    lvi.SubItems.Add(dr[1].ToString());
                    lvi.SubItems.Add(dr[2].ToString());
                    lvi.Checked = true;
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "DB error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void btnLogin_Click(object sender, EventArgs e)
        {
            SySal.OperaDb.OperaDbCredentials cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord();
            cred.DBUserName = "OPERAPUB";
            cred.DBPassword = txtPwd.Text;            
            try
            {
                OperaDbConn = cred.Connect();
                OperaDbConn.Open();
                txtPwd.Text = "";
                OperaDb.OperaDbDataReader rdr = new OperaDb.OperaDbCommand("SELECT DBLINK FROM PT_GENERAL ORDER BY DBLINK", OperaDbConn).ExecuteReader();
                while (rdr.Read())
                    cmbDBLink.Items.Add(rdr.GetString(0));
                if (cmbDBLink.Items.Count > 0) cmbDBLink.SelectedIndex = 0;
                rdr.Close();
                btnLogin.Enabled = false;
                btnOK.Enabled = true;
            }
            catch (Exception)
            {
                if (OperaDbConn != null)
                {
                    OperaDbConn.Close();
                    OperaDbConn = null;
                }
            }
        }

        private void OnClose(object sender, FormClosingEventArgs e)
        {
            if (OperaDbConn != null) OperaDbConn.Close();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            OperaDb.OperaDbTransaction trans = null;
            try
            {
                trans = OperaDbConn.BeginTransaction();
                OperaDb.OperaDbCommand cmd1 = new SySal.OperaDb.OperaDbCommand("call PP_ADD_PUBLICATION_JOB(:xid, '" + cmbDBLink.SelectedItem.ToString() + "', 'PUB_OPERATION', :jid)", OperaDbConn, null);
                cmd1.Parameters.Add("xid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Input);
                cmd1.Parameters.Add("jid", SySal.OperaDb.OperaDbType.Long, System.Data.ParameterDirection.Output);
                OperaDb.OperaDbCommand cmd2 = new SySal.OperaDb.OperaDbCommand("call PP_SCHEDULE_PUBLICATION_JOB(:jid)", OperaDbConn, null);
                cmd2.Parameters.Add("jid", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Input);
                foreach (ListViewItem lvi in lvProcOps.CheckedItems)
                {
                    cmd1.Parameters[0].Value = Convert.ToInt64(lvi.SubItems[0].Text);
                    cmd1.ExecuteNonQuery();
                    cmd2.Parameters[0].Value = cmd1.Parameters[1].Value;
                    cmd2.ExecuteNonQuery();
                }
                trans.Commit();
                MessageBox.Show("Operation publication correctly enqueued/scheduled", "Success", MessageBoxButtons.OK, MessageBoxIcon.Information);
            }
            catch (Exception x)
            {
                if (trans != null) trans.Rollback();
                MessageBox.Show(x.ToString(), "Cannot enqueue/schedule publication jobs", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            DialogResult = DialogResult.OK;
            Close();
        }
    }
}