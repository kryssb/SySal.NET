using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SetCSScanResult
{
    public partial class MainForm : Form
    {
        SySal.OperaDb.OperaDbConnection Conn;

        SySal.OperaDb.OperaDbTransaction Trans;

        SySal.OperaDb.OperaDbCredentials Cred;

        long IdUser;

        public MainForm()
        {
            InitializeComponent();
            Conn = (Cred = SySal.OperaDb.OperaDbCredentials.CreateFromRecord()).Connect();
            Conn.Open();
            IdUser = SySal.OperaDb.ComputingInfrastructure.User.CheckLogin(Cred.OPERAUserName, Cred.OPERAPassword, Conn, Trans);
            Trans = Conn.BeginTransaction();
        }

        private void OnLoad(object sender, EventArgs e)
        {
            labelUser.Text = Cred.OPERAUserName;
        }

        void RefreshList() 
        {
            lvCSScanOps.Items.Clear();
            lvCSScanOps.BeginUpdate();
            System.Data.DataSet ds = new DataSet();
            new SySal.OperaDb.OperaDbDataAdapter("select id_cs_eventbrick, event, run, id_event from tb_b_bmm_brick_extractions where id_cs_eventbrick is not null and id_cs_eventbrick not in (select id_cs_eventbrick from tb_cs_results) order by id_cs_eventbrick", Conn, Trans).Fill(ds);
            foreach (System.Data.DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvCSScanOps.Items.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add("");
            }
            lvCSScanOps.EndUpdate();
        }

        private void btnRefresh_Click(object sender, EventArgs e)
        {
            RefreshList();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Trans.Rollback();
            Trans = Conn.BeginTransaction();
            RefreshList();
        }

        private void btnReset_Click(object sender, EventArgs e)
        {
            if (lvCSScanOps.SelectedItems.Count == 1)
            {
                lvCSScanOps.SelectedItems[0].SubItems[lvCSScanOps.SelectedItems[0].SubItems.Count - 1].Text = "";
            }
        }

        private void btnCandOK_Click(object sender, EventArgs e)
        {
            if (lvCSScanOps.SelectedItems.Count == 1)
            {
                lvCSScanOps.SelectedItems[0].SubItems[lvCSScanOps.SelectedItems[0].SubItems.Count - 1].Text = "CS_CAND_OK_DEVELOP";
            }
        }

        private void btnBlack_Click(object sender, EventArgs e)
        {
            if (lvCSScanOps.SelectedItems.Count == 1)
            {
                lvCSScanOps.SelectedItems[0].SubItems[lvCSScanOps.SelectedItems[0].SubItems.Count - 1].Text = "BLACK_CS_DEVELOP";
            }
        }

        private void btnNoCand_Click(object sender, EventArgs e)
        {
            if (lvCSScanOps.SelectedItems.Count == 1)
            {
                lvCSScanOps.SelectedItems[0].SubItems[lvCSScanOps.SelectedItems[0].SubItems.Count - 1].Text = "BACK_TO_DETECTOR";
            }
        }

        private void btnSubmitChanges_Click(object sender, EventArgs e)
        {
            if (MessageBox.Show("Submitted changes are irreversible. Are you sure?", "Confirmation needed", MessageBoxButtons.YesNo) == DialogResult.Yes)
            {
                foreach (ListViewItem lvi in lvCSScanOps.Items)
                {
                    if (lvi.SubItems[lvi.SubItems.Count - 1].Text.Trim().Length > 0)
                    {
                        SySal.OperaDb.OperaDbCommand cmd = new SySal.OperaDb.OperaDbCommand("CALL PC_ADD_PROC_OPERATION_BRICK(81000000000000002,81000000005140007," + IdUser + "," + lvi.SubItems[0].Text + ",NULL,SYSTIMESTAMP,'Fake CS scan of " + lvi.SubItems[0].Text + "',:newid)", Conn, Trans);
                        long op;
                        cmd.Parameters.Add("newid", SySal.OperaDb.OperaDbType.Long, ParameterDirection.Output);
                        cmd.ExecuteNonQuery();
                        op = SySal.OperaDb.Convert.ToInt64(cmd.Parameters[0].Value);
                        new SySal.OperaDb.OperaDbCommand("INSERT INTO TB_CS_RESULTS (ID_CSSCAN_PROCOPID, ID_CS_EVENTBRICK, RESULT_STATUS) VALUES (" + op + ", " + lvi.SubItems[0].Text + ", '" + lvi.SubItems[lvi.SubItems.Count - 1].Text + "')", Conn, Trans).ExecuteNonQuery();
                        new SySal.OperaDb.OperaDbCommand("CALL PC_SUCCESS_OPERATION(" + op + ", SYSTIMESTAMP)", Conn, Trans).ExecuteNonQuery();                        
                    }
                }
                Trans.Commit();
                Trans = Conn.BeginTransaction();
                RefreshList();
            }
        }
    }
}