using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;
using SySal.OperaDb;

namespace OperaBrickManager
{
    public partial class frmInsertMarkFromDB : Form
    {

        public frmInsertMarkFromDB()
        {
            InitializeComponent();
        }

        public string NumberBrick
        {
            get
            {
                return txtIDBrick.Text;
            }
            set
            {
                txtIDBrick.Text = value;
            }
        }
        OperaDbConnection DBConn = OperaDbCredentials.CreateFromRecord().Connect(); 
       
        public int CsBrick;

        private void btnConnectDB_Click(object sender, EventArgs e)
        {
            try
            {
                lvCSMarks.Tag = GetData(txtIDBrick.Text);             
                if (lvCSMarks.Tag != "1")
                 
                {
                    MessageBox.Show("No Information on Brick inserted");
                    this.txtIDBrick.Text = "";
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "No Valid Brick", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.txtIDBrick.Text = "";
                Close();
            }
            

        
        }

        private string GetData(String idBrick)
        {
            string dataTag = "0";
            System.Data.DataSet ds = new DataSet();
            new OperaDbDataAdapter("select id_eventbrick from tb_templatemarksets where id_eventbrick >= 3000000 and mod(id_eventbrick,1000000) = mod(" + "'" + idBrick + "'" +" , 1000000) and id_mark = '1'", SySal.OperaDb.Schema.DB).Fill(ds);
            this.lvCSMarks.Items.Clear();

            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvCSMarks.Items.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[0].ToString());
                lvi.Tag = "1";
                dataTag = "1";


            }

            return dataTag;

        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            Close();
        }

        private void btnClearAll_Click(object sender, EventArgs e)
        {
            this.txtIDBrick.Text = "";
            lvCSMarks.Items.Clear();
            btnOK.Enabled = false;
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (lvCSMarks.SelectedItems.Count == 1)
            {
                this.CsBrick = System.Convert.ToInt32(lvCSMarks.SelectedItems[0].SubItems[0].Text);
                lvCSMarks.Clear();
            }

            this.txtIDBrick.Text = "";
            lvCSMarks.Items.Clear();
            DialogResult = DialogResult.OK;
            Close();
        }

        private void lvCSMarks_SelectedIndexChanged(object sender, EventArgs e)
        {
            btnOK.Enabled = true;
        }
        

    }
}