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
    public partial class frmAddFromDb : Form
    {
        public String m_Result1;
        public String m_Result2;
        

        public frmAddFromDb()
        {
            InitializeComponent();
        }



        private void Filled3(object sender, EventArgs e)
        {
            
            btnConnect.Enabled = ((this.txtIdBrick.Text.Length) > 0);
        
        }

        private void btnConnect_Click(object sender, EventArgs e)
        {
            try
            {
                
                lvZeroCoordinate.Tag = GetData(txtIdBrick.Text);
                    
                if (lvZeroCoordinate.Tag != "1")
                 
                {
                    MessageBox.Show("No information on brick inserted", "Input Error");
                    this.txtIdBrick.Text = "";
                }
            }
            catch (Exception x)
            {
                MessageBox.Show(x.Message, "No valid brick", MessageBoxButtons.OK, MessageBoxIcon.Error);
                this.txtIdBrick.Text = "";
                Close();
            }

            
        }


        private string GetData(String BrickId)
        {
            string dataTag = "0";
            System.Data.DataSet ds = new DataSet();
            new OperaDbDataAdapter("select id, minx, maxx, miny, maxy, id_set, id_brick from tb_eventbricks where  id >= 3000000 and mod(id,1000000) = mod("+ "'" + BrickId + "'"+ ", 1000000)", SySal.OperaDb.Schema.DB).Fill(ds);
            lvZeroCoordinate.Items.Clear();

            foreach (DataRow dr in ds.Tables[0].Rows)
            {
                ListViewItem lvi = lvZeroCoordinate.Items.Add(dr[0].ToString());
                lvi.SubItems.Add(dr[1].ToString());
                lvi.SubItems.Add(dr[2].ToString());
                lvi.SubItems.Add(dr[3].ToString());
                lvi.SubItems.Add(dr[4].ToString());
                lvi.SubItems.Add(dr[5].ToString());
                lvi.SubItems.Add(dr[6].ToString());

                lvi.Tag = "1";
                dataTag = "1";
                

            }
            
            return dataTag;
        }

        private void btnOk_Click(object sender, EventArgs e)
        {
                if (lvZeroCoordinate.SelectedItems.Count == 1)
                {
                    this.m_Result1 = lvZeroCoordinate.SelectedItems[0].SubItems[1].Text;
                    this.m_Result2 = lvZeroCoordinate.SelectedItems[0].SubItems[3].Text;
                    this.txtIdBrick.Text = "";
                    lvZeroCoordinate.Items.Clear();
                }
                /*else
                {
                    MessageBox.Show("you must selected a row");
                }*/

                txtIdBrick.Text = "";
                lvZeroCoordinate.Items.Clear();
                DialogResult = DialogResult.OK;
                Close();
        }


        private void btnClear_Click(object sender, EventArgs e)
        {
            this.txtIdBrick.Text = "";
            lvZeroCoordinate.Items.Clear();
            btnOk.Enabled = false;
            

        }

        private void lvZeroCoordinate_DoubleClick(object sender, EventArgs e)
        {
            btnOk.Enabled = true;
        }


    }

       
}