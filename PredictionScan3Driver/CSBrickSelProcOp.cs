using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.DAQSystem.Drivers.PredictionScan3Driver
{
    internal partial class CSBrickSelProcOp : Form
    {
        public CSBrickSelProcOp()
        {
            InitializeComponent();
        }

        public System.Data.DataRowCollection Rows;

        private void OnLoad(object sender, EventArgs e)
        {
            foreach (System.Data.DataRow dr in Rows)
            {
                ListViewItem lvi = new ListViewItem(dr[0].ToString());
                int i;
                for (i = 1; i < dr.ItemArray.Length; i++)
                    lvi.SubItems.Add(dr[i].ToString());
                lvi.Tag = Convert.ToUInt64(dr[0]);
                lvOps.Items.Add(lvi);
            }
            ProcOp = 0;
            txtMinGrains.Text = MinGrains.ToString();
        }

        public ulong ProcOp = 0;

        public bool UseMicrotracks = false;

        public uint MinGrains = 18;

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (lvOps.SelectedItems.Count == 1)
            {
                ProcOp = Convert.ToUInt64(lvOps.SelectedItems[0].Tag);                
                UseMicrotracks = chkIncludeMicrotracks.Checked;
                DialogResult = DialogResult.OK;
                Close();
            }            
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void OnMinGrainsLeave(object sender, EventArgs e)
        {
            try
            {
                MinGrains = Convert.ToUInt32(txtMinGrains.Text);
            }
            catch (Exception)
            {
                txtMinGrains.Text = MinGrains.ToString();
            }
        }
    }
}