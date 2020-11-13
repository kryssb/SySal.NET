using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace OperaBrickManager
{
    public partial class AddBrickSetForm : Form
    {
        public AddBrickSetForm()
        {
            InitializeComponent();
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.OK;
            Close();
        }

        public int DefaultId;
        public int RangeMin;
        public int RangeMax;
        public string SetName;
        public string TablespaceExt;

        void CheckData()
        {
            bool enabled = false;
            try
            {
                DefaultId = -1;
                if ((SetName = txtSetName.Text.Trim()).Length == 0) return;
                if ((TablespaceExt = txtTablespaceExt.Text.Trim()).Length == 0) return;
                if ((RangeMin = SySal.OperaDb.Convert.ToInt32(txtRangeMin.Text)) > (RangeMax = SySal.OperaDb.Convert.ToInt32(txtRangeMax.Text))) return;
                if (RangeMin <= 0) return;
                if (RangeMax <= 0) return;
                if ((txtDefaultId.Text.Trim().Length > 0) && ((DefaultId = SySal.OperaDb.Convert.ToInt32(txtDefaultId.Text)) < RangeMin || DefaultId > RangeMax)) return;
                if (DefaultId < 0) DefaultId = RangeMin;
                enabled = true;
            }
            catch (Exception) { }
            finally
            {
                btnOK.Enabled = enabled;
            }

        }

        private void OnDataChanged(object sender, EventArgs e)
        {
            CheckData();
        }
    }
}