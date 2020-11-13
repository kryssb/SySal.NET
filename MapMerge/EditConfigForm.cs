using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Processing.MapMerge
{
    public partial class EditConfigForm : Form
    {
        public EditConfigForm()
        {
            InitializeComponent();
        }

        public SySal.Processing.MapMerge.Configuration C;

        private bool ValidateDouble(TextBox txt, ref double v)
        {
            try
            {
                v = System.Convert.ToDouble(txt.Text, System.Globalization.CultureInfo.InvariantCulture);
                return true;
            }
            catch (Exception)
            {
                txt.Text = v.ToString(System.Globalization.CultureInfo.InvariantCulture);
                txt.Focus();
                return false;
            }
        }

        private bool ValidateInt(TextBox txt, ref int v)
        {
            try
            {                
                v = System.Convert.ToInt32(txt.Text);                
                return true;
            }
            catch (Exception)
            {
                txt.Text = v.ToString();
                txt.Focus();
                return false;
            }
        }

        private void btnOK_Click(object sender, EventArgs e)
        {
            if (ValidateAll(true))
            {
                DialogResult = DialogResult.OK;
                Close();
            }
        }

        private void OnLoad(object sender, EventArgs e)
        {
            chkFavorSpeed.Checked = C.FavorSpeedOverAccuracy;
            ValidateAll(false);            
        }

        private bool ValidateAll(bool stoponerror)
        {
            if (ValidateDouble(txtMapSize, ref C.MapSize) == false && stoponerror) return false;
            if (ValidateDouble(txtMaxOffset, ref C.MaxPosOffset) == false && stoponerror) return false;
            if (ValidateDouble(txtPosTolerance, ref C.PosTol) == false && stoponerror) return false;
            if (ValidateDouble(txtSlopeTolerance, ref C.SlopeTol) == false && stoponerror) return false;
            if (ValidateInt(txtMinMatches, ref C.MinMatches) == false && stoponerror) return false;
            C.FavorSpeedOverAccuracy = chkFavorSpeed.Checked;
            return true;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }
    }
}