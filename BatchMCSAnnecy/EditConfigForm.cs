using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Windows.Forms;

namespace SySal.Processing.MCSAnnecy
{
    public partial class EditConfigForm : Form
    {
        public EditConfigForm()
        {
            InitializeComponent();
        }

        public SySal.Processing.MCSAnnecy.Configuration C;

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
            ValidateAll(false);
            if (C.IgnoreLongitudinal == C.IgnoreTransverse) rdUse3D.Checked = true;
            else if (C.IgnoreTransverse) rdUseLongitudinal.Checked = true;
            else rdUseTransverse.Checked = true;
        }

        private bool ValidateAll(bool stoponerror)
        {
            if (ValidateDouble(txtRadLen, ref C.RadiationLength) == false && stoponerror) return false;
            if (ValidateDouble(txtDT0, ref C.SlopeError3D_0) == false && stoponerror) return false;
            if (ValidateDouble(txtDT1, ref C.SlopeError3D_1) == false && stoponerror) return false;
            if (ValidateDouble(txtDT2, ref C.SlopeError3D_2) == false && stoponerror) return false;
            if (ValidateDouble(txtDTx0, ref C.SlopeErrorLong_0) == false && stoponerror) return false;
            if (ValidateDouble(txtDTx1, ref C.SlopeErrorLong_1) == false && stoponerror) return false;
            if (ValidateDouble(txtDTx2, ref C.SlopeErrorLong_2) == false && stoponerror) return false;
            if (ValidateDouble(txtDTy0, ref C.SlopeErrorTransv_0) == false && stoponerror) return false;
            if (ValidateDouble(txtDTy1, ref C.SlopeErrorTransv_1) == false && stoponerror) return false;
            if (ValidateDouble(txtDTy2, ref C.SlopeErrorTransv_2) == false && stoponerror) return false;
            if (ValidateInt(txtMinEntries, ref C.MinEntries) == false && stoponerror) return false;
            if (rdUseTransverse.Checked)
            {
                C.IgnoreLongitudinal = true;
                C.IgnoreTransverse = false;
            }
            else if (rdUseLongitudinal.Checked)
            {
                C.IgnoreTransverse = true;
                C.IgnoreLongitudinal = false;
            }
            else
            {
                C.IgnoreLongitudinal = C.IgnoreTransverse = false;
            }
            return true;
        }

        private void btnCancel_Click(object sender, EventArgs e)
        {
            DialogResult = DialogResult.Cancel;
            Close();
        }
    }
}